"""Lazy Hagen PWM-MAC adapter for the converted toy classifiers."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import torch

from .toy import ConvertedToyModel, QuantizedAffine


HagenMode = Literal["mock", "hardware"]
HagenTiling = Literal["auto", "high-level", "host-128"]


def file_sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class HagenConfig:
    """Execution and provenance settings for one analog PWM stage."""

    mode: HagenMode = "mock"
    calibration_path: Path | None = None
    allow_environment_calibration: bool = False
    tiling: HagenTiling = "auto"
    tile_size: int = 128
    wait_between_events: int = 5
    num_sends: int | None = None
    hidden_shift: int = 1

    def __post_init__(self) -> None:
        if self.mode not in ("mock", "hardware"):
            raise ValueError("unsupported Hagen mode")
        if self.tiling not in ("auto", "high-level", "host-128"):
            raise ValueError("unsupported Hagen tiling mode")
        if self.tile_size <= 0 or self.tile_size > 128:
            raise ValueError("Hagen tile_size must lie in [1, 128]")
        if self.hidden_shift < 0 or self.hidden_shift > 7:
            raise ValueError("Hagen hidden_shift must lie in [0, 7]")
        if self.mode == "hardware":
            if self.calibration_path is None and not self.allow_environment_calibration:
                raise ValueError("formal Hagen hardware runs require calibration_path")
            if self.calibration_path is not None and not self.calibration_path.is_file():
                raise FileNotFoundError(self.calibration_path)


@dataclass(frozen=True)
class HagenResult:
    """One Hagen stage output plus physical execution metadata."""

    value: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)


class HagenPWMBackend:
    """Execute converted affine stages with hxtorch perceptron primitives."""

    def __init__(self, config: HagenConfig) -> None:
        self.config = config

    @staticmethod
    def dependencies_available() -> bool:
        try:
            import_module("hxtorch")
            import_module("hxtorch.perceptron")
        except ImportError:
            return False
        return True

    @staticmethod
    def _augment(value: torch.Tensor) -> torch.Tensor:
        constant = torch.full(
            (value.shape[0], 1),
            31.0,
            dtype=torch.float32,
            device=value.device,
        )
        return torch.cat((value.to(torch.float32), constant), dim=1)

    def _initialize_hardware(self, hxtorch: Any) -> bool:
        if self.config.mode == "mock":
            return False
        if self.config.calibration_path is None:
            hxtorch.init_hardware()
            return True
        errors: list[str] = []
        calibration_path = str(self.config.calibration_path)
        calibration_type = getattr(hxtorch, "CalibrationPath", None)
        candidates: list[Any] = []
        if callable(calibration_type):
            candidates.append(calibration_type(calibration_path))
        candidates.append(calibration_path)
        for candidate in candidates:
            try:
                hxtorch.init_hardware(candidate)
                return True
            except (TypeError, RuntimeError, ValueError) as error:
                errors.append(f"{type(candidate).__name__}: {error}")
        raise RuntimeError(
            "installed hxtorch could not initialize the explicit Hagen calibration: "
            + "; ".join(errors)
        )

    def _high_level_linear(
        self,
        hxtorch: Any,
        value: torch.Tensor,
        affine: QuantizedAffine,
        *,
        avg: int,
    ) -> torch.Tensor:
        layer = hxtorch.perceptron.nn.Linear(
            value.shape[1],
            affine.weight_with_bias.shape[0],
            bias=False,
            num_sends=self.config.num_sends,
            wait_between_events=self.config.wait_between_events,
            mock=self.config.mode == "mock",
            avg=avg,
        )
        layer.weight.data.copy_(affine.weight_with_bias.to(layer.weight.dtype))
        return layer(value)

    def _host_tiled_linear(
        self,
        hxtorch: Any,
        value: torch.Tensor,
        affine: QuantizedAffine,
        *,
        avg: int,
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        if avg != 1:
            raise RuntimeError("host-128 Hagen tiling does not implement Linear.avg")
        partials: list[torch.Tensor] = []
        schedule: list[dict[str, Any]] = []
        weight = affine.weight_with_bias.to(torch.float32)
        for start in range(0, value.shape[1], self.config.tile_size):
            stop = min(value.shape[1], start + self.config.tile_size)
            partial = hxtorch.perceptron.matmul(
                value[:, start:stop],
                weight[:, start:stop].T,
                num_sends=self.config.num_sends or 1,
                wait_between_events=self.config.wait_between_events,
                mock=self.config.mode == "mock",
            )
            partial = partial.detach().cpu()
            partials.append(partial.to(torch.int32))
            schedule.append(
                {
                    "start": start,
                    "stop": stop,
                    "minimum": float(partial.min()),
                    "maximum": float(partial.max()),
                    "saturation_rate": float(
                        ((partial <= -128) | (partial >= 127)).float().mean()
                    ),
                }
            )
        accumulated = torch.stack(partials).sum(dim=0)
        return accumulated.clamp(-128, 127).to(torch.float32), schedule

    def _linear(
        self,
        hxtorch: Any,
        value: torch.Tensor,
        affine: QuantizedAffine,
        *,
        avg: int,
    ) -> tuple[torch.Tensor, str, list[dict[str, Any]]]:
        if self.config.tiling == "high-level":
            return self._high_level_linear(hxtorch, value, affine, avg=avg), "high-level", []
        if self.config.tiling == "host-128":
            output, schedule = self._host_tiled_linear(hxtorch, value, affine, avg=avg)
            return output, "host-128", schedule
        try:
            return self._high_level_linear(hxtorch, value, affine, avg=avg), "high-level", []
        except (RuntimeError, ValueError) as high_level_error:
            if value.shape[1] <= 128:
                raise
            try:
                output, schedule = self._host_tiled_linear(
                    hxtorch, value, affine, avg=avg
                )
            except Exception as tiled_error:
                raise RuntimeError(
                    f"Hagen high-level and host tiling failed: high-level={high_level_error}; "
                    f"host-128={tiled_error}"
                ) from tiled_error
            return output, "host-128", schedule

    def _execute(
        self,
        value: torch.Tensor,
        affine: QuantizedAffine,
        *,
        avg: int,
        converting_relu_shift: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if not self.dependencies_available():
            raise RuntimeError(
                "hxtorch.perceptron is unavailable; use the EBRAINS-experimental kernel"
            )
        hxtorch = import_module("hxtorch")
        initialized = False
        started = perf_counter()
        try:
            initialized = self._initialize_hardware(hxtorch)
            output, tiling, schedule = self._linear(
                hxtorch,
                value,
                affine,
                avg=avg,
            )
            converting_relu = None
            if converting_relu_shift is not None:
                try:
                    output = hxtorch.perceptron.converting_relu(
                        output,
                        shift=converting_relu_shift,
                        mock=self.config.mode == "mock",
                    )
                except (AttributeError, TypeError):
                    output = torch.round(
                        output.to(torch.float64) / (2 ** converting_relu_shift)
                    ).clamp(0, 31)
                    converting_relu = "host-fallback"
                else:
                    converting_relu = "hxtorch"
            chip_identifier = None
            get_identifier = getattr(hxtorch, "get_unique_identifier", None)
            if callable(get_identifier) and self.config.mode == "hardware":
                chip_identifier = [str(item) for item in get_identifier()]
            return output.detach().cpu(), {
                "backend": f"hagen-{self.config.mode}",
                "hxtorch_version": getattr(hxtorch, "__version__", "unknown"),
                "chip_identifier": chip_identifier,
                "calibration_path": (
                    str(self.config.calibration_path)
                    if self.config.calibration_path is not None
                    else None
                ),
                "calibration_sha256": file_sha256(self.config.calibration_path),
                "avg": avg,
                "tiling": tiling,
                "tile_schedule": schedule,
                "input_shape": list(value.shape),
                "output_shape": list(output.shape),
                "elapsed_s": perf_counter() - started,
                "host_accumulated": tiling == "host-128",
                "converting_relu": converting_relu,
            }
        finally:
            if initialized:
                hxtorch.release_hardware()

    def first_layer(
        self,
        converted: ConvertedToyModel,
        input_uint5: torch.Tensor,
        *,
        avg: int = 1,
    ) -> HagenResult:
        """Execute first PWM affine and convert its output to hidden UInt5."""
        augmented = self._augment(input_uint5)
        hidden, metadata = self._execute(
            augmented,
            converted.first,
            avg=avg,
            converting_relu_shift=self.config.hidden_shift,
        )
        metadata["integer_reference_hidden_shift"] = converted.manifest.hidden_shift
        metadata["hagen_hidden_shift"] = self.config.hidden_shift
        return HagenResult(hidden.detach().cpu().to(torch.int32), metadata)

    def output_layer(
        self,
        converted: ConvertedToyModel,
        hidden_uint5: torch.Tensor,
    ) -> HagenResult:
        """Execute the output PWM affine and return Int8 logits."""
        augmented = self._augment(hidden_uint5)
        raw, metadata = self._execute(augmented, converted.second, avg=1)
        # A physical perceptron Linear already returns Int8.  The reference model's
        # output shift applies only to its int32 software accumulator.
        logits = torch.round(raw.to(torch.float64)).clamp(-128, 127)
        metadata["integer_reference_output_shift"] = converted.manifest.output_shift
        metadata["hagen_output_shift_applied"] = 0
        return HagenResult(logits.to(torch.int8), metadata)

    def recommend_hidden_shift(
        self,
        converted: ConvertedToyModel,
        input_uint5: torch.Tensor,
        target_hidden_uint5: torch.Tensor,
        *,
        candidates: tuple[int, ...] = (0, 1, 2, 3, 4),
    ) -> dict[str, Any]:
        """Select a label-free Hagen ReLU shift against converted activations."""
        if input_uint5.shape[0] != target_hidden_uint5.shape[0]:
            raise ValueError("shift calibration inputs and targets must share samples")
        augmented = self._augment(input_uint5)
        rows: list[dict[str, Any]] = []
        target = target_hidden_uint5.to(torch.float64)
        scale = float(target.square().mean()) + 1.0e-12
        for shift in candidates:
            output, metadata = self._execute(
                augmented,
                converted.first,
                avg=1,
                converting_relu_shift=shift,
            )
            output = output.to(torch.float64)
            mse = float((output - target).square().mean()) / scale
            saturation = float((output >= 31).float().mean())
            rows.append(
                {
                    "shift": shift,
                    "normalized_mse": mse,
                    "saturation_rate": saturation,
                    "score": mse + max(0.0, saturation - 0.01) * 100.0,
                    "metadata": metadata,
                }
            )
        selected = min(rows, key=lambda row: float(row["score"]))
        return {"selected": selected, "candidates": rows}

    def probe(self, converted: ConvertedToyModel) -> dict[str, Any]:
        """Probe native 128-row and architecture-sized first-layer execution."""
        probes: list[dict[str, Any]] = []
        for features in sorted({128, converted.architecture.input_features + 1}):
            features = min(features, converted.first.weight_with_bias.shape[1])
            value = torch.full((2, features), 15.0)
            affine = QuantizedAffine(
                converted.first.weight_with_bias[:, :features],
                converted.first.scale,
            )
            started = perf_counter()
            try:
                result, metadata = self._execute(value, affine, avg=1)
            except Exception as error:
                probes.append(
                    {
                        "features": features,
                        "success": False,
                        "error": f"{type(error).__name__}: {error}",
                        "elapsed_s": perf_counter() - started,
                    }
                )
            else:
                probes.append(
                    {
                        "features": features,
                        "success": True,
                        "minimum": float(result.min()),
                        "maximum": float(result.max()),
                        "saturation_rate": float(
                            ((result <= -128) | (result >= 127)).float().mean()
                        ),
                        **metadata,
                    }
                )
        return {"probes": probes, "config": self.config.__dict__}
