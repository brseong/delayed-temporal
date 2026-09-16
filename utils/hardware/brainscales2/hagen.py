"""Lazy Hagen PWM-MAC adapter for the converted toy classifiers."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator, Literal

import torch

from .toy import ConvertedToyModel, QuantizedAffine, ToyActivation
from utils.transforms.types import Potential, PotentialBounds


HagenMode = Literal["mock", "hardware"]
HagenTiling = Literal["auto", "high-level", "host-128"]
ReLUBoundary = Literal["implicit-lower-bound-host", "hagen-converting-relu"]


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


@dataclass(frozen=True)
class HagenFidelityResult:
    """Paired integer-reference and repeated physical affine observations."""

    ideal_first_accumulator: torch.Tensor
    physical_first_raw: torch.Tensor
    ideal_hidden_uint5: torch.Tensor
    physical_hidden_uint5: torch.Tensor
    ideal_output_int8: torch.Tensor
    physical_output_int8: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.ideal_first_accumulator.ndim != 2:
            raise ValueError("ideal first accumulator must have shape [sample, channel]")
        expected_first = (
            self.physical_first_raw.ndim == 3
            and self.physical_first_raw.shape[1:] == self.ideal_first_accumulator.shape
        )
        if not expected_first:
            raise ValueError(
                "physical first observations must have shape [trial, sample, channel]"
            )
        if self.ideal_hidden_uint5.shape != self.ideal_first_accumulator.shape:
            raise ValueError("ideal hidden output does not match first accumulator")
        if self.physical_hidden_uint5.shape != self.physical_first_raw.shape:
            raise ValueError("physical hidden observations do not match raw observations")
        if self.ideal_output_int8.ndim != 2:
            raise ValueError("ideal output must have shape [sample, channel]")
        if (
            self.physical_output_int8.ndim != 3
            or self.physical_output_int8.shape[0] != self.physical_first_raw.shape[0]
            or self.physical_output_int8.shape[1:] != self.ideal_output_int8.shape
        ):
            raise ValueError(
                "physical output observations must have shape [trial, sample, channel]"
            )


@dataclass(frozen=True)
class HagenAffineCorrection:
    """Label-free channelwise affine correction fitted on a calibration split."""

    calibration_samples: int
    evaluation_samples: int
    first_gain: torch.Tensor
    first_offset: torch.Tensor
    hidden_gain: torch.Tensor
    hidden_offset: torch.Tensor
    output_gain: torch.Tensor
    output_offset: torch.Tensor
    corrected_first_accumulator: torch.Tensor
    corrected_hidden_from_raw_uint5: torch.Tensor
    corrected_hidden_from_code_uint5: torch.Tensor
    corrected_output: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.calibration_samples <= 0 or self.evaluation_samples <= 0:
            raise ValueError("affine correction requires non-empty splits")
        if self.corrected_first_accumulator.ndim != 3:
            raise ValueError("corrected first-affine observations must be three-dimensional")
        if self.corrected_hidden_from_raw_uint5.ndim != 3:
            raise ValueError("corrected hidden observations must be three-dimensional")
        if (
            self.corrected_hidden_from_code_uint5.shape
            != self.corrected_hidden_from_raw_uint5.shape
        ):
            raise ValueError("corrected hidden paths must have matching shapes")
        if self.corrected_output.ndim != 3:
            raise ValueError("corrected output observations must be three-dimensional")


class HagenPWMBackend:
    """Execute converted affine stages with hxtorch perceptron primitives."""

    def __init__(self, config: HagenConfig) -> None:
        self.config = config
        self._active_hxtorch: Any | None = None

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

    @contextmanager
    def hardware_session(self) -> Iterator[None]:
        """Keep one hxtorch initialization active across related PWM calls."""
        if self._active_hxtorch is not None:
            yield
            return
        if not self.dependencies_available():
            raise RuntimeError(
                "hxtorch.perceptron is unavailable; use the EBRAINS-experimental kernel"
            )
        hxtorch = import_module("hxtorch")
        initialized = False
        try:
            initialized = self._initialize_hardware(hxtorch)
            self._active_hxtorch = hxtorch
            yield
        finally:
            self._active_hxtorch = None
            if initialized:
                hxtorch.release_hardware()

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
        relu_boundary: ReLUBoundary | None = None,
        activation_shift: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if (relu_boundary is None) != (activation_shift is None):
            raise ValueError("relu_boundary and activation_shift must be specified together")
        if not self.dependencies_available():
            raise RuntimeError(
                "hxtorch.perceptron is unavailable; use the EBRAINS-experimental kernel"
            )
        hxtorch = self._active_hxtorch or import_module("hxtorch")
        initialized = False
        started = perf_counter()
        try:
            if self._active_hxtorch is None:
                initialized = self._initialize_hardware(hxtorch)
            output, tiling, schedule = self._linear(
                hxtorch,
                value,
                affine,
                avg=avg,
            )
            activation_metadata: dict[str, Any] = {
                "relu_boundary": None,
                "converting_relu": None,
                "host_mediated_lower_bound": False,
            }
            if relu_boundary == "hagen-converting-relu":
                assert activation_shift is not None
                try:
                    output = hxtorch.perceptron.converting_relu(
                        output,
                        shift=activation_shift,
                        mock=self.config.mode == "mock",
                    )
                except (AttributeError, TypeError):
                    output = torch.round(
                        output.to(torch.float64) / (2 ** activation_shift)
                    ).clamp(0, 31)
                    activation_metadata["converting_relu"] = "host-fallback"
                else:
                    activation_metadata["converting_relu"] = "hxtorch"
                activation_metadata.update(
                    {
                        "relu_boundary": relu_boundary,
                        "activation_shift": activation_shift,
                    }
                )
            elif relu_boundary == "implicit-lower-bound-host":
                assert activation_shift is not None
                output, lower_bound_metadata = self._implicit_lower_bound_uint5(
                    output,
                    shift=activation_shift,
                )
                activation_metadata.update(lower_bound_metadata)
            elif relu_boundary is not None:
                raise ValueError(f"unsupported ReLU boundary: {relu_boundary}")
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
                **activation_metadata,
            }
        finally:
            if initialized:
                hxtorch.release_hardware()

    @staticmethod
    def _implicit_lower_bound_uint5(
        raw_preactivation: torch.Tensor,
        *,
        shift: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Lower raw PWM preactivations at the cached ``V_lb=0`` TTFS boundary.

        Hagen and the spiking graph run in separate hardware modes.  This is
        deliberately a host-mediated representation boundary, not a claim of a
        continuous on-chip Hagen-to-LIF lower clamp.
        """
        scaled = torch.round(raw_preactivation.detach().to(torch.float64) / (2 ** shift))
        potential = Potential(
            scaled.to(torch.float32),
            PotentialBounds(0.0, 31.0),
        )
        lower_clamped = int((potential.value < potential.domain.min).sum().item())
        upper_clamped = int((potential.value > potential.domain.max).sum().item())
        bounded = potential.domain.clamp(potential.value).to(torch.int32)
        return bounded, {
            "relu_boundary": "implicit-lower-bound-host",
            "converting_relu": None,
            "host_mediated_lower_bound": True,
            "lower_bound_v": 0.0,
            "upper_bound_v": 31.0,
            "activation_shift": shift,
            "lower_bound_clamped_values": lower_clamped,
            "upper_bound_clamped_values": upper_clamped,
            "raw_preactivation_minimum": float(raw_preactivation.min()),
            "raw_preactivation_maximum": float(raw_preactivation.max()),
        }

    @staticmethod
    def _host_sigmoid_uint5(
        raw_preactivation: torch.Tensor,
        *,
        input_scale: float,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Quantize a host sigmoid control onto the existing UInt5 TTFS rail.

        Public hxtorch exposes the Hagen MAC and the spiking graph as separate
        execution modes, and exposes no physical sigmoid/``phi_NL``-to-``psi_ED``
        composition.  This adapter is deliberately explicit about that boundary:
        it is useful for a network-level bounded-activation control, not evidence
        of a continuous on-chip sigmoid circuit.
        """
        preactivation = raw_preactivation.detach().to(torch.float64) * input_scale
        activation = torch.sigmoid(preactivation)
        potential = Potential(
            torch.round(31.0 * activation).to(torch.float32),
            PotentialBounds(0.0, 31.0),
        )
        bounded = potential.domain.clamp(potential.value).to(torch.int32)
        return bounded, {
            "activation": "sigmoid",
            "activation_adapter": "host-sigmoid-uint5",
            "host_mediated_activation": True,
            "sigmoid_physical_subcircuit": False,
            "sigmoid_input_scale": input_scale,
            "lower_bound_v": 0.0,
            "upper_bound_v": 31.0,
            "raw_preactivation_minimum": float(raw_preactivation.min()),
            "raw_preactivation_maximum": float(raw_preactivation.max()),
            "scaled_preactivation_minimum": float(preactivation.min()),
            "scaled_preactivation_maximum": float(preactivation.max()),
            "uint5_zero_code_rate": float((bounded == 0).to(torch.float64).mean()),
            "uint5_full_scale_rate": float((bounded == 31).to(torch.float64).mean()),
            "relu_boundary": None,
            "converting_relu": None,
            "host_mediated_lower_bound": False,
        }

    def first_layer_raw(
        self,
        converted: ConvertedToyModel,
        input_uint5: torch.Tensor,
        *,
        avg: int = 1,
    ) -> HagenResult:
        """Execute only the first physical affine before its activation boundary."""
        raw, metadata = self._execute(
            self._augment(input_uint5),
            converted.first,
            avg=avg,
        )
        metadata["stage"] = "first-affine-raw"
        return HagenResult(raw.detach().cpu(), metadata)

    def direct_linear(
        self,
        value: torch.Tensor,
        weight: torch.Tensor,
        *,
        avg: int = 1,
    ) -> HagenResult:
        """Execute a bias-free physical Linear for direct primitive measurement.

        This narrow entry point intentionally bypasses the converted-model
        adapters.  It is used to characterize the PWM integration primitive
        with a caller-supplied UInt5 duration and signed int6 drive.
        """
        value = value.detach().to(torch.float32)
        weight = weight.detach().to(torch.float32)
        if value.ndim != 2 or weight.ndim != 2:
            raise ValueError("direct Hagen inputs and weights must be matrices")
        if value.shape[1] != weight.shape[1]:
            raise ValueError("direct Hagen input and weight widths differ")
        if bool(((value < 0) | (value > 31)).any()):
            raise ValueError("direct Hagen input must lie in UInt5 [0, 31]")
        if bool(((weight < -63) | (weight > 63)).any()):
            raise ValueError("direct Hagen weight must lie in signed int6 [-63, 63]")
        if avg <= 0:
            raise ValueError("direct Hagen avg must be positive")
        started = perf_counter()
        with self.hardware_session():
            assert self._active_hxtorch is not None
            hxtorch = self._active_hxtorch
            layer = hxtorch.perceptron.nn.Linear(
                value.shape[1],
                weight.shape[0],
                bias=False,
                num_sends=self.config.num_sends,
                wait_between_events=self.config.wait_between_events,
                mock=self.config.mode == "mock",
                avg=avg,
            )
            layer.weight.data.copy_(weight.to(layer.weight.dtype))
            output = layer(value).detach().cpu()
            chip_identifier = None
            get_identifier = getattr(hxtorch, "get_unique_identifier", None)
            if callable(get_identifier) and self.config.mode == "hardware":
                chip_identifier = [str(item) for item in get_identifier()]
            return HagenResult(
                output,
                {
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
                    "bias": False,
                    "input_shape": list(value.shape),
                    "weight_shape": list(weight.shape),
                    "output_shape": list(output.shape),
                    "elapsed_s": perf_counter() - started,
                },
            )

    def first_layer(
        self,
        converted: ConvertedToyModel,
        input_uint5: torch.Tensor,
        *,
        avg: int = 1,
        relu_boundary: ReLUBoundary = "implicit-lower-bound-host",
        activation: ToyActivation | None = None,
    ) -> HagenResult:
        """Execute first PWM affine and apply the frozen hidden boundary adapter."""
        resolved_activation = activation or converted.manifest.activation
        if resolved_activation != converted.manifest.activation:
            raise ValueError(
                "requested hidden activation does not match the converted checkpoint: "
                f"{resolved_activation} != {converted.manifest.activation}"
            )
        if resolved_activation == "relu":
            if relu_boundary == "implicit-lower-bound-host":
                raw_result = self.first_layer_raw(converted, input_uint5, avg=avg)
                hidden, boundary_metadata = self._implicit_lower_bound_uint5(
                    raw_result.value,
                    shift=self.config.hidden_shift,
                )
                metadata = {**raw_result.metadata, **boundary_metadata}
            else:
                hidden, metadata = self._execute(
                    self._augment(input_uint5),
                    converted.first,
                    avg=avg,
                    relu_boundary=relu_boundary,
                    activation_shift=self.config.hidden_shift,
                )
            metadata["activation"] = "relu"
            metadata["activation_adapter"] = relu_boundary
            metadata["host_mediated_activation"] = (
                relu_boundary == "implicit-lower-bound-host"
            )
        elif resolved_activation == "sigmoid":
            raw_result = self.first_layer_raw(converted, input_uint5, avg=avg)
            hidden, sigmoid_metadata = self._host_sigmoid_uint5(
                raw_result.value,
                input_scale=converted.first.scale,
            )
            metadata = dict(raw_result.metadata)
            metadata.update(sigmoid_metadata)
        else:
            raise ValueError(f"unsupported hidden activation: {resolved_activation}")
        metadata["integer_reference_hidden_shift"] = converted.manifest.hidden_shift
        metadata["hagen_hidden_shift"] = self.config.hidden_shift
        metadata["hagen_hidden_shift_used"] = resolved_activation == "relu"
        return HagenResult(hidden.detach().cpu().to(torch.int32), metadata)

    def measure_fidelity(
        self,
        converted: ConvertedToyModel,
        input_uint5: torch.Tensor,
        *,
        trials: int,
        relu_boundary: ReLUBoundary = "implicit-lower-bound-host",
        activation: ToyActivation | None = None,
    ) -> HagenFidelityResult:
        """Pair repeated physical affine outputs with the frozen integer reference."""
        if trials <= 0:
            raise ValueError("Hagen fidelity trials must be positive")
        resolved_activation = activation or converted.manifest.activation
        if resolved_activation != converted.manifest.activation:
            raise ValueError("hidden activation does not match converted checkpoint")
        if resolved_activation == "relu" and relu_boundary != "implicit-lower-bound-host":
            raise ValueError(
                "paired raw Hagen fidelity requires implicit-lower-bound-host"
            )
        ideal_accumulator, ideal_hidden = converted.hidden_from_uint5(input_uint5)
        _, ideal_output = converted.output_from_hidden(ideal_hidden)
        raw_trials: list[torch.Tensor] = []
        hidden_trials: list[torch.Tensor] = []
        output_trials: list[torch.Tensor] = []
        trial_metadata: list[dict[str, Any]] = []
        with self.hardware_session():
            for trial in range(trials):
                raw_result = self.first_layer_raw(converted, input_uint5, avg=1)
                if resolved_activation == "relu":
                    physical_hidden, boundary_metadata = self._implicit_lower_bound_uint5(
                        raw_result.value,
                        shift=self.config.hidden_shift,
                    )
                else:
                    physical_hidden, boundary_metadata = self._host_sigmoid_uint5(
                        raw_result.value,
                        input_scale=converted.first.scale,
                    )
                output_result = self.output_layer(converted, ideal_hidden)
                raw_trials.append(raw_result.value.to(torch.float64))
                hidden_trials.append(physical_hidden.to(torch.int32))
                output_trials.append(output_result.value.to(torch.int8))
                trial_metadata.append(
                    {
                        "trial": trial,
                        "first": raw_result.metadata,
                        "hidden_boundary": boundary_metadata,
                        "output": output_result.metadata,
                    }
                )
        return HagenFidelityResult(
            ideal_first_accumulator=ideal_accumulator.detach().cpu().to(torch.int32),
            physical_first_raw=torch.stack(raw_trials),
            ideal_hidden_uint5=ideal_hidden.detach().cpu().to(torch.int32),
            physical_hidden_uint5=torch.stack(hidden_trials),
            ideal_output_int8=ideal_output.detach().cpu().to(torch.int8),
            physical_output_int8=torch.stack(output_trials),
            metadata={
                "trials": trials,
                "samples": int(input_uint5.shape[0]),
                "activation": resolved_activation,
                "relu_boundary": (
                    relu_boundary if resolved_activation == "relu" else None
                ),
                "integer_reference_hidden_shift": converted.manifest.hidden_shift,
                "hagen_hidden_shift": self.config.hidden_shift,
                "output_input": "ideal-hidden-uint5",
                "trial_metadata": trial_metadata,
            },
        )

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
        relu_boundary: ReLUBoundary = "implicit-lower-bound-host",
        activation: ToyActivation | None = None,
    ) -> dict[str, Any]:
        """Select a label-free ReLU shift or validate the fixed sigmoid adapter."""
        if input_uint5.shape[0] != target_hidden_uint5.shape[0]:
            raise ValueError("shift calibration inputs and targets must share samples")
        resolved_activation = activation or converted.manifest.activation
        if resolved_activation != converted.manifest.activation:
            raise ValueError("hidden activation does not match converted checkpoint")
        augmented = self._augment(input_uint5)
        rows: list[dict[str, Any]] = []
        target = target_hidden_uint5.to(torch.float64)
        scale = float(target.square().mean()) + 1.0e-12
        # Sigmoid uses the frozen first-affine scale, so there is no arbitrary
        # bit shift to sweep.  Keep the common payload schema for the notebook.
        candidate_shifts = candidates if resolved_activation == "relu" else (self.config.hidden_shift,)
        shared_raw: torch.Tensor | None = None
        shared_metadata: dict[str, Any] | None = None
        if (
            resolved_activation == "relu"
            and relu_boundary == "implicit-lower-bound-host"
        ):
            # The shift and lower clamp are host-side for this boundary. Run
            # the analog MAC once, then score every candidate from that same
            # physical observation instead of repeatedly reserving hardware.
            raw_result = self.first_layer_raw(converted, input_uint5, avg=1)
            shared_raw = raw_result.value
            shared_metadata = raw_result.metadata
        for shift in candidate_shifts:
            if shared_raw is not None:
                output, boundary_metadata = self._implicit_lower_bound_uint5(
                    shared_raw,
                    shift=shift,
                )
                metadata = {
                    **(shared_metadata or {}),
                    **boundary_metadata,
                    "shared_physical_shift_probe": True,
                }
            elif resolved_activation == "relu":
                output, metadata = self._execute(
                    augmented,
                    converted.first,
                    avg=1,
                    relu_boundary=relu_boundary,
                    activation_shift=shift,
                )
            else:
                raw, metadata = self._execute(augmented, converted.first, avg=1)
                output, sigmoid_metadata = self._host_sigmoid_uint5(
                    raw,
                    input_scale=converted.first.scale,
                )
                metadata.update(sigmoid_metadata)
            output = output.to(torch.float64)
            mse = float((output - target).square().mean()) / scale
            saturation = float((output >= 31).float().mean())
            rows.append(
                {
                    "shift": shift,
                    "normalized_mse": mse,
                    "saturation_rate": saturation,
                    "score": mse + max(0.0, saturation - 0.01) * 100.0,
                    "activation": resolved_activation,
                    "shift_used": resolved_activation == "relu",
                    "shared_physical_shift_probe": shared_raw is not None,
                    "metadata": metadata,
                }
            )
        selected = min(rows, key=lambda row: float(row["score"]))
        return {"selected": selected, "candidates": rows}

    def probe(self, converted: ConvertedToyModel) -> dict[str, Any]:
        """Probe native 128-row and architecture-sized first-layer execution."""
        probes: list[dict[str, Any]] = []
        width = converted.first.weight_with_bias.shape[1]
        for features in sorted({min(128, width), width}):
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


def _pearson_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.to(torch.float64).reshape(-1)
    right = right.to(torch.float64).reshape(-1)
    left = left - left.mean()
    right = right - right.mean()
    denominator = torch.sqrt(left.square().sum() * right.square().sum())
    if float(denominator) <= 0.0:
        return float("nan")
    return float((left * right).sum() / denominator)


def _affine_fit(reference: torch.Tensor, observed: torch.Tensor) -> dict[str, float]:
    reference = reference.to(torch.float64).reshape(-1)
    observed = observed.to(torch.float64).reshape(-1)
    centered = reference - reference.mean()
    denominator = centered.square().sum()
    if float(denominator) <= 0.0:
        return {
            "regression_slope": float("nan"),
            "regression_intercept": float("nan"),
            "coefficient_of_determination": float("nan"),
        }
    slope = ((observed - observed.mean()) * centered).sum() / denominator
    intercept = observed.mean() - slope * reference.mean()
    predicted = slope * reference + intercept
    residual = (observed - predicted).square().sum()
    total = (observed - observed.mean()).square().sum()
    r_squared = torch.where(
        total > 0.0,
        1.0 - residual / total,
        torch.tensor(float("nan"), dtype=torch.float64),
    )
    return {
        "regression_slope": float(slope),
        "regression_intercept": float(intercept),
        "coefficient_of_determination": float(r_squared),
    }


def _fit_channelwise_correction(
    reference: torch.Tensor,
    observations: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit reference = gain * trial_mean(observation) + offset per channel."""
    if reference.ndim != 2 or observations.ndim != 3:
        raise ValueError("affine correction expects two- and three-dimensional tensors")
    if observations.shape[1:] != reference.shape:
        raise ValueError("affine correction observations do not match their reference")
    target = reference.to(torch.float64)
    predictor = observations.to(torch.float64).mean(dim=0)
    predictor_centered = predictor - predictor.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    denominator = predictor_centered.square().sum(dim=0)
    numerator = (predictor_centered * target_centered).sum(dim=0)
    gain = torch.where(
        denominator > torch.finfo(torch.float64).eps,
        numerator / denominator,
        torch.zeros_like(denominator),
    )
    offset = target.mean(dim=0) - gain * predictor.mean(dim=0)
    return gain, offset


def fit_hagen_affine_correction(
    result: HagenFidelityResult,
    converted: ConvertedToyModel,
    *,
    calibration_samples: int,
) -> HagenAffineCorrection:
    """Fit label-free channel corrections and apply them only to later samples."""
    total_samples = int(result.ideal_hidden_uint5.shape[0])
    if calibration_samples <= 0 or calibration_samples >= total_samples:
        raise ValueError("calibration_samples must leave a non-empty evaluation split")
    calibration = slice(0, calibration_samples)
    evaluation = slice(calibration_samples, total_samples)

    first_gain, first_offset = _fit_channelwise_correction(
        result.ideal_first_accumulator[calibration],
        result.physical_first_raw[:, calibration],
    )
    hidden_gain, hidden_offset = _fit_channelwise_correction(
        result.ideal_hidden_uint5[calibration],
        result.physical_hidden_uint5[:, calibration],
    )
    output_gain, output_offset = _fit_channelwise_correction(
        result.ideal_output_int8[calibration],
        result.physical_output_int8[:, calibration],
    )

    corrected_first = (
        result.physical_first_raw[:, evaluation].to(torch.float64)
        * first_gain.reshape(1, 1, -1)
        + first_offset.reshape(1, 1, -1)
    )
    corrected_hidden_from_raw = converted.hidden_uint5_from_accumulator(
        corrected_first
    )
    corrected_hidden_from_code = torch.round(
        result.physical_hidden_uint5[:, evaluation].to(torch.float64)
        * hidden_gain.reshape(1, 1, -1)
        + hidden_offset.reshape(1, 1, -1)
    ).clamp(0, 31).to(torch.int32)
    corrected_output = (
        result.physical_output_int8[:, evaluation].to(torch.float64)
        * output_gain.reshape(1, 1, -1)
        + output_offset.reshape(1, 1, -1)
    ).clamp(-128, 127)
    return HagenAffineCorrection(
        calibration_samples=calibration_samples,
        evaluation_samples=total_samples - calibration_samples,
        first_gain=first_gain,
        first_offset=first_offset,
        hidden_gain=hidden_gain,
        hidden_offset=hidden_offset,
        output_gain=output_gain,
        output_offset=output_offset,
        corrected_first_accumulator=corrected_first,
        corrected_hidden_from_raw_uint5=corrected_hidden_from_raw,
        corrected_hidden_from_code_uint5=corrected_hidden_from_code,
        corrected_output=corrected_output,
        metadata={
            "fit": "per-channel-ordinary-least-squares",
            "predictor": "physical-trial-mean",
            "target": "frozen-integer-reference",
            "labels_used": False,
            "calibration_sample_range": [0, calibration_samples],
            "evaluation_sample_range": [calibration_samples, total_samples],
        },
    )


def _fidelity_statistics(
    stage: str,
    reference: torch.Tensor,
    observations: torch.Tensor,
    *,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    include_argmax: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if reference.ndim != 2 or observations.ndim != 3:
        raise ValueError("fidelity comparison expects [sample, channel] references")
    if observations.shape[1:] != reference.shape:
        raise ValueError("fidelity observations do not match their reference")
    reference64 = reference.to(torch.float64)
    observed64 = observations.to(torch.float64)
    trial_mean = observed64.mean(dim=0)
    residual = observed64 - reference64.unsqueeze(0)
    mean_residual = trial_mean - reference64
    trial_noise = observed64.std(dim=0, unbiased=False)
    summary: dict[str, Any] = {
        "stage": stage,
        "trials": int(observations.shape[0]),
        "samples": int(observations.shape[1]),
        "channels": int(observations.shape[2]),
        "all_trial_bias": float(residual.mean()),
        "all_trial_mae": float(residual.abs().mean()),
        "all_trial_rmse": float(torch.sqrt(residual.square().mean())),
        "trial_mean_bias": float(mean_residual.mean()),
        "trial_mean_mae": float(mean_residual.abs().mean()),
        "trial_mean_rmse": float(torch.sqrt(mean_residual.square().mean())),
        "trial_noise_std": float(trial_noise.mean()),
        "exact_match_rate": float(
            (observations == reference.unsqueeze(0)).to(torch.float64).mean()
        ),
        "pearson_correlation": _pearson_correlation(reference64, trial_mean),
        **_affine_fit(reference64, trial_mean),
    }
    if lower_bound is not None and upper_bound is not None:
        summary.update(
            {
                "ideal_saturation_rate": float(
                    (
                        (reference64 <= lower_bound)
                        | (reference64 >= upper_bound)
                    ).to(torch.float64).mean()
                ),
                "physical_saturation_rate": float(
                    (
                        (observed64 <= lower_bound)
                        | (observed64 >= upper_bound)
                    ).to(torch.float64).mean()
                ),
            }
        )
    if include_argmax:
        reference_prediction = reference64.argmax(dim=-1)
        summary.update(
            {
                "all_trial_argmax_agreement": float(
                    (
                        observed64.argmax(dim=-1)
                        == reference_prediction.unsqueeze(0)
                    ).to(torch.float64).mean()
                ),
                "trial_mean_argmax_agreement": float(
                    (trial_mean.argmax(dim=-1) == reference_prediction)
                    .to(torch.float64)
                    .mean()
                ),
            }
        )
    channels: list[dict[str, Any]] = []
    for channel in range(reference.shape[1]):
        channel_reference = reference64[:, channel]
        channel_observed = observed64[:, :, channel]
        channel_mean = channel_observed.mean(dim=0)
        channel_residual = channel_mean - channel_reference
        channels.append(
            {
                "stage": stage,
                "channel": channel,
                "trial_mean_bias": float(channel_residual.mean()),
                "trial_mean_mae": float(channel_residual.abs().mean()),
                "trial_mean_rmse": float(
                    torch.sqrt(channel_residual.square().mean())
                ),
                "trial_noise_std": float(
                    channel_observed.std(dim=0, unbiased=False).mean()
                ),
                "pearson_correlation": _pearson_correlation(
                    channel_reference, channel_mean
                ),
                **_affine_fit(channel_reference, channel_mean),
            }
        )
    return summary, channels


def summarize_hagen_fidelity(
    result: HagenFidelityResult,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Separate systematic affine error from repeated physical variation."""
    first_raw, first_raw_channels = _fidelity_statistics(
        "first-affine-raw",
        result.ideal_first_accumulator,
        result.physical_first_raw,
    )
    hidden, hidden_channels = _fidelity_statistics(
        "hidden-uint5",
        result.ideal_hidden_uint5,
        result.physical_hidden_uint5,
        lower_bound=0.0,
        upper_bound=31.0,
    )
    output, output_channels = _fidelity_statistics(
        "output-int8",
        result.ideal_output_int8,
        result.physical_output_int8,
        lower_bound=-128.0,
        upper_bound=127.0,
        include_argmax=True,
    )
    return (
        {
            "schema_version": 1,
            "first_affine_raw": first_raw,
            "hidden_uint5": hidden,
            "output_int8": output,
            "measurement": result.metadata,
        },
        first_raw_channels + hidden_channels + output_channels,
    )


def summarize_hagen_affine_correction(
    result: HagenFidelityResult,
    correction: HagenAffineCorrection,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Report uncorrected and corrected errors on the evaluation split only."""
    start = correction.calibration_samples
    ideal_first = result.ideal_first_accumulator[start:]
    ideal_hidden = result.ideal_hidden_uint5[start:]
    ideal_output = result.ideal_output_int8[start:]
    comparisons = (
        (
            "first-affine-raw-uncorrected",
            ideal_first,
            result.physical_first_raw[:, start:],
            None,
            None,
            False,
        ),
        (
            "first-affine-raw-corrected",
            ideal_first,
            correction.corrected_first_accumulator,
            None,
            None,
            False,
        ),
        (
            "hidden-uint5-uncorrected",
            ideal_hidden,
            result.physical_hidden_uint5[:, start:],
            0.0,
            31.0,
            False,
        ),
        (
            "hidden-uint5-corrected-from-raw",
            ideal_hidden,
            correction.corrected_hidden_from_raw_uint5,
            0.0,
            31.0,
            False,
        ),
        (
            "hidden-uint5-corrected-from-code",
            ideal_hidden,
            correction.corrected_hidden_from_code_uint5,
            0.0,
            31.0,
            False,
        ),
        (
            "output-int8-uncorrected",
            ideal_output,
            result.physical_output_int8[:, start:],
            -128.0,
            127.0,
            True,
        ),
        (
            "output-int8-corrected",
            ideal_output,
            correction.corrected_output,
            -128.0,
            127.0,
            True,
        ),
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "calibration": {
            **correction.metadata,
            "calibration_samples": correction.calibration_samples,
            "evaluation_samples": correction.evaluation_samples,
            "first_gain": correction.first_gain.tolist(),
            "first_offset": correction.first_offset.tolist(),
            "hidden_gain": correction.hidden_gain.tolist(),
            "hidden_offset": correction.hidden_offset.tolist(),
            "output_gain": correction.output_gain.tolist(),
            "output_offset": correction.output_offset.tolist(),
        },
        "evaluation": {},
    }
    channel_rows: list[dict[str, Any]] = []
    for stage, reference, observations, lower, upper, argmax in comparisons:
        statistics, channels = _fidelity_statistics(
            stage,
            reference,
            observations,
            lower_bound=lower,
            upper_bound=upper,
            include_argmax=argmax,
        )
        summary["evaluation"][stage] = statistics
        channel_rows.extend(channels)
    return summary, channel_rows
