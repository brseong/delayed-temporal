"""Stable artifact serialization for independent primitive measurements."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable
import csv
import json
import math

import torch

from .primitive_noise import (
    PRIMITIVES,
    PrimitiveNoiseConfig,
    PrimitiveObservation,
    PrimitiveValidation,
    primitive_result_key,
)


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_value(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(child) for child in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_json_value(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: "" if value is None else value for key, value in row.items()}
            )


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_payload(
    observation: PrimitiveObservation,
    trial_slice: slice,
    split: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "primitive": observation.primitive,
        "stage": observation.stage,
        "split": split,
        "output_kind": observation.output_kind,
        "input_code": observation.input_code.detach().cpu(),
        "ideal_variable": observation.ideal_variable.detach().cpu(),
        "observed": observation.observed[trial_slice].detach().cpu(),
        "delivered": observation.delivered[trial_slice].detach().cpu(),
        "spike_count": observation.spike_count[trial_slice].detach().cpu(),
        "saturated": observation.saturated[trial_slice].detach().cpu(),
        "physical_coordinates": observation.physical_coordinates,
        "fit_point_mask": (
            observation.fit_point_mask.detach().cpu()
            if observation.fit_point_mask is not None
            else None
        ),
        "auxiliary_input": (
            observation.auxiliary_input.detach().cpu()
            if observation.auxiliary_input is not None
            else None
        ),
        "baseline": (
            observation.baseline[trial_slice].detach().cpu()
            if observation.baseline is not None
            else None
        ),
        "precharge_cadc": (
            observation.precharge_cadc[trial_slice].detach().cpu()
            if observation.precharge_cadc is not None
            else None
        ),
        "metadata": observation.metadata,
    }
    return payload


def _write_raw_chunks(
    raw_dir: Path,
    observations: list[PrimitiveObservation],
    config: PrimitiveNoiseConfig,
) -> dict[str, Any]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for observation in observations:
        key = primitive_result_key(observation)
        splits = (
            ("calibration", slice(0, config.calibration_repeats)),
            ("validation", slice(config.calibration_repeats, config.repeats)),
        )
        for split, trial_slice in splits:
            path = raw_dir / f"{key}_{split}.pt"
            torch.save(_raw_payload(observation, trial_slice, split), path)
            entries.append(
                {
                    "primitive": observation.primitive,
                    "stage": observation.stage,
                    "split": split,
                    "path": path.name,
                    "sha256": _file_sha256(path),
                    "shape": list(observation.observed[trial_slice].shape),
                }
            )
    index = {"schema_version": 1, "chunks": entries}
    _write_json(raw_dir / "index.json", index)
    return index


def _calibration_payload(
    validations: list[PrimitiveValidation],
) -> dict[str, Any]:
    lookup = {
        (validation.primitive, validation.stage): validation
        for validation in validations
    }
    primitives: dict[str, Any] = {}
    for primitive in PRIMITIVES:
        if primitive == "phi-np":
            static = lookup.get((primitive, "static"))
            dynamic = lookup.get((primitive, "dynamic"))
            validated = bool(
                static is not None
                and dynamic is not None
                and static.validated
                and dynamic.validated
            )
            selected = dynamic
            stages = {
                "static": _validation_summary(static) if static else None,
                "dynamic": _validation_summary(dynamic) if dynamic else None,
            }
            diagnostic_only = static is not None and (
                dynamic is None or not dynamic.validated
            )
        else:
            selected = lookup.get((primitive, "transfer"))
            validated = bool(selected is not None and selected.validated)
            stages = {
                "transfer": _validation_summary(selected) if selected else None
            }
            diagnostic_only = False
        primitives[primitive] = {
            "validated": validated,
            "diagnostic_only": diagnostic_only,
            "stages": stages,
            "transfer_parameters": (
                list(selected.calibration_parameters)
                if selected is not None and validated
                else None
            ),
            "noise": selected.noise if selected is not None and validated else None,
        }
    return {
        "schema_version": 1,
        "validated": all(primitives[item]["validated"] for item in PRIMITIVES),
        "interpretation": "independent-marginal",
        "composed_hardware_distribution_claimed": False,
        "primitives": primitives,
    }


def _validate_hardware_provenance(
    observations: list[PrimitiveObservation],
    config: PrimitiveNoiseConfig,
) -> None:
    chip_identifiers: set[str] = set()
    for observation in observations:
        backend = str(observation.metadata.get("backend", ""))
        if "hardware" not in backend:
            continue
        identifier = observation.metadata.get("chip_identifier")
        if not identifier and observation.metadata.get("per_code"):
            identifier = observation.metadata["per_code"][0].get("chip_identifier")
        if not identifier:
            raise ValueError(
                f"hardware observation has no chip identifier: {observation.primitive}"
            )
        chip_identifiers.add(json.dumps(identifier, sort_keys=True))
        expected_path = (
            config.hagen_calibration_path
            if observation.primitive == "psi-int"
            else config.spiking_calibration_path
        )
        expected_checksum = (
            _file_sha256(expected_path) if expected_path is not None else None
        )
        actual_checksum = observation.metadata.get("calibration_sha256")
        if expected_checksum != actual_checksum:
            raise ValueError(
                f"calibration checksum mismatch for {observation.primitive}: "
                f"{actual_checksum} != {expected_checksum}"
            )
    if len(chip_identifiers) > 1:
        raise ValueError("primitive observations were acquired from different chips")


def _validation_summary(validation: PrimitiveValidation) -> dict[str, Any]:
    return {
        "validated": validation.validated,
        "diagnostic_only": validation.diagnostic_only,
        "gates": validation.gates,
        "noise": validation.noise,
    }


def _plot_validation(
    output_dir: Path,
    observation: PrimitiveObservation,
    validation: PrimitiveValidation,
    config: PrimitiveNoiseConfig,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    key = primitive_result_key(observation)
    validation_slice = slice(config.calibration_repeats, config.repeats)
    values = observation.observed[validation_slice].to(torch.float64)
    delivered = observation.delivered[validation_slice] & ~observation.saturated[
        validation_slice
    ]
    means = torch.full(
        (observation.point_count, observation.device_count),
        float("nan"),
        dtype=torch.float64,
    )
    residuals: list[torch.Tensor] = []
    for device, parameters in enumerate(validation.calibration_parameters):
        if observation.primitive == "psi-ne":
            prediction = parameters["baseline_cadc"] + parameters[
                "response_scale_cadc"
            ] * torch.exp(
                -(
                    config.observation_time_s
                    - observation.ideal_variable.to(torch.float64)
                )
                / parameters["tau_effective_s"]
            )
        elif observation.primitive == "phi-np":
            prediction = parameters["offset_s"] + parameters[
                "slope_s_per_code"
            ] * observation.ideal_variable
        elif observation.primitive == "phi-nl":
            prediction = parameters["offset_s"] + parameters[
                "log_slope_s"
            ] * observation.ideal_variable
        else:
            prediction = parameters["offset_code"] + parameters[
                "gain"
            ] * observation.ideal_variable
        for point in range(observation.point_count):
            mask = delivered[:, point, device]
            if bool(mask.any()):
                selected = values[:, point, device][mask]
                means[point, device] = selected.mean()
                residuals.append(selected - prediction[point])

    figure, axes = plt.subplots(2, 2, figsize=(10.0, 8.0))
    axis = axes[0, 0]
    x = observation.ideal_variable.detach().cpu().to(torch.float64)
    for device in range(observation.device_count):
        axis.plot(x, means[:, device], marker=".", alpha=0.45)
    axis.set_xlabel("ideal primitive variable")
    axis.set_ylabel(observation.output_kind)
    axis.set_title("held-out transfer")

    residual = torch.cat(residuals) if residuals else torch.empty(0)
    axes[0, 1].hist(residual.numpy(), bins=40)
    axes[0, 1].set_title("held-out residual histogram")
    axes[0, 1].set_xlabel(observation.output_kind)

    ordered = torch.sort(residual).values
    if ordered.numel():
        empirical = (
            torch.arange(ordered.numel(), dtype=torch.float64) + 0.5
        ) / ordered.numel()
        axes[1, 0].plot(ordered, empirical)
        axes[1, 0].set_title("held-out residual empirical CDF")
        axes[1, 0].set_xlabel(observation.output_kind)
        axes[1, 0].set_ylabel("probability")
        probabilities = empirical.clamp(1.0e-6, 1.0 - 1.0e-6)
        normal_quantiles = math.sqrt(2.0) * torch.erfinv(2.0 * probabilities - 1.0)
        axes[1, 1].scatter(normal_quantiles, ordered, s=5)
    axes[1, 1].set_title("normal quantile plot")
    axes[1, 1].set_xlabel("standard-normal quantile")
    axes[1, 1].set_ylabel(observation.output_kind)
    figure.suptitle(key)
    figure.tight_layout()
    figure.savefig(output_dir / "transfer_and_residuals.png", dpi=180)
    plt.close(figure)

    failure = (~observation.delivered).to(torch.float64).mean(dim=(0, 2))
    saturation = observation.saturated.to(torch.float64).mean(dim=(0, 2))
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    axis.plot(observation.input_code, failure, marker="o", label="miss rate")
    axis.plot(
        observation.input_code,
        saturation,
        marker="o",
        label="saturation rate",
    )
    axis.set_xlabel("input code")
    axis.set_ylabel("rate")
    axis.set_ylim(-0.02, 1.02)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "miss_and_saturation.png", dpi=180)
    plt.close(figure)


def write_primitive_noise_artifacts(
    output_dir: Path,
    *,
    config: PrimitiveNoiseConfig,
    observations: list[PrimitiveObservation],
    validations: list[PrimitiveValidation],
    environment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write raw split chunks, per-stage analysis, and the combined calibration."""
    _validate_hardware_provenance(observations, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_index = _write_raw_chunks(output_dir / "raw", observations, config)
    validation_lookup = {
        (validation.primitive, validation.stage): validation
        for validation in validations
    }
    for observation in observations:
        key = primitive_result_key(observation)
        validation = validation_lookup.get((observation.primitive, observation.stage))
        if validation is None:
            continue
        stage_dir = output_dir / key
        stage_dir.mkdir(parents=True, exist_ok=True)
        _write_rows(stage_dir / "transfer.csv", validation.transfer_rows)
        _write_rows(stage_dir / "device_stats.csv", validation.device_statistics)
        _write_json(stage_dir / "noise.json", validation.noise)
        _write_json(stage_dir / "validation.json", _validation_summary(validation))
        _plot_validation(stage_dir, observation, validation, config)

    calibration = _calibration_payload(validations)
    _write_json(output_dir / "primitive_noise_calibration.json", calibration)
    manifest = {
        "schema_version": 1,
        "experiment": "brainscales2-primitive-noise",
        "interpretation": "independent-marginal",
        "replicas_are_pooled": False,
        "transformer_forward_included": False,
        "config": config.to_manifest_dict(),
        "environment": environment or {},
        "raw_index": raw_index,
        "observations": [
            {
                "key": primitive_result_key(observation),
                "primitive": observation.primitive,
                "stage": observation.stage,
                "output_kind": observation.output_kind,
                "shape": list(observation.observed.shape),
                "physical_coordinates": list(observation.physical_coordinates),
                "metadata": observation.metadata,
            }
            for observation in observations
        ],
        "validated": calibration["validated"],
    }
    _write_json(output_dir / "manifest.json", manifest)
    return calibration


def load_primitive_observations(
    output_dir: Path,
    config: PrimitiveNoiseConfig,
) -> list[PrimitiveObservation]:
    """Reconstruct observations from checksum-verified calibration/validation chunks."""
    index_path = output_dir / "raw" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for entry in index.get("chunks", []):
        path = index_path.parent / entry["path"]
        if _file_sha256(path) != entry["sha256"]:
            raise ValueError(f"raw primitive chunk checksum changed: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        key = (str(payload["primitive"]), str(payload["stage"]))
        grouped.setdefault(key, {})[str(payload["split"])] = payload

    observations: list[PrimitiveObservation] = []
    for (primitive, stage), splits in grouped.items():
        if set(splits) != {"calibration", "validation"}:
            raise ValueError(f"primitive split is incomplete: {primitive}/{stage}")
        calibration = splits["calibration"]
        validation = splits["validation"]
        expected_coordinates = (
            tuple(range(config.device_count))
            if primitive == "psi-int"
            else config.physical_coordinates
        )
        if tuple(calibration["physical_coordinates"]) != expected_coordinates:
            raise ValueError("raw primitive coordinates differ from configuration")
        if calibration["input_code"].tolist() != validation["input_code"].tolist():
            raise ValueError("primitive input grid changed between splits")

        def concatenate(name: str) -> torch.Tensor:
            return torch.cat((calibration[name], validation[name]), dim=0)

        def optional_concatenate(name: str) -> torch.Tensor | None:
            left = calibration.get(name)
            right = validation.get(name)
            if left is None and right is None:
                return None
            if left is None or right is None:
                raise ValueError(f"optional raw field differs between splits: {name}")
            return torch.cat((left, right), dim=0)

        observations.append(
            PrimitiveObservation(
                primitive=primitive,  # type: ignore[arg-type]
                stage=stage,  # type: ignore[arg-type]
                output_kind=calibration["output_kind"],
                input_code=calibration["input_code"],
                ideal_variable=calibration["ideal_variable"],
                observed=concatenate("observed"),
                delivered=concatenate("delivered"),
                spike_count=concatenate("spike_count"),
                saturated=concatenate("saturated"),
                physical_coordinates=tuple(calibration["physical_coordinates"]),
                fit_point_mask=calibration.get("fit_point_mask"),
                auxiliary_input=calibration.get("auxiliary_input"),
                baseline=optional_concatenate("baseline"),
                precharge_cadc=optional_concatenate("precharge_cadc"),
                metadata=calibration.get("metadata", {}),
            )
        )
    if not observations:
        raise ValueError("raw primitive index contains no observations")
    return observations
