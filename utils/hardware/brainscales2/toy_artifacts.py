"""Metrics and stable artifact serialization for toy ANN2SNN HIL runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
import csv
import json
import math

import torch

from .toy_pooling import ToyPoolResult


PoolingDomain = Literal["ttfs", "potential"]


@dataclass(frozen=True)
class ToyConditionEvaluation:
    """Classifier logits paired with one temporal-pool observation bundle."""

    key: str
    pool_size: int
    pooling_domain: PoolingDomain
    pool_result: ToyPoolResult
    nominal_hidden_uint5: torch.Tensor
    logits: torch.Tensor
    oracle_miss_repair_logits: torch.Tensor
    torch_readout_logits: torch.Tensor
    torch_oracle_miss_repair_logits: torch.Tensor
    pwm_metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if self.logits.ndim != 3:
            raise ValueError("condition logits must have shape [trial, sample, class]")
        if self.logits.shape[:2] != self.pool_result.decoded_uint5.shape[:2]:
            raise ValueError("condition logits do not match pool trials and samples")
        expected_hidden = self.pool_result.decoded_uint5.shape[1:]
        if self.nominal_hidden_uint5.shape != expected_hidden:
            raise ValueError("nominal hidden activations must have shape [sample, hidden]")
        for name, logits in (
            ("oracle miss repair", self.oracle_miss_repair_logits),
            ("torch readout", self.torch_readout_logits),
            ("torch oracle miss repair", self.torch_oracle_miss_repair_logits),
        ):
            if logits.shape != self.logits.shape:
                raise ValueError(f"{name} logits must match condition logits")


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


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: "" if isinstance(value, float) and not math.isfinite(value) else value
                    for key, value in row.items()
                }
            )


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == labels).float().mean())


def _nll(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            labels.reshape(-1),
        )
    )


def _paired_accuracy_ci(
    current: torch.Tensor,
    baseline: torch.Tensor,
    *,
    seed: int,
    iterations: int,
) -> tuple[float, float]:
    if current.shape != baseline.shape or current.ndim != 2:
        return math.nan, math.nan
    sample_effect = current.float().mean(dim=0) - baseline.float().mean(dim=0)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    estimates = torch.empty(iterations, dtype=torch.float64)
    for iteration in range(iterations):
        index = torch.randint(
            sample_effect.numel(),
            (sample_effect.numel(),),
            generator=generator,
        )
        estimates[iteration] = sample_effect[index].mean()
    interval = torch.quantile(estimates, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return float(interval[0]), float(interval[1])


def _masked_rate(value: torch.Tensor, mask: torch.Tensor) -> float:
    """Return a boolean rate on a declared support, or NaN for empty support."""
    if value.shape != mask.shape:
        raise ValueError("rate value and support mask must have identical shapes")
    return float(value[mask].float().mean()) if bool(mask.any()) else math.nan


def _activation_analysis(
    evaluation: ToyConditionEvaluation,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Stratify misses and non-miss UInt5 errors by the actual pool input code."""
    result = evaluation.pool_result
    nominal = evaluation.nominal_hidden_uint5.to(torch.int32)
    nominal_per_trial = nominal.unsqueeze(0).expand_as(result.decoded_uint5)
    nominal_per_replica = nominal_per_trial.unsqueeze(-1).expand_as(result.fired)
    zero_pool = nominal_per_trial == 0
    positive_pool = nominal_per_trial > 0
    zero_replica = nominal_per_replica == 0
    positive_replica = nominal_per_replica > 0
    nonmiss = ~result.all_miss
    error = result.decoded_uint5.to(torch.float64) - nominal_per_trial.to(torch.float64)

    aggregate = {
        "nominal_zero_pool_observations": int(zero_pool.sum()),
        "nominal_positive_pool_observations": int(positive_pool.sum()),
        "neuron_miss_rate_nominal_zero": _masked_rate(~result.fired, zero_replica),
        "neuron_miss_rate_nominal_positive": _masked_rate(
            ~result.fired, positive_replica
        ),
        "all_miss_rate_nominal_zero": _masked_rate(result.all_miss, zero_pool),
        "all_miss_rate_nominal_positive": _masked_rate(
            result.all_miss, positive_pool
        ),
        "nonmiss_activation_count": int(nonmiss.sum()),
        "nonmiss_activation_mae_uint5": (
            float(error[nonmiss].abs().mean()) if bool(nonmiss.any()) else math.nan
        ),
        "nonmiss_activation_bias_uint5": (
            float(error[nonmiss].mean()) if bool(nonmiss.any()) else math.nan
        ),
    }

    code_rows: list[dict[str, Any]] = []
    for code in range(32):
        pool_support = nominal_per_trial == code
        replica_support = nominal_per_replica == code
        code_nonmiss = pool_support & nonmiss
        code_error = error[code_nonmiss]
        code_rows.append(
            {
                "condition": evaluation.key,
                "pool_size": evaluation.pool_size,
                "pooling_domain": evaluation.pooling_domain,
                "hagen_avg": (
                    evaluation.pool_size if evaluation.pooling_domain == "potential" else 1
                ),
                "temporal_pool_size": result.pool_size,
                "placement": result.placement,
                "mapping": result.mapping,
                "nominal_code": code,
                "pool_observations": int(pool_support.sum()),
                "nonmiss_count": int(code_nonmiss.sum()),
                "all_miss_count": int((pool_support & result.all_miss).sum()),
                "all_miss_rate": _masked_rate(result.all_miss, pool_support),
                "replica_observations": int(replica_support.sum()),
                "neuron_miss_count": int((replica_support & ~result.fired).sum()),
                "neuron_miss_rate": _masked_rate(~result.fired, replica_support),
                "activation_mae_uint5": (
                    float(code_error.abs().mean()) if code_error.numel() else math.nan
                ),
                "activation_bias_uint5": (
                    float(code_error.mean()) if code_error.numel() else math.nan
                ),
                "activation_exact_rate": (
                    float((code_error == 0).float().mean())
                    if code_error.numel()
                    else math.nan
                ),
            }
        )
    return aggregate, code_rows


def activation_error_by_code(
    evaluations: list[ToyConditionEvaluation],
) -> list[dict[str, Any]]:
    """Return the complete UInt5 code-conditioned miss and error table."""
    return [
        row
        for evaluation in evaluations
        for row in _activation_analysis(evaluation)[1]
    ]


def summarize_toy_evaluations(
    labels: torch.Tensor,
    float_logits: torch.Tensor,
    ideal_logits: torch.Tensor,
    evaluations: list[ToyConditionEvaluation],
    *,
    bootstrap_iterations: int = 1_000,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Compute accuracy drops, recovery, miss rates, and paired intervals."""
    rows: list[dict[str, Any]] = []
    float_accuracy = _accuracy(float_logits, labels)
    ideal_accuracy = _accuracy(ideal_logits, labels)
    rows.extend(
        [
            {
                "condition": "float-ann",
                "accuracy": float_accuracy,
                "nll": _nll(float_logits, labels),
                "conversion_drop": 0.0,
                "hardware_drop": 0.0,
            },
            {
                "condition": "ideal-converted",
                "accuracy": ideal_accuracy,
                "nll": _nll(ideal_logits, labels),
                "conversion_drop": float_accuracy - ideal_accuracy,
                "hardware_drop": 0.0,
            },
        ]
    )
    correctness: dict[tuple[str, str, str, int], torch.Tensor] = {}
    for evaluation in evaluations:
        repeated_labels = labels.reshape(1, -1).expand(evaluation.logits.shape[0], -1)
        correct = evaluation.logits.argmax(dim=-1) == repeated_labels
        oracle_correct = (
            evaluation.oracle_miss_repair_logits.argmax(dim=-1) == repeated_labels
        )
        torch_correct = evaluation.torch_readout_logits.argmax(dim=-1) == repeated_labels
        torch_oracle_correct = (
            evaluation.torch_oracle_miss_repair_logits.argmax(dim=-1)
            == repeated_labels
        )
        result = evaluation.pool_result
        activation_metrics, _ = _activation_analysis(evaluation)
        sample_all_miss = result.all_miss.any(dim=-1)
        valid = ~sample_all_miss
        oracle_accuracy = (
            float(correct[valid].float().mean()) if bool(valid.any()) else math.nan
        )
        latency_residual = (
            result.pooled_first_spike_s - result.nominal_input_s.reshape(1, *result.nominal_input_s.shape)
        )
        finite_latency = latency_residual[torch.isfinite(latency_residual)]
        latency_variance = (
            float(finite_latency.var(unbiased=True))
            if finite_latency.numel() >= 2
            else math.nan
        )
        condition_group = (
            evaluation.pooling_domain,
            result.placement,
            result.mapping,
            evaluation.pool_size,
        )
        correctness[condition_group] = correct
        rows.append(
            {
                "condition": evaluation.key,
                "pool_size": evaluation.pool_size,
                "pooling_domain": evaluation.pooling_domain,
                "hagen_avg": (
                    evaluation.pool_size if evaluation.pooling_domain == "potential" else 1
                ),
                "temporal_pool_size": result.pool_size,
                "placement": result.placement,
                "mapping": result.mapping,
                "trials": evaluation.logits.shape[0],
                "accuracy": float(correct.float().mean()),
                "nll": _nll(evaluation.logits, repeated_labels),
                "oracle_miss_repair_accuracy": float(oracle_correct.float().mean()),
                "oracle_miss_repair_nll": _nll(
                    evaluation.oracle_miss_repair_logits, repeated_labels
                ),
                "oracle_miss_repair_gain": float(
                    oracle_correct.float().mean() - correct.float().mean()
                ),
                "torch_readout_accuracy": float(torch_correct.float().mean()),
                "torch_readout_nll": _nll(
                    evaluation.torch_readout_logits, repeated_labels
                ),
                "torch_readout_gain_vs_physical": float(
                    torch_correct.float().mean() - correct.float().mean()
                ),
                "torch_oracle_miss_repair_accuracy": float(
                    torch_oracle_correct.float().mean()
                ),
                "torch_oracle_miss_repair_nll": _nll(
                    evaluation.torch_oracle_miss_repair_logits, repeated_labels
                ),
                "torch_oracle_miss_repair_gain": float(
                    torch_oracle_correct.float().mean() - torch_correct.float().mean()
                ),
                "conversion_drop": float_accuracy - ideal_accuracy,
                "hardware_drop": ideal_accuracy - float(correct.float().mean()),
                "recovery_vs_m1": math.nan,
                "recovery_ci_low": math.nan,
                "recovery_ci_high": math.nan,
                "neuron_miss_rate": float((~result.fired).float().mean()),
                "all_miss_rate": float(result.all_miss.float().mean()),
                "sample_any_all_miss_rate": float(sample_all_miss.float().mean()),
                "oracle_valid_accuracy": oracle_accuracy,
                "latency_variance_s2": latency_variance,
                "multi_spike_rate": float((result.spike_count > 1).float().mean()),
                **activation_metrics,
            }
        )
    for row in rows:
        if "pool_size" not in row:
            continue
        group = (
            str(row["pooling_domain"]),
            str(row["placement"]),
            str(row["mapping"]),
        )
        current_key = (*group, int(row["pool_size"]))
        baseline_key = (*group, 1)
        if current_key not in correctness or baseline_key not in correctness:
            continue
        current = correctness[current_key]
        baseline = correctness[baseline_key]
        row["recovery_vs_m1"] = float(current.float().mean() - baseline.float().mean())
        low, high = _paired_accuracy_ci(
            current,
            baseline,
            seed=seed + int(row["pool_size"]),
            iterations=bootstrap_iterations,
        )
        row["recovery_ci_low"] = low
        row["recovery_ci_high"] = high
    return rows


def write_toy_artifacts(
    output_dir: Path,
    *,
    labels: torch.Tensor,
    float_logits: torch.Tensor,
    ideal_logits: torch.Tensor,
    ideal_hidden_uint5: torch.Tensor,
    evaluations: list[ToyConditionEvaluation],
    manifest: dict[str, Any],
    runtime: dict[str, Any],
    bootstrap_iterations: int = 1_000,
    seed: int = 0,
    event_csv_sample_limit: int = 128,
    event_csv_trial_limit: int = 2,
) -> list[dict[str, Any]]:
    """Write network predictions, raw events, tensors, metrics, and figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = summarize_toy_evaluations(
        labels,
        float_logits,
        ideal_logits,
        evaluations,
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
    )
    manifest = {
        **manifest,
        "event_csv_coverage": {
            "sample_limit_per_condition": event_csv_sample_limit,
            "trial_limit_per_condition": event_csv_trial_limit,
            "full_raw_tensor": "intermediates.pt",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(_json_value({"schema_version": 1, **manifest}), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "runtime.json").write_text(
        json.dumps(_json_value(runtime), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_rows(output_dir / "metrics.csv", metrics)
    _write_rows(
        output_dir / "activation_error_by_code.csv",
        activation_error_by_code(evaluations),
    )

    prediction_rows: list[dict[str, Any]] = []
    for sample in range(labels.numel()):
        prediction_rows.append(
            {
                "condition": "float-ann",
                "analysis_variant": "reference",
                "trial": 0,
                "sample": sample,
                "label": int(labels[sample]),
                "prediction": int(float_logits[sample].argmax()),
                "correct": int(float_logits[sample].argmax() == labels[sample]),
                "logits": json.dumps(float_logits[sample].tolist()),
            }
        )
        prediction_rows.append(
            {
                "condition": "ideal-converted",
                "analysis_variant": "reference",
                "trial": 0,
                "sample": sample,
                "label": int(labels[sample]),
                "prediction": int(ideal_logits[sample].argmax()),
                "correct": int(ideal_logits[sample].argmax() == labels[sample]),
                "logits": json.dumps(ideal_logits[sample].tolist()),
            }
        )
    for evaluation in evaluations:
        variants = (
            ("physical", evaluation.logits),
            ("physical-oracle-miss-repair", evaluation.oracle_miss_repair_logits),
            ("torch-readout", evaluation.torch_readout_logits),
            (
                "torch-readout-oracle-miss-repair",
                evaluation.torch_oracle_miss_repair_logits,
            ),
        )
        for analysis_variant, variant_logits in variants:
            for trial in range(variant_logits.shape[0]):
                for sample in range(labels.numel()):
                    logits = variant_logits[trial, sample]
                    prediction_rows.append(
                        {
                            "condition": evaluation.key,
                            "analysis_variant": analysis_variant,
                            "trial": trial,
                            "sample": sample,
                            "label": int(labels[sample]),
                            "prediction": int(logits.argmax()),
                            "correct": int(logits.argmax() == labels[sample]),
                            "logits": json.dumps(logits.tolist()),
                        }
                    )
    _write_rows(output_dir / "predictions.csv", prediction_rows)

    event_fields = (
        "condition",
        "trial",
        "sample",
        "logical_neuron",
        "replica",
        "physical_coordinate",
        "nominal_activation_uint5",
        "nominal_input_s",
        "first_spike_s",
        "fired",
        "spike_count",
    )
    with (output_dir / "events.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=event_fields)
        writer.writeheader()
        for evaluation in evaluations:
            result = evaluation.pool_result
            trials, samples, logical_neurons, replicas = result.first_spike_s.shape
            csv_trials = min(trials, event_csv_trial_limit)
            csv_samples = min(samples, event_csv_sample_limit)
            for trial in range(csv_trials):
                for sample in range(csv_samples):
                    for logical in range(logical_neurons):
                        for replica in range(replicas):
                            fired = bool(result.fired[trial, sample, logical, replica])
                            writer.writerow(
                                {
                                    "condition": evaluation.key,
                                    "trial": trial,
                                    "sample": sample,
                                    "logical_neuron": logical,
                                    "replica": replica,
                                    "physical_coordinate": int(
                                        result.physical_coordinates[logical, replica]
                                    ),
                                    "nominal_activation_uint5": int(
                                        evaluation.nominal_hidden_uint5[sample, logical]
                                    ),
                                    "nominal_input_s": float(result.nominal_input_s[sample, logical]),
                                    "first_spike_s": (
                                        float(result.first_spike_s[trial, sample, logical, replica])
                                        if fired
                                        else ""
                                    ),
                                    "fired": int(fired),
                                    "spike_count": int(
                                        result.spike_count[trial, sample, logical, replica]
                                    ),
                                }
                            )
    torch.save(
        {
            "labels": labels,
            "float_logits": float_logits,
            "ideal_logits": ideal_logits,
            "ideal_hidden_uint5": ideal_hidden_uint5,
            "conditions": {
                evaluation.key: {
                    "nominal_hidden_uint5": evaluation.nominal_hidden_uint5,
                    "first_spike_s": evaluation.pool_result.first_spike_s,
                    "fired": evaluation.pool_result.fired,
                    "spike_count": evaluation.pool_result.spike_count,
                    "nominal_input_s": evaluation.pool_result.nominal_input_s,
                    "pooled_first_spike_s": evaluation.pool_result.pooled_first_spike_s,
                    "decoded_uint5": evaluation.pool_result.decoded_uint5,
                    "all_miss": evaluation.pool_result.all_miss,
                    "physical_coordinates": evaluation.pool_result.physical_coordinates,
                    "logits": evaluation.logits,
                    "oracle_miss_repair_logits": evaluation.oracle_miss_repair_logits,
                    "torch_readout_logits": evaluation.torch_readout_logits,
                    "torch_oracle_miss_repair_logits": (
                        evaluation.torch_oracle_miss_repair_logits
                    ),
                }
                for evaluation in evaluations
            },
        },
        output_dir / "intermediates.pt",
    )
    _write_figures(output_dir, labels, evaluations, metrics)
    return metrics


def _write_figures(
    output_dir: Path,
    labels: torch.Tensor,
    evaluations: list[ToyConditionEvaluation],
    metrics: list[dict[str, Any]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    condition_rows = [row for row in metrics if "pool_size" in row]
    if condition_rows:
        figure, axis = plt.subplots(figsize=(7.5, 4.5))
        groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for row in condition_rows:
            key = (str(row["pooling_domain"]), str(row["placement"]), str(row["mapping"]))
            groups.setdefault(key, []).append(row)
        for key, rows in groups.items():
            rows.sort(key=lambda row: int(row["pool_size"]))
            axis.plot(
                [int(row["pool_size"]) for row in rows],
                [float(row["accuracy"]) for row in rows],
                marker="o",
                label="/".join(key),
            )
        axis.set_xlabel("pool size")
        axis.set_ylabel("classification accuracy")
        axis.set_xscale("log", base=2)
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(output_dir / "accuracy_vs_pool.png", dpi=180)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(7.5, 4.5))
        for key, rows in groups.items():
            usable = [row for row in rows if math.isfinite(float(row["latency_variance_s2"]))]
            if not usable:
                continue
            axis.plot(
                [1.0 / int(row["pool_size"]) for row in usable],
                [float(row["latency_variance_s2"]) for row in usable],
                marker="o",
                label="/".join(key),
            )
        axis.set_xlabel("1 / pool size")
        axis.set_ylabel("pooled latency variance [s²]")
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(output_dir / "variance_vs_inverse_pool.png", dpi=180)
        plt.close(figure)

    if evaluations:
        evaluation = evaluations[-1]
        prediction = evaluation.logits.mean(dim=0).argmax(dim=-1)
        classes = evaluation.logits.shape[-1]
        confusion = torch.zeros((classes, classes), dtype=torch.int64)
        for expected, actual in zip(labels, prediction):
            confusion[int(expected), int(actual)] += 1
        figure, axis = plt.subplots(figsize=(5.0, 4.5))
        image = axis.imshow(confusion.numpy(), cmap="Blues")
        axis.set_xlabel("predicted")
        axis.set_ylabel("label")
        figure.colorbar(image, ax=axis)
        figure.tight_layout()
        figure.savefig(output_dir / "confusion_matrix.png", dpi=180)
        plt.close(figure)
