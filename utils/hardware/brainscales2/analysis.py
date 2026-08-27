"""Miss-aware pooling and variance-floor analysis for BSS-2 observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import math

import torch

from .config import BrainScaleS2PoolConfig, CADCDiagnosticResult, PoolRunResult


Estimator = Literal["corrected-mean", "mean", "median", "earliest"]


@dataclass(frozen=True)
class PoolCalibration:
    """Calibration-only target trajectory and persistent per-neuron offsets."""

    target_s: torch.Tensor
    neuron_offset_s: torch.Tensor
    calibration_trials: int


def _nanmean(value: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
    finite = torch.isfinite(value)
    numerator = torch.where(finite, value, torch.zeros_like(value)).sum(dim=dim)
    denominator = finite.sum(dim=dim)
    return torch.where(
        denominator > 0,
        numerator / denominator.clamp_min(1),
        torch.full_like(numerator, torch.nan),
    )


def calibrate_pool(result: PoolRunResult) -> PoolCalibration:
    """Estimate sample targets and static neuron offsets on the first trial half."""
    trial_count = result.first_spike_s.shape[0]
    calibration_trials = max(1, trial_count // 2)
    observed = result.first_spike_s[:calibration_trials]
    target = _nanmean(observed, dim=(0, 2))
    residual = observed - target.reshape(1, -1, 1)
    offsets = _nanmean(residual, dim=(0, 1))
    offsets = torch.where(torch.isfinite(offsets), offsets, torch.zeros_like(offsets))
    corrected_target = _nanmean(
        observed - offsets.reshape(1, 1, -1),
        dim=(0, 2),
    )
    return PoolCalibration(
        target_s=corrected_target,
        neuron_offset_s=offsets,
        calibration_trials=calibration_trials,
    )


def pool_first_spikes(
    result: PoolRunResult,
    calibration: PoolCalibration,
    estimator: Estimator,
    *,
    evaluation_only: bool = True,
) -> torch.Tensor:
    """Pool valid events while retaining NaN for an all-miss pool."""
    start = calibration.calibration_trials if evaluation_only else 0
    observed = result.first_spike_s[start:]
    if estimator == "corrected-mean":
        observed = observed - calibration.neuron_offset_s.reshape(1, 1, -1)
        return _nanmean(observed, dim=-1)
    if estimator == "mean":
        return _nanmean(observed, dim=-1)
    if estimator == "median":
        return torch.nanmedian(observed, dim=-1).values
    if estimator == "earliest":
        replacement = torch.full_like(observed, torch.inf)
        minimum = torch.where(torch.isfinite(observed), observed, replacement).amin(dim=-1)
        return torch.where(
            torch.isfinite(minimum), minimum, torch.full_like(minimum, torch.nan)
        )
    raise ValueError(f"unsupported estimator: {estimator}")


def _finite_variance(value: torch.Tensor) -> tuple[float, int]:
    finite = value[torch.isfinite(value)]
    if finite.numel() < 2:
        return math.nan, int(finite.numel())
    return float(finite.var(unbiased=True).item()), int(finite.numel())


def mean_pairwise_correlation(
    result: PoolRunResult,
    calibration: PoolCalibration,
) -> float:
    """Compute average finite pairwise correlation after static-offset removal."""
    if result.pool_size < 2:
        return math.nan
    observed = result.first_spike_s[calibration.calibration_trials :]
    corrected = observed - calibration.neuron_offset_s.reshape(1, 1, -1)
    corrected = corrected - calibration.target_s.reshape(1, -1, 1)
    flattened = corrected.reshape(-1, result.pool_size)
    correlations: list[float] = []
    for left in range(result.pool_size):
        for right in range(left + 1, result.pool_size):
            valid = torch.isfinite(flattened[:, left]) & torch.isfinite(
                flattened[:, right]
            )
            pair = flattened[valid][:, (left, right)]
            if pair.shape[0] < 3:
                continue
            centered = pair - pair.mean(dim=0, keepdim=True)
            denominator = torch.sqrt(
                centered[:, 0].square().sum() * centered[:, 1].square().sum()
            )
            if float(denominator) > 0.0:
                correlations.append(
                    float((centered[:, 0] * centered[:, 1]).sum() / denominator)
                )
    if not correlations:
        return math.nan
    return sum(correlations) / len(correlations)


def summarize_pool_result(
    result: PoolRunResult,
    estimator: Estimator,
) -> dict[str, float | int | str]:
    """Summarize held-out pooling residuals for one physical condition."""
    calibration = calibrate_pool(result)
    pooled = pool_first_spikes(result, calibration, estimator)
    residual = pooled - calibration.target_s.reshape(1, -1)
    finite = residual[torch.isfinite(residual)]
    variance, valid_count = _finite_variance(residual)
    total_pools = residual.numel()
    eval_spikes = result.first_spike_s[calibration.calibration_trials :]
    eval_counts = result.spike_count[calibration.calibration_trials :]
    return {
        "pool_size": result.pool_size,
        "placement": result.placement,
        "routing": result.routing,
        "estimator": estimator,
        "evaluation_trials": result.first_spike_s.shape[0]
        - calibration.calibration_trials,
        "valid_pools": valid_count,
        "all_miss_rate": 1.0 - valid_count / max(1, total_pools),
        "neuron_miss_rate": float((~torch.isfinite(eval_spikes)).float().mean()),
        "multi_spike_rate": float((eval_counts > 1).float().mean()),
        "bias_s": float(finite.mean()) if finite.numel() else math.nan,
        "mae_s": float(finite.abs().mean()) if finite.numel() else math.nan,
        "variance_s2": variance,
        "mean_pairwise_correlation": mean_pairwise_correlation(result, calibration),
    }


def fit_variance_floor(
    summaries: list[dict[str, float | int | str]],
) -> dict[str, float]:
    """Fit Var(pool)=a/M+c with valid-count weighted least squares."""
    usable = [
        summary
        for summary in summaries
        if math.isfinite(float(summary["variance_s2"]))
        and int(summary["valid_pools"]) >= 2
    ]
    if len(usable) < 2:
        return {"a_s2": math.nan, "c_s2": math.nan}
    x = torch.tensor(
        [[1.0 / int(row["pool_size"]), 1.0] for row in usable],
        dtype=torch.float64,
    )
    y = torch.tensor(
        [float(row["variance_s2"]) for row in usable], dtype=torch.float64
    )
    weights = torch.tensor(
        [max(1, int(row["valid_pools"]) - 1) for row in usable],
        dtype=torch.float64,
    )
    weighted_x = x * torch.sqrt(weights).reshape(-1, 1)
    weighted_y = y * torch.sqrt(weights)
    solution = torch.linalg.lstsq(weighted_x, weighted_y).solution
    return {"a_s2": float(solution[0]), "c_s2": float(solution[1])}


def bootstrap_variance_floor(
    results: list[PoolRunResult],
    estimator: Estimator,
    *,
    iterations: int = 500,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap trials independently within each pool size and refit a and c."""
    if len(results) < 2 or iterations <= 0:
        return {
            "a_ci_low_s2": math.nan,
            "a_ci_high_s2": math.nan,
            "c_ci_low_s2": math.nan,
            "c_ci_high_s2": math.nan,
        }
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    prepared: list[tuple[PoolRunResult, PoolCalibration, torch.Tensor]] = []
    for result in results:
        calibration = calibrate_pool(result)
        pooled = pool_first_spikes(result, calibration, estimator)
        residual = pooled - calibration.target_s.reshape(1, -1)
        prepared.append((result, calibration, residual))

    estimates: list[tuple[float, float]] = []
    for _ in range(iterations):
        summaries: list[dict[str, float | int | str]] = []
        for result, _, residual in prepared:
            trial_count = residual.shape[0]
            sampled_trials = torch.randint(
                trial_count,
                (trial_count,),
                generator=generator,
            )
            variance, valid_count = _finite_variance(residual[sampled_trials])
            summaries.append(
                {
                    "pool_size": result.pool_size,
                    "variance_s2": variance,
                    "valid_pools": valid_count,
                }
            )
        fitted = fit_variance_floor(summaries)
        if math.isfinite(fitted["a_s2"]) and math.isfinite(fitted["c_s2"]):
            estimates.append((fitted["a_s2"], fitted["c_s2"]))
    if not estimates:
        return {
            "a_ci_low_s2": math.nan,
            "a_ci_high_s2": math.nan,
            "c_ci_low_s2": math.nan,
            "c_ci_high_s2": math.nan,
        }
    estimate_tensor = torch.tensor(estimates, dtype=torch.float64)
    quantiles = torch.quantile(
        estimate_tensor,
        torch.tensor([0.025, 0.975], dtype=torch.float64),
        dim=0,
    )
    return {
        "a_ci_low_s2": float(quantiles[0, 0]),
        "a_ci_high_s2": float(quantiles[1, 0]),
        "c_ci_low_s2": float(quantiles[0, 1]),
        "c_ci_high_s2": float(quantiles[1, 1]),
    }


def score_operating_point(result: PoolRunResult) -> dict[str, float]:
    """Score delivery while rejecting multi-spike and pre-input activity."""
    fired_rate = float(result.fired.float().mean())
    multi_spike_rate = float((result.spike_count > 1).float().mean())
    input_time = result.nominal_input_s.reshape(1, -1, 1)
    premature = result.fired & (result.first_spike_s < input_time - 1.0e-9)
    premature_spike_rate = float(premature.float().mean())
    score = (
        abs(fired_rate - 0.97)
        + 10.0 * multi_spike_rate
        + 10.0 * premature_spike_rate
    )
    return {
        "fired_rate": fired_rate,
        "multi_spike_rate": multi_spike_rate,
        "premature_spike_rate": premature_spike_rate,
        "score": score,
    }


def analyze_cadc_diagnostic(
    result: CADCDiagnosticResult,
    config: BrainScaleS2PoolConfig,
) -> dict[str, object]:
    """Measure PSP separation without equating CADC and parameter units."""
    pre_stimulus = result.time_s < result.stimulus_time_s
    post_stimulus = result.time_s >= result.stimulus_time_s
    if int(pre_stimulus.sum()) < 2 or int(post_stimulus.sum()) < 2:
        raise ValueError("CADC diagnostic requires samples before and after stimulus")

    baseline_reference = result.baseline_cadc[:, pre_stimulus].median(dim=1).values
    paired_psp = result.stimulated_cadc - result.baseline_cadc
    peak_delta = paired_psp[:, post_stimulus].amax(dim=1)
    baseline_excursion = (
        result.baseline_cadc - baseline_reference.unsqueeze(1)
    )[:, post_stimulus].abs().amax(dim=1)

    baseline_fired = result.baseline_spikes.sum(dim=1) > 0
    stimulated_fired = result.stimulated_spikes[:, post_stimulus].sum(dim=1) > 0
    baseline_fired_rate = float(baseline_fired.float().mean())
    stimulated_fired_rate = float(stimulated_fired.float().mean())

    peak_flat = peak_delta.reshape(-1).to(torch.float64)
    excursion_flat = baseline_excursion.reshape(-1).to(torch.float64)
    signal_floor = float(torch.quantile(peak_flat, 0.10))
    signal_median = float(torch.quantile(peak_flat, 0.50))
    noise_ceiling = max(0.0, float(torch.quantile(excursion_flat, 0.99)))

    minimum_gap = noise_ceiling + 2.0
    maximum_gap = 0.85 * signal_floor
    already_viable = baseline_fired_rate <= 0.01 and stimulated_fired_rate >= 0.90
    trace_viable = (
        baseline_fired_rate <= 0.01
        and math.isfinite(maximum_gap)
        and maximum_gap > minimum_gap
    )
    viable = already_viable or trace_viable

    selected: dict[str, float] | None = (
        {
            "threshold": float(config.threshold),
            "synaptic_weight": float(config.synaptic_weight),
            "i_synin_gm": float(config.i_synin_gm),
        }
        if already_viable
        else None
    )

    per_neuron: list[dict[str, float | int]] = []
    for neuron, coordinate in enumerate(result.physical_coordinates):
        per_neuron.append(
            {
                "neuron": neuron,
                "physical_coordinate": coordinate,
                "baseline_cadc": float(baseline_reference[:, neuron].mean()),
                "baseline_excursion_q99": float(
                    torch.quantile(
                        baseline_excursion[:, neuron].to(torch.float64), 0.99
                    )
                ),
                "psp_peak_q10": float(
                    torch.quantile(peak_delta[:, neuron].to(torch.float64), 0.10)
                ),
                "psp_peak_median": float(peak_delta[:, neuron].median()),
                "baseline_fired_rate": float(
                    baseline_fired[:, neuron].float().mean()
                ),
                "stimulated_fired_rate": float(
                    stimulated_fired[:, neuron].float().mean()
                ),
            }
        )

    return {
        "viable": viable,
        "reason": (
            "current operating point already produces input-triggered spikes"
            if already_viable
            else (
                "single PSP is measurable; select threshold with a raw-spike sweep"
                if trace_viable
                else "single PSP is not separable; increase i_synin_gm or input fan-in"
            )
        ),
        "selected": selected,
        "aggregate": {
            "baseline_fired_rate": baseline_fired_rate,
            "stimulated_fired_rate": stimulated_fired_rate,
            "noise_excursion_q99_cadc": noise_ceiling,
            "psp_peak_q10_cadc": signal_floor,
            "psp_peak_median_cadc": signal_median,
            "minimum_threshold_gap_cadc": minimum_gap,
            "maximum_threshold_gap_cadc": maximum_gap,
        },
        "per_neuron": per_neuron,
    }
