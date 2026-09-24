"""Shared identities for the BrainScaleS-2 screening median model sweep."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import json
import math
from pathlib import Path
from typing import Any, Iterable

from scripts.runtime import identity


HARDWARE_SUMMARY_SHA256 = (
    "639a908ac4b86440ef706afcd98467117cbf0f1e5424ca3e8de5f4ef8802f6c0"
)
INITIAL_ALPHAS = ("0.003", "0.01", "0.03", "0.1", "0.3", "1")
MODELS = ("cct7", "imagenet_vit_small", "imagenet_vit_base")
SEEDS = (0, 1, 2)
DEADLINE_MARGIN_SIGMA_RATIO = 4.0


@dataclass(frozen=True)
class EncoderMeasurement:
    """One held-out screening median normalized by its own encoder span."""

    encoder: str
    physical_coordinate: int
    calibration_rt: float
    validation_rt: float
    calibration_sigma_s: float
    validation_sigma_s: float
    signal_span_s: float


@dataclass(frozen=True)
class ScreeningMedianPair:
    """Authenticated NP and NL screening median measurements."""

    summary_path: str
    summary_sha256: str
    phi_np: EncoderMeasurement
    phi_nl: EncoderMeasurement

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Cell:
    """One stochastic model-evaluation replica."""

    phase: str
    model: str
    alpha: str
    seed: int
    evaluation_samples: int

    @property
    def identity(self) -> tuple[str, str, str, int, int]:
        return (
            self.phase,
            self.model,
            self.alpha,
            self.seed,
            self.evaluation_samples,
        )

    @property
    def relative_path(self) -> Path:
        return (
            Path(self.phase)
            / self.model
            / f"alpha_{alpha_slug(self.alpha)}"
            / f"seed_{self.seed}"
        )


def _measurement(payload: dict[str, Any], encoder: str) -> EncoderMeasurement:
    median = payload["screening_median"]
    measurement = EncoderMeasurement(
        encoder=encoder,
        physical_coordinate=int(median["physical_coordinate"]),
        calibration_rt=float(median["calibration_rt"]),
        validation_rt=float(median["validation_rt"]),
        calibration_sigma_s=float(median["calibration_sigma_s"]),
        validation_sigma_s=float(median["validation_sigma_s"]),
        signal_span_s=float(median["signal_span_s"]),
    )
    for field in (
        measurement.calibration_rt,
        measurement.validation_rt,
        measurement.calibration_sigma_s,
        measurement.validation_sigma_s,
        measurement.signal_span_s,
    ):
        if not math.isfinite(field) or field <= 0.0:
            raise ValueError(f"{encoder} screening median contains a non-positive value")
    observed = measurement.validation_sigma_s / measurement.signal_span_s
    if not math.isclose(
        observed,
        measurement.validation_rt,
        rel_tol=2.0e-12,
        abs_tol=0.0,
    ):
        raise ValueError(f"{encoder} validation r_t is not normalized by its own span")
    return measurement


def load_screening_median(path: Path) -> ScreeningMedianPair:
    """Load only the authenticated held-out screening median pair."""

    resolved = path.resolve(strict=True)
    digest = identity.sha256_file(resolved)
    if digest != HARDWARE_SUMMARY_SHA256:
        raise ValueError("BrainScaleS-2 encoder summary checksum differs")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("BrainScaleS-2 encoder summary schema differs")
    pair = ScreeningMedianPair(
        summary_path=str(resolved),
        summary_sha256=digest,
        phi_np=_measurement(payload["phi_np"], "phi_np"),
        phi_nl=_measurement(payload["phi_nl"], "phi_nl"),
    )
    if (
        pair.phi_np.physical_coordinate != 184
        or pair.phi_nl.physical_coordinate != 1
    ):
        raise ValueError("screening median coordinates differ from the frozen evidence")
    return pair


def canonical_alpha(value: str | float | Decimal) -> str:
    """Return a positive finite multiplier with one stable decimal spelling."""

    if isinstance(value, bool):
        raise TypeError("alpha must be a decimal number")
    try:
        decimal = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError("alpha must be a decimal number") from error
    if not decimal.is_finite() or decimal <= 0:
        raise ValueError("alpha must be finite and positive")
    normalized = format(decimal.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized


def canonical_alphas(values: Iterable[str | float | Decimal]) -> tuple[str, ...]:
    """Canonicalize, deduplicate, and numerically order requested multipliers."""

    result = {canonical_alpha(value) for value in values}
    if not result:
        raise ValueError("at least one alpha is required")
    return tuple(sorted(result, key=Decimal))


def alpha_slug(value: str | float | Decimal) -> str:
    """Encode one canonical multiplier as a filesystem-safe directory name."""

    canonical = canonical_alpha(value)
    return canonical.replace(".", "p")


def scaled_fractions(
    pair: ScreeningMedianPair,
    alpha: str | float | Decimal,
) -> tuple[float, float]:
    """Scale the measured NP/NL pair without changing their ratio."""

    multiplier = Decimal(canonical_alpha(alpha))
    linear = Decimal(str(pair.phi_np.validation_rt)) * multiplier
    logarithmic = Decimal(str(pair.phi_nl.validation_rt)) * multiplier
    return float(linear), float(logarithmic)


def expected_cells(
    *,
    phase: str,
    alphas: Iterable[str | float | Decimal],
) -> tuple[Cell, ...]:
    """Build the deterministic three-model, three-seed cell population."""

    if phase not in {"smoke", "formal"}:
        raise ValueError("phase must be smoke or formal")
    sample_counts = {
        "smoke": {model: 500 for model in MODELS},
        "formal": {
            "cct7": 10_000,
            "imagenet_vit_small": 5_000,
            "imagenet_vit_base": 5_000,
        },
    }
    rows = tuple(
        Cell(
            phase=phase,
            model=model,
            alpha=alpha,
            seed=seed,
            evaluation_samples=sample_counts[phase][model],
        )
        for model in MODELS
        for alpha in canonical_alphas(alphas)
        for seed in SEEDS
    )
    if len(rows) != len({row.identity for row in rows}):
        raise AssertionError("screening median campaign contains duplicate cells")
    return rows


def protocol_id(identity_payload: dict[str, Any]) -> str:
    """Hash the scientific protocol independently of requested alpha values."""

    forbidden = {"alphas", "gpus", "runtime_root", "created_at", "requests"}
    if forbidden.intersection(identity_payload):
        raise ValueError("execution-only fields cannot enter the protocol identity")
    return identity.json_sha256(identity_payload)
