"""Canonical identities for the screening-median text-model sweep."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from scripts.experiments.screening_median_model_sweep import canonical_alphas


TEXT_MODELS = ("roberta_base", "gpt2")
TEXT_SEEDS = (0, 1, 2)
TEXT_INITIAL_ALPHAS = ("0.003", "0.01", "0.03", "0.05", "0.1", "0.3", "1")
TEXT_EVALUATION_SAMPLES = {"roberta_base": 872, "gpt2": 2_891}
TEXT_SMOKE_SAMPLES = {"roberta_base": 64, "gpt2": 64}


@dataclass(frozen=True)
class TextCell:
    """One seeded text-model evaluation under a scaled measured noise pair."""

    phase: str
    model: str
    alpha: str
    seed: int
    evaluation_samples: int

    @property
    def identity(self) -> tuple[str, str, str, int, int]:
        return self.phase, self.model, self.alpha, self.seed, self.evaluation_samples

    @property
    def relative_path(self) -> Path:
        return (
            Path(self.phase)
            / self.model
            / f"alpha_{self.alpha.replace('.', 'p')}"
            / f"seed_{self.seed}"
        )


def expected_text_cells(
    *, phase: str, alphas: Iterable[str]
) -> tuple[TextCell, ...]:
    """Return the deterministic model, multiplier, and seed product."""

    if phase not in {"smoke", "formal"}:
        raise ValueError("phase must be smoke or formal")
    samples = TEXT_SMOKE_SAMPLES if phase == "smoke" else TEXT_EVALUATION_SAMPLES
    rows = tuple(
        TextCell(
            phase=phase,
            model=model,
            alpha=alpha,
            seed=seed,
            evaluation_samples=samples[model],
        )
        for model in TEXT_MODELS
        for alpha in canonical_alphas(alphas)
        for seed in TEXT_SEEDS
    )
    if len(rows) != len({row.identity for row in rows}):
        raise AssertionError("text timing-noise sweep contains duplicate cells")
    return rows
