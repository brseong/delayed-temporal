"""Verify the analysis-only GELU atomic-operator ablation implementation."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
import sys


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from scripts.analysis.gelu_operator_ablation_vit import (
    gelu_operator_ablation,
    install_gelu_operator_ablation,
)
from utils.transformers.models.spiking_vit import modeling_spiking_vit
from utils.transforms.functions import gelu_approximation
from utils.transforms.noise import (
    get_gaussian_noise_stats,
    get_gaussian_time_noise,
    set_gaussian_time_noise,
)
from utils.transforms.types import PotentialBounds


_OPERATORS = ("multiplication", "exponential", "division")


def verify_gelu_operator_ablation() -> None:
    """Check deterministic parity, selection validation, and patch isolation.

    Every dense-operator subset must reduce to the maintained temporal composition
    when Gaussian noise is disabled. This establishes that an accuracy difference
    in the stochastic sweep comes from removing the selected operator's sampled
    events rather than from changing the nominal GELU formula or propagated rails.
    """
    # Disable Gaussian sampling before constructing the reference. A symmetric test
    # interval exercises negative, zero, and positive branches of the cubic gate.
    set_gaussian_time_noise(enabled=False)
    input_value = torch.linspace(-3.0, 3.0, 257, dtype=torch.float32)
    domain = PotentialBounds(-3.0, 3.0)
    reference_value, reference_domain = gelu_approximation(
        input_value,
        domain,
        theta=2000.0,
    )

    # Exhaust all eight subsets, including the fully temporal and fully dense ends
    # of the experiment matrix. Bounds are checked exactly because their propagation
    # is algebraic and should not depend on observed tensor values.
    for count in range(len(_OPERATORS) + 1):
        for selected in combinations(_OPERATORS, count):
            actual_value, actual_domain = gelu_operator_ablation(
                input_value,
                domain,
                dense_operators=frozenset(selected),
                theta=2000.0,
            )
            torch.testing.assert_close(
                actual_value,
                reference_value,
                rtol=2.0e-6,
                atol=2.0e-7,
            )
            assert actual_domain == reference_domain, (
                selected,
                actual_domain,
                reference_domain,
            )

    # Reject misspelled selections before evaluating any operator; silently ignoring
    # one would mislabel an expensive model-scale run.
    try:
        gelu_operator_ablation(
            input_value,
            domain,
            dense_operators=frozenset({"multiplicaton"}),
            theta=2000.0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("accepted an unknown GELU operator name")

    # Installation must replace only the symbol resolved by the local ViT adapter.
    # Restore it afterward so this verification has no process-wide residual effect.
    original_vit_symbol = modeling_spiking_vit.gelu_approximation
    try:
        install_gelu_operator_ablation(frozenset({"division"}))
        assert modeling_spiking_vit.gelu_approximation is not original_vit_symbol
        assert gelu_approximation is original_vit_symbol
    finally:
        modeling_spiking_vit.gelu_approximation = original_vit_symbol


def verify_gelu_operator_event_selection() -> None:
    """Check that each selected GELU operator stops exactly its own event draws."""
    input_value = torch.linspace(-1.0, 1.0, 17, dtype=torch.float32)
    domain = PotentialBounds(-1.0, 1.0)
    element_count = input_value.numel()

    # Zero standard deviation keeps every event deterministic while still entering
    # the event-aware paths and counters. It therefore isolates routing from random
    # numerical differences and consumes no generator samples.
    for count in range(len(_OPERATORS) + 1):
        for selected_tuple in combinations(_OPERATORS, count):
            selected = frozenset(selected_tuple)
            set_gaussian_time_noise(
                enabled=True,
                time_std=0.0,
                seed=7,
                device="cpu",
            )
            gelu_operator_ablation(
                input_value,
                domain,
                dense_operators=selected,
                theta=2000.0,
            )
            stats = get_gaussian_noise_stats()

            # GELU contains seven multiplication calls. Each call samples one event
            # per tensor element plus one scalar reference shared by the call.
            expected_multiplication_events = (
                0 if "multiplication" in selected else 7 * element_count
            )
            expected_multiplication_references = (
                0 if "multiplication" in selected else 7
            )
            assert stats.get("multiplication.data", {}).get("events", 0) == (
                expected_multiplication_events
            )
            assert stats.get("multiplication.reference", {}).get("events", 0) == (
                expected_multiplication_references
            )

            # The tanh decomposition owns exactly one exponential input and one
            # complete division per element. Division selection also removes its
            # internal exponential-difference re-encoding event.
            expected_exponential_events = (
                0 if "exponential" in selected else element_count
            )
            expected_division_events = (
                0 if "division" in selected else element_count
            )
            assert stats.get("exponential.input", {}).get("events", 0) == (
                expected_exponential_events
            )
            assert stats.get("division.numerator", {}).get("events", 0) == (
                expected_division_events
            )
            assert stats.get("division.denominator", {}).get("events", 0) == (
                expected_division_events
            )
            assert stats.get(
                "exponential_difference.internal",
                {},
            ).get("events", 0) == expected_division_events

            # With sigma zero all nominal events arrive, so any miss would indicate
            # an endpoint-classification or dense-routing regression in this fixture.
            assert all(site["misses"] == 0 for site in stats.values())

    # Repeat with nonzero sigma and compare final generator states. Dense helpers
    # shadow their omitted draws, so every condition must leave later model sites at
    # the same point in the run-wide stream as the fully noisy GELU condition.
    generator_states: dict[frozenset[str], torch.Tensor] = {}
    for count in range(len(_OPERATORS) + 1):
        for selected_tuple in combinations(_OPERATORS, count):
            selected = frozenset(selected_tuple)
            set_gaussian_time_noise(
                enabled=True,
                time_std=1.2648e-6,
                seed=11,
                device="cpu",
            )
            gelu_operator_ablation(
                input_value,
                domain,
                dense_operators=selected,
                theta=2000.0,
            )
            generator = get_gaussian_time_noise().generator
            assert isinstance(generator, torch.Generator)
            generator_states[selected] = generator.get_state().clone()

    # Equality here is stronger than matching draw counts: it catches a changed
    # tensor shape, dtype-specific sampling path, or reordered scalar reference.
    fully_noisy_state = generator_states[frozenset()]
    for selected, state in generator_states.items():
        assert torch.equal(state, fully_noisy_state), selected


if __name__ == "__main__":
    verify_gelu_operator_ablation()
    verify_gelu_operator_event_selection()
    print("GELU atomic-operator ablation verification passed.")
