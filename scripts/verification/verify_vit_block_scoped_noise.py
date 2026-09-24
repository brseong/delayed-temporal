"""Verify exact evaluation prefixes and ViT encoder-block timing-noise scope."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation.error_analysis_vit import select_evaluation_examples
from utils.transforms.noise import (
    gaussian_time_noise_is_active,
    gaussian_time_noise_scope,
    get_gaussian_noise_stats,
    get_gaussian_time_noise,
    set_gaussian_time_noise,
)
from utils.transforms.potential_to_spike import neg_identity_transform
from utils.transforms.types import Potential, PotentialBounds, SpikeSample
from utils.transformers.models.spiking_vit.modeling_spiking_vit import ViTEncoder


class FakeDataset:
    """Minimal ordered dataset used to verify prefix selection."""

    def __init__(self, values: tuple[int, ...]):
        self.values = values

    def __len__(self) -> int:
        return len(self.values)

    def select(self, indices: range) -> "FakeDataset":
        return FakeDataset(tuple(self.values[index] for index in indices))


class ScopeRecorder(nn.Module):
    """Record block activation and emit one event only in active scopes."""

    def __init__(self, observations: list[bool]):
        super().__init__()
        self.observations = observations

    def forward(self, potential: Potential) -> Potential:
        active = gaussian_time_noise_is_active()
        self.observations.append(active)
        if active:
            event = neg_identity_transform(
                potential.value,
                potential.domain,
                return_spike_sample=True,
                noise_site="verification.encoder",
            )
            assert isinstance(event, SpikeSample)
        return potential


def verify_prefix_selection() -> None:
    dataset = FakeDataset(tuple(range(6000)))
    assert select_evaluation_examples(
        dataset, evaluation_samples=500, quick_test=False
    ).values == tuple(range(500))
    assert len(
        select_evaluation_examples(dataset, evaluation_samples=5000, quick_test=False)
    ) == 5000
    assert len(
        select_evaluation_examples(dataset, evaluation_samples=0, quick_test=True)
    ) == 5000
    assert select_evaluation_examples(
        dataset, evaluation_samples=0, quick_test=False
    ) is dataset
    for samples, quick in ((500, True), (6001, False), (-1, False)):
        try:
            select_evaluation_examples(
                dataset, evaluation_samples=samples, quick_test=quick
            )
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("invalid evaluation prefix was accepted")


def verify_scope_rng_and_statistics() -> None:
    domain = PotentialBounds(-1.0, 1.0)
    value = torch.tensor([0.25], dtype=torch.float64)
    set_gaussian_time_noise(
        enabled=True,
        time_std_fraction=0.2,
        seed=71,
        explicit_scope_required=True,
    )
    generator = get_gaussian_time_noise().generator
    assert generator is not None
    before = generator.get_state().clone()
    with gaussian_time_noise_scope(active=False, label="vit.encoder.block.0"):
        event = neg_identity_transform(
            value,
            domain,
            return_spike_sample=True,
            noise_site="verification.inactive",
        )
    assert isinstance(event, SpikeSample)
    assert bool(event.fired.all())
    assert torch.equal(event.time, value.new_tensor([float(domain.max)]) - value)
    assert torch.equal(before, generator.get_state())
    assert get_gaussian_noise_stats() == {}

    with gaussian_time_noise_scope(active=True, label="vit.encoder.block.0"):
        neg_identity_transform(
            value,
            domain,
            return_spike_sample=True,
            noise_site="verification.active",
        )
    assert not torch.equal(before, generator.get_state())
    assert set(get_gaussian_noise_stats()) == {
        "vit.encoder.block.0/verification.active"
    }
    assert not gaussian_time_noise_is_active()


def verify_vit_prefix_scope() -> None:
    observations: list[bool] = []
    encoder = ViTEncoder.__new__(ViTEncoder)
    nn.Module.__init__(encoder)
    encoder.layer = nn.ModuleList([ScopeRecorder(observations) for _ in range(3)])
    encoder.time_noise_vit_first_block_count = 2
    set_gaussian_time_noise(
        enabled=True,
        time_std_fraction=0.0,
        seed=72,
        explicit_scope_required=True,
    )
    potential = Potential(
        torch.tensor([0.0], dtype=torch.float64),
        PotentialBounds(-1.0, 1.0),
    )
    output = encoder(potential)
    assert output == potential
    assert observations == [True, True, False]
    assert set(get_gaussian_noise_stats()) == {
        "vit.encoder.block.0/verification.encoder",
        "vit.encoder.block.1/verification.encoder",
    }

    observations.clear()
    encoder.time_noise_vit_first_block_count = 0
    set_gaussian_time_noise(
        enabled=True,
        time_std_fraction=1.0,
        seed=73,
        explicit_scope_required=True,
    )
    assert encoder(potential) == potential
    assert observations == [False, False, False]
    assert get_gaussian_noise_stats() == {}

    set_gaussian_time_noise(enabled=True, time_std_fraction=0.0, seed=74)
    neg_identity_transform(
        potential.value,
        potential.domain,
        return_spike_sample=True,
        noise_site="verification.global",
    )
    assert set(get_gaussian_noise_stats()) == {"verification.global"}


# @lat: [[evaluation#Evaluation and Verification#ViT-B Cumulative Encoder Block Timing Noise#Verification#Block scope]]
def main() -> None:
    try:
        verify_prefix_selection()
        verify_scope_rng_and_statistics()
        verify_vit_prefix_scope()
    finally:
        set_gaussian_time_noise(enabled=False)
    print("ViT block-scoped timing-noise verification passed.")


if __name__ == "__main__":
    main()
