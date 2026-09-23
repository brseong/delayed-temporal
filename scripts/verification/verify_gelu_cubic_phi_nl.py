"""Verify the isolated GELU cubic construction comparison."""

from __future__ import annotations

from pathlib import Path
import sys


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from scripts.analysis.gelu_cubic_phi_nl_vit import (
    gelu_with_multiplication_cube,
    gelu_with_phi_nl_psi_ed_cube,
    install_gelu_cubic_implementation,
    install_phi_nl_psi_ed_cube,
    phi_nl_psi_ed_cube,
)
from utils.transformers.models.spiking_gpt2 import modeling_spiking_gpt2
from utils.transformers.models.spiking_roberta import modeling_spiking_roberta
from utils.transformers.models.spiking_vit import modeling_spiking_vit
from utils.transforms import functions
from utils.transforms.functions import gelu_approximation
from utils.transforms.noise import (
    get_gaussian_noise_stats,
    set_gaussian_time_noise,
)
from utils.transforms.types import PotentialBounds


# @lat: [[evaluation#Evaluation and Verification#Historical Noise and Ablation Sweeps#GELU Cubic Construction Comparison]]
def verify_phi_nl_psi_ed_cube() -> None:
    """Check signed values, scale invariance, bounds, and ViT patch isolation."""
    set_gaussian_time_noise(enabled=False)
    domain = PotentialBounds(-3.0, 3.0)
    input64 = torch.linspace(-3.0, 3.0, 601, dtype=torch.float64)

    reference_domain = None
    for tau_s in (0.5, 1.0, 2.0):
        actual, actual_domain = phi_nl_psi_ed_cube(
            input64,
            domain,
            tau_s=tau_s,
            magnitude_floor=1.0e-5,
        )
        torch.testing.assert_close(
            actual,
            input64.pow(3),
            rtol=2.0e-13,
            atol=3.0e-13,
        )
        if reference_domain is None:
            reference_domain = actual_domain
        else:
            assert actual_domain == reference_domain

    near_zero = torch.tensor(
        [-1.0e-6, 0.0, 1.0e-6],
        dtype=torch.float64,
    )
    zeroed, _ = phi_nl_psi_ed_cube(
        near_zero,
        domain,
        tau_s=1.0,
        magnitude_floor=1.0e-5,
    )
    torch.testing.assert_close(zeroed, torch.zeros_like(zeroed))

    range_limited, range_limited_domain = phi_nl_psi_ed_cube(
        torch.tensor([-3.0, -2.0, 0.0, 2.0, 3.0], dtype=torch.float64),
        domain,
        tau_s=1.0,
        magnitude_floor=1.0e-5,
    )
    torch.testing.assert_close(
        range_limited,
        torch.tensor([-27.0, -8.0, 0.0, 8.0, 27.0], dtype=torch.float64),
        rtol=2.0e-13,
        atol=3.0e-13,
    )
    assert range_limited_domain == PotentialBounds(-27.0, 27.0)

    input32 = torch.linspace(-3.0, 3.0, 6001, dtype=torch.float32)
    alternative, alternative_domain = gelu_with_phi_nl_psi_ed_cube(
        input32,
        domain,
    )
    baseline, baseline_domain = gelu_approximation(
        input32,
        domain,
    )
    torch.testing.assert_close(
        alternative,
        baseline,
        rtol=8.0e-4,
        atol=7.0e-4,
    )
    assert alternative_domain == baseline_domain

    multiplication, multiplication_domain = gelu_with_multiplication_cube(
        input32,
        domain,
    )
    torch.testing.assert_close(
        baseline,
        multiplication,
        rtol=8.0e-4,
        atol=7.0e-4,
    )
    assert multiplication_domain == baseline_domain

    # ViT, RoBERTa and GPT-2 all resolve the same canonical production GELU.
    assert modeling_spiking_vit.gelu_approximation is functions.gelu_approximation
    assert modeling_spiking_roberta.gelu_approximation is functions.gelu_approximation
    assert modeling_spiking_gpt2.gelu_approximation is functions.gelu_approximation

    original_vit_symbol = modeling_spiking_vit.gelu_approximation
    try:
        install_phi_nl_psi_ed_cube(magnitude_floor=1.0e-5)
        assert modeling_spiking_vit.gelu_approximation is not original_vit_symbol
        assert gelu_approximation is original_vit_symbol
        install_gelu_cubic_implementation(
            "multiplication",
            magnitude_floor=1.0e-5,
        )
        assert modeling_spiking_vit.gelu_approximation is not original_vit_symbol
    finally:
        modeling_spiking_vit.gelu_approximation = original_vit_symbol

    set_gaussian_time_noise(enabled=True, time_std_fraction=0.0, seed=0, device="cpu")
    try:
        gaussian_zero, gaussian_domain = phi_nl_psi_ed_cube(
            input32,
            domain,
            tau_s=1.0,
            magnitude_floor=1.0e-5,
        )
        stats = get_gaussian_noise_stats()
        assert stats["gelu.cubic.log_positive"]["events"] == input32.numel()
        assert stats["gelu.cubic.log_negative"]["events"] == input32.numel()
        assert stats["gelu.cubic.log_reference"]["events"] == 1
        assert stats["exponential_difference.internal"]["events"] == 2 * input32.numel()

        set_gaussian_time_noise(enabled=False)
        deterministic, deterministic_domain = phi_nl_psi_ed_cube(
            input32,
            domain,
            tau_s=1.0,
            magnitude_floor=1.0e-5,
        )
        torch.testing.assert_close(gaussian_zero, deterministic)
        assert gaussian_domain.max == deterministic_domain.max

        noisy_replicas = []
        for seed in (17, 17, 18):
            set_gaussian_time_noise(
                enabled=True,
                time_std_fraction=1.0e-3,
                deadline_margin_std_ratio=4.0e-3,
                seed=seed,
                device="cpu",
            )
            noisy, _ = phi_nl_psi_ed_cube(
                input32,
                domain,
                tau_s=1.0,
                magnitude_floor=1.0e-5,
            )
            noisy_replicas.append(noisy)
        assert torch.equal(noisy_replicas[0], noisy_replicas[1])
        assert not torch.equal(noisy_replicas[0], noisy_replicas[2])
    finally:
        set_gaussian_time_noise(enabled=False)


if __name__ == "__main__":
    verify_phi_nl_psi_ed_cube()
    print("GELU cubic phi_NL/psi_ED verification passed.")
