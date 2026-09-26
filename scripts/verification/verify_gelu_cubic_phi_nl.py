"""Verify the isolated GELU cubic construction comparison."""

from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from scripts.analysis.gelu_cubic_phi_nl_vit import (
    gelu_with_multiplication_cube,
    make_gelu_cubic_implementation,
    GeluCubicImplementation,
)
from utils.transformers.models.spiking_gpt2 import modeling_spiking_gpt2
from utils.transformers.models.spiking_roberta import modeling_spiking_roberta
from utils.transformers.models.spiking_vit import modeling_spiking_vit
from transformers.models.vit.configuration_vit import ViTConfig
from utils.transformers.models.spiking_vit.modeling_spiking_vit import (
    ViTForImageClassification,
    ViTIntermediate,
)
from utils.transforms import functions
from utils.transforms.functions import gelu_approximation, gelu_cubic_power_operator
from utils.transforms.noise import (
    get_gaussian_noise_stats,
    set_gaussian_time_noise,
)
from utils.transforms.types import PotentialBounds


def verify_phi_nl_psi_ed_cube() -> None:
    """Check signed values, scale invariance, bounds, and independent ViT construction."""
    set_gaussian_time_noise(enabled=False)
    domain = PotentialBounds(-3.0, 3.0)
    input64 = torch.linspace(-3.0, 3.0, 601, dtype=torch.float64)

    reference_domain = None
    for tau_s in (0.5, 1.0, 2.0):
        actual, actual_domain = gelu_cubic_power_operator(
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
    zeroed, _ = gelu_cubic_power_operator(
        near_zero,
        domain,
        tau_s=1.0,
        magnitude_floor=1.0e-5,
    )
    torch.testing.assert_close(zeroed, torch.zeros_like(zeroed))

    range_limited, range_limited_domain = gelu_cubic_power_operator(
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
    baseline, baseline_domain = gelu_approximation(
        input32,
        domain,
    )
    torch.testing.assert_close(
        baseline,
        torch.nn.functional.gelu(input32, approximate="tanh"),
        rtol=8.0e-4,
        atol=7.0e-4,
    )

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

    first = ViTIntermediate(
        ViTConfig(hidden_size=4, intermediate_size=8),
        gelu_operator=make_gelu_cubic_implementation(
            GeluCubicImplementation.MULTIPLICATION, magnitude_floor=1.0e-5,
        ),
    )
    second = ViTIntermediate(ViTConfig(hidden_size=4, intermediate_size=8))
    assert first.gelu_operator is not second.gelu_operator
    assert second.gelu_operator is gelu_approximation
    assert modeling_spiking_vit.gelu_approximation is gelu_approximation
    selected, selected_domain = first.gelu_operator(input32, domain)
    torch.testing.assert_close(selected, multiplication)
    assert selected_domain == multiplication_domain

    set_gaussian_time_noise(enabled=True, time_std_fraction=0.0, seed=0, device="cpu")
    try:
        gaussian_zero, gaussian_domain = gelu_cubic_power_operator(
            input32,
            domain,
            tau_s=1.0,
            magnitude_floor=1.0e-5,
        )
        stats = get_gaussian_noise_stats()
        assert stats["gelu.cubic.log_positive"]["events"] == int((input32 >= 1e-5).sum())
        assert stats["gelu.cubic.log_negative"]["events"] == int((input32 <= -1e-5).sum())
        assert stats["gelu.cubic.log_reference"]["events"] == 1
        assert stats["exponential_difference.internal"]["events"] == int((input32.abs() >= 1e-5).sum())

        set_gaussian_time_noise(enabled=False)
        deterministic, deterministic_domain = gelu_cubic_power_operator(
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
            noisy, _ = gelu_cubic_power_operator(
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


def verify_gelu_checkpoint_construction() -> None:
    """Load real weights with a constructor-selected GELU and execute every block."""
    set_gaussian_time_noise(enabled=False)
    # Transformers accepts the local adapter's extra serialized fields through its
    # configuration factory; the upstream dataclass constructor omits their types.
    config = ViTConfig.from_dict({
        "image_size": 4, "patch_size": 2, "hidden_size": 8, "intermediate_size": 16,
        "num_hidden_layers": 2, "num_attention_heads": 2, "num_labels": 3,
        "layer_norm_eps": 1e-6, "pixel_value_min": -1.0, "pixel_value_max": 1.0,
        "use_spiking_layernorm": True, "use_spiking_mlp": True,
        "spiking_mlp_exact_gelu": False,
    })
    config._attn_implementation = "eager"
    reference = ViTForImageClassification(config)
    reference.eval()
    selected = make_gelu_cubic_implementation(GeluCubicImplementation.PHI_NL_PSI_ED, magnitude_floor=1e-5)
    calls: list[PotentialBounds] = []

    def observe_gelu(
        values: torch.Tensor, bounds: PotentialBounds,
    ) -> tuple[torch.Tensor, PotentialBounds]:
        calls.append(bounds)
        return selected(values, bounds)

    with TemporaryDirectory(prefix="vit-gelu-construction-") as directory:
        reference.save_pretrained(directory)
        loaded = ViTForImageClassification.from_pretrained(
            directory, config=config, local_files_only=True,
            attn_implementation="eager", gelu_operator=observe_gelu,
        )
        loaded.eval()
        assert loaded.state_dict().keys() == reference.state_dict().keys()
        for name, parameter in loaded.state_dict().items():
            torch.testing.assert_close(parameter, reference.state_dict()[name])
        blocks = [module for module in loaded.modules() if isinstance(module, ViTIntermediate)]
        assert len(blocks) == config.num_hidden_layers
        assert all(block.gelu_operator is observe_gelu for block in blocks)
        assert all(
            module.gelu_operator is gelu_approximation
            for module in reference.modules() if isinstance(module, ViTIntermediate)
        )
        pixels = torch.linspace(-0.9, 0.9, 96).reshape(2, 3, 4, 4)
        with torch.inference_mode():
            expected = reference(pixel_values=pixels).logits
            actual = loaded(pixel_values=pixels).logits
        assert len(calls) == config.num_hidden_layers
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    verify_phi_nl_psi_ed_cube()
    verify_gelu_checkpoint_construction()
    print("GELU cubic phi_NL/psi_ED verification passed.")
