"""CPU regression checks for the configured ViT LayerNorm epsilon."""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from itertools import product
from pathlib import Path
import sys
from unittest.mock import patch

import torch
from torch import nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from utils.transformers.models import spiking_ops
from utils.transformers.models.spiking_ops import SpikingLayerNorm, _apply_norm
from utils.transformers.models.spiking_vit.configuration_spiking_vit import ViTConfig
from utils.transformers.models.spiking_vit.modeling_spiking_vit import ViTModel
from utils.transforms.noise import set_gaussian_time_noise
from utils.transforms.types import Potential, PotentialBounds

FLAGS = tuple(product((False, True), repeat=3))
EPSILONS = (1.0e-12, 1.0e-6, 1.0e-5)
EXPECTED_NAMES = {
    "encoder.layer.0.layernorm_before", "encoder.layer.0.layernorm_after",
    "encoder.layer.1.layernorm_before", "encoder.layer.1.layernorm_after",
    "layernorm",
}


def _model(eps: float, flags: tuple[bool, bool, bool], *, spiking: bool = True) -> ViTModel:
    """Build actual small ViT modules without loading weights, images, or a device."""
    config = ViTConfig(
        hidden_size=4, intermediate_size=8, num_hidden_layers=2, num_attention_heads=2,
        image_size=16, patch_size=8, num_channels=3, layer_norm_eps=eps,
        theta=40.0, tau_s=1.0, use_spiking_layernorm=spiking,
        spiking_ln_mul=flags[0], spiking_ln_log=flags[1], spiking_ln_expdiff=flags[2],
    )
    with redirect_stdout(StringIO()):
        model = ViTModel(config, add_pooling_layer=False).double().eval()
    return model


def _norms(model: ViTModel) -> dict[str, nn.Module]:
    norms = {name: module for name, module in model.named_modules()
             if isinstance(module, (SpikingLayerNorm, nn.LayerNorm))}
    assert set(norms) == EXPECTED_NAMES
    return norms


# @lat: [[calibration#Layer-wise Calibration#Frozen Execution#ViT LayerNorm Epsilon]]
def verify_checkpoint_epsilon_all_sites() -> None:
    """Keep the configured epsilon at every actual ViT normalization site."""
    for eps, flags, spiking in product(EPSILONS, FLAGS, (False, True)):
        model = _model(eps, flags, spiking=spiking)
        assert model.config.layer_norm_eps == eps
        for name, norm in _norms(model).items():
            assert norm.eps == eps, (name, flags, spiking, norm.eps, eps)
            assert norm.normalized_shape == (4,)
            assert norm.weight.shape == norm.bias.shape == (4,)
            assert norm.weight.dtype == norm.bias.dtype == torch.float64
            assert norm.weight.device.type == norm.bias.device.type == "cpu"
            if spiking:
                assert isinstance(norm, SpikingLayerNorm)
                assert norm.clip_margin == 1.0e-5
                assert norm.theta == 40.0 and norm.tau_s == 1.0
                assert (norm.use_spiking_mul, norm.use_spiking_log,
                        norm.use_spiking_expdiff) == flags
            else:
                assert isinstance(norm, nn.LayerNorm)


def verify_low_variance_reference() -> None:
    """Expose the former epsilon mismatch without reaching the logarithmic floor."""
    values = torch.tensor([[[-0.001, 0.001, -0.001, 0.001]]], dtype=torch.float64)
    potential = Potential(values, PotentialBounds(-0.01, 0.01))
    for eps, flags, spiking in product(EPSILONS, FLAGS, (False, True)):
        for name, norm in _norms(_model(eps, flags, spiking=spiking)).items():
            with torch.no_grad():
                norm.weight.copy_(torch.tensor([0.6, 0.8, 1.2, 1.4], dtype=torch.float64))
                norm.bias.copy_(torch.tensor([-0.1, 0.0, 0.0, 0.1], dtype=torch.float64))
                actual = _apply_norm(norm, potential)
                reference = nn.functional.layer_norm(
                    values, norm.normalized_shape, norm.weight, norm.bias, eps,
                )
            torch.testing.assert_close(actual.value, reference, rtol=2.0e-10, atol=2.0e-10,
                                       msg=lambda message: f"{name}, {flags}, {spiking}, eps={eps}: {message}")
            assert bool(torch.isfinite(actual.value).all())
            assert actual.value.device.type == "cpu" and actual.value.dtype == torch.float64
            if eps == 1.0e-12:
                former = nn.functional.layer_norm(
                    values, norm.normalized_shape, norm.weight, norm.bias, 1.0e-5,
                )
                assert float((actual.value - former).detach().abs().max()) > 0.5


def verify_epsilon_is_not_log_floor() -> None:
    """Epsilon changes the variance stabilizer but not positive encoding bounds."""
    values = torch.tensor([[-0.001, 0.001, -0.001, 0.001]], dtype=torch.float64)
    original = spiking_ops.neg_log_transform
    for eps in (1.0e-12, 1.0e-3):
        layer = _model(eps, (True, True, True)).layernorm
        calls = []

        def capture(value, domain, *, tau_s=1.0, **kwargs):
            calls.append((value.detach().clone(), domain, tau_s))
            return original(value, domain, tau_s=tau_s, **kwargs)

        with torch.no_grad(), patch.object(spiking_ops, "neg_log_transform", capture):
            layer(Potential(values, PotentialBounds(-0.01, 0.01)))
        assert len(calls) == 3
        variance, positive, negative = calls
        assert layer.eps == eps and layer.clip_margin == 1.0e-5
        assert variance[1] == PotentialBounds(layer.clip_margin ** 2, 1600.0)
        assert variance[2] == 0.5
        torch.testing.assert_close(variance[0], values.new_tensor([[1.0e-6 + eps]]),
                                   rtol=2.0e-10, atol=1.0e-16)
        for value, domain, tau_s in (positive, negative):
            assert domain == PotentialBounds(1.0e-5, 40.0)
            assert tau_s == 1.0
            assert float(value.min()) == 1.0e-5
            assert float(value.max()) == 0.001


def verify_centering_clamp_unchanged() -> None:
    """Preserve the existing magnitude-before-variance approximation unchanged."""
    values = torch.tensor([[-30.0, 30.0, 30.0, 30.0]], dtype=torch.float64)
    potential = Potential(values, PotentialBounds(-30.0, 30.0))
    eps = 1.0e-12
    centered = (values - values.mean(dim=-1, keepdim=True)).clamp(-40.0, 40.0)
    variance = (centered.square().mean(dim=-1, keepdim=True) + eps).clamp(1.0e-5 ** 2, 1600.0)
    clipped_reference = centered / variance.sqrt()
    dense_reference = nn.functional.layer_norm(values, (4,), eps=eps)
    for flags in FLAGS:
        layer = _model(eps, flags).layernorm
        with torch.no_grad():
            actual = layer(potential).value
        expected = clipped_reference if any(flags) else dense_reference
        torch.testing.assert_close(actual, expected, rtol=2.0e-11, atol=2.0e-11)
        if any(flags):
            assert float((actual - dense_reference).abs().max()) > 0.05


def main() -> None:
    torch.set_num_threads(1)
    set_gaussian_time_noise(enabled=False)
    try:
        for check in (verify_checkpoint_epsilon_all_sites, verify_low_variance_reference,
                      verify_epsilon_is_not_log_floor, verify_centering_clamp_unchanged):
            check()
        print("ViT LayerNorm epsilon: four verification groups passed")
    finally:
        set_gaussian_time_noise(enabled=False)


if __name__ == "__main__":
    main()
