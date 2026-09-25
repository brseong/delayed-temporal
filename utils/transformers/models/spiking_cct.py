"""CCT-7 adapter built on the maintained TTFS ViT encoder blocks.

The CCT encoder block has the same pre-normalization attention and MLP order as
the ViT adapter. This module owns the CCT-specific topology and checkpoint-key
mapping; converted arithmetic is delegated to the maintained temporal operators.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from utils.transforms.functions import multiplication_operator, softmin_function
from utils.transforms.noise import clamp_gaussian_output
from utils.transforms.types import Potential, PotentialBounds
from utils.transformers.calibration import calibrated_potential, model_calibration_is_bound
from utils.transformers.models.spiking_ops import (
    SpikingConv2d,
    SpikingLayerNorm,
    SpikingLinear,
)
from utils.transformers.models.spiking_vit.configuration_spiking_vit import ViTConfig
from utils.transformers.models.spiking_vit.modeling_spiking_vit import (
    ViTEncoder,
    ViTLayer,
    _apply_norm,
)


CCT7_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CCT7_CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def cct7_config(*, converted: bool) -> ViTConfig:
    """Return the fixed official CCT-7/3x1 encoder geometry."""

    config = ViTConfig(
        hidden_size=256,
        num_hidden_layers=7,
        num_attention_heads=4,
        intermediate_size=512,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        layer_norm_eps=1.0e-5,
        image_size=32,
        patch_size=2,
        num_channels=3,
        qkv_bias=False,
        tau_s=1.0,
        use_spiking_layernorm=converted,
        spiking_ln_mul=converted,
        spiking_ln_log=converted,
        spiking_ln_expdiff=converted,
        use_spiking_mlp=converted,
        spiking_mlp_exact_gelu=False,
    )
    config._attn_implementation = "spiking_sdpa" if converted else "eager"
    return config


# @lat: [[evaluation#Compact Transformer Diagnostic]]
class CCT7ForImageClassification(nn.Module):
    """Official CCT-7/3x1 topology with converted learned operations.

    The converted path retains ``Potential`` metadata from the convolutional
    tokenizer through learned sequence pooling and the classifier. The seven
    attention/LayerNorm/MLP blocks remain owned by the existing ViT adapter.
    """

    def __init__(self, *, converted: bool) -> None:
        super().__init__()
        self.config = cct7_config(converted=converted)
        self.converted = bool(converted)
        convolution = SpikingConv2d if converted else nn.Conv2d
        self.tokenizer_conv = convolution(
            3, 256, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.tokenizer_activation = nn.ReLU()
        self.tokenizer_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.positional_emb = nn.Parameter(torch.zeros(1, 256, 256))
        self.encoder = CCTEncoder(self.config)
        if converted:
            self.final_norm: nn.Module = SpikingLayerNorm(
                256,
                eps=1.0e-5,
                tau_s=1.0,
                clip_margin=1.0e-5,
                use_spiking_mul=True,
                use_spiking_log=True,
                use_spiking_expdiff=True,
            )
        else:
            self.final_norm = nn.LayerNorm(256, eps=1.0e-5)
        affine = SpikingLinear if converted else nn.Linear
        self.attention_pool = affine(256, 1)
        self.classifier = affine(256, 10)

    @staticmethod
    def _input_domain() -> PotentialBounds:
        """Return the fixed scalar range produced by CIFAR-10 normalization."""

        channel_lower = tuple(
            -mean / std
            for mean, std in zip(CCT7_CIFAR10_MEAN, CCT7_CIFAR10_STD, strict=True)
        )
        channel_upper = tuple(
            (1.0 - mean) / std
            for mean, std in zip(CCT7_CIFAR10_MEAN, CCT7_CIFAR10_STD, strict=True)
        )
        return PotentialBounds(min(channel_lower), max(channel_upper))

    def _tokenizer_domain(self) -> PotentialBounds:
        """Derive one analytic tokenizer-output range from weights and preprocessing."""

        weight = self.tokenizer_conv.weight.detach().to(dtype=torch.float64)
        means = torch.tensor(CCT7_CIFAR10_MEAN, dtype=torch.float64, device=weight.device)
        stds = torch.tensor(CCT7_CIFAR10_STD, dtype=torch.float64, device=weight.device)
        lower = (-means / stds).view(1, 3, 1, 1)
        upper = ((1.0 - means) / stds).view(1, 3, 1, 1)
        low_terms = torch.minimum(weight * lower, weight * upper)
        high_terms = torch.maximum(weight * lower, weight * upper)
        conv_lower = low_terms.sum(dim=(1, 2, 3)).min().item()
        conv_upper = high_terms.sum(dim=(1, 2, 3)).max().item()
        # ReLU and max pooling preserve this non-negative scalar envelope.
        token_lower = max(0.0, conv_lower)
        token_upper = max(0.0, conv_upper)
        position = self.positional_emb.detach()
        result = PotentialBounds(
            token_lower + float(position.min().item()),
            token_upper + float(position.max().item()),
        )
        if not math.isfinite(float(result.min)) or not math.isfinite(float(result.max)):
            raise ValueError("CCT tokenizer bounds must be finite")
        return result

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.converted:
            convolution = self.tokenizer_conv(
                Potential(pixel_values, self._input_domain())
            )
            activated = self.tokenizer_activation(convolution.value)
            activated_domain = PotentialBounds(
                max(0.0, float(convolution.domain.min)),
                max(0.0, float(convolution.domain.max)),
            )
            pooled_spatial = self.tokenizer_pool(activated)
            tokens = pooled_spatial.flatten(2).transpose(1, 2) + self.positional_emb
            position = self.positional_emb.detach()
            token_domain = PotentialBounds(
                activated_domain.min + float(position.min().item()),
                activated_domain.max + float(position.max().item()),
            )
        else:
            tokens = self.tokenizer_pool(
                self.tokenizer_activation(self.tokenizer_conv(pixel_values))
            )
            tokens = tokens.flatten(2).transpose(1, 2) + self.positional_emb
            token_domain = self._tokenizer_domain()

        pot = self.encoder(Potential(tokens, token_domain))
        if isinstance(self.final_norm, SpikingLayerNorm):
            pot = self.final_norm(pot)
            if not isinstance(self.attention_pool, SpikingLinear):
                raise RuntimeError("converted sequence pool requires SpikingLinear")
            score = self.attention_pool(pot)
            score_domain = PotentialBounds(-score.domain.max, -score.domain.min)
            weights, weight_domain = softmin_function(
                -score.value.transpose(-1, -2),
                score_domain,
                tau=1.0,
            )
            weighted, _ = multiplication_operator(
                pot.value,
                pot.domain,
                weights.transpose(-1, -2),
                weight_domain,
            )
            pooled = clamp_gaussian_output(
                weighted.sum(dim=1),
                pot.domain,
                site="sequence_pool.output",
                name="sequence_pool_output",
            )
            if not isinstance(self.classifier, SpikingLinear):
                raise RuntimeError("converted classifier requires SpikingLinear")
            return self.classifier(Potential(pooled, pot.domain)).value
        else:
            hidden = self.final_norm(pot.value)
            weights = torch.softmax(self.attention_pool(hidden), dim=1).transpose(-1, -2)
            pooled = torch.matmul(weights, hidden).squeeze(-2)
            return self.classifier(pooled)


class CCTLayer(ViTLayer):
    """CCT residual ordering while reusing every maintained ViT operator module."""

    def forward(self, pot: Potential) -> Potential:
        pot_norm1 = _apply_norm(self.layernorm_before, pot)
        pot_attn = self.attention(pot_norm1)
        res1_value = pot_attn.value + pot.value
        res1_bounds = PotentialBounds(
            pot_attn.domain.min + pot.domain.min,
            pot_attn.domain.max + pot.domain.max,
        )
        if model_calibration_is_bound(self):
            pot_res1 = calibrated_potential(
                self,
                "attention_residual",
                res1_value,
                collection_bounds=res1_bounds,
            )
        else:
            pot_res1 = Potential(res1_value, res1_bounds)

        # Unlike standard ViT, CCT makes the normalized attention residual the
        # skip tensor around its MLP.  Operator implementations remain unchanged.
        pot_norm2 = _apply_norm(self.layernorm_after, pot_res1)
        pot_inter = self.intermediate(pot_norm2)
        pot_output = self.output(pot_inter, pot_norm2)
        if model_calibration_is_bound(self):
            return calibrated_potential(
                self,
                "output",
                pot_output.value,
                collection_bounds=pot_output.domain,
            )
        return pot_output


class CCTEncoder(ViTEncoder):
    """Seven CCT blocks under the ViT encoder calibration ownership boundary."""

    def __init__(self, config: ViTConfig) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.layer = nn.ModuleList(
            CCTLayer(config) for _ in range(config.num_hidden_layers)
        )
        self.gradient_checkpointing = False
        first_block_count = getattr(config, "time_noise_vit_first_block_count", None)
        if first_block_count is not None:
            if isinstance(first_block_count, bool) or not isinstance(
                first_block_count, int
            ):
                raise TypeError("time_noise_vit_first_block_count must be an integer")
            if not 0 <= first_block_count <= len(self.layer):
                raise ValueError(
                    "time_noise_vit_first_block_count must be inside the encoder depth"
                )
        self.time_noise_vit_first_block_count = first_block_count


def load_official_cct7_checkpoint(
    model: CCT7ForImageClassification,
    checkpoint: str | Path,
) -> None:
    """Map the official SHI-Labs CCT-7 checkpoint into the shared ViT blocks."""

    state = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError("CCT checkpoint must contain a state dictionary")

    with torch.no_grad():
        model.tokenizer_conv.weight.copy_(state["tokenizer.conv_layers.0.0.weight"])
        model.positional_emb.copy_(state["classifier.positional_emb"])
        model.attention_pool.weight.copy_(state["classifier.attention_pool.weight"])
        model.attention_pool.bias.copy_(state["classifier.attention_pool.bias"])
        model.final_norm.weight.copy_(state["classifier.norm.weight"])
        model.final_norm.bias.copy_(state["classifier.norm.bias"])
        model.classifier.weight.copy_(state["classifier.fc.weight"])
        model.classifier.bias.copy_(state["classifier.fc.bias"])

        for index, layer in enumerate(model.encoder.layer):
            prefix = f"classifier.blocks.{index}"
            layer.layernorm_before.weight.copy_(state[f"{prefix}.pre_norm.weight"])
            layer.layernorm_before.bias.copy_(state[f"{prefix}.pre_norm.bias"])
            layer.layernorm_after.weight.copy_(state[f"{prefix}.norm1.weight"])
            layer.layernorm_after.bias.copy_(state[f"{prefix}.norm1.bias"])

            qkv = state[f"{prefix}.self_attn.qkv.weight"]
            query, key, value = qkv.chunk(3, dim=0)
            layer.attention.attention.query.weight.copy_(query)
            layer.attention.attention.key.weight.copy_(key)
            layer.attention.attention.value.weight.copy_(value)
            layer.attention.output.dense.weight.copy_(state[f"{prefix}.self_attn.proj.weight"])
            layer.attention.output.dense.bias.copy_(state[f"{prefix}.self_attn.proj.bias"])
            layer.intermediate.dense.weight.copy_(state[f"{prefix}.linear1.weight"])
            layer.intermediate.dense.bias.copy_(state[f"{prefix}.linear1.bias"])
            layer.output.dense.weight.copy_(state[f"{prefix}.linear2.weight"])
            layer.output.dense.bias.copy_(state[f"{prefix}.linear2.bias"])
