#!/usr/bin/env python3
"""Verify complete operator replacement through maintained model output heads."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
import sys
from typing import Iterable

import torch
from torch import nn
from transformers.pytorch_utils import Conv1D

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.vit_comparison_costs import estimate_vit_cost
from utils.transforms.noise import (
    clear_gaussian_noise_stats,
    get_gaussian_noise_stats,
    set_gaussian_time_noise,
)
from utils.transforms.types import Potential, PotentialBounds
from utils.transformers.models.spiking_ops import (
    SpikingConv2d,
    SpikingLayerNorm,
    SpikingLinear,
    _apply_dropout,
)


def _imports():
    """Import local model classes while suppressing vendored docstring diagnostics."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from utils.transformers.models.spiking_bert.configuration_bert import BertConfig
        from utils.transformers.models.spiking_bert.modeling_spiking_bert import (
            BertForSequenceClassification,
        )
        from utils.transformers.models.spiking_gpt2.configuration_gpt2 import GPT2Config
        from utils.transformers.models.spiking_gpt2.modeling_spiking_gpt2 import (
            GPT2DoubleHeadsModel,
            GPT2ForQuestionAnswering,
            GPT2ForSequenceClassification,
            GPT2ForTokenClassification,
            GPT2LMHeadModel,
            SpikingConv1D,
        )
        from utils.transformers.models.spiking_roberta.configuration_roberta import RobertaConfig
        from utils.transformers.models.spiking_roberta.modeling_spiking_roberta import (
            RobertaForCausalLM,
            RobertaForMaskedLM,
            RobertaForMultipleChoice,
            RobertaForQuestionAnswering,
            RobertaForSequenceClassification,
            RobertaForTokenClassification,
        )
        from utils.transformers.models.spiking_vit.configuration_spiking_vit import ViTConfig
        from utils.transformers.models.spiking_vit.modeling_spiking_vit import (
            ViTForImageClassification,
            ViTForMaskedImageModeling,
        )
    return locals()


def _configs(classes):
    vit = classes["ViTConfig"](
        hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=16, image_size=8, patch_size=4, encoder_stride=4,
        num_channels=3, num_labels=3, pixel_value_min=-2.0,
        pixel_value_max=2.0, hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    bert = classes["BertConfig"](
        vocab_size=32, hidden_size=8, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=16,
        max_position_embeddings=16, type_vocab_size=2, num_labels=2,
        hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
    )
    roberta = classes["RobertaConfig"](
        vocab_size=32, hidden_size=8, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=16,
        max_position_embeddings=16, type_vocab_size=1, num_labels=2,
        hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
        pad_token_id=0,
    )
    gpt2 = classes["GPT2Config"](
        vocab_size=32, n_embd=8, n_layer=1, n_head=2, n_positions=16,
        n_ctx=16, n_inner=16, num_labels=2, pad_token_id=0,
        embd_pdrop=0.0, attn_pdrop=0.0, resid_pdrop=0.0,
        summary_first_dropout=0.0, summary_last_dropout=0.0,
    )
    for config in (vit, bert, roberta, gpt2):
        config._attn_implementation = "eager"
    return vit, bert, roberta, gpt2


def _models(classes) -> Iterable[nn.Module]:
    vit, bert, roberta, gpt2 = _configs(classes)
    constructors = (
        (classes["ViTForImageClassification"], vit),
        (classes["ViTForMaskedImageModeling"], vit),
        (classes["BertForSequenceClassification"], bert),
        (classes["RobertaForCausalLM"], roberta),
        (classes["RobertaForMaskedLM"], roberta),
        (classes["RobertaForMultipleChoice"], roberta),
        (classes["RobertaForQuestionAnswering"], roberta),
        (classes["RobertaForSequenceClassification"], roberta),
        (classes["RobertaForTokenClassification"], roberta),
        (classes["GPT2DoubleHeadsModel"], gpt2),
        (classes["GPT2ForQuestionAnswering"], gpt2),
        (classes["GPT2ForSequenceClassification"], gpt2),
        (classes["GPT2ForTokenClassification"], gpt2),
        (classes["GPT2LMHeadModel"], gpt2),
    )
    for constructor, config in constructors:
        yield constructor(config).eval()


# @lat: [[output-heads#Output Head Coverage#Verification]]
def verify_enabled_topologies_contain_no_dense_learned_layers() -> None:
    """Default task wrappers contain no ordinary affine or normalization module."""
    classes = _imports()
    spiking_conv1d = classes["SpikingConv1D"]
    violations: list[str] = []
    for model in _models(classes):
        for name, module in model.named_modules():
            dense = (
                isinstance(module, nn.Linear) and not isinstance(module, SpikingLinear)
                or isinstance(module, nn.Conv2d) and not isinstance(module, SpikingConv2d)
                or isinstance(module, nn.LayerNorm) and not isinstance(module, SpikingLayerNorm)
                or isinstance(module, Conv1D) and not isinstance(module, spiking_conv1d)
            )
            if dense:
                violations.append(f"{type(model).__name__}:{name}:{type(module).__name__}")
    assert not violations, "dense learned modules remain:\n" + "\n".join(violations)

    _, _, roberta, gpt2 = _configs(classes)
    tied_cases = (
        (
            classes["RobertaForCausalLM"](roberta).eval(),
            "roberta.embeddings.word_embeddings",
        ),
        (
            classes["RobertaForMaskedLM"](roberta).eval(),
            "roberta.embeddings.word_embeddings",
        ),
        (classes["GPT2DoubleHeadsModel"](gpt2).eval(), "transformer.wte"),
        (classes["GPT2LMHeadModel"](gpt2).eval(), "transformer.wte"),
    )
    for model, input_name in tied_cases:
        output = model.get_output_embeddings()
        assert isinstance(output, SpikingLinear)
        assert output.weight is model.get_submodule(input_name).weight
        try:
            model.set_output_embeddings(nn.Linear(output.in_features, output.out_features))
        except TypeError:
            pass
        else:
            raise AssertionError("dense output embeddings were accepted")


def verify_evaluated_head_boundaries() -> None:
    """The four evaluated task wrappers pass Potential directly to their final head."""
    classes = _imports()
    vit, bert, roberta, gpt2 = _configs(classes)
    ids = torch.tensor([[1, 2, 3, 4]])
    mask = torch.ones_like(ids)
    cases = (
        (
            classes["ViTForImageClassification"](vit).eval(),
            "classifier", {"pixel_values": torch.randn(1, 3, 8, 8)}, (1, 3),
        ),
        (
            classes["BertForSequenceClassification"](bert).eval(),
            "classifier", {"input_ids": ids, "attention_mask": mask}, (1, 2),
        ),
        (
            classes["RobertaForSequenceClassification"](roberta).eval(),
            "classifier.out_proj", {"input_ids": ids, "attention_mask": mask}, (1, 2),
        ),
        (
            classes["GPT2LMHeadModel"](gpt2).eval(),
            "lm_head", {"input_ids": ids, "attention_mask": mask}, (1, 4, 32),
        ),
    )
    for model, name, kwargs, shape in cases:
        head = model.get_submodule(name)
        observed: list[type] = []
        hook = head.register_forward_pre_hook(
            lambda _module, inputs: observed.append(type(inputs[0]))
        )
        try:
            output = model(**kwargs)
        finally:
            hook.remove()
        assert observed == [Potential], (type(model).__name__, name, observed)
        assert tuple(output.logits.shape) == shape
        assert bool(torch.isfinite(output.logits).all())


def verify_dropout_and_noise_contract() -> None:
    """Dropout preserves fixed bounds and a final head participates in timing noise."""
    value = torch.tensor([[-0.5, 0.25]], dtype=torch.float64)
    potential = Potential(value, PotentialBounds(-1.0, 1.0))
    dropout = nn.Dropout(0.25).eval()
    assert _apply_dropout(dropout, potential).domain == potential.domain
    dropout.train()
    assert _apply_dropout(dropout, potential).domain == PotentialBounds(-4 / 3, 4 / 3)

    head = SpikingLinear(2, 3, dtype=torch.float64).eval()
    clear_gaussian_noise_stats()
    set_gaussian_time_noise(
        enabled=True, time_std_fraction=1.0e-5,
        deadline_margin_std_ratio=4.0, seed=7,
    )
    try:
        output = head(potential)
        stats = get_gaussian_noise_stats()
    finally:
        set_gaussian_time_noise(enabled=False)
        clear_gaussian_noise_stats()
    assert output.value.shape == (1, 3)
    assert stats["linear.data"]["events"] == 2
    assert stats["linear.reference"]["events"] == 1


def verify_cost_model_uses_the_evaluated_head() -> None:
    """ViT SOP accounting names the same classifier executed by evaluation."""
    result = estimate_vit_cost({
        "model_type": "vit", "hidden_act": "gelu", "image_size": 8,
        "patch_size": 4, "num_channels": 3, "hidden_size": 8,
        "intermediate_size": 16, "num_attention_heads": 2,
        "num_hidden_layers": 1, "num_labels": 3,
    })
    names = {row["component"] for row in result["breakdown"]}
    assert "classification_head" in names
    assert not any("assumed" in name for name in names)
    assert result["cost_model_version"] == "vit_composed_sop_v2"


def main() -> None:
    torch.manual_seed(20260923)
    torch.set_default_dtype(torch.float64)
    checks = (
        verify_enabled_topologies_contain_no_dense_learned_layers,
        verify_evaluated_head_boundaries,
        verify_dropout_and_noise_contract,
        verify_cost_model_uses_the_evaluated_head,
    )
    for check in checks:
        check()
        print(f"PASS: {check.__name__}")
    print("Output-head conversion verification passed")


if __name__ == "__main__":
    main()
