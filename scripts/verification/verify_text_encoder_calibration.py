#!/usr/bin/env python3
"""Verify complete BERT and RoBERTa calibration with small CPU models."""

from __future__ import annotations

import itertools
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.verification.verify_calibration import _expect_raises
from utils.transforms.calibration import (
    CalibrationMode, calibration_table_to_dict, create_calibration_collector,
    create_calibration_runtime, get_calibration_clipping_report,
    load_calibration_table, save_calibration_table, validate_calibration_metadata,
    validate_calibration_table_specs, observe_calibration_activation,
    start_histogram_calibration_pass, finalize_calibration_collection,
)
from utils.transforms.noise import set_gaussian_time_noise
from utils.transforms.types import Potential, PotentialBounds
from utils.transformers.calibration import bind_model_calibration, clear_model_calibration, model_calibration_is_bound
from utils.transformers.integrations.spiking_sdpa_attention import spiking_sdpa_attention_forward
from utils.transformers.models.spiking_ops import SpikingLayerNorm
from utils.transformers.models.spiking_bert.configuration_bert import BertConfig
from utils.transformers.models.spiking_bert.modeling_spiking_bert import (
    ALL_ATTENTION_FUNCTIONS, BertForSequenceClassification, BertModel,
)
from utils.transformers.models.spiking_roberta.configuration_roberta import RobertaConfig
from utils.transformers.models.spiking_roberta.modeling_spiking_roberta import (
    RobertaForSequenceClassification, RobertaForMaskedLM, RobertaModel,
)
from utils.transformers.models.text_calibration import (
    build_text_calibration_metadata, calibrate_text_potential,
    collect_text_calibration_table, text_calibration_specs,
)


def make_model(family="bert", *, dtype=torch.float64, depth=1, head="classification", **changes):
    options = dict(
        vocab_size=32, hidden_size=8, intermediate_size=16, num_hidden_layers=depth,
        num_attention_heads=2, max_position_embeddings=16, num_labels=2, tau_s=1.0, layer_norm_eps=1e-12, clip_margin=1e-5,
        hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0, pad_token_id=0,
    )
    options.update(changes)
    cfg = (BertConfig if family == "bert" else RobertaConfig)(**options)
    cls = BertForSequenceClassification if family == "bert" else (
        RobertaForMaskedLM if head == "lm" else RobertaForSequenceClassification
    )
    model = cls(cfg).to(dtype=dtype).eval()
    cfg._attn_implementation = "spiking_sdpa"
    return model


def identity(model, family="bert", *, samples=2):
    return build_text_calibration_metadata(
        model_family=family, model_id="tiny-" + family, dataset_id="fixture",
        calibration_split="train", calibration_dataset_fingerprint="fixed-order",
        calibration_samples=samples, calibration_seed=0, tokenizer={"vocab": {"a": 2}},
        config=model.config, max_length=4, attention_implementation="spiking_sdpa",
        dtype=str(next(model.parameters()).dtype).removeprefix("torch."),
        evaluation_options={"checkpoint_sha256": "fixture", "batch_size": 2},
    )


def fixture():
    return [
        dict(input_ids=torch.tensor([2, 3, 4, 0]), attention_mask=torch.tensor([1, 1, 1, 0]),
             token_type_ids=torch.zeros(4, dtype=torch.long), labels=torch.tensor(1)),
        dict(input_ids=torch.tensor([2, 4, 5, 6]), attention_mask=torch.ones(4, dtype=torch.long),
             token_type_ids=torch.zeros(4, dtype=torch.long), labels=torch.tensor(0)),
    ]


def inputs(batch=2):
    return next(iter(DataLoader(fixture(), batch_size=batch)))


def collect(model, family="bert"):
    info = identity(model, family)
    specs = text_calibration_specs(model)
    collector = create_calibration_collector(info, specs, bin_count=16)
    table = collect_text_calibration_table(
        model, DataLoader(fixture(), batch_size=2), collector,
        device=torch.device("cpu"), expected_samples=2,
    )
    return table


# @lat: [[text-calibration#Text Model Calibration#Encoder Coverage]]
def verify_topology():
    for family, depth in itertools.product(("bert", "roberta"), (1, 2, 12, 24)):
        with torch.device("meta"):
            model = make_model(family, depth=depth)
        specs = text_calibration_specs(model)
        assert len(specs) == 9 * depth + 2
        keys = {(spec.module_name, spec.tensor_name) for spec in specs}
        assert len(keys) == len(specs)
        assert (family + ".embeddings.LayerNorm", "centered_input") in keys
        for index in range(depth):
            prefix = f"{family}.encoder.layer.{index}"
            for tensor in ("query", "key", "value", "attention_score"):
                assert (prefix + ".attention.self", tensor) in keys
            for owner in ("attention.output", "output"):
                assert (prefix + "." + owner, "residual") in keys
                assert (prefix + "." + owner + ".LayerNorm", "centered_input") in keys
            assert (prefix + ".intermediate", "activation_input") in keys
        assert not any(key[1] == "input" for key in keys)
    model = make_model("roberta", head="lm")
    keys = {(spec.module_name, spec.tensor_name) for spec in text_calibration_specs(model)}
    assert len(keys) == 12
    assert ("lm_head", "activation_input") in keys
    assert ("lm_head.layer_norm", "centered_input") in keys
    assert not any("pooler" in key[0] for key in keys)
    for name, module in model.named_modules():
        if isinstance(module, SpikingLayerNorm):
            assert module.eps == 1e-12 and module.clip_margin == 1e-5, name


def verify_collection_and_frozen_execution():
    for family, dtype, head in (
        ("bert", torch.float32, "classification"), ("bert", torch.float64, "classification"),
        ("roberta", torch.float32, "classification"), ("roberta", torch.float64, "classification"),
        ("roberta", torch.float32, "lm"), ("roberta", torch.float64, "lm"),
    ):
        torch.manual_seed(7)
        model = make_model(family, dtype=dtype, head=head)
        before = {key: tensor.clone() for key, tensor in model.state_dict().items()}
        table = collect(model, family)
        assert all(not model_calibration_is_bound(module) for module in model.modules())
        assert set(model.state_dict()) == set(before)
        assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
        validate_calibration_table_specs(table, text_calibration_specs(model))
        with tempfile.TemporaryDirectory(prefix="text-calibration-test-", dir=ROOT / "artifacts" / "runtime") as folder:
            path = Path(folder) / "table.json"
            save_calibration_table(table, path)
            assert load_calibration_table(path) == table
        runtime = create_calibration_runtime(CalibrationMode.INFERENCE, table, expected_metadata=identity(model, family))
        bind_model_calibration(model, runtime)
        snapshot = calibration_table_to_dict(table)
        batch = inputs()
        batch.pop("labels")
        with torch.no_grad():
            output = model(**batch).logits
            repeat = model(**batch).logits
            split = torch.cat([model(**{key: value[i:i + 1] for key, value in batch.items()}).logits for i in range(2)])
        assert torch.isfinite(output).all()
        assert torch.equal(output, repeat)
        torch.testing.assert_close(output, split, atol=2e-5 if dtype == torch.float32 else 1e-10, rtol=1e-5)
        assert calibration_table_to_dict(table) == snapshot
        report = get_calibration_clipping_report(runtime)
        assert len(report) == len(table.layers)
        assert all(row.num_values > 0 for row in report)
        clear_model_calibration(model)


def verify_ablations_and_fixed_bounds():
    for family, flags in itertools.product(("bert", "roberta"), itertools.product((False, True), repeat=3)):
        model = make_model(
            family, spiking_ln_mul=flags[0], spiking_ln_log=flags[1], spiking_ln_expdiff=flags[2],
        )
        specs = text_calibration_specs(model)
        assert len(specs) == 8 + 3 * int(any(flags))
        table = collect(model, family)
        runtime = create_calibration_runtime(CalibrationMode.INFERENCE, table, expected_metadata=identity(model, family))
        bind_model_calibration(model, runtime)
        batch = inputs(); batch.pop("labels")
        assert torch.isfinite(model(**batch).logits).all()
        clear_model_calibration(model)
    for family in ("bert", "roberta"):
        model = make_model(family, use_spiking_layernorm=False, use_spiking_mlp=False)
        model.config._attn_implementation = "eager"
        assert len(text_calibration_specs(model)) == 2
        collect(model, family)
    plain = torch.nn.Module()
    value = Potential(torch.tensor([2.0]), PotentialBounds(-3.0, 3.0))
    assert calibrate_text_potential(plain, "unused", value) is value


def verify_rejections():
    model = make_model()
    info = identity(model)
    for field in ("text_calibration_policy_version", "layer_norm_eps", "layer_norm_clip_margin"):
        changed = dict(info.model_options); changed[field] = 99
        _expect_raises(ValueError, lambda changed=changed: validate_calibration_metadata(
            replace(info, model_options=tuple(sorted(changed.items()))), info,
        ), "model_options")
    for attribute in ("is_decoder", "add_cross_attention"):
        previous = getattr(model.config, attribute, False)
        setattr(model.config, attribute, True)
        _expect_raises(ValueError, lambda: text_calibration_specs(model), "decoder")
        setattr(model.config, attribute, previous)
    specs = text_calibration_specs(model)
    collector = create_calibration_collector(info, specs, bin_count=16)
    _expect_raises(ValueError, lambda: collect_text_calibration_table(
        model, DataLoader(fixture(), batch_size=2, shuffle=True), collector,
        device=torch.device("cpu"), expected_samples=2,
    ), "sequential")
    _expect_raises(ValueError, lambda: collect_text_calibration_table(
        model, DataLoader(fixture(), batch_size=2), collector,
        device=torch.device("cpu"), expected_samples=3,
    ), "expected_samples")
    _expect_raises(ValueError, lambda: collect_text_calibration_table(
        model, DataLoader(fixture(), batch_size=2), collector,
        device=torch.device("cpu"), expected_samples=2, dtype=torch.float32,
    ), "dtype")
    class ChangingDataset:
        calls = 0
        def __len__(self):
            return 2
        def __getitem__(self, index):
            row = fixture()[index]
            self.calls += 1
            if self.calls > 2:
                row["input_ids"] = row["input_ids"] + 1
            return row
    _expect_raises(ValueError, lambda: collect_text_calibration_table(
        model, DataLoader(ChangingDataset(), batch_size=2),
        create_calibration_collector(info, specs, bin_count=16),
        device=torch.device("cpu"), expected_samples=2,
    ), "changed between passes")
    assert all(not model_calibration_is_bound(module) for module in model.modules())


def verify_attention_selected_ranges_and_noise():
    for family in ("bert", "roberta"):
        model = make_model(family)
        base = getattr(model, family)
        attention = base.encoder.layer[0].attention.self
        owner = family + ".encoder.layer.0.attention.self"
        specs = tuple(spec for spec in text_calibration_specs(model) if spec.module_name == owner)
        with torch.no_grad():
            for name, bias in (("query", 1.0), ("key", 50.0), ("value", 50.0)):
                getattr(attention, name).weight.zero_()
                getattr(attention, name).bias.fill_(bias)
        info = identity(model, family)
        collector = create_calibration_collector(info, specs, bin_count=16)
        bind_model_calibration(model, collector)
        attention(Potential(torch.ones(1, 2, 8, dtype=torch.float64), PotentialBounds(-2, 2)))
        assert collector.min_max_states[(owner, "key")].observed_max > 40.0
        assert collector.min_max_states[(owner, "value")].observed_max > 40.0
        clear_model_calibration(model)
        collector = create_calibration_collector(info, specs, bin_count=16)
        for phase in range(2):
            for spec in specs:
                observed = torch.tensor([-60.0 / 1.1, 60.0 / 1.1], dtype=torch.float64)
                observe_calibration_activation(collector, spec.module_name, spec.tensor_name, observed)
            if phase == 0:
                start_histogram_calibration_pass(collector)
        table = finalize_calibration_collection(collector)
        runtime = create_calibration_runtime(CalibrationMode.INFERENCE, table, expected_metadata=info)
        bind_model_calibration(model, runtime)
        hidden = Potential(torch.ones(1, 2, 8, dtype=torch.float64), PotentialBounds(-2, 2))
        result, _ = attention(hidden)
        torch.testing.assert_close(result.value, torch.full_like(result.value, 50.0), rtol=1e-10, atol=1e-10)
        assert result.domain.max > 40.0
        snapshot = calibration_table_to_dict(table)
        for std in (0.0, 1e-9):
            set_gaussian_time_noise(enabled=True, time_std_fraction=std, seed=3)
            first, _ = attention(hidden)
            set_gaussian_time_noise(enabled=True, time_std_fraction=std, seed=3)
            second, _ = attention(hidden)
            assert torch.isfinite(first.value).all()
            assert torch.equal(first.value, second.value)
            assert first.domain == second.domain == result.domain
        set_gaussian_time_noise(enabled=False)
        assert calibration_table_to_dict(table) == snapshot
        clear_model_calibration(model)


def main():
    ALL_ATTENTION_FUNCTIONS["spiking_sdpa"] = spiking_sdpa_attention_forward
    set_gaussian_time_noise(enabled=False)
    for verify in (verify_topology, verify_collection_and_frozen_execution,
                   verify_ablations_and_fixed_bounds, verify_rejections,
                   verify_attention_selected_ranges_and_noise):
        verify()
        print("PASS", verify.__name__, flush=True)


if __name__ == "__main__":
    main()
