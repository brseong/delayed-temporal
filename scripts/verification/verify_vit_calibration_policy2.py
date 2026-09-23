#!/usr/bin/env python3
"""Verify explicit ViT calibration sites, frozen ranges and compatibility."""

from __future__ import annotations

import itertools
import math
import sys
import tempfile
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.verification.verify_calibration import _expect_raises
from utils.transforms.calibration import (
    CalibrationMode, CalibrationRangePolicy, LayerCalibrationSpec,
    calibration_table_from_dict, calibration_table_to_dict,
    create_calibration_collector, create_calibration_runtime,
    finalize_calibration_collection, get_calibration_clipping_report,
    observe_calibration_activation, start_histogram_calibration_pass,
    load_calibration_table, save_calibration_table,
    validate_calibration_metadata, validate_calibration_table_specs,
)
from utils.transforms.noise import set_gaussian_time_noise
from utils.transforms.types import Potential, PotentialBounds
from utils.transformers.calibration import (
    VIT_CALIBRATION_POLICY_VERSION, bind_model_calibration, clear_model_calibration,
    model_calibration_is_bound, validate_symmetric_encoder_bounds,
    vit_calibration_uses_explicit_bounds,
)
from utils.transformers.integrations.spiking_sdpa_attention import spiking_sdpa_attention_forward
from utils.transformers.models.spiking_ops import SpikingLayerNorm
from utils.transformers.models.spiking_vit.calibration import (
    build_vit_calibration_metadata, collect_vit_calibration_table, vit_calibration_specs,
)
from utils.transformers.models.spiking_vit.configuration_spiking_vit import ViTConfig
from utils.transformers.models.spiking_vit.modeling_spiking_vit import (
    ALL_ATTENTION_FUNCTIONS, ViTForImageClassification,
)


def config(**changes):
    values = dict(
        image_size=4, patch_size=2, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, num_labels=3, tau_s=1.0, layer_norm_eps=1e-12, clip_margin=1e-5,
        hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
        pixel_value_min=-1.0, pixel_value_max=1.0,
    )
    values.update(changes)
    return ViTConfig(**values)


def metadata(model_config):
    return build_vit_calibration_metadata(
        model_id="tiny-vit", dataset_id="imagenet-1k", calibration_split="train",
        calibration_dataset_fingerprint="fixed-training-inputs", calibration_samples=2,
        calibration_seed=0, processor=SimpleNamespace(), config=model_config,
        dtype="float64", attention_implementation="spiking_sdpa",
    )


def specs(model):
    return vit_calibration_specs(
        model, lower_quantile=0.0, upper_quantile=1.0, margin_fraction=0.05
    )


def make_table(identity, site_specs, values):
    collector = create_calibration_collector(identity, site_specs, bin_count=16)
    for phase in range(2):
        for spec, value in zip(site_specs, values, strict=True):
            observe_calibration_activation(collector, spec.module_name, spec.tensor_name, value)
        if phase == 0:
            start_histogram_calibration_pass(collector)
    return finalize_calibration_collection(collector)


# @lat: [[calibration#Layer-wise Calibration#Frozen Execution#ViT Calibration Policy 3]]
def verify_topology_and_ablations():
    """Discover real modules without loading checkpoint weights or using a GPU."""
    for depth, hidden, heads in ((12, 384, 6), (12, 768, 12), (24, 1024, 16)):
        cfg = config(num_hidden_layers=depth, hidden_size=hidden,
                     intermediate_size=4 * hidden, num_attention_heads=heads)
        with torch.device("meta"), redirect_stdout(StringIO()):
            model = ViTForImageClassification(cfg).double()
        cfg._attn_implementation = "spiking_sdpa"
        rows = specs(model)
        keys = {(row.module_name, row.tensor_name) for row in rows}
        assert len(keys) == len(rows) == 9 * depth + 1
        assert ("vit.layernorm", "centered_input") in keys
        for index in range(depth):
            prefix = f"vit.encoder.layer.{index}"
            for name in ("query", "key", "value", "attention_score"):
                assert (prefix + ".attention.attention", name) in keys
            for suffix in ("layernorm_before", "layernorm_after"):
                assert (prefix + "." + suffix, "centered_input") in keys
        assert all(row.fixed_max > 0 for row in rows if row.tensor_name == "attention_score")

    for attention, mlp, exact, layernorm, flags in itertools.product(
        ("eager", "spiking_sdpa"), (False, True), (False, True), (False, True),
        itertools.product((False, True), repeat=3),
    ):
        cfg = config(use_spiking_mlp=mlp, spiking_mlp_exact_gelu=exact,
                     use_spiking_layernorm=layernorm, spiking_ln_mul=flags[0],
                     spiking_ln_log=flags[1], spiking_ln_expdiff=flags[2])
        with torch.device("meta"), redirect_stdout(StringIO()):
            model = ViTForImageClassification(cfg).double()
        cfg._attn_implementation = attention
        rows = specs(model)
        expected = 2 + int(mlp and not exact) + 4 * int(attention == "spiking_sdpa")
        expected += 3 * int(layernorm and any(flags))
        assert len(rows) == expected


def verify_range_validation_and_identity():
    """Reject bad new ranges and unknown policies before any model is bound."""
    for dtype in (torch.float32, torch.float64):
        original = PotentialBounds(-3.0, 3.0)
        assert validate_symmetric_encoder_bounds(original, dtype, name="query") is original
        assert validate_symmetric_encoder_bounds(
            original, dtype, name="centered_input", positive_floor=1e-5
        ) is original
        for bounds in (PotentialBounds(0.0, 0.0), PotentialBounds(-1.0, 2.0),
                       PotentialBounds(-1e308, 1e308)):
            _expect_raises(ValueError, lambda bounds=bounds: validate_symmetric_encoder_bounds(
                bounds, dtype, name="bad.site"
            ), "bad.site")
    _expect_raises(ValueError, lambda: validate_symmetric_encoder_bounds(
        PotentialBounds(-1e-5, 1e-5), torch.float64, name="small", positive_floor=1e-5
    ), "small")
    _expect_raises(ValueError, lambda: validate_symmetric_encoder_bounds(
        PotentialBounds(-1e20, 1e20), torch.float32, name="square", positive_floor=1e-5
    ), "square")
    _expect_raises(ValueError, lambda: validate_symmetric_encoder_bounds(
        PotentialBounds(-1, 1), torch.float32, name="floor", positive_floor=1e-30
    ), "floor")
    _expect_raises(ValueError, lambda: validate_symmetric_encoder_bounds(
        PotentialBounds(-4e-23, 4e-23), torch.float32, name="squared", positive_floor=3e-23
    ), "squared logarithmic endpoints")

    cfg = config()
    identity = metadata(cfg)
    options = dict(identity.model_options)
    assert options["vit_calibration_policy_version"] == VIT_CALIBRATION_POLICY_VERSION
    assert options["operator_backed_output_head_version"] == 1
    assert options["output_bounds_version"] == 4
    assert options["layer_norm_eps"] == 1e-12
    assert options["layer_norm_clip_margin"] == identity.clip_margin == 1e-5
    for name in (
        "vit_calibration_policy_version",
        "operator_backed_output_head_version",
        "layer_norm_eps",
        "layer_norm_clip_margin",
    ):
        changed = tuple(option for option in identity.model_options if option[0] != name)
        _expect_raises(ValueError, lambda changed=changed: validate_calibration_metadata(
            replace(identity, model_options=changed), identity
        ), "model_options")
    validate_calibration_metadata(identity, identity)

    norm = SpikingLayerNorm(4, eps=1e-12).double()
    model = nn.Module()
    model.add_module("norm", norm)
    spec = LayerCalibrationSpec("norm", "centered_input", CalibrationRangePolicy.SIGNED_SYMMETRIC,
                                0.0, 1.0, 0.05)
    assert not vit_calibration_uses_explicit_bounds(norm)
    for field, value in (("layer_norm_eps", 1e-5), ("layer_norm_clip_margin", 2e-5)):
        changed = dict(identity.model_options)
        changed[field] = value
        mismatch = replace(identity, model_options=tuple(sorted(changed.items())))
        collector = create_calibration_collector(mismatch, (spec,), bin_count=16)
        _expect_raises(ValueError, lambda: bind_model_calibration(model, collector), "norm.centered_input")
        assert not model_calibration_is_bound(norm)
    for endpoints in ((0.0, 0.0), (-1e-6, 1e-6)):
        table = make_table(identity, (spec,), (torch.tensor(endpoints, dtype=torch.float64),))
        runtime = create_calibration_runtime(CalibrationMode.INFERENCE, table, expected_metadata=identity)
        _expect_raises(ValueError, lambda: bind_model_calibration(model, runtime), "norm.centered_input")
        assert not model_calibration_is_bound(norm)
        assert all(row.num_values == 0 for row in get_calibration_clipping_report(runtime))

    for supplied in (1, 2, 4, True):
        bad_options = dict(identity.model_options, vit_calibration_policy_version=supplied)
        bad_identity = replace(identity, model_options=tuple(sorted(bad_options.items())))
        collector = create_calibration_collector(bad_identity, (spec,), bin_count=16)
        _expect_raises(ValueError, lambda: bind_model_calibration(model, collector), "vit_calibration_policy_version")
        assert not model_calibration_is_bound(norm)

    bad_options = dict(identity.model_options, operator_backed_output_head_version=2)
    bad_identity = replace(identity, model_options=tuple(sorted(bad_options.items())))
    collector = create_calibration_collector(bad_identity, (spec,), bin_count=16)
    _expect_raises(
        ValueError,
        lambda: bind_model_calibration(model, collector),
        "operator-backed output head",
    )
    assert not model_calibration_is_bound(norm)
    legacy = replace(identity, model_options=tuple(
        option for option in identity.model_options if option[0] != "vit_calibration_policy_version"
    ))
    collector = create_calibration_collector(legacy, (spec,), bin_count=16)
    bind_model_calibration(model, collector)
    assert not vit_calibration_uses_explicit_bounds(norm)
    clear_model_calibration(model)

    attention = nn.Module()
    attention.add_module("query", nn.Linear(4, 4).float())
    model = nn.Module()
    model.add_module("attention", attention)
    query_spec = replace(spec, module_name="attention", tensor_name="query")
    table = make_table(identity, (query_spec,), (torch.tensor([-1e39, 1e39], dtype=torch.float64),))
    runtime = create_calibration_runtime(CalibrationMode.INFERENCE, table, expected_metadata=identity)
    _expect_raises(ValueError, lambda: bind_model_calibration(model, runtime), "attention.query")
    assert not model_calibration_is_bound(attention)


class FixedInputs(Dataset):
    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return {"pixel_values": self.values[index]}


def verify_centered_collection_and_reuse():
    """Collect signed centered values before clipping and freeze one reused table."""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = SpikingLayerNorm(4, eps=1e-12).double()

        def forward(self, pixel_values):
            return self.norm(Potential(pixel_values, PotentialBounds(-30, 30))).value

    model = Model().eval()
    identity = metadata(config())
    spec = LayerCalibrationSpec("norm", "centered_input", CalibrationRangePolicy.SIGNED_SYMMETRIC,
                                0.0, 1.0, 0.05)
    values = torch.tensor([[-30, 30, 30, 30], [30, -30, -30, -30]], dtype=torch.float64)
    collector = create_calibration_collector(identity, (spec,), bin_count=16)
    table = collect_vit_calibration_table(
        model, DataLoader(FixedInputs(values), batch_size=1), collector,
        device=torch.device("cpu"), dtype=torch.float64, expected_samples=2,
    )
    row = table.layers[0]
    assert row.num_values == 8 and row.observed_min == -45 and row.observed_max == 45
    assert row.bounds.min == -49.5 and row.bounds.max == 49.5
    assert row.histogram.num_values == 8
    assert row.histogram.underflows == row.histogram.overflows == 0
    assert not model_calibration_is_bound(model.norm)
    runtime = create_calibration_runtime(CalibrationMode.INFERENCE, table, expected_metadata=identity)
    bind_model_calibration(model, runtime)
    try:
        batched = model(values)
        separate = torch.cat([model(values[index:index+1]) for index in range(2)])
        torch.testing.assert_close(batched, separate, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(batched, nn.functional.layer_norm(values, (4,), eps=1e-12),
                                   rtol=1e-12, atol=1e-12)
        assert runtime.table == table
    finally:
        clear_model_calibration(model)


def verify_model_two_pass_and_persistence():
    """Execute both collection passes and a frozen tiny-model prediction."""
    torch.manual_seed(913)
    cfg = config()
    model = ViTForImageClassification(cfg).double().eval()
    cfg._attn_implementation = "spiking_sdpa"
    ALL_ATTENTION_FUNCTIONS.register("spiking_sdpa", spiking_sdpa_attention_forward)
    identity = metadata(cfg)
    rows = specs(model)
    assert len(rows) == 10
    values = torch.linspace(-0.9, 0.9, 96, dtype=torch.float64).reshape(2, 3, 4, 4)
    collector = create_calibration_collector(identity, rows, bin_count=16)
    table = collect_vit_calibration_table(
        model, DataLoader(FixedInputs(values), batch_size=1), collector,
        device=torch.device("cpu"), dtype=torch.float64, expected_samples=2,
    )
    assert len(table.layers) == 10
    assert all(layer.num_values > 0 and layer.histogram.underflows == layer.histogram.overflows == 0
               for layer in table.layers)
    assert not any(model_calibration_is_bound(module) for module in model.modules())
    validate_calibration_table_specs(table, rows)
    _expect_raises(ValueError, lambda: validate_calibration_table_specs(table, rows[:-1]), "site mismatch")
    assert calibration_table_from_dict(calibration_table_to_dict(table)) == table
    with tempfile.TemporaryDirectory(prefix="vit-calibration-policy2-") as directory:
        path = Path(directory) / "calibration.json"
        save_calibration_table(table, path)
        assert load_calibration_table(path) == table
        before = path.read_bytes()
        runtime = create_calibration_runtime(CalibrationMode.VALIDATE, table, expected_metadata=identity)
        bind_model_calibration(model, runtime)
        try:
            output = model(values).logits
            replay = model(values).logits
            assert output.shape == (2, 3) and bool(torch.isfinite(output).all())
            torch.testing.assert_close(output, replay, rtol=0, atol=0)
            assert all(row.num_values > 0 for row in get_calibration_clipping_report(runtime))
            assert path.read_bytes() == before
        finally:
            clear_model_calibration(model)


def main():
    torch.set_num_threads(1)
    set_gaussian_time_noise(enabled=False)
    with torch.no_grad():
        for check in (verify_topology_and_ablations, verify_range_validation_and_identity,
                      verify_centered_collection_and_reuse, verify_model_two_pass_and_persistence):
            check()
            print(f"PASS {check.__name__}")
    print("ViT calibration policy 3: four verification groups passed")


if __name__ == "__main__":
    main()
