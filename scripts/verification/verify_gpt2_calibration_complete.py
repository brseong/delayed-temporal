#!/usr/bin/env python3
"""Verify complete GPT-2 calibration ranges and local evaluation progress."""

from __future__ import annotations

import itertools
import json
import math
import sys
import tempfile
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import AttentionInterface
from transformers.cache_utils import DynamicCache
from utils.transforms.calibration import (
    CalibrationMode, calibration_table_to_dict, create_calibration_collector,
    create_calibration_runtime, finalize_calibration_collection,
    get_calibration_clipping_report, load_calibration_table,
    observe_calibration_activation, save_calibration_table,
    start_histogram_calibration_pass, validate_calibration_metadata,
    validate_calibration_table_specs,
)
from utils.transforms.noise import set_gaussian_time_noise
from utils.transforms.types import Potential, PotentialBounds
from utils.transformers.calibration import (
    TEXT_CALIBRATION_POLICY_VERSION, bind_model_calibration, clear_model_calibration,
    model_calibration_is_bound,
)
from utils.transformers.integrations.spiking_sdpa_attention import spiking_sdpa_attention_forward
from utils.transformers.models.spiking_gpt2.calibration import (
    build_gpt2_calibration_metadata, collect_gpt2_calibration_table, gpt2_calibration_specs,
)
from utils.transformers.models.spiking_gpt2.configuration_gpt2 import GPT2Config
from utils.transformers.models.spiking_gpt2.modeling_spiking_gpt2 import GPT2LMHeadModel

AttentionInterface.register("spiking_sdpa", spiking_sdpa_attention_forward)


def expect_error(exception, action, message=""):
    try:
        action()
    except exception as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected {exception.__name__}")


def config(**changes):
    values = dict(
        vocab_size=32, n_positions=8, n_embd=8, n_layer=1, n_head=2, n_inner=16,
        theta=40.0, tau_s=1.0, layer_norm_epsilon=1e-5, clip_margin=1e-5,
        use_spiking_layernorm=True, use_spiking_mlp=True,
        spiking_ln_mul=True, spiking_ln_log=True, spiking_ln_expdiff=True,
        activation_function="gelu_new", resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        use_cache=False,
    )
    values.update(changes)
    cfg = GPT2Config(**values)
    cfg._attn_implementation = "spiking_sdpa"
    return cfg


def metadata(cfg, *, dtype="float64"):
    return build_gpt2_calibration_metadata(
        model_id="tiny-gpt2", dataset_id="tiny-text", calibration_split="train",
        calibration_dataset_fingerprint="fixed-training-inputs",
        calibration_samples=2, calibration_seed=0, tokenizer=SimpleNamespace(),
        config=cfg, max_length=4, attention_implementation=cfg._attn_implementation,
        dtype=dtype,
    )


def specs(model):
    return gpt2_calibration_specs(
        model, lower_quantile=0.0, upper_quantile=1.0, margin_fraction=0.05,
    )


def synthetic_table(identity, site_specs):
    collector = create_calibration_collector(identity, site_specs, bin_count=16)
    for phase in range(2):
        for spec in site_specs:
            radius = {"query": 60.0, "key": 70.0, "value": 80.0,
                      "attention_score": 100.0, "centered_input": 90.0}.get(spec.tensor_name, 30.0)
            observe_calibration_activation(
                collector, spec.module_name, spec.tensor_name,
                torch.tensor([-radius / 1.1, radius / 1.1], dtype=torch.float64),
            )
        if phase == 0:
            start_histogram_calibration_pass(collector)
    return finalize_calibration_collection(collector)


# @lat: [[text-calibration#Text Model Calibration#GPT-2 Coverage]]
def verify_topology_and_metadata():
    for depth in (1, 12, 24):
        cfg = config(n_layer=depth)
        with torch.device("meta"):
            model = GPT2LMHeadModel(cfg).double().eval()
        rows = specs(model)
        keys = {(row.module_name, row.tensor_name) for row in rows}
        assert len(rows) == len(keys) == 9 * depth + 1
        assert ("transformer.ln_f", "centered_input") in keys
        assert all(row.fixed_max > cfg.theta for row in rows if row.tensor_name == "attention_score")
        for index in range(depth):
            prefix = f"transformer.h.{index}"
            for name in ("query", "key", "value", "attention_score"):
                assert (prefix + ".attn", name) in keys
            assert (prefix + ".mlp", "activation_input") in keys

    for backend, use_ln, use_mlp, flags in itertools.product(
        ("eager", "spiking_sdpa"), (False, True), (False, True),
        tuple(itertools.product((False, True), repeat=3)),
    ):
        cfg = config(use_spiking_layernorm=use_ln, use_spiking_mlp=use_mlp,
                     spiking_ln_mul=flags[0], spiking_ln_log=flags[1], spiking_ln_expdiff=flags[2])
        cfg._attn_implementation = backend
        with torch.device("meta"):
            model = GPT2LMHeadModel(cfg).double().eval()
        assert len(specs(model)) == 3 + 4 * (backend == "spiking_sdpa") + 3 * (use_ln and any(flags))

    cfg = config()
    identity = metadata(cfg)
    options = dict(identity.model_options)
    assert options["text_calibration_policy_version"] == TEXT_CALIBRATION_POLICY_VERSION
    assert options["layer_norm_eps"] == cfg.layer_norm_epsilon
    assert options["layer_norm_clip_margin"] == cfg.clip_margin
    assert options["output_bounds_version"] == 3
    assert identity.dtype == "float64"
    changed = metadata(cfg, dtype="float32")
    expect_error(ValueError, lambda: validate_calibration_metadata(identity, changed))
    expect_error(ValueError, lambda: metadata(cfg, dtype="float16"))
    expect_error(ValueError, lambda: validate_calibration_metadata(
        identity, metadata(config(layer_norm_epsilon=1e-6)),
    ))
    changed_options = dict(identity.model_options)
    changed_options["checkpoint_sha256"] = "different-checkpoint"
    expect_error(ValueError, lambda: validate_calibration_metadata(
        identity, replace(identity, model_options=tuple(sorted(changed_options.items()))),
    ))


def verify_collection_and_replay():
    for dtype in (torch.float32, torch.float64):
        for backend in ("eager", "spiking_sdpa"):
            torch.manual_seed(1539)
            cfg = config()
            cfg._attn_implementation = backend
            model = GPT2LMHeadModel(cfg).to(dtype=dtype).eval()
            identity = metadata(cfg, dtype=str(dtype).removeprefix("torch."))
            rows = specs(model)
            collector = create_calibration_collector(identity, rows, bin_count=2048)
            examples = [
                {"input_ids": torch.tensor([1, 3, 4, 2]), "attention_mask": torch.ones(4, dtype=torch.long)},
                {"input_ids": torch.tensor([1, 5, 6, 2]), "attention_mask": torch.ones(4, dtype=torch.long)},
            ]
            loader = DataLoader(examples, batch_size=1, shuffle=False)
            output = StringIO()
            set_gaussian_time_noise(enabled=False)
            with redirect_stdout(output):
                table = collect_gpt2_calibration_table(
                    model, loader, collector, device=torch.device("cpu"), expected_samples=2,
                )
            events = [json.loads(line) for line in output.getvalue().splitlines()]
            assert len(events) == 4 and [event["pass"] for event in events] == [1, 1, 2, 2]
            assert events[-1]["observed_samples"] == 2
            assert not model_calibration_is_bound(model.transformer.h[0])
            assert len(table.layers) == len(rows)
            assert all(layer.num_values > 0 for layer in table.layers)
            assert all(layer.bounds.max > 0 and layer.bounds.min == -layer.bounds.max for layer in table.layers)
            with tempfile.TemporaryDirectory(prefix="gpt2-calibration-test-") as directory:
                path = Path(directory) / "table.json"
                save_calibration_table(table, path)
                assert calibration_table_to_dict(load_calibration_table(path)) == calibration_table_to_dict(table)
            validate_calibration_table_specs(table, rows)
            expect_error(ValueError, lambda: validate_calibration_table_specs(table, rows[:-1]))

            old_options = tuple((key, value) for key, value in identity.model_options
                                if key != "text_calibration_policy_version")
            old_table = replace(table, metadata=replace(identity, model_options=old_options))
            expect_error(ValueError, lambda: create_calibration_runtime(
                CalibrationMode.VALIDATE, old_table, expected_metadata=identity,
            ))
            state = create_calibration_runtime(CalibrationMode.VALIDATE, table, expected_metadata=identity)
            bind_model_calibration(model, state)
            tokens = torch.stack([item["input_ids"] for item in examples])
            mask = torch.ones_like(tokens)
            with torch.no_grad():
                together = model(input_ids=tokens, attention_mask=mask, use_cache=False).logits
                separately = torch.cat([
                    model(input_ids=tokens[index:index + 1], attention_mask=mask[index:index + 1],
                          use_cache=False).logits
                    for index in range(2)
                ])
            assert torch.isfinite(together).all()
            torch.testing.assert_close(together, separately, rtol=2e-4, atol=2e-5)
            assert calibration_table_to_dict(state.table) == calibration_table_to_dict(table)
            assert all(item.num_values > 0 for item in get_calibration_clipping_report(state))
            if dtype == torch.float64:
                set_gaussian_time_noise(enabled=True, time_std=0.0, seed=0, device=torch.device("cpu"))
                with torch.no_grad():
                    zero_noise = model(input_ids=tokens, attention_mask=mask, use_cache=False).logits
                assert torch.isfinite(zero_noise).all()
                torch.testing.assert_close(together, zero_noise, rtol=1e-5, atol=1e-6)
            clear_model_calibration(model, expected_state=state)
            set_gaussian_time_noise(enabled=False)

    class ChangedOrder:
        def __init__(self):
            self.calls = 0

        def __len__(self):
            return 2

        def __getitem__(self, index):
            phase = self.calls // 2
            self.calls += 1
            return examples[index if phase == 0 else 1 - index]

    with redirect_stdout(StringIO()):
        expect_error(ValueError, lambda: collect_gpt2_calibration_table(
            model, DataLoader(ChangedOrder(), batch_size=1, shuffle=False),
            create_calibration_collector(identity, rows, bin_count=16),
            device=torch.device("cpu"), expected_samples=2,
        ), "token order")
    assert not model_calibration_is_bound(model.transformer.h[0])


def verify_selected_ranges_and_cache():
    torch.manual_seed(1562)
    cfg = config()
    model = GPT2LMHeadModel(cfg).double().eval()
    table = synthetic_table(metadata(cfg), specs(model))
    state = create_calibration_runtime(CalibrationMode.VALIDATE, table, expected_metadata=metadata(cfg))
    bind_model_calibration(model, state)
    attention = model.transformer.h[0].attn
    fused = torch.tensor([0.01] * 8 + [50.0] * 8 + [50.0] * 8, dtype=torch.float64).reshape(1, 1, 24)
    projected = Potential(fused, PotentialBounds(-100.0, 100.0))
    received = {}

    def capture(module, query, key, value, mask, **kwargs):
        received.update(kwargs)
        assert torch.all(key == 50) and torch.all(value == 50)
        return spiking_sdpa_attention_forward(module, query, key, value, mask, **kwargs)

    AttentionInterface.register("spiking_sdpa", capture)
    cache = DynamicCache(config=cfg)
    try:
        with patch.object(attention.c_attn, "forward", return_value=projected), \
             patch.object(attention.c_proj, "forward", side_effect=lambda potential: potential):
            first, _ = attention(Potential(torch.zeros(1, 1, 8, dtype=torch.float64),
                                           PotentialBounds(-1.0, 1.0)),
                                 past_key_values=cache, cache_position=torch.tensor([0]))
            second, _ = attention(Potential(torch.zeros(1, 1, 8, dtype=torch.float64),
                                            PotentialBounds(-1.0, 1.0)),
                                  past_key_values=cache, cache_position=torch.tensor([1]))
        assert received["query_bounds"].max > 40
        assert received["key_bounds"].max > 40
        assert received["value_bounds"].max > 40
        assert first.domain == second.domain == received["value_bounds"]
        torch.testing.assert_close(first.value, torch.full_like(first.value, 50.0), rtol=1e-9, atol=1e-9)
        assert cache.get_seq_length(0) == 2
        cache._delayed_temporal_calibration_bounds.clear()
        expect_error(ValueError, lambda: attention(
            Potential(torch.zeros(1, 1, 8, dtype=torch.float64), PotentialBounds(-1.0, 1.0)),
            past_key_values=cache, cache_position=torch.tensor([2]),
        ), "cache")
    finally:
        AttentionInterface.register("spiking_sdpa", spiking_sdpa_attention_forward)

    layernorm = model.transformer.ln_f
    x = torch.tensor([[[-50.0, 50.0, -20.0, 20.0, -2.0, 2.0, -1.0, 1.0]]], dtype=torch.float64)
    result = layernorm(Potential(x, PotentialBounds(-100.0, 100.0)))
    expected = torch.nn.functional.layer_norm(
        x, (8,), layernorm.weight, layernorm.bias, layernorm.eps,
    )
    torch.testing.assert_close(result.value, expected, rtol=1e-9, atol=1e-9)
    ln_counts = next(item for item in get_calibration_clipping_report(state)
                     if item.module_name == "transformer.ln_f")
    assert ln_counts.underflows == ln_counts.overflows == 0
    clear_model_calibration(model, expected_state=state)


def verify_progress_and_optional_tensorboard():
    from scripts.evaluation.error_analysis_gpt2 import (
        parse_arguments, print_gpt2_progress, validate_gpt2_calibration_arguments,
    )
    from utils.transformers.optional_tensorboard import create_summary_writer
    with patch.object(sys, "argv", ["gpt2", "--model_backend", "spiking", "--dtype", "float64",
                                   "--calibration-mode", "collect", "--calibration-path", "table.json",
                                   "--no-tensorboard"]):
        args = parse_arguments()
    assert args.tensorboard is False
    assert validate_gpt2_calibration_arguments(args) is CalibrationMode.COLLECT
    with tempfile.TemporaryDirectory(prefix="gpt2-tensorboard-test-") as directory:
        path = Path(directory) / "runs"
        writer = create_summary_writer(log_dir=str(path), enabled=args.tensorboard)
        writer.add_scalar("loss", 0.2, 0)
        writer.close()
        assert not path.exists()
        expect_error(FileExistsError, lambda: validate_gpt2_calibration_arguments(
            replace(args, calibration_path=directory),
        ))
    stream = StringIO()
    with redirect_stdout(stream):
        print_gpt2_progress(batch_index=2, total_batches=4, total_examples=6,
                            total_loss=6.0, valid_batches=2, elapsed_seconds=5.0)
        print_gpt2_progress(batch_index=1, total_batches=4, total_examples=3,
                            total_loss=0.0, valid_batches=0, elapsed_seconds=2.0)
    rows = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert rows[0]["average_loss"] == 3.0
    assert rows[0]["perplexity"] == math.exp(3.0)
    assert rows[0]["estimated_remaining_seconds"] == 5.0
    assert rows[1]["average_loss"] is None and rows[1]["perplexity"] is None


def verify_strict_collection_and_outputs():
    from scripts.evaluation.error_analysis_gpt2 import (
        gpt2_loss_metrics, validate_gpt2_batch_output,
    )
    cfg = config()
    model = GPT2LMHeadModel(cfg).double().eval()
    identity = metadata(cfg)
    rows = specs(model)
    examples = [
        {"input_ids": torch.tensor([1, 3, 4, 2]), "attention_mask": torch.ones(4, dtype=torch.long)},
        {"input_ids": torch.tensor([1, 5, 6, 2]), "attention_mask": torch.ones(4, dtype=torch.long)},
    ]
    loader = DataLoader(examples, batch_size=1, shuffle=False)
    old_options = tuple((key, value) for key, value in identity.model_options
                        if key != "text_calibration_policy_version")
    cases = (
        (identity, tuple(row for row in rows if row.tensor_name not in {"query", "key", "value"}), "sites"),
        (replace(identity, model_family="bert"), rows, "family"),
        (replace(identity, dtype="float32"), rows, "dtype"),
        (replace(identity, model_options=old_options), rows, "policy"),
        (replace(identity, theta=20.0), rows, "configuration"),
    )
    set_gaussian_time_noise(enabled=False)
    for invalid_identity, invalid_rows, message in cases:
        collector = create_calibration_collector(invalid_identity, invalid_rows, bin_count=16)
        with patch.object(model, "forward", wraps=model.forward) as forward:
            expect_error(ValueError, lambda: collect_gpt2_calibration_table(
                model, loader, collector, device=torch.device("cpu"), expected_samples=2,
            ), message)
            forward.assert_not_called()
        assert not model_calibration_is_bound(model.transformer.h[0])

    output = SimpleNamespace(logits=torch.ones(1, 4, 32), loss=torch.tensor(2.0))
    validate_gpt2_batch_output(output, batch_index=1, calibrated=True)
    for name in ("logits", "loss"):
        for bad in (float("nan"), float("inf"), -float("inf")):
            corrupted = SimpleNamespace(**vars(output))
            setattr(corrupted, name, torch.tensor(bad))
            expect_error(FloatingPointError, lambda: validate_gpt2_batch_output(
                corrupted, batch_index=2, calibrated=True,
            ), name)
            validate_gpt2_batch_output(corrupted, batch_index=2, calibrated=False)
    for calibrated in (False, True):
        avg, ppl = gpt2_loss_metrics(total_loss=6.0, total_steps=2, expected_batches=2, calibrated=calibrated)
        assert avg == 3.0 and ppl == math.exp(3.0)
    for total_loss, steps in ((0.0, 0), (3.0, 1), (float("inf"), 2), (float("nan"), 2), (2000.0, 2)):
        expect_error(FloatingPointError, lambda: gpt2_loss_metrics(
            total_loss=total_loss, total_steps=steps, expected_batches=2, calibrated=True,
        ))


def main():
    torch.set_num_threads(1)
    for verify in (
        verify_topology_and_metadata, verify_collection_and_replay,
        verify_selected_ranges_and_cache, verify_progress_and_optional_tensorboard,
        verify_strict_collection_and_outputs,
    ):
        verify()
        print(f"PASS {verify.__name__}", flush=True)
    set_gaussian_time_noise(enabled=False)


if __name__ == "__main__":
    main()
