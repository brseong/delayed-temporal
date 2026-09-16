#!/usr/bin/env python3
"""Verify text evaluator collection, frozen reuse and local accuracy without assets."""

from __future__ import annotations

import contextlib
from dataclasses import replace
import importlib
import io
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from datasets import Dataset

from scripts.evaluation import text_calibration_runtime
from scripts.evaluation.text_calibration_runtime import (
    ClassificationProgress,
    make_text_dataloader,
    model_state_sha256,
    validate_text_calibration_arguments,
)
from utils.transforms.calibration import load_calibration_table
from utils.transformers.optional_tensorboard import create_summary_writer


class TinyTokenizer:
    name_or_path = "local-text-test"
    vocab_size = 32
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    unk_token_id = 3
    sep_token_id = 2
    cls_token_id = 1
    mask_token_id = 4
    padding_side = "right"
    truncation_side = "right"
    special_tokens_map = {}

    def get_vocab(self):
        return {str(index): index for index in range(self.vocab_size)}

    def __call__(self, texts, *, padding, truncation, max_length):
        assert padding == "max_length" and truncation is True
        ids, masks = [], []
        for text in texts:
            sequence = [1] + [5 + ord(char) % 20 for char in text][:max_length - 2] + [2]
            length = len(sequence)
            ids.append(sequence + [0] * (max_length - length))
            masks.append([1] * length + [0] * (max_length - length))
        return {
            "input_ids": ids, "attention_mask": masks,
            "token_type_ids": [[0] * max_length for _ in texts],
        }


def reject(function, error_type=(ValueError, RuntimeError, TypeError, FileNotFoundError)):
    try:
        function()
    except error_type:
        return
    raise AssertionError("invalid request was accepted")


def family_arguments(family, path):
    module = importlib.import_module("scripts.evaluation.error_analysis_" + family)
    with patch.object(sys, "argv", ["evaluator", "--device", "cpu", "--no-tensorboard"]):
        args = module.parse_arguments()
    args = replace(
        args, model_backend="spiking", model_id="local-tiny-" + family,
        experiment_name="unit_text_" + family + "_" + path.parent.name,
        dtype="float64", batch_size=2, max_length=8,
        calibration_mode="collect", calibration_path=str(path),
        calibration_samples=4, calibration_bins=32, theta=40.0,
    )
    return module, args


def verify_arguments_and_progress(root):
    _, args = family_arguments("bert", root / "arguments.json")
    validate_text_calibration_arguments(args)
    for changes in (
        {"model_backend": "hf"}, {"dtype": "float16"}, {"calibration_samples": 0},
        {"calibration_bins": 1}, {"calibration_seed": -1},
        {"calibration_lower_quantile": 0.9, "calibration_upper_quantile": 0.1},
        {"calibration_margin_fraction": float("nan")},
        {"gaussian_time_noise": True}, {"collect_quantiles": True},
        {"calibration_mode": "inference"}, {"max_length": 0},
    ):
        reject(lambda changes=changes: validate_text_calibration_arguments(replace(args, **changes)))
    progress = ClassificationProgress(3, 2)
    with contextlib.redirect_stdout(io.StringIO()) as output:
        progress.update(torch.tensor([0, 1]), torch.tensor([0, 0]))
        progress.update(torch.tensor([1]), torch.tensor([1]))
        assert progress.final_accuracy() == 2 / 3
    assert output.getvalue().count("Evaluation progress:") == 2
    assert "Correct/total: 2/3" in output.getvalue()
    reject(ClassificationProgress(1, 1).final_accuracy)
    print("PASS arguments and immediate local progress")


def verify_dataset_inputs():
    dataset = Dataset.from_dict({"sentence": ["one", "two", "three"], "label": [0, 1, 0]})
    loader = make_text_dataloader(
        dataset, TinyTokenizer(), text_column="sentence", max_length=8,
        batch_size=2, include_labels=True,
    )
    batches = list(loader)
    assert batches[0]["labels"].tolist() == [0, 1]
    assert "token_type_ids" in batches[0]
    collection = make_text_dataloader(
        dataset.remove_columns("label"), TinyTokenizer(), text_column="sentence",
        max_length=8, batch_size=2, include_labels=False,
    )
    assert all("labels" not in batch for batch in collection)
    reject(lambda: make_text_dataloader(
        dataset.remove_columns("label"), TinyTokenizer(), text_column="sentence",
        max_length=8, batch_size=2, include_labels=True,
    ))
    print("PASS sample order, labels and model input columns")


# @lat: [[text-calibration#Text Model Calibration#Evaluator Lifecycle]]
def verify_family_lifecycle(root, family):
    module, args = family_arguments(family, root / (family + ".json"))
    prefix = "Bert" if family == "bert" else "Roberta"
    config_type = getattr(module, prefix + "Config")
    model_type = getattr(module, prefix + "ForSequenceClassification")
    train = Dataset.from_dict({
        "sentence": ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"],
        "label": [0, 1, 0, 1, 0, 1],
    })
    validation = Dataset.from_dict({"sentence": ["seven", "eight", "nine"], "label": [0, 1, 0]})
    loads = []
    created = []

    def fake_load(_name, *_config, split, cache_dir):
        assert cache_dir == args.cache_dir
        loads.append(split)
        return train if split == "train" else validation

    def fake_config(*_args, **_kwargs):
        return config_type(
            vocab_size=32, hidden_size=8, intermediate_size=16,
            num_hidden_layers=1, num_attention_heads=2,
            max_position_embeddings=32, num_labels=2,
            hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
            layer_norm_eps=1.0e-12, theta=40.0, tau_s=1.0,
            pad_token_id=0,
        )

    def fake_model(_model_id, *, config, attn_implementation):
        assert attn_implementation == "spiking_sdpa"
        config._attn_implementation = attn_implementation
        torch.manual_seed(157)
        model = model_type(config)
        created.append(model)
        return model

    def no_tensorboard(*, log_dir, enabled):
        assert enabled is False
        assert not Path(log_dir).exists()
        return create_summary_writer(log_dir=log_dir, enabled=False)

    def run(run_args):
        with (
            patch.object(module, "load_dataset", side_effect=fake_load),
            patch.object(module.AutoTokenizer, "from_pretrained", return_value=TinyTokenizer()),
            patch.object(config_type, "from_pretrained", side_effect=fake_config),
            patch.object(model_type, "from_pretrained", side_effect=fake_model),
            patch.object(module, "create_summary_writer", side_effect=no_tensorboard),
            patch.object(module.wandb, "init"), patch.object(module.wandb, "log"),
            patch.object(module.wandb, "finish"),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            getattr(module, "evaluate_" + family + "_model")(run_args)
        return output.getvalue()

    collected = run(args)
    assert loads == ["train"], loads
    assert "Starting evaluation" not in collected and "Accuracy:" not in collected
    assert not (root / "runs").exists()
    table = load_calibration_table(args.calibration_path)
    assert table.metadata.model_family == family
    assert dict(table.metadata.model_options)["text_calibration_policy_version"] == 1
    assert len(table.layers) == 11
    assert all(layer.num_values == layer.histogram.num_values for layer in table.layers)
    for model in created:
        assert not model.training
        assert next(model.parameters()).dtype == torch.float64
        assert model.config.use_cache is False

    loads.clear()
    frozen = run(replace(args, calibration_mode="inference"))
    assert loads == ["train", "validation"], loads
    assert frozen.count("Evaluation progress:") == 2
    assert "total=3/3" in frozen and "Correct/total:" in frozen and "Prediction SHA256:" in frozen
    assert "Calibration[" in frozen
    old_bytes = Path(args.calibration_path).read_bytes()
    reject(lambda: run(args), FileExistsError)
    assert Path(args.calibration_path).read_bytes() == old_bytes
    reject(lambda: run(replace(args, calibration_mode="inference", dtype="float32")))
    reject(lambda: run(replace(args, calibration_mode="inference", calibration_seed=1)))
    reject(lambda: run(replace(args, calibration_mode="inference", calibration_bins=64)))
    reject(lambda: run(replace(args, calibration_mode="inference", max_length=7)))
    # Frozen setup must reject corrupt sites and incompatible family or epsilon.
    options = dict(table.metadata.model_options)
    old_policy = dict(options, text_calibration_policy_version=0)
    wrong_epsilon = dict(options, layer_norm_eps=1.0e-5)
    invalid_tables = (
        replace(table, layers=table.layers[:-1]),
        replace(table, layers=table.layers + (table.layers[0],)),
        replace(table, metadata=replace(table.metadata, model_family="gpt2")),
        replace(table, metadata=replace(table.metadata, model_options=tuple(sorted(old_policy.items())))),
        replace(table, metadata=replace(table.metadata, model_options=tuple(sorted(wrong_epsilon.items())))),
    )
    for invalid in invalid_tables:
        with patch.object(text_calibration_runtime, "load_calibration_table", return_value=invalid):
            reject(lambda: run(replace(args, calibration_mode="inference")))
    with patch.object(text_calibration_runtime, "model_state_sha256", return_value="0" * 64):
        # Evaluators import the hash helper directly; patch their actual binding.
        with patch.object(module, "model_state_sha256", return_value="0" * 64):
            reject(lambda: run(replace(args, calibration_mode="inference")))
    changed = created[-1]
    before_hash = model_state_sha256(changed)
    with torch.no_grad():
        next(changed.parameters()).add_(0.001)
    assert model_state_sha256(changed) != before_hash
    print(f"PASS {family}: two training passes, frozen reuse, metadata refusal, logging without tracking")


def main():
    runtime = ROOT / "artifacts" / "runtime" / "verification-text-evaluators"
    runtime.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=runtime, prefix="unit-") as directory:
        root = Path(directory)
        verify_arguments_and_progress(root)
        verify_dataset_inputs()
        verify_family_lifecycle(root, "bert")
        verify_family_lifecycle(root, "roberta")
    print("All text evaluator calibration checks passed")


if __name__ == "__main__":
    main()
