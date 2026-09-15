#!/usr/bin/env python3
"""Verify stable text tokenizer identity without hiding preprocessing changes."""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers

from utils.transforms.calibration import validate_calibration_metadata
from utils.transformers.models.spiking_gpt2.calibration import build_gpt2_calibration_metadata
from utils.transformers.models.spiking_gpt2.configuration_gpt2 import GPT2Config
from utils.transformers.models.text_calibration import build_text_calibration_metadata
from utils.transformers.tokenizer_identity import tokenizer_backend_sha256


def tokenizer_fixture():
    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "[PAD]": 1, "Hello": 2, "world": 3}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = SimpleNamespace(
        name_or_path="fixed-tokenizer", vocab_size=4, backend_tokenizer=backend,
        get_vocab=backend.get_vocab, bos_token_id=0, eos_token_id=0,
        pad_token_id=1, unk_token_id=0, padding_side="right", truncation_side="right",
        special_tokens_map={"pad_token": "[PAD]", "unk_token": "[UNK]"},
    )
    return tokenizer


def metadata(family, tokenizer, *, max_length=8):
    common = dict(
        model_id="fixed-model", dataset_id="fixed-data", calibration_split="train",
        calibration_dataset_fingerprint="fixed-subset", calibration_samples=2,
        calibration_seed=0, tokenizer=tokenizer, max_length=max_length,
        attention_implementation="spiking_sdpa", dtype="float64",
    )
    if family == "gpt2":
        config = GPT2Config(n_positions=64, theta=40.0, tau_s=1.0, clip_margin=1e-5)
        return build_gpt2_calibration_metadata(config=config, **common)
    config = SimpleNamespace(
        max_position_embeddings=64, pad_token_id=1, theta=40.0, tau_s=1.0,
        layer_norm_eps=1e-12, clip_margin=1e-5,
    )
    return build_text_calibration_metadata(model_family=family, config=config, **common)


def expect_mismatch(left, right):
    try:
        validate_calibration_metadata(left, right)
    except ValueError:
        return
    raise AssertionError("changed tokenization must reject the calibration identity")


# @lat: [[text-calibration#Text Model Calibration#Evaluator Lifecycle]]
def verify_request_state_independence():
    for family in ("bert", "roberta", "gpt2"):
        tokenizer = tokenizer_fixture()
        backend = tokenizer.backend_tokenizer
        before_structure = backend.to_str()
        before = metadata(family, tokenizer)
        before_hash = tokenizer_backend_sha256(backend)
        backend.enable_padding(length=8, pad_id=1, pad_token="[PAD]")
        backend.enable_truncation(max_length=8)
        backend.encode_batch(["Hello world", "Hello"])
        assert backend.to_str() != before_structure
        assert tokenizer_backend_sha256(backend) == before_hash
        validate_calibration_metadata(before, metadata(family, tokenizer))
        backend.no_padding()
        backend.no_truncation()
        validate_calibration_metadata(before, metadata(family, tokenizer))
        expect_mismatch(before, metadata(family, tokenizer, max_length=16))
        for field, value in (("padding_side", "left"), ("truncation_side", "left"), ("pad_token_id", 0)):
            original = getattr(tokenizer, field)
            setattr(tokenizer, field, value)
            expect_mismatch(before, metadata(family, tokenizer))
            setattr(tokenizer, field, original)
        for field, value in (("padding", False), ("truncation", False)):
            preprocessing = json.loads(before.preprocessing)
            preprocessing[field] = value
            changed = replace(before, preprocessing=json.dumps(preprocessing, sort_keys=True, separators=(",", ":")))
            expect_mismatch(before, changed)
        backend.normalizer = normalizers.Lowercase()
        expect_mismatch(before, metadata(family, tokenizer))
        assert backend.encode("Hello").ids == [0]


def verify_structure_changes_are_rejected():
    tokenizer = tokenizer_fixture()
    structure = json.loads(tokenizer.backend_tokenizer.to_str())
    baseline = tokenizer_backend_sha256(tokenizer.backend_tokenizer)
    mutations = {
        "model": {"type": "WordLevel", "vocab": {"[UNK]": 0, "[PAD]": 1, "world": 2, "Hello": 3}, "unk_token": "[UNK]"},
        "normalizer": {"type": "Lowercase"},
        "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": {"type": "ByteLevel", "trim_offsets": False, "add_prefix_space": False, "use_regex": True},
        "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": False},
        "added_tokens": [{"id": 4, "content": "added", "single_word": False, "lstrip": False, "rstrip": False, "normalized": True, "special": False}],
    }
    for name, value in mutations.items():
        changed = copy.deepcopy(structure)
        changed[name] = value
        backend = SimpleNamespace(to_str=lambda changed=changed: json.dumps(changed))
        assert tokenizer_backend_sha256(backend) != baseline, name
    reordered = SimpleNamespace(to_str=lambda: json.dumps(structure, sort_keys=True, indent=4))
    assert tokenizer_backend_sha256(reordered) == baseline
    assert tokenizer_backend_sha256(None) is None
    try:
        tokenizer_backend_sha256(SimpleNamespace(to_str=lambda: "[]"))
    except ValueError:
        pass
    else:
        raise AssertionError("non-object backend must be rejected")


def main():
    verify_request_state_independence()
    verify_structure_changes_are_rejected()
    print("text tokenizer identity verification passed")


if __name__ == "__main__":
    main()
