"""Stable tokenizer identity for fixed text preprocessing contracts."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def tokenizer_backend_sha256(backend: Any) -> str | None:
    """Hash tokenization structure, excluding mutable request padding/truncation.

    Hugging Face sets these two backend fields while encoding a batch. A cached
    dataset can bypass encoding entirely, leaving them unset on an otherwise
    identical tokenizer. Callers separately persist the actual padding strategy,
    truncation setting, maximum length, sides and special token IDs. Every other
    backend field, including model, vocabulary, added tokens, normalizer,
    pre-tokenizer, post-processor and decoder, remains part of this identity.
    """
    if backend is None or not hasattr(backend, "to_str"):
        return None
    structure = json.loads(backend.to_str())
    if not isinstance(structure, dict):
        raise ValueError("tokenizer backend must serialize to a JSON object")
    structure.pop("padding", None)
    structure.pop("truncation", None)
    return hashlib.sha256(json.dumps(
        structure, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()
