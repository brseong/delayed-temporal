"""Fixed activation-range declarations for the Llama decoder."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from torch import nn

from utils.transforms.calibration import (
    CalibrationMetadata,
    CalibrationRangePolicy,
    LayerCalibrationSpec,
)
from utils.transformers.calibration import (
    OPERATOR_BACKED_OUTPUT_HEAD_VERSION,
    TEXT_CALIBRATION_POLICY_VERSION,
)
from utils.transformers.integrations.spiking_sdpa_attention import (
    attention_score_representability_bounds,
)
from utils.transformers.models.spiking_llama.modeling_spiking_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaMLP,
)
from utils.transformers.tokenizer_identity import tokenizer_backend_sha256


def llama_calibration_specs(
    model: LlamaForCausalLM,
    *,
    lower_quantile: float = 0.0,
    upper_quantile: float = 1.0,
    margin_fraction: float = 0.05,
) -> tuple[LayerCalibrationSpec, ...]:
    """Select the data-dependent decoder ranges without duplicating residual sites."""
    if not isinstance(model, LlamaForCausalLM):
        raise TypeError("Llama calibration requires LlamaForCausalLM")
    specs: list[LayerCalibrationSpec] = []

    def signed(name: str, tensor: str) -> LayerCalibrationSpec:
        return LayerCalibrationSpec(
            module_name=name,
            tensor_name=tensor,
            range_policy=CalibrationRangePolicy.SIGNED_SYMMETRIC,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
            margin_fraction=margin_fraction,
        )

    for name, module in model.named_modules():
        if isinstance(module, LlamaDecoderLayer):
            specs.extend((signed(name, "attention_residual"), signed(name, "output")))
        elif isinstance(module, LlamaMLP) and module.use_spiking_mlp:
            specs.extend((signed(name, "gate"), signed(name, "up")))
        elif isinstance(module, LlamaAttention) and module.config._attn_implementation == "spiking_sdpa":
            specs.extend(signed(name, tensor) for tensor in ("query", "key", "value"))
            ceiling = attention_score_representability_bounds(
                float(module.config.tau_s),
                int(module.config.max_position_embeddings),
                module.q_proj.weight.dtype,
            )
            specs.append(LayerCalibrationSpec(
                module_name=name,
                tensor_name="attention_score",
                range_policy=CalibrationRangePolicy.SIGNED_SYMMETRIC_CEILING,
                lower_quantile=lower_quantile,
                upper_quantile=upper_quantile,
                margin_fraction=margin_fraction,
                fixed_min=float(ceiling.min),
                fixed_max=float(ceiling.max),
            ))
    if not specs:
        raise ValueError("Llama calibration has no active sites")
    return tuple(specs)


def checkpoint_sha256(model_path: Path) -> str:
    """Hash all local checkpoint shards in their stable filename order."""
    files = sorted(model_path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError("Llama calibration requires local safetensors weights")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def llama_calibration_metadata(
    model: LlamaForCausalLM,
    tokenizer: object,
    *,
    model_path: Path,
    calibration_fingerprint: str,
    max_length: int,
    calibration_samples: int,
    calibration_seed: int,
) -> CalibrationMetadata:
    """Bind a frozen table to checkpoint, tokenizer, population, and operator path."""
    if not isinstance(model, LlamaForCausalLM):
        raise TypeError("Llama metadata requires LlamaForCausalLM")
    resolved = model_path.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError("Llama calibration requires a local checkpoint directory")
    backend = getattr(tokenizer, "backend_tokenizer", None)
    preprocessing = json.dumps({
        "padding": "max_length",
        "truncation": True,
        "max_length": max_length,
        "empty_text_filter": "strip_length_greater_than_zero",
        "subset_selection": "seeded_training_permutation_prefix",
        "subset_seed": calibration_seed,
        "subset_samples": calibration_samples,
        "subset_fingerprint": calibration_fingerprint,
        "vocab_sha256": hashlib.sha256(json.dumps(
            tokenizer.get_vocab(), sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest(),
        "backend_sha256": tokenizer_backend_sha256(backend) if backend is not None else None,
    }, sort_keys=True, separators=(",", ":"), allow_nan=False)
    config = model.config
    config_payload = config.to_dict()
    for key in ("_name_or_path", "_commit_hash", "transformers_version", "torch_dtype", "dtype"):
        config_payload.pop(key, None)
    config_digest = hashlib.sha256(json.dumps(
        config_payload, sort_keys=True, separators=(",", ":"), default=str,
    ).encode()).hexdigest()
    options = tuple(sorted((
        ("attention_implementation", config._attn_implementation),
        ("checkpoint_sha256", checkpoint_sha256(resolved)),
        ("config_sha256", config_digest),
        ("hidden_act", config.hidden_act),
        ("operator_backed_output_head_version", OPERATOR_BACKED_OUTPUT_HEAD_VERSION),
        ("rms_norm_eps", float(config.rms_norm_eps)),
        ("rmsnorm_clip_margin", float(config.rmsnorm_clip_margin)),
        ("text_calibration_policy_version", TEXT_CALIBRATION_POLICY_VERSION),
        ("use_spiking_mlp", bool(config.use_spiking_mlp)),
    )))
    return CalibrationMetadata(
        model_family="llama",
        model_id=str(resolved),
        dataset_id="wikitext-2",
        dataset_split="train",
        preprocessing=preprocessing,
        dtype=str(next(model.parameters()).dtype).removeprefix("torch."),
        tau_s=float(config.tau_s),
        tau_m=float(config.tau_s),
        clip_margin=float(config.rmsnorm_clip_margin),
        max_sequence_length=max_length,
        input_shape=(max_length,),
        model_options=options,
    )
