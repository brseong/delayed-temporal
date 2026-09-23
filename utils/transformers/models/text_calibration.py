"""Shared calibration of fixed ranges for the maintained text model adapters."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler

from utils.transforms.calibration import (
    CalibrationCollectorState, CalibrationMetadata, CalibrationRangePolicy,
    CalibrationTable, LayerCalibrationSpec, finalize_calibration_collection,
    start_histogram_calibration_pass,
)
from utils.transforms.functions import GELU_OUTPUT_MIN, OUTPUT_BOUNDS_VERSION
from utils.transforms.noise import get_gaussian_time_noise
from utils.transforms.types import Potential, PotentialBounds
from utils.transformers.calibration import (
    TEXT_CALIBRATION_POLICY_VERSION, bind_model_calibration,
    calibrated_potential, clear_model_calibration, model_calibration_is_bound,
    validate_symmetric_encoder_bounds,
)
from utils.transformers.integrations.spiking_sdpa_attention import (
    attention_score_representability_bounds,
)
from utils.transformers.tokenizer_identity import tokenizer_backend_sha256


def calibrate_text_potential(module: nn.Module, tensor_name: str, value: Potential) -> Potential:
    """Observe the raw tensor before clipping and preserve unbound execution."""
    if not model_calibration_is_bound(module):
        return value
    radius = max(abs(float(value.domain.min)), abs(float(value.domain.max)))
    bounds = PotentialBounds(-radius, radius)
    validate_symmetric_encoder_bounds(bounds, value.value.dtype, name=tensor_name)
    selected = calibrated_potential(
        module, tensor_name, value.value, collection_bounds=bounds,
    )
    validate_symmetric_encoder_bounds(selected.domain, selected.value.dtype, name=tensor_name)
    return selected


def text_calibration_specs(
    model: nn.Module, *, lower_quantile: float = 0.0,
    upper_quantile: float = 1.0, margin_fraction: float = 0.05,
) -> tuple[LayerCalibrationSpec, ...]:
    """Discover every executed text model site, retaining fixed output bounds."""
    from transformers.activations import GELUActivation
    from utils.transformers.models.spiking_ops import SpikingLayerNorm
    from utils.transformers.models.spiking_bert.modeling_spiking_bert import (
        BertModel, BertSelfAttention, BertSelfOutput, BertIntermediate,
        BertOutput, BertPooler,
    )
    from utils.transformers.models.spiking_roberta.modeling_spiking_roberta import (
        RobertaModel, RobertaSelfAttention, RobertaSelfOutput, RobertaIntermediate,
        RobertaOutput, RobertaPooler, RobertaClassificationHead, RobertaLMHead,
    )

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if isinstance(model, nn.DataParallel):
        raise RuntimeError("text calibration requires an unwrapped model")
    family = getattr(getattr(model, "config", None), "model_type", None)
    if family in ("gpt2", "spiking_gpt2", "spiking-gpt2"):
        from utils.transformers.models.spiking_gpt2.calibration import gpt2_calibration_specs
        return gpt2_calibration_specs(
            model, lower_quantile=lower_quantile, upper_quantile=upper_quantile,
            margin_fraction=margin_fraction,
        )
    bases = [module for module in model.modules() if isinstance(module, (BertModel, RobertaModel))]
    if len(bases) != 1:
        raise ValueError("text calibration requires exactly one BERT or RoBERTa model")
    config = bases[0].config
    if bool(getattr(config, "is_decoder", False)) or bool(getattr(config, "add_cross_attention", False)):
        raise ValueError("encoder calibration does not support decoder configurations or attention across sequences")
    if getattr(config, "_attn_implementation", None) not in ("eager", "sdpa", "spiking_sdpa"):
        raise ValueError("unsupported text calibration attention implementation")

    specs: list[LayerCalibrationSpec] = []

    def add(name: str, tensor: str, ceiling: PotentialBounds | None = None) -> None:
        options = {} if ceiling is None else dict(fixed_min=ceiling.min, fixed_max=ceiling.max)
        specs.append(LayerCalibrationSpec(
            module_name=name, tensor_name=tensor,
            range_policy=(CalibrationRangePolicy.SIGNED_SYMMETRIC if ceiling is None
                          else CalibrationRangePolicy.SIGNED_SYMMETRIC_CEILING),
            lower_quantile=lower_quantile, upper_quantile=upper_quantile,
            margin_fraction=margin_fraction, **options,
        ))

    for name, module in sorted(model.named_modules()):
        if isinstance(module, (BertSelfOutput, BertOutput, RobertaSelfOutput, RobertaOutput)):
            add(name, "residual")
        elif isinstance(module, (BertSelfAttention, RobertaSelfAttention)):
            if module.config._attn_implementation == "spiking_sdpa":
                for tensor in ("query", "key", "value"):
                    add(name, tensor)
                ceiling = attention_score_representability_bounds(
                    float(getattr(config, "tau_s", 1.0)),
                    int(config.max_position_embeddings), module.query.weight.dtype,
                )
                add(name, "attention_score", ceiling)
        elif isinstance(module, (BertIntermediate, RobertaIntermediate)):
            enabled = getattr(module, "_use_spiking_mlp", getattr(module, "use_spiking_mlp", True))
            if enabled and isinstance(module.intermediate_act_fn, GELUActivation):
                add(name, "activation_input")
        elif isinstance(module, (BertPooler, RobertaPooler, RobertaClassificationHead, RobertaLMHead)):
            if module.use_spiking_mlp:
                add(name, "activation_input")
        elif isinstance(module, SpikingLayerNorm):
            if any((module.use_spiking_mul, module.use_spiking_log, module.use_spiking_expdiff)):
                add(name, "centered_input")
    if not specs:
        raise ValueError("text model contains no supported calibration sites")
    keys = [(spec.module_name, spec.tensor_name) for spec in specs]
    if len(set(keys)) != len(keys):
        raise ValueError("text calibration sites must be unique")
    return tuple(specs)


def build_text_calibration_metadata(
    *, model_family: str, model_id: str, dataset_id: str, calibration_split: str,
    calibration_dataset_fingerprint: str, calibration_samples: int,
    calibration_seed: int, tokenizer: Any, config: Any, max_length: int,
    attention_implementation: str, dtype: str = "float32", text_column: str = "text",
    tokenizer_identifier: str | None = None, tokenizer_revision: str | None = None,
    evaluation_options: Mapping[str, Any] | None = None,
) -> CalibrationMetadata:
    """Identify exact text preprocessing, checkpoint configuration and fixed ranges."""
    text_fields = dict(
        model_family=model_family, model_id=model_id, dataset_id=dataset_id,
        calibration_split=calibration_split,
        calibration_dataset_fingerprint=calibration_dataset_fingerprint,
        attention_implementation=attention_implementation, dtype=dtype, text_column=text_column,
    )
    for name, value in text_fields.items():
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(f"{name} must be a nonempty string without surrounding whitespace")
    if model_family not in ("bert", "roberta", "gpt2"):
        raise ValueError("unsupported text calibration model family")
    if dtype not in ("float32", "float64"):
        raise ValueError("text calibration supports float32 and float64")
    for name, value, minimum in (
        ("calibration_samples", calibration_samples, 1),
        ("calibration_seed", calibration_seed, 0), ("max_length", max_length, 1),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer greater than or equal to {minimum}")
    capacity = getattr(config, "max_position_embeddings", None)
    if type(capacity) is not int or max_length > capacity:
        raise ValueError("max_length exceeds the configured position capacity")
    if model_family == "roberta":
        padding_index = getattr(config, "pad_token_id", None)
        if type(padding_index) is not int or padding_index < 0 or max_length + padding_index >= capacity:
            raise ValueError("max_length exceeds the position capacity after the padding offset")
    tokenizer_get = tokenizer.get if isinstance(tokenizer, Mapping) else lambda key, default=None: getattr(tokenizer, key, default)
    vocab = tokenizer_get("vocab", None)
    if hasattr(tokenizer, "get_vocab"):
        vocab = tokenizer.get_vocab()
    vocab_sha256 = None
    if vocab is not None:
        vocab_sha256 = hashlib.sha256(json.dumps(
            vocab, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()).hexdigest()
    backend = tokenizer_get("backend_tokenizer", None)
    backend_sha256 = tokenizer_backend_sha256(backend)
    preprocessing = dict(
        tokenizer_class=type(tokenizer).__name__,
        tokenizer_name_or_path=tokenizer_identifier or str(tokenizer_get("name_or_path", "")),
        tokenizer_revision=tokenizer_revision, vocab_sha256=vocab_sha256,
        backend_sha256=backend_sha256, vocab_size=tokenizer_get("vocab_size"),
        special_tokens_map={str(key): str(value) for key, value in tokenizer_get("special_tokens_map", {}).items()},
        bos_token_id=tokenizer_get("bos_token_id"), eos_token_id=tokenizer_get("eos_token_id"),
        pad_token_id=tokenizer_get("pad_token_id"), unk_token_id=tokenizer_get("unk_token_id"),
        mask_token_id=tokenizer_get("mask_token_id"), sep_token_id=tokenizer_get("sep_token_id"),
        cls_token_id=tokenizer_get("cls_token_id"),
        padding_side=tokenizer_get("padding_side"), truncation_side=tokenizer_get("truncation_side"),
        padding="max_length", truncation=True, max_length=max_length, text_column=text_column,
        subset_selection="seeded_training_permutation_prefix", subset_seed=calibration_seed,
        subset_samples=calibration_samples, subset_fingerprint=calibration_dataset_fingerprint,
    )
    options: dict[str, Any] = {
        "text_calibration_policy_version": TEXT_CALIBRATION_POLICY_VERSION,
        "output_bounds_version": OUTPUT_BOUNDS_VERSION, "gelu_output_min": GELU_OUTPUT_MIN,
        "attention_implementation": attention_implementation,
        "hidden_act": str(getattr(config, "hidden_act", getattr(config, "activation_function", ""))),
        "layer_norm_eps": float(getattr(config, "layer_norm_eps", getattr(config, "layer_norm_epsilon", 1.0e-5))),
        "layer_norm_clip_margin": float(getattr(config, "clip_margin", 1.0e-5)),
        "max_position_embeddings": capacity,
        "use_cache": bool(getattr(config, "use_cache", False)),
    }
    for name in ("use_spiking_mlp", "use_spiking_layernorm", "spiking_ln_mul", "spiking_ln_log", "spiking_ln_expdiff"):
        options[name] = bool(getattr(config, name, True))
    if evaluation_options is not None:
        for name, value in evaluation_options.items():
            if not isinstance(name, str) or type(value) not in (str, int, float, bool, type(None)):
                raise ValueError("evaluation options must contain string keys and JSON scalar values")
            options["evaluation_" + name] = value
    json.dumps(options, sort_keys=True, allow_nan=False)
    return CalibrationMetadata(
        model_family=model_family, model_id=model_id, dataset_id=dataset_id,
        dataset_split=calibration_split,
        preprocessing=json.dumps(preprocessing, sort_keys=True, separators=(",", ":"), allow_nan=False),
        dtype=dtype, tau_s=float(getattr(config, "tau_s")),
        tau_m=float(getattr(config, "tau_s")), clip_margin=float(getattr(config, "clip_margin", 1.0e-5)),
        max_sequence_length=max_length, input_shape=(max_length,),
        model_options=tuple(sorted(options.items())),
    )


def collect_text_calibration_table(
    model: nn.Module, dataloader: DataLoader, collector: CalibrationCollectorState,
    *, device: torch.device, expected_samples: int, dtype: torch.dtype | None = None,
) -> CalibrationTable:
    """Collect two identical ordered token passes without labels or task loss."""
    if not isinstance(model, nn.Module) or isinstance(model, nn.DataParallel):
        raise RuntimeError("text calibration requires an unwrapped model")
    if model.training:
        raise RuntimeError("text calibration requires model.eval()")
    if not isinstance(dataloader, DataLoader) or not isinstance(dataloader.sampler, SequentialSampler):
        raise ValueError("text calibration requires a sequential DataLoader")
    if not isinstance(collector, CalibrationCollectorState):
        raise TypeError("collector must be a CalibrationCollectorState")
    if type(expected_samples) is not int or expected_samples <= 0 or len(dataloader.dataset) != expected_samples:
        raise ValueError("calibration dataset length must equal positive expected_samples")
    if get_gaussian_time_noise().enabled:
        raise RuntimeError("text calibration requires Gaussian timing noise off")
    actual_dtype = next(model.parameters()).dtype
    if actual_dtype not in (torch.float32, torch.float64) or collector.metadata.dtype != str(actual_dtype).removeprefix("torch."):
        raise ValueError("collection dtype differs from the model")
    if dtype is not None and actual_dtype != dtype:
        raise ValueError("collection dtype differs from the model")
    actual_family = str(getattr(model.config, "model_type", "")).removeprefix("spiking-").removeprefix("spiking_")
    if collector.metadata.model_family != actual_family:
        raise ValueError("calibration model family differs from the model")
    first_spec = next(iter(collector.site_specs.values()))
    expected_specs = text_calibration_specs(
        model, lower_quantile=first_spec.lower_quantile, upper_quantile=first_spec.upper_quantile,
        margin_fraction=first_spec.margin_fraction,
    )
    if dict(collector.site_specs) != {(spec.module_name, spec.tensor_name): spec for spec in expected_specs}:
        raise ValueError("calibration sites differ from the active model")
    pass_digests: list[str] = []
    started = time.monotonic()
    bind_model_calibration(model, collector)
    try:
        for pass_index in range(2):
            digest = hashlib.sha256()
            observed = 0
            with torch.no_grad():
                for batch_index, batch in enumerate(dataloader, start=1):
                    if not isinstance(batch, Mapping) or "input_ids" not in batch:
                        raise ValueError("calibration batch must contain input_ids")
                    inputs = {}
                    for key in ("input_ids", "attention_mask", "token_type_ids", "position_ids"):
                        if key not in batch:
                            continue
                        value = batch[key]
                        if not isinstance(value, torch.Tensor) or value.ndim != 2:
                            raise ValueError(f"{key} must be a tensor with two dimensions")
                        if key == "input_ids" and value.dtype not in (torch.int32, torch.int64):
                            raise ValueError("input_ids must contain integer token indices")
                        if value.shape != batch["input_ids"].shape or value.shape[1] != collector.metadata.max_sequence_length:
                            raise ValueError(f"{key} shape differs from the fixed calibration input")
                        digest.update(key.encode())
                        digest.update(str((tuple(value.shape), str(value.dtype))).encode())
                        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
                        inputs[key] = value.to(device=device)
                    observed += int(inputs["input_ids"].shape[0])
                    model(**inputs)
                    print(
                        f"Calibration progress: pass={pass_index + 1}/2 "
                        f"batch={batch_index}/{len(dataloader)} "
                        f"samples={observed}/{expected_samples} "
                        f"elapsed_seconds={time.monotonic() - started:.2f}",
                        flush=True,
                    )
            if observed != expected_samples:
                raise ValueError("calibration pass sample count differs from expected_samples")
            pass_digests.append(digest.hexdigest())
            if pass_index == 0:
                start_histogram_calibration_pass(collector)
        if pass_digests[0] != pass_digests[1]:
            raise ValueError("calibration token order or preprocessing changed between passes")
        return finalize_calibration_collection(collector)
    finally:
        clear_model_calibration(model)
