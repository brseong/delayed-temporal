"""GPT-2-specific declarations for layer-wise fixed-range calibration."""

import json
import time
import hashlib
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler

from utils.transforms.calibration import (
    CalibrationCollectorState,
    CalibrationMetadata,
    CalibrationRangePolicy,
    CalibrationTable,
    LayerCalibrationSpec,
    finalize_calibration_collection,
    start_histogram_calibration_pass,
)
from utils.transforms.functions import GELU_OUTPUT_MIN, OUTPUT_BOUNDS_VERSION
from utils.transforms.noise import get_gaussian_time_noise
from utils.transformers.calibration import (
    TEXT_CALIBRATION_POLICY_VERSION,
    bind_model_calibration,
    clear_model_calibration,
)
from utils.transformers.models.spiking_ops import SpikingLayerNorm
from utils.transformers.integrations.spiking_sdpa_attention import (
    attention_score_representability_bounds,
)
from utils.transformers.tokenizer_identity import tokenizer_backend_sha256


def gpt2_calibration_specs(
    model: nn.Module,
    *,
    lower_quantile: float,
    upper_quantile: float,
    margin_fraction: float,
) -> tuple[LayerCalibrationSpec, ...]:
    """Declare every GPT-2 activation requiring a selected fixed range.

    Residuals, MLP inputs, query/key/value projections, attention scores and active
    LayerNorm inputs are discovered from actual modules. Embeddings and bounded
    activation outputs retain their analytically derived intervals.

    Args:
        model: Unwrapped GPT-2 model or task wrapper.
        lower_quantile: Lower signed histogram cutoff shared by all sites.
        upper_quantile: Upper signed histogram cutoff shared by all sites.
        margin_fraction: Per-side expansion after symmetric quantile selection.

    Returns:
        Specifications for exactly the sites executed by the configured modules.

    Raises:
        TypeError: If ``model`` is not an unwrapped PyTorch module.
        RuntimeError: If DataParallel would make module identities replica-dependent.
        ValueError: If the wrapper does not contain exactly one GPT2Model or contains
            no GPT2Block modules.
    """
    # Import exact classes locally so reading calibration utilities does not trigger
    # Hugging Face model registration unless architecture discovery is requested.
    from utils.transformers.models.spiking_gpt2.modeling_spiking_gpt2 import (
        GPT2Attention,
        GPT2Block,
        GPT2MLP,
        GPT2Model,
    )

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if isinstance(model, nn.DataParallel):
        raise RuntimeError("GPT-2 calibration requires an unwrapped model")

    # Stable names come from the complete wrapper, allowing bare GPT2Model and
    # GPT2LMHeadModel without guessing task-specific prefixes.
    model_names = tuple(
        sorted(
            name
            for name, module in model.named_modules()
            if isinstance(module, GPT2Model)
        )
    )
    if len(model_names) != 1:
        raise ValueError("model must contain exactly one GPT2Model module")
    if any(getattr(module.config, "add_cross_attention", False)
           for module in model.modules() if isinstance(module, GPT2Model)):
        raise ValueError("GPT-2 calibration does not support cross-attention")
    block_names = tuple(
        sorted(
            name
            for name, module in model.named_modules()
            if isinstance(module, GPT2Block)
        )
    )
    if not block_names:
        raise ValueError("model contains no GPT2Block modules")

    # Residual resets carry signed streams into affine PWM. A symmetric policy uses
    # both observed extrema and guarantees the shared zero reference remains
    # representable after every frozen range reset.
    specs: list[LayerCalibrationSpec] = []
    for module_name in block_names:
        for tensor_name in ("attention_residual", "output"):
            specs.append(
                LayerCalibrationSpec(
                    module_name=module_name,
                    tensor_name=tensor_name,
                    range_policy=CalibrationRangePolicy.SIGNED_SYMMETRIC,
                    lower_quantile=lower_quantile,
                    upper_quantile=upper_quantile,
                    margin_fraction=margin_fraction,
                )
            )

    for module_name, module in sorted(model.named_modules()):
        if isinstance(module, GPT2MLP):
            tensor_name = "activation_input"
        elif isinstance(module, SpikingLayerNorm) and any((
            module.use_spiking_mul, module.use_spiking_log, module.use_spiking_expdiff,
        )):
            tensor_name = "centered_input"
        else:
            continue
        specs.append(LayerCalibrationSpec(
            module_name=module_name,
            tensor_name=tensor_name,
            range_policy=CalibrationRangePolicy.SIGNED_SYMMETRIC,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
            margin_fraction=margin_fraction,
        ))

    # Dense attention has no bounded temporal Q/K/V or score encoder.
    attention_modules = tuple(
        sorted(
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, GPT2Attention)
            and module.config._attn_implementation == "spiking_sdpa"
        )
    )
    for module_name, module in attention_modules:
        for tensor_name in ("query", "key", "value"):
            specs.append(LayerCalibrationSpec(
                module_name=module_name,
                tensor_name=tensor_name,
                range_policy=CalibrationRangePolicy.SIGNED_SYMMETRIC,
                lower_quantile=lower_quantile,
                upper_quantile=upper_quantile,
                margin_fraction=margin_fraction,
            ))
        # GPT-2 cache growth is bounded by the configured position capacity, not the
        # current token batch. The combined Q/K/V projection weight supplies the
        # execution dtype that determines the exponential representability floor.
        ceiling = attention_score_representability_bounds(
            float(getattr(module.config, "tau_s", 1.0)),
            int(module.config.max_position_embeddings),
            module.c_attn.weight.dtype,
        )
        specs.append(
            LayerCalibrationSpec(
                module_name=module_name,
                tensor_name="attention_score",
                range_policy=CalibrationRangePolicy.SIGNED_SYMMETRIC_CEILING,
                lower_quantile=lower_quantile,
                upper_quantile=upper_quantile,
                margin_fraction=margin_fraction,
                fixed_min=float(ceiling.min),
                fixed_max=float(ceiling.max),
            )
        )
    return tuple(specs)


def _gpt2_config_sha256(config: Any) -> str:
    """Hash numerical configuration without checkpoint location or dtype aliases."""
    config_dict = config.to_dict()
    for name in ("_name_or_path", "_commit_hash", "transformers_version", "torch_dtype", "dtype"):
        config_dict.pop(name, None)
    return hashlib.sha256(json.dumps(config_dict, sort_keys=True, default=str).encode()).hexdigest()


def build_gpt2_calibration_metadata(
    *,
    model_id: str,
    dataset_id: str,
    calibration_split: str,
    calibration_dataset_fingerprint: str,
    calibration_samples: int,
    calibration_seed: int,
    tokenizer: Any,
    config: Any,
    max_length: int,
    attention_implementation: str,
    dtype: str = "float32",
    checkpoint_sha256: str = "",
) -> CalibrationMetadata:
    """Build the complete reusable identity of one GPT-2 calibration artifact.

    The selected WikiText revision and tokenizer configuration identify the exact
    token population replayed by collection. Sequence capacity, TTFS constants, and
    every evaluator ablation that changes the residual stream are persisted so a
    frozen range table cannot be loaded into a numerically different GPT-2 path.

    Args:
        model_id: Pretrained checkpoint identifier.
        dataset_id: Dataset and configuration identifier used by the evaluator.
        calibration_split: Training split name used only for calibration.
        calibration_dataset_fingerprint: Fingerprint after empty-text filtering and
            deterministic subset selection.
        calibration_samples: Exact number of texts replayed in both passes.
        calibration_seed: Seed for the training-split permutation.
        tokenizer: Tokenizer that pads and truncates every selected text.
        config: Loaded spiking GPT-2 configuration.
        max_length: Fixed padded token length presented to the model.
        attention_implementation: Effective eager or spiking attention backend.

    Returns:
        Immutable metadata accepted by collection and frozen runtime setup.

    Raises:
        TypeError: If scalar identities have incompatible types.
        ValueError: If required identities, sequence capacity, tokenizer state, or
            preprocessing fields are invalid or not JSON-compatible.
    """
    # Validate exact external identities before inspecting tokenizer or model state.
    # An empty subset fingerprint cannot distinguish two source-data revisions.
    text_values = {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "calibration_split": calibration_split,
        "calibration_dataset_fingerprint": calibration_dataset_fingerprint,
        "attention_implementation": attention_implementation,
        "dtype": dtype,
    }
    for name, value in text_values.items():
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string")
        if not value or value != value.strip():
            raise ValueError(f"{name} must be non-empty without surrounding whitespace")
    for name, value in (
        ("calibration_samples", calibration_samples),
        ("calibration_seed", calibration_seed),
        ("max_length", max_length),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if calibration_samples <= 0:
        raise ValueError("calibration_samples must be positive")
    if calibration_seed < 0:
        raise ValueError("calibration_seed must be non-negative")
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if dtype not in ("float32", "float64"):
        raise ValueError("dtype must be float32 or float64")
    if not isinstance(checkpoint_sha256, str):
        raise TypeError("checkpoint_sha256 must be a string")

    # The artifact is valid only up to the same padded request length. Refuse a
    # tokenizer length beyond the learned position table instead of persisting an
    # input capacity the checkpoint cannot execute.
    position_capacity = getattr(config, "max_position_embeddings", None)
    if isinstance(position_capacity, bool) or not isinstance(position_capacity, int):
        raise ValueError("GPT-2 config must define max_position_embeddings")
    if max_length > position_capacity:
        raise ValueError("max_length exceeds GPT-2 position capacity")

    # Store tokenizer fields that can change token IDs, padding, or truncation. JSON
    # normalization gives exact deterministic equality across save and load while the
    # subset fingerprint identifies the already filtered and selected raw examples.
    preprocessing_fields = {
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "padding_side": getattr(tokenizer, "padding_side", None),
        "truncation_side": getattr(tokenizer, "truncation_side", None),
        "padding": "max_length",
        "truncation": True,
        "max_length": max_length,
        "empty_text_filter": "strip_length_greater_than_zero",
        "subset_selection": "seeded_training_permutation_prefix",
        "subset_seed": calibration_seed,
        "subset_samples": calibration_samples,
        "subset_fingerprint": calibration_dataset_fingerprint,
    }
    if hasattr(tokenizer, "get_vocab"):
        preprocessing_fields["vocab_sha256"] = hashlib.sha256(json.dumps(
            tokenizer.get_vocab(), sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()).hexdigest()
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None and hasattr(backend, "to_str"):
        preprocessing_fields["backend_sha256"] = tokenizer_backend_sha256(backend)
    try:
        preprocessing = json.dumps(
            preprocessing_fields,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "tokenizer calibration metadata must be finite and JSON-compatible"
        ) from error

    from utils.transformers.models.spiking_gpt2.modeling_spiking_gpt2 import (
        resolve_gpt2_mlp_activation_implementation,
    )

    # Persist all configured paths that alter residual distributions. Training-only
    # dropout probabilities are included because model.eval() makes their effective
    # behavior zero, while recording them still rejects a different checkpoint config.
    config_sha256 = _gpt2_config_sha256(config)
    model_options = tuple(
        sorted(
            (
                ("activation_function", str(getattr(config, "activation_function", ""))),
                (
                    "mlp_activation_implementation",
                    resolve_gpt2_mlp_activation_implementation(config),
                ),
                ("checkpoint_sha256", checkpoint_sha256),
                ("config_sha256", config_sha256),
                ("attention_implementation", attention_implementation),
                ("gelu_output_min", GELU_OUTPUT_MIN),
                ("output_bounds_version", OUTPUT_BOUNDS_VERSION),
                ("text_calibration_policy_version", TEXT_CALIBRATION_POLICY_VERSION),
                ("layer_norm_eps", float(config.layer_norm_epsilon)),
                ("layer_norm_clip_margin", float(getattr(config, "clip_margin", 1.0e-5))),
                ("attn_pdrop", float(getattr(config, "attn_pdrop", 0.0))),
                ("embd_pdrop", float(getattr(config, "embd_pdrop", 0.0))),
                ("resid_pdrop", float(getattr(config, "resid_pdrop", 0.0))),
                ("spiking_ln_expdiff", bool(getattr(config, "spiking_ln_expdiff", True))),
                ("spiking_ln_log", bool(getattr(config, "spiking_ln_log", True))),
                ("spiking_ln_mul", bool(getattr(config, "spiking_ln_mul", True))),
                ("use_spiking_layernorm", bool(getattr(config, "use_spiking_layernorm", True))),
                ("use_spiking_mlp", bool(getattr(config, "use_spiking_mlp", True))),
            )
        )
    )

    # The metadata records the evaluator dtype and the single configured time scale.
    tau_s = float(getattr(config, "tau_s"))
    return CalibrationMetadata(
        model_family="gpt2",
        model_id=model_id,
        dataset_id=dataset_id,
        dataset_split=calibration_split,
        preprocessing=preprocessing,
        dtype=dtype,
        tau_s=tau_s,
        tau_m=tau_s,
        clip_margin=float(getattr(config, "clip_margin", 1.0e-5)),
        max_sequence_length=max_length,
        input_shape=(max_length,),
        model_options=model_options,
    )


def collect_gpt2_calibration_table(
    model: nn.Module,
    dataloader: DataLoader,
    collector: CalibrationCollectorState,
    *,
    device: torch.device,
    expected_samples: int,
) -> CalibrationTable:
    """Run deterministic GPT-2 min-max and histogram passes and finalize a table.

    Both passes replay one sequential tokenized training subset with the model in
    evaluation mode and timing noise disabled. Labels and task loss are excluded;
    only ``input_ids`` and ``attention_mask`` enter the clean residual computation.
    Calibration bindings are always removed even if a model forward fails.

    Args:
        model: Unwrapped spiking GPT-2 causal-language-model wrapper.
        dataloader: Sequential loader over the fixed tokenized subset.
        collector: Empty collector whose site declarations match the model.
        device: Device receiving token IDs and masks.
        expected_samples: Exact number of selected texts replayed in each pass.

    Returns:
        Final immutable calibration table after identical two-pass populations.

    Raises:
        TypeError: If model, loader, collector, device, or sample controls are invalid.
        ValueError: If loader order, dataset size, or pass populations differ.
        RuntimeError: If collection is replicated, noisy, training, already bound, or
            receives a batch without integer token IDs and an attention mask.
    """
    # Mutable observer state requires one unwrapped model and deterministic sample
    # order. Validate the complete topology before publishing a calibration binding.
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if isinstance(model, nn.DataParallel):
        raise RuntimeError("GPT-2 calibration collection requires an unwrapped model")
    if model.training:
        raise RuntimeError("GPT-2 calibration collection requires model.eval()")
    if not isinstance(dataloader, DataLoader):
        raise TypeError("dataloader must be a torch.utils.data.DataLoader")
    if not isinstance(dataloader.sampler, SequentialSampler):
        raise ValueError("calibration DataLoader must use sequential sampling")
    if not isinstance(collector, CalibrationCollectorState):
        raise TypeError("collector must be a CalibrationCollectorState")
    if not isinstance(device, torch.device):
        raise TypeError("device must be a torch.device")
    if isinstance(expected_samples, bool) or not isinstance(expected_samples, int):
        raise TypeError("expected_samples must be an integer")
    if expected_samples <= 0:
        raise ValueError("expected_samples must be positive")
    if len(dataloader.dataset) != expected_samples:
        raise ValueError("calibration dataset length does not match expected_samples")
    if get_gaussian_time_noise().enabled:
        raise RuntimeError("calibration collection requires Gaussian timing noise off")
    from utils.transformers.models.spiking_gpt2.modeling_spiking_gpt2 import GPT2Model

    model_configs = [module.config for module in model.modules() if isinstance(module, GPT2Model)]
    if len(model_configs) != 1:
        raise ValueError("calibration requires exactly one GPT2Model")
    config = model_configs[0]
    metadata = collector.metadata
    options = dict(metadata.model_options)
    if metadata.model_family != "gpt2":
        raise ValueError("calibration model family differs from the model")
    actual_dtype = next(model.parameters()).dtype
    if actual_dtype not in (torch.float32, torch.float64) or metadata.dtype != str(actual_dtype).removeprefix("torch."):
        raise ValueError("calibration dtype differs from the model")
    version = options.get("text_calibration_policy_version")
    if type(version) is not int or version != TEXT_CALIBRATION_POLICY_VERSION:
        raise ValueError("calibration requires the current text calibration policy")
    if (
        metadata.tau_s != float(config.tau_s)
        or metadata.tau_m != float(config.tau_s)
        or metadata.clip_margin != float(getattr(config, "clip_margin", 1.0e-5))
        or options.get("config_sha256") != _gpt2_config_sha256(config)
        or options.get("output_bounds_version") != OUTPUT_BOUNDS_VERSION
        or options.get("attention_implementation") != config._attn_implementation
    ):
        raise ValueError("calibration metadata differs from the model configuration")
    if not collector.site_specs:
        raise ValueError("calibration sites must not be empty")
    first_spec = next(iter(collector.site_specs.values()))
    expected_specs = gpt2_calibration_specs(
        model, lower_quantile=first_spec.lower_quantile,
        upper_quantile=first_spec.upper_quantile, margin_fraction=first_spec.margin_fraction,
    )
    if dict(collector.site_specs) != {(spec.module_name, spec.tensor_name): spec for spec in expected_specs}:
        raise ValueError("calibration sites differ from the active model")

    # Keep one binding across both passes. The first pass discovers only extrema;
    # histogram bin edges are fixed before replaying the identical selected texts.
    bind_model_calibration(model, collector)
    try:
        pass_counts: list[int] = []
        pass_digests: list[str] = []
        collection_started = time.monotonic()
        for pass_index in range(2):
            observed_samples = 0
            digest = hashlib.sha256()

            # Token identities remain integer-valued and are never cast to the model
            # floating dtype. Disabling cache avoids retaining generation state across
            # independent calibration batches and between the two passes.
            with torch.no_grad():
                for batch in dataloader:
                    if not isinstance(batch, dict):
                        raise RuntimeError("calibration batch must be a dictionary")
                    if "input_ids" not in batch or "attention_mask" not in batch:
                        raise RuntimeError(
                            "calibration batch must contain input_ids and attention_mask"
                        )
                    input_ids = batch["input_ids"]
                    attention_mask = batch["attention_mask"]
                    if not isinstance(input_ids, torch.Tensor) or not isinstance(
                        attention_mask, torch.Tensor
                    ):
                        raise RuntimeError("calibration token fields must be tensors")
                    if input_ids.is_floating_point() or attention_mask.is_floating_point():
                        raise RuntimeError("calibration token fields must be integer tensors")
                    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
                        raise RuntimeError(
                            "calibration input_ids and attention_mask must share [batch, sequence] shape"
                        )
                    if input_ids.shape[1] != metadata.max_sequence_length:
                        raise ValueError("calibration token length differs from metadata")
                    observed_samples += int(input_ids.shape[0])
                    for name, value in (("input_ids", input_ids), ("attention_mask", attention_mask)):
                        digest.update(name.encode())
                        digest.update(str((tuple(value.shape), str(value.dtype))).encode())
                        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
                    model(
                        input_ids=input_ids.to(device=device),
                        attention_mask=attention_mask.to(device=device),
                        use_cache=False,
                    )
                    completed = pass_index * expected_samples + observed_samples
                    elapsed = time.monotonic() - collection_started
                    print(json.dumps({
                        "event": "calibration_progress", "pass": pass_index + 1,
                        "passes": 2, "observed_samples": observed_samples,
                        "expected_samples": expected_samples,
                        "elapsed_seconds": round(elapsed, 3),
                        "estimated_remaining_seconds": round(
                            elapsed * (2 * expected_samples - completed) / completed, 3,
                        ),
                    }, sort_keys=True, allow_nan=False), flush=True)

            # Exact counts catch custom collation or dataset behavior that drops or
            # duplicates examples despite a nominally correct dataset length.
            if observed_samples != expected_samples:
                raise ValueError(
                    "calibration pass sample count does not match expected_samples"
                )
            pass_counts.append(observed_samples)
            pass_digests.append(digest.hexdigest())
            if pass_index == 0:
                start_histogram_calibration_pass(collector)

        # Preserve the replay invariant explicitly before immutable finalization.
        if pass_counts[0] != pass_counts[1]:
            raise ValueError("calibration passes consumed different populations")
        if pass_digests[0] != pass_digests[1]:
            raise ValueError("calibration token order or preprocessing changed between passes")
        return finalize_calibration_collection(collector)
    finally:
        clear_model_calibration(model, expected_state=collector)
