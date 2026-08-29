"""GPT-2-specific declarations for layer-wise fixed-range calibration."""

import json
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
from utils.transforms.noise import get_gaussian_time_noise
from utils.transformers.calibration import (
    bind_model_calibration,
    clear_model_calibration,
)
from utils.transformers.integrations.spiking_sdpa_attention import (
    attention_score_representability_bounds,
)


def gpt2_calibration_specs(
    model: nn.Module,
    *,
    lower_quantile: float,
    upper_quantile: float,
    margin_fraction: float,
) -> tuple[LayerCalibrationSpec, ...]:
    """Declare GPT-2 residual and optional spiking-attention calibration sites.

    One signed-symmetric entry range constrains token-plus-position embeddings before
    the first block. Every block contributes self-attention and MLP residual outputs.
    A spiking attention module additionally freezes its raw softmin score rail below
    the representability ceiling derived from model capacity, temporal scale, and
    the query/key/value projection dtype.

    Args:
        model: Unwrapped GPT-2 model or task wrapper.
        lower_quantile: Lower signed histogram cutoff shared by all sites.
        upper_quantile: Upper signed histogram cutoff shared by all sites.
        margin_fraction: Per-side expansion after symmetric quantile selection.

    Returns:
        One model-input specification, two specifications per block, and one score
        specification per spiking attention layer.

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
    block_names = tuple(
        sorted(
            name
            for name, module in model.named_modules()
            if isinstance(module, GPT2Block)
        )
    )
    if not block_names:
        raise ValueError("model contains no GPT2Block modules")

    # All three site types carry signed residual streams into affine PWM. A symmetric
    # policy calibrates both tails and guarantees the shared zero reference remains
    # representable after every frozen range reset.
    specs = [
        LayerCalibrationSpec(
            module_name=model_names[0],
            tensor_name="input",
            range_policy=CalibrationRangePolicy.SIGNED_SYMMETRIC,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
            margin_fraction=margin_fraction,
        )
    ]
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

    # Only the spiking backend consumes calibrated softmin score rails. The eager
    # architecture therefore preserves the artifact schema used before score-range
    # calibration was introduced.
    attention_modules = tuple(
        sorted(
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, GPT2Attention)
            and module.config._attn_implementation == "spiking_sdpa"
        )
    )
    for module_name, module in attention_modules:
        # GPT-2 cache growth is bounded by the configured position capacity, not the
        # current token batch. The combined Q/K/V projection weight supplies the
        # execution dtype that determines the exponential representability floor.
        ceiling = attention_score_representability_bounds(
            float(getattr(module.config, "theta", 10.0)),
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

    # Persist all configured paths that alter residual distributions. Training-only
    # dropout probabilities are included because model.eval() makes their effective
    # behavior zero, while recording them still rejects a different checkpoint config.
    model_options = tuple(
        sorted(
            (
                ("activation_function", str(getattr(config, "activation_function", ""))),
                ("attention_implementation", attention_implementation),
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

    # GPT-2 evaluator execution is float32 today. The common metadata schema still
    # has a tau_m slot, so it stores the same value as tau_s rather than a second
    # attention scale.
    theta = float(getattr(config, "theta"))
    tau_s = float(getattr(config, "tau_s"))
    return CalibrationMetadata(
        model_family="gpt2",
        model_id=model_id,
        dataset_id=dataset_id,
        dataset_split=calibration_split,
        preprocessing=preprocessing,
        dtype="float32",
        theta=theta,
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

    # Keep one binding across both passes. The first pass discovers only extrema;
    # histogram bin edges are fixed before replaying the identical selected texts.
    bind_model_calibration(model, collector)
    try:
        pass_counts: list[int] = []
        for pass_index in range(2):
            observed_samples = 0

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
                    observed_samples += int(input_ids.shape[0])
                    model(
                        input_ids=input_ids.to(device=device),
                        attention_mask=attention_mask.to(device=device),
                        use_cache=False,
                    )

            # Exact counts catch custom collation or dataset behavior that drops or
            # duplicates examples despite a nominally correct dataset length.
            if observed_samples != expected_samples:
                raise ValueError(
                    "calibration pass sample count does not match expected_samples"
                )
            pass_counts.append(observed_samples)
            if pass_index == 0:
                start_histogram_calibration_pass(collector)

        # Preserve the replay invariant explicitly before immutable finalization.
        if pass_counts[0] != pass_counts[1]:
            raise ValueError("calibration passes consumed different populations")
        return finalize_calibration_collection(collector)
    finally:
        clear_model_calibration(model, expected_state=collector)
