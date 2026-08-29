"""ViT-specific fixed-range derivation for layer-wise calibration."""

import collections.abc
import dataclasses
import json
import math
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
from utils.transforms.types import PotentialBounds
from utils.transformers.calibration import (
    bind_model_calibration,
    clear_model_calibration,
    select_calibration_subset,
)
from utils.transformers.integrations.spiking_sdpa_attention import (
    attention_score_representability_bounds,
)


def _normalize_json_metadata(value: Any) -> Any:
    """Convert processor metadata containers into canonical JSON-compatible values.

    Transformers may expose image geometry through frozen dataclasses such as
    ``SizeDict`` rather than plain dictionaries. Calibration identity needs their
    complete field values, but must not depend on that library-specific container
    type or Python object representation.

    Args:
        value: Nested scalar, mapping, sequence, or dataclass metadata value.

    Returns:
        An equivalent tree containing only JSON scalar, list, and string-keyed dict
        values. Floating-point finiteness is checked later by strict ``json.dumps``.

    Raises:
        TypeError: If a mapping key is not a string or a value uses an unsupported
            runtime object type.
    """
    # Dataclass conversion preserves every declared field, including None-valued
    # alternative geometry modes. Process it before generic containers because
    # Transformers SizeDict does not implement collections.abc.Mapping.
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _normalize_json_metadata(dataclasses.asdict(value))

    # Normalize mappings recursively and sort only during JSON serialization. String
    # keys are required so distinct Python key types cannot collapse to one JSON name.
    if isinstance(value, collections.abc.Mapping):
        normalized_mapping: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("processor metadata mapping keys must be strings")
            normalized_mapping[key] = _normalize_json_metadata(item)
        return normalized_mapping

    # Tuple/list identity is intentionally normalized to JSON arrays. Processor
    # channel vectors use both forms across Transformers versions but have the same
    # deterministic preprocessing meaning.
    if isinstance(value, (tuple, list)):
        return [_normalize_json_metadata(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value

    # Do not stringify unknown objects: repr output can be version-dependent and
    # would make artifact compatibility appear stable without preserving semantics.
    raise TypeError(
        f"unsupported processor metadata type: {type(value).__name__}"
    )


def image_processor_pixel_bounds(
    processor: Any,
    *,
    num_channels: int,
) -> PotentialBounds:
    """Derive the fixed tensor range produced from uint8 images by a processor.

    The ViT evaluator supplies PIL images, whose channel values lie in ``[0, 255]``.
    Rescaling maps those endpoints linearly, and optional channel-wise normalization
    maps them again through ``(x - mean) / std``. The final scalar envelope covers
    every configured channel and includes zero for signed-PWM reference encoding.

    Args:
        processor: Loaded image processor used by the evaluation data transform.
        num_channels: Number of channels expected by the model configuration.

    Returns:
        A finite ordered global pixel range containing zero.

    Raises:
        TypeError: If the channel count or processor metadata is not numeric.
        ValueError: If channel metadata is incompatible or has invalid scale values.
    """
    # Reject bool explicitly because Python treats it as an integer. A positive
    # channel count is required before scalar processor fields can be broadcast.
    if isinstance(num_channels, bool) or not isinstance(num_channels, int):
        raise TypeError("num_channels must be an integer")
    if num_channels <= 0:
        raise ValueError("num_channels must be positive")

    # Normalize scalar or per-channel processor fields to one explicit tuple. A
    # singleton is broadcast deliberately; any other length mismatch would make the
    # calibration identity ambiguous and therefore fails before model construction.
    def channel_values(value: Any, *, name: str) -> tuple[float, ...]:
        try:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values = (float(value),) * num_channels
            else:
                values = tuple(float(item) for item in value)
                if len(values) == 1:
                    values = values * num_channels
        except (TypeError, ValueError) as error:
            raise TypeError(f"image processor {name} must be numeric") from error
        if len(values) != num_channels:
            raise ValueError(
                f"image processor {name} must contain one or {num_channels} values"
            )
        if not all(math.isfinite(item) for item in values):
            raise ValueError(f"image processor {name} must be finite")
        return values

    # PIL inputs begin on the closed uint8 interval. Rescaling is a positive linear
    # map in maintained processors; rejecting non-positive factors prevents silently
    # reversing endpoints or collapsing the physical encoder window.
    lower = 0.0
    upper = 255.0
    if bool(getattr(processor, "do_rescale", False)):
        factor_values = channel_values(
            getattr(processor, "rescale_factor", None),
            name="rescale_factor",
        )
        if any(factor <= 0.0 for factor in factor_values):
            raise ValueError("image processor rescale_factor must be positive")
        channel_lowers = tuple(lower * factor for factor in factor_values)
        channel_uppers = tuple(upper * factor for factor in factor_values)
    else:
        channel_lowers = (lower,) * num_channels
        channel_uppers = (upper,) * num_channels

    # Normalization is monotone only for a positive standard deviation. Evaluate both
    # original endpoints per channel, then reduce them to the scalar PotentialBounds
    # representation carried by the patch projection.
    if bool(getattr(processor, "do_normalize", False)):
        means = channel_values(
            getattr(processor, "image_mean", None),
            name="image_mean",
        )
        standard_deviations = channel_values(
            getattr(processor, "image_std", None),
            name="image_std",
        )
        if any(std <= 0.0 for std in standard_deviations):
            raise ValueError("image processor image_std must be positive")
        channel_lowers = tuple(
            (value - mean) / std
            for value, mean, std in zip(
                channel_lowers,
                means,
                standard_deviations,
                strict=True,
            )
        )
        channel_uppers = tuple(
            (value - mean) / std
            for value, mean, std in zip(
                channel_uppers,
                means,
                standard_deviations,
                strict=True,
            )
        )

    # Signed PWM always encodes a zero reference on the same interval. Widening a
    # one-sided preprocessing envelope to zero changes no input values and guarantees
    # that reference is representable without a batch-dependent special case.
    return PotentialBounds(
        min(0.0, *channel_lowers),
        max(0.0, *channel_uppers),
    )


def vit_residual_calibration_specs(
    model: nn.Module,
    *,
    lower_quantile: float,
    upper_quantile: float,
    margin_fraction: float,
) -> tuple[LayerCalibrationSpec, ...]:
    """Declare both signed residual calibration sites for every ViT block.

    Stable names are taken from ``model.named_modules()`` after the complete wrapper
    has been constructed, so the same function supports a bare ``ViTModel`` and task
    wrappers such as ``ViTForImageClassification`` without guessing a name prefix.

    Args:
        model: Unwrapped ViT model or task wrapper containing ``ViTLayer`` modules.
        lower_quantile: Lower signed histogram cutoff used at every residual site.
        upper_quantile: Upper signed histogram cutoff used at every residual site.
        margin_fraction: Per-side expansion applied after symmetric range selection.

    Returns:
        Deterministically ordered layer specifications, two per ViT block.

    Raises:
        TypeError: If the model is not an unwrapped PyTorch module.
        ValueError: If no ViT blocks are found. Quantile and margin validation remains
            centralized in calibration collector construction.
        RuntimeError: If ``DataParallel`` would add unstable replica name prefixes.
    """
    # Import locally to avoid making processor-only range derivation initialize the
    # full Hugging Face model adapter. The class identity still provides an exact,
    # architecture-aware selection rather than matching informal class-name strings.
    from utils.transformers.models.spiking_vit.modeling_spiking_vit import ViTLayer

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if isinstance(model, nn.DataParallel):
        raise RuntimeError(
            "ViT calibration specifications require an unwrapped model"
        )

    # ``named_modules`` follows registered module order, which is checkpoint-stable
    # for the fixed architecture. Sort names explicitly so persistence never depends
    # on incidental traversal changes in an outer task wrapper.
    layer_names = tuple(
        sorted(
            name
            for name, module in model.named_modules()
            if isinstance(module, ViTLayer)
        )
    )
    if not layer_names:
        raise ValueError("model contains no ViTLayer modules")

    # Both post-add tensors cross zero and serve as affine inputs later in the block
    # stack. One signed-symmetric policy therefore calibrates both tails and guarantees
    # a zero-containing PWM rail at every depth reset.
    specs: list[LayerCalibrationSpec] = []
    for module_name in layer_names:
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
    return tuple(specs)


def vit_calibration_specs(
    model: nn.Module,
    *,
    lower_quantile: float,
    upper_quantile: float,
    margin_fraction: float,
) -> tuple[LayerCalibrationSpec, ...]:
    """Declare ViT residual and optional spiking-attention calibration sites.

    The encoder entry resets embedding output to one signed range before the first
    affine projection. Each later block contributes the two residual sites returned
    by :func:`vit_residual_calibration_specs`. When the selected backend is spiking
    attention, every attention module additionally freezes its raw softmin score rail
    below a configuration- and dtype-derived representability ceiling.

    Args:
        model: Unwrapped ViT model or task wrapper.
        lower_quantile: Lower signed histogram cutoff shared by calibrated sites.
        upper_quantile: Upper signed histogram cutoff shared by calibrated sites.
        margin_fraction: Per-side expansion after symmetric range selection.

    Returns:
        One encoder-input specification, two residual specifications per block, and
        one score specification per spiking attention layer.

    Raises:
        TypeError: If ``model`` is not an unwrapped PyTorch module.
        ValueError: If the model does not contain exactly one ViT encoder.
    """
    # Import locally for the same reason as ViTLayer above: processor-only users do
    # not need to initialize model registration, while actual model inspection uses
    # exact class identities rather than name suffixes.
    from utils.transformers.models.spiking_vit.modeling_spiking_vit import (
        ViTEncoder,
        ViTSelfAttention,
    )

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if isinstance(model, nn.DataParallel):
        raise RuntimeError("ViT calibration specifications require an unwrapped model")
    encoder_names = tuple(
        sorted(
            name
            for name, module in model.named_modules()
            if isinstance(module, ViTEncoder)
        )
    )
    if len(encoder_names) != 1:
        raise ValueError("model must contain exactly one ViTEncoder module")

    # Encoder outputs are signed and immediately feed affine PWM. Use the same
    # symmetric policy as residual resets so every persisted range contains zero.
    entry_spec = LayerCalibrationSpec(
        module_name=encoder_names[0],
        tensor_name="input",
        range_policy=CalibrationRangePolicy.SIGNED_SYMMETRIC,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        margin_fraction=margin_fraction,
    )
    residual_specs = vit_residual_calibration_specs(
        model,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        margin_fraction=margin_fraction,
    )

    # Attention score calibration belongs only to the maintained spiking backend.
    # Exact class discovery gives each layer its stable owning module name, while an
    # eager artifact retains the existing entry-and-residual schema unchanged.
    attention_specs: list[LayerCalibrationSpec] = []
    for module_name, module in sorted(model.named_modules()):
        if not isinstance(module, ViTSelfAttention):
            continue
        if module.config._attn_implementation != "spiking_sdpa":
            continue

        # Derive one request-independent source capacity from the configured patch
        # grid. The query projection weight supplies the actual execution dtype after
        # model conversion, device transfer, and evaluator precision selection.
        image_size = module.config.image_size
        patch_size = module.config.patch_size
        image_hw = (
            tuple(image_size)
            if isinstance(image_size, collections.abc.Iterable)
            else (image_size, image_size)
        )
        patch_hw = (
            tuple(patch_size)
            if isinstance(patch_size, collections.abc.Iterable)
            else (patch_size, patch_size)
        )
        source_length_max = (
            (int(image_hw[0]) // int(patch_hw[0]))
            * (int(image_hw[1]) // int(patch_hw[1]))
            + 1
        )
        ceiling = attention_score_representability_bounds(
            float(getattr(module.config, "theta", 10.0)),
            float(getattr(module.config, "tau_s", 1.0)),
            source_length_max,
            module.query.weight.dtype,
        )
        attention_specs.append(
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
    return (entry_spec, *residual_specs, *attention_specs)


def build_vit_calibration_metadata(
    *,
    model_id: str,
    dataset_id: str,
    calibration_split: str,
    calibration_dataset_fingerprint: str,
    calibration_samples: int,
    calibration_seed: int,
    processor: Any,
    config: Any,
    dtype: str,
    attention_implementation: str,
) -> CalibrationMetadata:
    """Build the complete reusable identity of one ViT calibration artifact.

    The dataset fingerprint, deterministic permutation seed, prefix length, and image
    processor fields identify the exact representative input population. Model and
    numerical fields identify the clean converted network whose residual distributions
    were measured. Robustness noise is intentionally absent so one table can be reused.

    Args:
        model_id: Pretrained checkpoint identifier.
        dataset_id: Hugging Face dataset identifier.
        calibration_split: Training split name used only for calibration.
        calibration_dataset_fingerprint: Fingerprint after deterministic subset
            selection, which changes if source data or selected indices change.
        calibration_samples: Exact number of examples in both collection passes.
        calibration_seed: Seed used to permute the training split before taking a prefix.
        processor: Image processor used to construct every calibration tensor.
        config: Loaded spiking ViT configuration.
        dtype: Stable evaluator precision name such as ``float32``.
        attention_implementation: Effective eager or spiking attention backend.

    Returns:
        Immutable metadata accepted by calibration collection and frozen runtime setup.

    Raises:
        TypeError: If text or integer identities have invalid types.
        ValueError: If required identities, image geometry, or processor fields are
            missing or invalid.
    """
    # Validate the exact subset identity before serializing processor state. Booleans
    # are excluded from integer fields, and an empty fingerprint cannot distinguish
    # a reconstructed dataset revision from the original collection population.
    text_values = {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "calibration_split": calibration_split,
        "calibration_dataset_fingerprint": calibration_dataset_fingerprint,
        "dtype": dtype,
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
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if calibration_samples <= 0:
        raise ValueError("calibration_samples must be positive")
    if calibration_seed < 0:
        raise ValueError("calibration_seed must be non-negative")

    # Store only deterministic processor fields that mathematically affect the model
    # tensor. JSON normalization converts tuples and mappings to a stable artifact
    # identity while rejecting non-finite numeric constants.
    processor_fields = {
        "do_resize": bool(getattr(processor, "do_resize", False)),
        "size": getattr(processor, "size", None),
        "do_center_crop": bool(getattr(processor, "do_center_crop", False)),
        "crop_size": getattr(processor, "crop_size", None),
        "do_rescale": bool(getattr(processor, "do_rescale", False)),
        "rescale_factor": getattr(processor, "rescale_factor", None),
        "do_normalize": bool(getattr(processor, "do_normalize", False)),
        "image_mean": getattr(processor, "image_mean", None),
        "image_std": getattr(processor, "image_std", None),
        "subset_selection": "seeded_training_permutation_prefix",
        "subset_seed": calibration_seed,
        "subset_samples": calibration_samples,
        "subset_fingerprint": calibration_dataset_fingerprint,
    }
    try:
        normalized_processor_fields = _normalize_json_metadata(processor_fields)
        preprocessing = json.dumps(
            normalized_processor_fields,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "image processor calibration metadata must be finite and JSON-compatible"
        ) from error

    # Normalize scalar or pair image geometry to the channels-first capacity stored
    # in the calibration schema. A table for one resolution must never constrain a
    # larger interpolated request without an explicitly different artifact.
    image_size = getattr(config, "image_size", None)
    if isinstance(image_size, collections.abc.Iterable) and not isinstance(
        image_size,
        (str, bytes),
    ):
        image_hw = tuple(int(value) for value in image_size)
    elif image_size is not None:
        image_hw = (int(image_size), int(image_size))
    else:
        image_hw = ()
    num_channels = getattr(config, "num_channels", None)
    if (
        len(image_hw) != 2
        or any(value <= 0 for value in image_hw)
        or isinstance(num_channels, bool)
        or not isinstance(num_channels, int)
        or num_channels <= 0
    ):
        raise ValueError("ViT config must define positive channels and image geometry")

    # Persist every model-path choice that changes residual distributions. Sort the
    # pairs explicitly because metadata equality is exact during artifact loading.
    model_options = tuple(
        sorted(
            (
                ("attention_implementation", attention_implementation),
                ("hidden_act", str(getattr(config, "hidden_act", ""))),
                ("spiking_ln_expdiff", bool(getattr(config, "spiking_ln_expdiff", True))),
                ("spiking_ln_log", bool(getattr(config, "spiking_ln_log", True))),
                ("spiking_ln_mul", bool(getattr(config, "spiking_ln_mul", True))),
                ("spiking_mlp_exact_gelu", bool(getattr(config, "spiking_mlp_exact_gelu", False))),
                ("use_spiking_layernorm", bool(getattr(config, "use_spiking_layernorm", True))),
                ("use_spiking_mlp", bool(getattr(config, "use_spiking_mlp", True))),
            )
        )
    )

    # CalibrationMetadata validation remains centralized in collector/runtime setup.
    # Conversion here uses only ordinary immutable scalars and tuples.
    return CalibrationMetadata(
        model_family="vit",
        model_id=model_id,
        dataset_id=dataset_id,
        dataset_split=calibration_split,
        preprocessing=preprocessing,
        dtype=dtype,
        theta=float(getattr(config, "theta")),
        tau_s=float(getattr(config, "tau_s")),
        tau_m=float(getattr(config, "tau_m")),
        clip_margin=float(getattr(config, "clip_margin", 1.0e-5)),
        max_sequence_length=None,
        input_shape=(num_channels, image_hw[0], image_hw[1]),
        model_options=model_options,
    )


def collect_vit_calibration_table(
    model: nn.Module,
    dataloader: DataLoader,
    collector: CalibrationCollectorState,
    *,
    device: torch.device,
    dtype: torch.dtype,
    expected_samples: int,
) -> CalibrationTable:
    """Run deterministic ViT min-max and histogram passes and finalize a table.

    Both passes reuse one sequential DataLoader over the already selected training
    subset. The model remains in evaluation mode with Gaussian timing noise disabled;
    calibration bindings are removed in ``finally`` even when a forward or table
    invariant fails, leaving the mutable collector available for diagnosis.

    Args:
        model: Unwrapped spiking ViT task model containing declared calibration sites.
        dataloader: Sequential loader over the fixed calibration subset.
        collector: Empty collector whose specs match the model's named modules.
        device: Device receiving each preprocessed pixel batch.
        dtype: Floating model input dtype.
        expected_samples: Exact subset population replayed in each pass.

    Returns:
        Final immutable calibration table after identical two-pass populations.

    Raises:
        TypeError: If arguments have incompatible types.
        ValueError: If sample counts or loader ordering are inconsistent.
        RuntimeError: If the model is training, replicated, noisy, already bound, or
            a batch does not provide a floating ``pixel_values`` tensor.
    """
    # Validate collection topology before installing mutable observers. Sequential
    # sampling is required so both iterations replay the same selected examples in
    # the same order even when only a prefix-sized calibration artifact is requested.
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if isinstance(model, nn.DataParallel):
        raise RuntimeError("ViT calibration collection requires an unwrapped model")
    if model.training:
        raise RuntimeError("ViT calibration collection requires model.eval()")
    if not isinstance(dataloader, DataLoader):
        raise TypeError("dataloader must be a torch.utils.data.DataLoader")
    if not isinstance(dataloader.sampler, SequentialSampler):
        raise ValueError("calibration DataLoader must use sequential sampling")
    if not isinstance(collector, CalibrationCollectorState):
        raise TypeError("collector must be a CalibrationCollectorState")
    if not isinstance(device, torch.device):
        raise TypeError("device must be a torch.device")
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise TypeError("dtype must be a floating torch.dtype")
    if isinstance(expected_samples, bool) or not isinstance(expected_samples, int):
        raise TypeError("expected_samples must be an integer")
    if expected_samples <= 0:
        raise ValueError("expected_samples must be positive")
    if len(dataloader.dataset) != expected_samples:
        raise ValueError("calibration dataset length does not match expected_samples")
    if get_gaussian_time_noise().enabled:
        raise RuntimeError("calibration collection requires Gaussian timing noise off")

    # Bind once across both passes so every module retains the same stable name and
    # collector identity. Forward failures still trigger complete unbinding below.
    bind_model_calibration(model, collector)
    try:
        pass_counts: list[int] = []
        for pass_index in range(2):
            observed_samples = 0

            # Calibration needs activations only; labels and task losses are excluded.
            # no_grad also guarantees observers never inherit an autograd graph.
            with torch.no_grad():
                for batch in dataloader:
                    if not isinstance(batch, dict) or "pixel_values" not in batch:
                        raise RuntimeError(
                            "calibration batch must contain pixel_values"
                        )
                    pixel_values = batch["pixel_values"]
                    if not isinstance(pixel_values, torch.Tensor):
                        raise RuntimeError("pixel_values must be a torch.Tensor")
                    if not pixel_values.is_floating_point():
                        raise RuntimeError("pixel_values must be floating point")
                    observed_samples += int(pixel_values.shape[0])
                    model(pixel_values.to(device=device, dtype=dtype))

            # Each pass must consume exactly the selected dataset. This catches a
            # custom collator or iterable behavior that drops or duplicates examples.
            if observed_samples != expected_samples:
                raise ValueError(
                    "calibration pass sample count does not match expected_samples"
                )
            pass_counts.append(observed_samples)
            if pass_index == 0:
                start_histogram_calibration_pass(collector)

        # The equality is redundant with exact expected counts but documents and
        # enforces the replay invariant directly before immutable finalization.
        if pass_counts[0] != pass_counts[1]:
            raise ValueError("calibration passes consumed different populations")
        return finalize_calibration_collection(collector)
    finally:
        clear_model_calibration(model, expected_state=collector)
