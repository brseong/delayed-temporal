"""ViT-specific fixed-range derivation for layer-wise calibration."""

import math
from typing import Any

from torch import nn

from utils.transforms.calibration import (
    CalibrationRangePolicy,
    LayerCalibrationSpec,
)
from utils.transforms.types import PotentialBounds


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
