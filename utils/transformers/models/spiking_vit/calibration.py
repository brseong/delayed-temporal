"""ViT-specific fixed-range derivation for layer-wise calibration."""

import math
from typing import Any

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
