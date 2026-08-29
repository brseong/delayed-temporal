"""GPT-2-specific declarations for layer-wise fixed-range calibration."""

from torch import nn

from utils.transforms.calibration import (
    CalibrationRangePolicy,
    LayerCalibrationSpec,
)


def gpt2_calibration_specs(
    model: nn.Module,
    *,
    lower_quantile: float,
    upper_quantile: float,
    margin_fraction: float,
) -> tuple[LayerCalibrationSpec, ...]:
    """Declare GPT-2 model-entry and pre-norm residual calibration sites.

    One signed-symmetric entry range constrains token-plus-position embeddings before
    the first block. Every block contributes self-attention and MLP residual outputs,
    resetting the pre-norm stream twice per depth step without calibrating bounded
    activations or LayerNorm internals.

    Args:
        model: Unwrapped GPT-2 model or task wrapper.
        lower_quantile: Lower signed histogram cutoff shared by all sites.
        upper_quantile: Upper signed histogram cutoff shared by all sites.
        margin_fraction: Per-side expansion after symmetric quantile selection.

    Returns:
        One model-input specification followed by two specifications per block.

    Raises:
        TypeError: If ``model`` is not an unwrapped PyTorch module.
        RuntimeError: If DataParallel would make module identities replica-dependent.
        ValueError: If the wrapper does not contain exactly one GPT2Model or contains
            no GPT2Block modules.
    """
    # Import exact classes locally so reading calibration utilities does not trigger
    # Hugging Face model registration unless architecture discovery is requested.
    from utils.transformers.models.spiking_gpt2.modeling_spiking_gpt2 import (
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
    return tuple(specs)
