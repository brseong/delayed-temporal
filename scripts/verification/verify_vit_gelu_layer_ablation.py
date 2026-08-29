"""Verify selective dense-formula GELU configuration for the local ViT adapter."""

from pathlib import Path
from types import SimpleNamespace
import sys


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.evaluation.error_analysis_vit import configure_vit_exact_gelu_layers


def _make_dummy_vit(depth: int = 12) -> SimpleNamespace:
    """Build only the module topology inspected by the selector.

    A structural fixture keeps this verification independent of checkpoints and
    CUDA while retaining the exact attribute path used by the local ViT model.
    """
    # Every dummy intermediate begins on the temporal branch, matching a normal
    # model constructed without the global all-layer exact-GELU flag.
    layers = [
        SimpleNamespace(
            intermediate=SimpleNamespace(_spiking_mlp_exact_gelu=False)
        )
        for _ in range(depth)
    ]

    # Mirror ``model.vit.encoder.layer`` without constructing heavyweight model
    # parameters or invoking Hugging Face checkpoint loading.
    return SimpleNamespace(
        vit=SimpleNamespace(encoder=SimpleNamespace(layer=layers))
    )


def verify_vit_gelu_layer_ablation() -> None:
    """Check selective mutation, validation, and all-or-nothing failure behavior."""
    # A sparse selection must modify exactly those blocks and leave every other
    # temporal GELU active for a valid leave-one-layer-out experiment.
    model = _make_dummy_vit()
    configure_vit_exact_gelu_layers(model, (0, 5, 11))
    enabled = tuple(
        index
        for index, layer in enumerate(model.vit.encoder.layer)
        if layer.intermediate._spiking_mlp_exact_gelu
    )
    assert enabled == (0, 5, 11)

    # Empty input must remain a no-op, while malformed index sets must fail loudly
    # instead of silently changing the requested experimental condition.
    untouched = _make_dummy_vit()
    configure_vit_exact_gelu_layers(untouched, ())
    assert not any(
        layer.intermediate._spiking_mlp_exact_gelu
        for layer in untouched.vit.encoder.layer
    )
    for invalid in ((-1,), (12,), (3, 3)):
        candidate = _make_dummy_vit()
        try:
            configure_vit_exact_gelu_layers(candidate, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid layer selection: {invalid}")
        assert not any(
            layer.intermediate._spiking_mlp_exact_gelu
            for layer in candidate.vit.encoder.layer
        )

    # A malformed intermediate later in the selection must not leave an earlier
    # valid layer toggled, which would make retry behavior order-dependent.
    malformed = _make_dummy_vit()
    del malformed.vit.encoder.layer[5].intermediate._spiking_mlp_exact_gelu
    try:
        configure_vit_exact_gelu_layers(malformed, (0, 5))
    except RuntimeError:
        pass
    else:
        raise AssertionError("accepted a ViT layer without a selectable GELU")
    assert not malformed.vit.encoder.layer[0].intermediate._spiking_mlp_exact_gelu


if __name__ == "__main__":
    verify_vit_gelu_layer_ablation()
    print("ViT per-layer GELU ablation verification passed.")
