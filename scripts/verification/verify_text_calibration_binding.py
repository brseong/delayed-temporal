"""CPU checks for shared, versioned text calibration range binding."""

from dataclasses import replace
from itertools import product
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.verification.verify_layernorm_calibrated_bounds import (
    FLAGS, KEY, _capture, _layer, _metadata, _reference, _table,
)
from utils.transformers.calibration import (
    bind_model_calibration, calibration_uses_explicit_bounds,
    model_calibration_is_bound, vit_calibration_uses_explicit_bounds,
)
from utils.transforms.calibration import CalibrationMode, create_calibration_runtime
from utils.transforms.noise import set_gaussian_time_noise
from utils.transforms.types import Potential, PotentialBounds


def metadata(family="bert", **changes):
    base = _metadata(policy=None)
    options = dict(
        base.model_options,
        text_calibration_policy_version=2,
        operator_backed_output_head_version=1,
    )
    return replace(
        base,
        model_family=family,
        **changes,
        model_options=tuple(sorted(options.items())),
    )


def runtime(meta, radius=60):
    table = replace(_table(radius), metadata=meta)
    return create_calibration_runtime(CalibrationMode.INFERENCE, table, expected_metadata=meta)


# @lat: [[text-calibration#Text Model Calibration#Shared Range Binding]]
def verify_text_layernorm_binding():
    """Every family uses selected bounds in all active LayerNorm combinations."""
    value = torch.tensor([[-50, -10, 10, 50], [-80, -20, 20, 80]], dtype=torch.float64)
    potential = Potential(value, PotentialBounds(-100, 100))
    for family, flags, radius in product(("bert", "roberta", "gpt2"), FLAGS, (60, 1)):
        layer = _layer(flags)
        state = runtime(metadata(family), radius)
        bind_model_calibration(layer, state)
        assert calibration_uses_explicit_bounds(layer)
        assert not vit_calibration_uses_explicit_bounds(layer)
        clean = None
        for enabled, std in ((False, 0), (True, 0), (True, 1e-9)):
            result, logs, multiplications, clamps = _capture(
                layer, potential, enabled=enabled, std=std,
            )
            assert torch.isfinite(result.value).all()
            if not enabled:
                clean = result.value
                torch.testing.assert_close(clean, _reference(layer, value, radius),
                                           rtol=2e-10, atol=2e-10)
            elif std == 0:
                torch.testing.assert_close(result.value, clean, rtol=2e-10, atol=2e-10)
            if any(flags):
                expected = ([radius, radius] if flags[0] else []) + ([6] if flags[2] else [])
                assert multiplications == expected
                log_radius = (radius**2 + layer.eps) ** 0.5
                assert clamps["var_x"][0] == PotentialBounds(
                    layer.clip_margin**2,
                    log_radius**2,
                )
                if flags[1]:
                    assert all(abs(item[2].max - logs[0][2].max) < 1e-12 for item in logs)
        assert state.clipping_counts[KEY].num_values == (24 if any(flags) else 0)
    set_gaussian_time_noise(enabled=False)


def verify_rejection_is_atomic():
    """Version, family, dtype and LayerNorm identity errors publish no bindings."""
    base = metadata()
    variants = [replace(base, model_family="vit"), replace(base, dtype="float32")]
    for key, value in (("text_calibration_policy_version", 1),
                       ("text_calibration_policy_version", 3),
                       ("text_calibration_policy_version", True),
                       ("vit_calibration_policy_version", 2),
                       ("operator_backed_output_head_version", 2),
                       ("layer_norm_eps", 1e-5),
                       ("layer_norm_clip_margin", 1e-4)):
        options = dict(base.model_options)
        options[key] = value
        variants.append(replace(base, model_options=tuple(sorted(options.items()))))
    for meta in variants:
        layer = _layer()
        try:
            bind_model_calibration(layer, runtime(meta))
        except ValueError:
            assert not model_calibration_is_bound(layer)
        else:
            raise AssertionError("incompatible text table was accepted")
    layer = _layer()
    try:
        bind_model_calibration(layer, runtime(base, 1e-6))
    except ValueError:
        assert not model_calibration_is_bound(layer)
    else:
        raise AssertionError("range below log floor was accepted")


def main():
    verify_text_layernorm_binding()
    verify_rejection_is_atomic()
    print("Text calibration shared binding verification passed")


if __name__ == "__main__":
    main()
