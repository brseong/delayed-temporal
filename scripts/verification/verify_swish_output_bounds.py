"""CPU checks for fixed Swish output bounds and calibration compatibility."""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from utils.transforms.calibration import (
    CalibrationHistogram,
    CalibrationMode,
    CalibrationRange,
    CalibrationRangePolicy,
    LayerCalibrationSpec,
    MinMaxObserverState,
    create_calibration_runtime,
    create_calibration_table,
    create_layer_calibration,
    load_calibration_table,
    save_calibration_table,
    validate_calibration_metadata,
)
from utils.transforms.functions import (
    GELU_OUTPUT_MIN,
    OUTPUT_BOUNDS_VERSION,
    SWISH_OUTPUT_MIN,
    clamp_swish_output,
    swish_output_bounds,
)
from utils.transforms.noise import (
    get_gaussian_noise_stats,
    get_gaussian_time_noise,
    set_gaussian_time_noise,
)
from utils.transforms.types import Potential, PotentialBounds


# @lat: [[bounds-audit#2026-09-14 Bound Corrections#Swish Output Bounds]]
def verify_swish_bounds() -> None:
    """Check constant endpoints for either beta sign and the exact zero case."""
    assert SWISH_OUTPUT_MIN == -0.278465

    # Solve the derivative independently of the bounds implementation.
    lower, upper = -2.0, -1.0
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        sigmoid = 1.0 / (1.0 + math.exp(-midpoint))
        derivative = sigmoid * (1.0 + midpoint * (1.0 - sigmoid))
        if derivative < 0.0:
            lower = midpoint
        else:
            upper = midpoint
    minimum_input = 0.5 * (lower + upper)
    minimum = minimum_input / (1.0 + math.exp(-minimum_input))
    assert SWISH_OUTPUT_MIN < minimum < SWISH_OUTPUT_MIN + 1.0e-6

    for domain in (
        PotentialBounds(-4.0, 4.0),
        PotentialBounds(-4.0, -2.0),
        PotentialBounds(2.0, 4.0),
        PotentialBounds(0.0, 0.0),
    ):
        values = torch.linspace(domain.min, domain.max, 513, dtype=torch.float64)
        for beta in (0.0, 0.5, 1.0, 2.0, -0.5, -1.0, -2.0):
            if beta > 0.0:
                expected = PotentialBounds(SWISH_OUTPUT_MIN / beta, max(0.0, domain.max))
            elif beta < 0.0:
                expected = PotentialBounds(min(0.0, domain.min), SWISH_OUTPUT_MIN / beta)
            else:
                expected = PotentialBounds(0.5 * domain.min, 0.5 * domain.max)
            assert swish_output_bounds(domain, beta=beta) == expected
            reference = values * torch.sigmoid(beta * values)
            assert bool((reference >= expected.min).all())
            assert bool((reference <= expected.max).all())
            clean, clean_domain = clamp_swish_output(reference, domain, beta=beta)
            assert clean_domain == expected
            assert torch.equal(clean, reference)
            _, subset_domain = clamp_swish_output(reference[::4], domain, beta=beta)
            assert subset_domain == expected


def verify_swish_clamp_diagnostics() -> None:
    """Count strict output clipping without adding timing events or random draws."""
    domain = PotentialBounds(-4.0, 4.0)
    for beta in (1.0, -1.0, 0.0):
        bounds = swish_output_bounds(domain, beta=beta)
        raw = torch.tensor(
            [bounds.min - 1.0, bounds.min, 0.0, bounds.max, bounds.max + 1.0],
            dtype=torch.float64,
        )
        set_gaussian_time_noise(enabled=True, time_std=0.1, seed=73, device="cpu")
        try:
            before = get_gaussian_time_noise().generator.get_state().clone()
            output, output_domain = clamp_swish_output(raw, domain, beta=beta)
            assert output_domain == bounds
            assert torch.equal(output, raw.clamp(bounds.min, bounds.max))
            assert torch.equal(before, get_gaussian_time_noise().generator.get_state())
            stats = get_gaussian_noise_stats()
            assert set(stats) == {"swish.output"}
            assert stats["swish.output"]["outputs"] == 5
            assert stats["swish.output"]["output_underflows"] == 1
            assert stats["swish.output"]["output_overflows"] == 1
            assert stats["swish.output"]["events"] == 0
            assert stats["swish.output"]["misses"] == 0
        finally:
            set_gaussian_time_noise(enabled=False)
        clean, _ = clamp_swish_output(raw, domain, beta=beta)
        assert torch.equal(clean, output)
        assert get_gaussian_noise_stats() == {}


def verify_direct_swish_adapters() -> None:
    """Check SiLU and Swish in ViT and both GPT-2 projection paths."""
    from utils.transformers.models.spiking_vit.configuration_spiking_vit import ViTConfig
    from utils.transformers.models.spiking_vit import modeling_spiking_vit as vit
    from utils.transformers.models.spiking_gpt2.configuration_gpt2 import GPT2Config
    from utils.transformers.models.spiking_gpt2 import modeling_spiking_gpt2 as gpt2

    domain = PotentialBounds(-4.0, 4.0)
    expected_domain = PotentialBounds(SWISH_OUTPUT_MIN, 4.0)
    samples = (
        torch.tensor([[[-3.0, -1.27846, 0.5, 3.0]]], dtype=torch.float64),
        torch.tensor([[[0.1, 0.2, 0.3, 0.4]]], dtype=torch.float64),
    )
    for activation in ("silu", "swish"):
        config = ViTConfig(
            hidden_size=4, intermediate_size=4, num_hidden_layers=1,
            num_attention_heads=1, theta=40.0, use_spiking_mlp=False,
            hidden_act=activation,
        )
        module = vit.ViTIntermediate(config).double().eval()
        with torch.no_grad():
            module.dense.weight.copy_(torch.eye(4, dtype=torch.float64))
            module.dense.bias.zero_()
            for values in samples:
                with patch.object(vit, "clamp_swish_output", wraps=clamp_swish_output) as helper:
                    output = module(Potential(values, domain))
                helper.assert_called_once()
                assert output.domain == expected_domain
                torch.testing.assert_close(output.value, torch.nn.functional.silu(values), rtol=1e-10, atol=1e-10)

        for spiking in (False, True):
            config = GPT2Config(
                n_embd=4, n_layer=1, n_head=1, n_positions=4,
                theta=40.0, use_spiking_mlp=spiking,
                activation_function=activation, resid_pdrop=0.0,
            )
            module = gpt2.GPT2MLP(4, config).double().eval()
            with torch.no_grad():
                for projection in (module.c_fc, module.c_proj):
                    projection.weight.copy_(torch.eye(4, dtype=torch.float64))
                    projection.bias.zero_()
                for values in samples:
                    with patch.object(gpt2, "clamp_swish_output", wraps=clamp_swish_output) as helper:
                        output = module(Potential(values, domain))
                    helper.assert_called_once()
                    assert output.domain == expected_domain
                    torch.testing.assert_close(output.value, torch.nn.functional.silu(values), rtol=1e-10, atol=1e-10)


def verify_output_bounds_calibration_identity() -> None:
    """Reject old calibration identities in both maintained model families."""
    from utils.transformers.models.spiking_vit.calibration import build_vit_calibration_metadata
    from utils.transformers.models.spiking_vit.configuration_spiking_vit import ViTConfig
    from utils.transformers.models.spiking_gpt2.calibration import build_gpt2_calibration_metadata
    from utils.transformers.models.spiking_gpt2.configuration_gpt2 import GPT2Config

    common = dict(
        model_id="test-checkpoint", dataset_id="test-dataset", calibration_split="train",
        calibration_dataset_fingerprint="test-training-subset", calibration_samples=4,
        calibration_seed=0, attention_implementation="eager",
    )
    vit_metadata = build_vit_calibration_metadata(
        **common, processor=SimpleNamespace(),
        config=ViTConfig(image_size=4, num_channels=3, theta=40.0, tau_s=1.0),
        dtype="float64",
    )
    gpt2_metadata = build_gpt2_calibration_metadata(
        **common, tokenizer=SimpleNamespace(),
        config=GPT2Config(n_positions=4, theta=40.0, tau_s=1.0), max_length=4,
    )
    assert OUTPUT_BOUNDS_VERSION == 3
    for metadata in (vit_metadata, gpt2_metadata):
        assert dict(metadata.model_options)["gelu_output_min"] == GELU_OUTPUT_MIN
        assert dict(metadata.model_options)["output_bounds_version"] == OUTPUT_BOUNDS_VERSION
        validate_calibration_metadata(metadata, metadata)
        remaining = tuple(pair for pair in metadata.model_options if pair[0] != "output_bounds_version")
        for options in (
            remaining,
            tuple(sorted((*remaining, ("output_bounds_version", 1)))),
            tuple(sorted((*remaining, ("output_bounds_version", 2)))),
            tuple(sorted((*remaining, ("output_bounds_version", 4)))),
        ):
            try:
                validate_calibration_metadata(replace(metadata, model_options=options), metadata)
            except ValueError as error:
                assert "model_options" in str(error)
            else:
                raise AssertionError("old output bounds calibration identity was accepted")

        # Persist one valid layer with each default model identity, then exercise
        # the same compatibility check used before frozen inference. Reading an old
        # file remains supported for inspection; installing its bounds must fail.
        layer = create_layer_calibration(
            LayerCalibrationSpec(
                module_name="block",
                tensor_name="output",
                range_policy=CalibrationRangePolicy.SIGNED_SYMMETRIC,
                lower_quantile=0.0,
                upper_quantile=1.0,
                margin_fraction=0.0,
            ),
            MinMaxObserverState(-1.0, 1.0, 4),
            CalibrationHistogram(
                bounds=CalibrationRange(-1.0, 1.0),
                bin_counts=(2, 2),
                num_values=4,
                underflows=0,
                overflows=0,
            ),
        )
        table = create_calibration_table(metadata, (layer,))
        runtime_root = REPOSITORY_ROOT / "artifacts" / "runtime"
        runtime_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            dir=runtime_root, prefix="layernorm-bound-identity-"
        ) as directory:
            path = Path(directory) / f"{metadata.model_family}.json"
            save_calibration_table(table, path)
            loaded = load_calibration_table(path)
            assert loaded == table
            for mode in (CalibrationMode.VALIDATE, CalibrationMode.INFERENCE):
                runtime = create_calibration_runtime(
                    mode, loaded, expected_metadata=metadata
                )
                assert runtime.table == table

            obsolete_metadata = replace(
                metadata,
                model_options=tuple(
                    sorted((*remaining, ("output_bounds_version", 2)))
                ),
            )
            obsolete_table = create_calibration_table(obsolete_metadata, (layer,))
            obsolete_path = Path(directory) / f"{metadata.model_family}-v2.json"
            save_calibration_table(obsolete_table, obsolete_path)
            obsolete_loaded = load_calibration_table(obsolete_path)
            assert obsolete_loaded == obsolete_table
            for mode in (CalibrationMode.VALIDATE, CalibrationMode.INFERENCE):
                try:
                    create_calibration_runtime(
                        mode, obsolete_loaded, expected_metadata=metadata
                    )
                except ValueError as error:
                    assert "model_options" in str(error)
                else:
                    raise AssertionError(
                        "obsolete persisted output bounds were accepted for frozen inference"
                    )


def main() -> None:
    """Run without loading external checkpoints, datasets, or GPUs."""
    torch.set_num_threads(1)
    set_gaussian_time_noise(enabled=False)
    try:
        for verify in (
            verify_swish_bounds,
            verify_swish_clamp_diagnostics,
            verify_direct_swish_adapters,
            verify_output_bounds_calibration_identity,
        ):
            verify()
            print(f"PASS {verify.__name__}")
    finally:
        set_gaussian_time_noise(enabled=False)


if __name__ == "__main__":
    main()
