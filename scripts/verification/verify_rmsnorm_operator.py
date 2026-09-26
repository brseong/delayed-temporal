"""Verify RMSNorm algebra, declared bounds, event readout and Llama integration."""

from __future__ import annotations

from contextlib import ExitStack
from itertools import product
import math
from pathlib import Path
import sys
from unittest.mock import patch

import sympy as sp
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers.models.llama.modeling_llama import LlamaRMSNorm as HFRMSNorm
from utils.transforms import functions, noise, spike_to_potential
from utils.transforms.clock import get_clock_update_stats, set_clock_driven
from utils.transforms.types import Potential, PotentialBounds, SpikeSample
from utils.transformers.models.spiking_llama import modeling_spiking_llama as llama
from utils.transformers.models.spiking_llama.configuration_llama import LlamaConfig


def reference(x: torch.Tensor, eps: float, floor: float) -> torch.Tensor:
    normalized = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
    return torch.where(x.abs() >= floor, normalized, torch.zeros_like(normalized))


def verify_algebra_and_domains() -> None:
    magnitude, moment, upper, tau = sp.symbols("magnitude moment upper tau", positive=True)
    t_magnitude = tau * sp.log(upper / magnitude)
    t_moment = tau / 2 * sp.log(upper**2 / moment)
    assert sp.simplify(sp.exp((t_moment - t_magnitude) / tau) - magnitude / sp.sqrt(moment)) == 0

    largest = {}
    for dtype in (torch.float64, torch.float32):
        max_error = 0.0
        tolerance = 2e-11 if dtype == torch.float64 else 3e-7
        for eps, tau, radius in product((1e-12, 1e-6, 0.2), (0.5, 1.0, 2.0), (1.0, 40.172)):
            values = torch.tensor([
                [0, 0, 0, 0], [1, 1, 1, 1], [-1, -1, -1, -1],
                [-1, 0.5, 0.25, 0], [1, 0, 0, 0],
                [-1e-7, 1e-7, -1e-8, 1e-8],
            ], dtype=dtype) * radius
            expected = reference(values, eps, 1e-8)
            actual, bounds = functions.rmsnorm_function(
                values, PotentialBounds(-radius, radius), eps=eps, tau_s=tau,
            )
            torch.testing.assert_close(actual[:-1], expected[:-1], atol=tolerance, rtol=tolerance)
            # Tiny inputs in a wide affine code expose timestamp cancellation.
            # Keep this stress case separate from ordinary-input equivalence.
            stress_tolerance = 1e-9 if dtype == torch.float64 and eps == 1e-12 else tolerance
            torch.testing.assert_close(actual[-1], expected[-1], atol=stress_tolerance, rtol=0.0)
            assert bounds == PotentialBounds(-2.0, 2.0)
            assert torch.all((actual >= bounds.min) & (actual <= bounds.max))
            assert torch.equal(actual[0], torch.zeros_like(actual[0]))
            max_error = max(max_error, (actual - expected).abs().max().item())
        largest[str(dtype)] = max_error

    for limits in ((0.0, 0.0), (0.5, 2.0), (-3.0, -0.5), (-3.0, 2.0)):
        values = torch.linspace(*limits, 16, dtype=torch.float64).reshape(4, 4)
        actual, _ = functions.rmsnorm_function(values, PotentialBounds(*limits))
        torch.testing.assert_close(actual, reference(values, 1e-6, 1e-8), atol=2e-12, rtol=2e-12)

    # A deliberately coarse positive floor must not change the variance epsilon.
    values = torch.tensor([[0.01, -0.01, 0.0, 0.0]], dtype=torch.float64)
    actual, _ = functions.rmsnorm_function(
        values, PotentialBounds(-1.0, 1.0), eps=1e-6, clip_margin=0.1,
    )
    torch.testing.assert_close(actual, reference(values, 1e-6, 1e-3), atol=2e-12, rtol=2e-12)
    print(f"RMSNorm maximum absolute errors: {largest}")


class Trace(ExitStack):
    """Force selected events past the deadline while retaining the real sampler."""

    def __init__(
        self, missed_site: str | set[str] | None = None, *,
        shifts: dict[str, float] | None = None, margin: float = 0.0,
    ):
        super().__init__()
        self.missed_sites = {missed_site} if isinstance(missed_site, str) else (missed_site or set())
        self.shifts = shifts or {}
        self.margin = margin
        self.events: dict[str, list[SpikeSample]] = {}
        self.inputs: dict[str, list[torch.Tensor]] = {}
        self.denominators: list[SpikeSample] = []
        self.current_site = None

    def __enter__(self):
        super().__enter__()
        noise.set_gaussian_time_noise(
            enabled=True, time_std_fraction=0.01, seed=23,
            deadline_margin_std_ratio=self.margin,
        )
        self.callback(noise.set_gaussian_time_noise, enabled=False)
        original_sample = noise._sample_gaussian_spike_time
        original_ed = functions.exponential_difference_operator

        def sample(nominal, **kwargs):
            raw = nominal + self.shifts.get(self.current_site, 0.0) * kwargs["domain"].range
            if self.current_site in self.missed_sites:
                raw = torch.full_like(
                    nominal, kwargs["domain"].max + kwargs["deadline_margin"] + 1.0,
                )
            kwargs.update(time_mean=raw - nominal, time_std=0.0)
            return original_sample(nominal, **kwargs)

        def wrap_encoder(encoder):
            def call(*args, **kwargs):
                previous = self.current_site
                self.current_site = kwargs.get("noise_site")
                try:
                    self.inputs.setdefault(self.current_site, []).append(args[0].detach().clone())
                    event = encoder(*args, **kwargs)
                    if isinstance(event, SpikeSample):
                        self.events.setdefault(self.current_site, []).append(event)
                    return event
                finally:
                    self.current_site = previous
            return call

        def ed(a, da, b, db, **kwargs):
            self.denominators.append(b)
            return original_ed(a, da, b, db, **kwargs)

        self.enter_context(patch.object(noise, "_sample_gaussian_spike_time", sample))
        self.enter_context(patch.object(functions, "neg_log_transform", wrap_encoder(functions.neg_log_transform)))
        self.enter_context(patch.object(
            functions, "neg_identity_transform", wrap_encoder(functions.neg_identity_transform),
        ))
        self.enter_context(patch.object(
            spike_to_potential, "neg_identity_transform",
            wrap_encoder(spike_to_potential.neg_identity_transform),
        ))
        self.enter_context(patch.object(functions, "exponential_difference_operator", ed))
        return self


def verify_events_and_noise() -> None:
    x = torch.tensor([[-0.9, 0.2, 0.7, 0.0], [0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    domain = PotentialBounds(-1.0, 1.0)
    clean, output_domain = functions.rmsnorm_function(x, domain)
    for site in (None, "rmsnorm.log_mean_square", "rmsnorm.log_positive",
                 "rmsnorm.log_negative", "exponential_difference.internal"):
        with Trace(site) as trace:
            actual, bounds = functions.rmsnorm_function(x, domain)
        denominator = trace.events["rmsnorm.log_mean_square"][0]
        assert len(trace.events["rmsnorm.log_mean_square"]) == 1
        assert all(event is denominator for event in trace.denominators)
        assert len(trace.denominators) == 2
        expected_parts = []
        internal_events = trace.events["exponential_difference.internal"]
        for index, name in enumerate(("rmsnorm.log_positive", "rmsnorm.log_negative")):
            event = trace.events[name][0]
            assert event.domain is denominator.domain
            assert event.observation_deadline == denominator.observation_deadline
            cutoff = event.observation_deadline
            da = event.fired * (cutoff - event.time)
            db = denominator.fired * (cutoff - denominator.time)
            intermediate = (db - da).clamp(-event.domain.max, event.domain.max)
            response = torch.exp(-intermediate) * internal_events[index].fired
            active = (x if index == 0 else -x) >= 1e-8
            expected_parts.append(torch.where(active, response, 0.0))
        expected = (expected_parts[0] - expected_parts[1]).clamp(-2.0, 2.0)
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
        assert bounds == output_domain
        assert torch.equal(actual[1], torch.zeros_like(actual[1]))
        if site is None:
            torch.testing.assert_close(actual, clean, atol=2e-12, rtol=2e-12)
        else:
            assert not torch.allclose(actual, clean)

    outputs = []
    for _ in range(2):
        noise.set_gaussian_time_noise(enabled=True, time_std_fraction=1e-3, seed=31)
        output, bounds = functions.rmsnorm_function(x, domain)
        outputs.append(output)
        assert bounds == output_domain and torch.isfinite(output).all()
    assert torch.equal(outputs[0], outputs[1])
    assert not torch.equal(outputs[0], clean)
    noise.set_gaussian_time_noise(enabled=False)


def verify_clock_and_validation() -> None:
    x = torch.tensor([[-0.9, 0.2, 0.7, 0.0]], dtype=torch.float64)
    domain = PotentialBounds(-1.0, 1.0)
    expected = reference(x, 1e-6, 1e-8)
    for settings in ({"time_step": 0.001}, {"time_steps_per_window": 1024}):
        set_clock_driven(enabled=True, **settings)
        actual, bounds = functions.rmsnorm_function(x, domain)
        assert bounds == PotentialBounds(-2.0, 2.0)
        torch.testing.assert_close(actual, expected, atol=0.08, rtol=0.08)
        stats = get_clock_update_stats()
        assert stats["pwm"]["time_steps"] > 0 and stats["exponential"]["time_steps"] > 0
    set_clock_driven(enabled=False)

    noise.set_gaussian_time_noise(enabled=True, time_std_fraction=1e-3, seed=7)
    generator = noise.get_gaussian_time_noise().generator
    for kwargs in ({"eps": 0.0}, {"eps": -1.0}, {"tau_s": 0.0},
                   {"tau_s": float("inf")}, {"clip_margin": 0.0}, {"clip_margin": True}):
        state = generator.get_state().clone()
        try:
            functions.rmsnorm_function(x, domain, **kwargs)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(f"invalid RMSNorm configuration accepted: {kwargs}")
        assert torch.equal(generator.get_state(), state)
    noise.set_gaussian_time_noise(enabled=False)


def verify_llama_rmsnorm() -> None:
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        hf = HFRMSNorm(4, eps=1e-6).to(dtype=dtype)
        module = llama.LlamaRMSNorm(4, eps=1e-6, tau_s=0.5).to(dtype=dtype)
        with torch.no_grad():
            hf.weight.copy_(torch.tensor([-2.0, 0.5, 0.0, 3.0], dtype=dtype))
        module.load_state_dict(hf.state_dict(), strict=True)
        x = torch.tensor([[0, 0, 0, 0], [-0.9, 0.2, 0.7, 0.0], [1, 1, 1, 1]], dtype=dtype)
        expected = hf(x)
        with patch.object(torch, "rsqrt", side_effect=AssertionError("dense RMSNorm bypass")), \
             patch.object(llama, "rmsnorm_function", wraps=functions.rmsnorm_function) as operator:
            actual = module(Potential(x, PotentialBounds(-1.0, 1.0)))
        operator.assert_called_once()
        tolerance = 2e-5 if dtype == torch.float32 else 0.02
        torch.testing.assert_close(actual.value, expected, atol=tolerance, rtol=tolerance)
        assert actual.value.dtype == dtype
        assert actual.domain == PotentialBounds(-6.0, 6.0)

        module.tau_s = 2.0
        try:
            module(Potential(x, PotentialBounds(-1.0, 1.0)))
        except RuntimeError as error:
            assert "bounds were frozen" in str(error)
        else:
            raise AssertionError("stale RMSNorm configuration accepted")
        module.freeze_parameter_bounds(refresh=True)
        module(Potential(x, PotentialBounds(-1.0, 1.0)))

    config = LlamaConfig(tau_s=0.5, rmsnorm_clip_margin=1e-9)
    restored = LlamaConfig.from_dict(config.to_dict())
    assert restored.tau_s == 0.5 and restored.rmsnorm_clip_margin == 1e-9



def verify_complete_event_readout() -> None:
    x = torch.tensor([[-1.7, 0.2, 2.6, 0.0], [0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    domain = PotentialBounds(-2.0, 3.0)
    sites = ("multiplication.data", "multiplication.reference", "rmsnorm.log_mean_square",
             "rmsnorm.log_positive", "rmsnorm.log_negative", "exponential_difference.internal")
    cases = 0

    def duration(event):
        return torch.where(event.fired, (event.observation_deadline - event.time).clamp_min(0.0), 0.0)

    for tau, margin, shifted in product((0.5, 1.0, 2.0), (0.0, 2.0), (False, True)):
        shifts = dict(zip(sites, (-0.04, 0.03, -0.01, -0.02, 0.03, 0.015))) if shifted else {}
        for bits in product((False, True), repeat=len(sites)):
            missed = {site for site, flag in zip(sites, bits) if flag}
            with Trace(missed, shifts=shifts, margin=margin) as trace:
                actual, bounds = functions.rmsnorm_function(x, domain, tau_s=tau)

            # Independently reconstruct the square, including the generic product
            # clamp followed by the structural nonnegative square clamp.
            data, zero = (trace.events[site][0] for site in sites[:2])
            assert zero.time.ndim == 0
            square = (x * (duration(data) - duration(zero))).clamp(-6.0, 9.0).clamp(0.0, 9.0)
            moment = square.mean(-1, keepdim=True) + 1e-6
            torch.testing.assert_close(
                trace.inputs["rmsnorm.log_mean_square"][0], moment, atol=2e-14, rtol=2e-14,
            )
            denominator = trace.events["rmsnorm.log_mean_square"][0]
            assert len(trace.events["rmsnorm.log_mean_square"]) == 1
            assert all(event is denominator for event in trace.denominators)
            assert len(trace.denominators) == 2
            parts = []
            for index, site in enumerate(sites[3:5]):
                event = trace.events[site][0]
                assert event.domain is denominator.domain
                assert event.observation_deadline == denominator.observation_deadline
                deadline = event.domain.max
                intermediate = (duration(denominator) - duration(event)).clamp(-deadline, deadline)
                torch.testing.assert_close(
                    trace.inputs["exponential_difference.internal"][index],
                    intermediate, atol=2e-14, rtol=2e-14,
                )
                internal = trace.events["exponential_difference.internal"][index]
                response = torch.where(
                    internal.fired, torch.exp((internal.time - deadline) / tau), 0.0,
                ).clamp(0.0, math.exp(deadline / tau))
                magnitude = (x if index == 0 else -x).clamp_min(0.0)
                parts.append(torch.where(magnitude >= 1e-8, response, 0.0))
            expected = (parts[0] - parts[1]).clamp(-2.0, 2.0)
            torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-12)
            assert bounds == PotentialBounds(-2.0, 2.0)
            assert torch.equal(actual[1], torch.zeros_like(actual[1]))
            cases += 1
    assert cases == 768
    print(f"RMSNorm complete event cases: {cases}")


def verify_floor_and_input_masks() -> None:
    # The floor modifies the dense function even with exact delivered events.
    floor, eps = 1e-8, 1e-6
    x = torch.tensor([[0.5 * floor, -0.5 * floor, 0.0, 0.0]], dtype=torch.float64)
    actual, _ = functions.rmsnorm_function(x, PotentialBounds(-1.0, 1.0))
    dense = reference(x, eps, 0.0)
    assert torch.equal(actual, torch.zeros_like(x))
    assert dense.abs().max().item() > 0.0
    assert (actual - dense).abs().max().item() < floor / math.sqrt(eps)

    # Identical numerator and denominator event inputs to ED do not determine
    # the final output: the original-input active mask also carries information.
    x = torch.tensor([[floor, 0.0], [0.0, floor]], dtype=torch.float64)
    with Trace() as trace:
        actual, _ = functions.rmsnorm_function(x, PotentialBounds(-1.0, 1.0))
    for site in ("rmsnorm.log_mean_square", "rmsnorm.log_positive", "rmsnorm.log_negative"):
        event = trace.events[site][0]
        assert torch.equal(event.time[0], event.time[1])
        assert torch.equal(event.fired[0], event.fired[1])
    assert actual[0, 0] > 0.0 and actual[0, 1] == 0.0
    assert actual[1, 0] == 0.0 and actual[1, 1] > 0.0

    # Missing both external ED events is not the same as missing its internal
    # event: zero differential input is re-encoded and decodes to one.
    x = torch.tensor([[0.9, -0.2]], dtype=torch.float64)
    missed = {"rmsnorm.log_mean_square", "rmsnorm.log_positive", "rmsnorm.log_negative"}
    with Trace(missed):
        actual, _ = functions.rmsnorm_function(x, PotentialBounds(-1.0, 1.0))
    torch.testing.assert_close(actual, x.sign(), atol=2e-14, rtol=0.0)
    with Trace(missed | {"exponential_difference.internal"}):
        actual, _ = functions.rmsnorm_function(x, PotentialBounds(-1.0, 1.0))
    assert torch.equal(actual, torch.zeros_like(actual))
    print("RMSNorm floor and original-input mask counterexamples reproduced.")


def verify_numerical_precision_and_bounds() -> None:
    errors = {}
    for eps, radius, magnitude, tau in product(
        (1e-12, 1e-6), (1.0, 40.172, 10000.0),
        (1e-7, 1e-3, 1.0), (0.5, 1.0, 2.0),
    ):
        x = torch.tensor([[-1.0, 0.5, 0.25, 0.0]], dtype=torch.float32) * magnitude
        dense = reference(x.double(), eps, 0.0)
        actual, bounds = functions.rmsnorm_function(
            x, PotentialBounds(-radius, radius), eps=eps, tau_s=tau,
        )
        torch.testing.assert_close(actual.double(), dense, atol=3e-7, rtol=0.0)
        assert actual.dtype == x.dtype
        assert bool(((actual.double() >= bounds.min) & (actual.double() <= bounds.max)).all())
        if eps == 1e-6 and magnitude == 1e-3 and tau == 1.0:
            errors[radius] = (actual.double() - dense).abs().max().item()
    print(f"RMSNorm corrected errors at default epsilon: {errors}")

    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        for lower, upper in ((-math.sqrt(3), math.sqrt(3)), (-0.1, 0.7), (0.0, 0.0)):
            original = PotentialBounds(lower, upper)
            rounded = original.outward_rounded(dtype)
            assert rounded.min <= original.min and rounded.max >= original.max
            encoded = torch.tensor([rounded.min, rounded.max], dtype=dtype).double()
            assert encoded.tolist() == [rounded.min, rounded.max]
            assert rounded.outward_rounded(dtype) == rounded
            endpoints = encoded.to(dtype)
            toward_zero_width = torch.nextafter(
                endpoints, torch.tensor([math.inf, -math.inf], dtype=dtype),
            ).double()
            assert toward_zero_width[0].item() > original.min
            assert toward_zero_width[1].item() < original.max

        for width, gain, sign in product((3, 7, 17), (0.0, 1.0, -1.3), (-1.0, 1.0)):
            module = llama.LlamaRMSNorm(width).to(dtype=dtype)
            with torch.no_grad():
                module.weight.fill_(gain)
            values = torch.zeros((1, width), dtype=dtype)
            values[0, 0] = sign
            result = module(Potential(values, PotentialBounds(-1.0, 1.0)))
            value = result.value.double()
            assert bool(((value >= result.domain.min) & (value <= result.domain.max)).all())
            assert result.value.dtype == dtype
            # Exercise the direct composition's output conversion as well.
            normalized, bounds = functions.rmsnorm_function(values, PotentialBounds(-1.0, 1.0))
            value = normalized.double()
            assert bool(((value >= bounds.min) & (value <= bounds.max)).all())

        replicas = []
        for _ in range(2):
            noise.set_gaussian_time_noise(enabled=True, time_std_fraction=0.001, seed=71)
            values = torch.tensor([[0.9, -0.2, 0.0]], dtype=dtype)
            result, bounds = functions.rmsnorm_function(values, PotentialBounds(-1.0, 1.0))
            replicas.append(result)
            assert result.dtype == dtype and torch.isfinite(result).all()
            assert bool(((result.double() >= bounds.min) & (result.double() <= bounds.max)).all())
        assert torch.equal(replicas[0], replicas[1])
        noise.set_gaussian_time_noise(enabled=False)

    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        try:
            PotentialBounds(-1e100, 1e100).outward_rounded(dtype)
        except ValueError:
            pass
        else:
            raise AssertionError("unrepresentable finite output bounds accepted")

    module = llama.LlamaRMSNorm(3)
    module.freeze_parameter_bounds()
    module.half()
    try:
        module.freeze_parameter_bounds()
    except RuntimeError:
        pass
    else:
        raise AssertionError("dtype change bypassed frozen parameter bounds")
    module.freeze_parameter_bounds(refresh=True)
    # Mixed activation/weight dtypes must use the actual promoted output dtype.
    values = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    result = module(Potential(values, PotentialBounds(-1.0, 1.0)))
    assert result.value.dtype == torch.float32
    assert result.value.double().max().item() <= result.domain.max


def verify_shape_and_model_coverage() -> None:
    generator = torch.Generator().manual_seed(829)
    largest = 0.0
    for width in (1, 2, 17, 128, 4096):
        values = 0.2 + 0.8 * torch.rand((2, width, 3), dtype=torch.float64, generator=generator)
        values = values * torch.where(torch.rand(values.shape, generator=generator) > 0.5, 1, -1)
        x = values.transpose(1, 2)
        for tau in (0.25, 3.0):
            actual, bounds = functions.rmsnorm_function(x, PotentialBounds(-1.0, 1.0), tau_s=tau)
            dense = reference(x, 1e-6, 0.0)
            torch.testing.assert_close(actual, dense, atol=2e-12, rtol=2e-12)
            assert bounds == PotentialBounds(-math.sqrt(width), math.sqrt(width))
            assert actual.shape == x.shape
            largest = max(largest, (actual - dense).abs().max().item())

    config = LlamaConfig(
        vocab_size=16, hidden_size=8, intermediate_size=12, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, tau_s=0.5, rmsnorm_clip_margin=1e-9,
    )
    model = llama.LlamaForCausalLM(config).eval()
    norms = [module for module in model.modules() if isinstance(module, llama.LlamaRMSNorm)]
    assert len(norms) == 5
    assert all(module.tau_s == 0.5 and module.clip_margin == 1e-9 for module in norms)
    with torch.no_grad(), patch.object(torch, "rsqrt", side_effect=AssertionError("dense RMSNorm")), \
         patch.object(llama, "rmsnorm_function", wraps=functions.rmsnorm_function) as operator:
        output = model(torch.tensor([[1, 2, 3]]))
    assert operator.call_count == 5 and torch.isfinite(output.logits).all()
    print(f"RMSNorm noncontiguous shape checks maximum error: {largest}")



def main() -> None:
    set_clock_driven(enabled=False)
    noise.set_gaussian_time_noise(enabled=False)
    try:
        for check in (verify_algebra_and_domains, verify_events_and_noise,
                      verify_clock_and_validation, verify_llama_rmsnorm,
                      verify_complete_event_readout, verify_floor_and_input_masks,
                      verify_numerical_precision_and_bounds, verify_shape_and_model_coverage):
            check()
            print(f"PASS {check.__name__}", flush=True)
    finally:
        set_clock_driven(enabled=False)
        noise.set_gaussian_time_noise(enabled=False)


if __name__ == "__main__":
    main()
