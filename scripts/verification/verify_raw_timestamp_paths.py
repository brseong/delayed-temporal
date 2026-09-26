"""Exercise delivered timestamps outside nominal windows at every consumer."""

from __future__ import annotations

from contextlib import ExitStack
from itertools import product
import math
from pathlib import Path
import sys
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.transforms import functions, noise, spike_to_potential
from utils.transforms.potential_to_spike import neg_identity_transform, neg_log_transform
from utils.transforms.primitive import (
    signed_pulse_width_duration, signed_pulse_width_modulation_operator,
)
from utils.transforms.types import Potential, PotentialBounds, SpikeSample, TimeBounds
from utils.transformers.integrations import spiking_sdpa_attention as attention
from utils.transformers.models import spiking_ops
from utils.transformers.models.spiking_gpt2 import modeling_spiking_gpt2 as gpt2
from scripts.verification.verify_layernorm_calibrated_bounds import _bind, _layer, _reference


DTYPE = torch.float64
CONSUMER_MODULES = (functions, spike_to_potential, spiking_ops, gpt2, attention)


class Trace(ExitStack):
    """Control timing draws, retain real encoders, and inspect unclipped potentials.

    Values in ``targets`` are raw event times selected by site. Unselected sites
    use zero timing error. Every event still traverses the real sampler, including
    the delivery decision. The configured margin equals one nominal window.
    """

    def __init__(self, targets=None):
        super().__init__()
        self.targets = targets or {}
        self.events: dict[str, list[SpikeSample]] = {}
        self.outputs: dict[str, list[torch.Tensor]] = {}
        self.current_site = None

    def __enter__(self):
        super().__enter__()
        noise.set_gaussian_time_noise(
            enabled=True, time_std_fraction=0.25,
            deadline_margin_std_ratio=4.0, seed=41,
        )
        self.callback(noise.set_gaussian_time_noise, enabled=False)
        original_sample = noise._sample_gaussian_spike_time
        original_clamp = noise.clamp_gaussian_output

        def sample(nominal, **kwargs):
            target = self.targets.get(self.current_site)
            raw = nominal if target is None else torch.as_tensor(
                target(nominal, kwargs["domain"]), dtype=nominal.dtype,
                device=nominal.device,
            ).expand_as(nominal)
            kwargs.update(time_std=0.0, time_mean=raw - nominal)
            return original_sample(nominal, **kwargs)

        def capture(value, domain, *, site, name):
            self.outputs.setdefault(site, []).append(value.detach().clone())
            return original_clamp(value, domain, site=site, name=name)

        def wrap_encoder(encoder):
            def call(*args, **kwargs):
                previous = self.current_site
                self.current_site = kwargs.get("noise_site")
                try:
                    event = encoder(*args, **kwargs)
                    if isinstance(event, SpikeSample):
                        self.events.setdefault(self.current_site, []).append(event)
                    return event
                finally:
                    self.current_site = previous
            return call

        self.enter_context(patch.object(noise, "_sample_gaussian_spike_time", sample))
        for module in CONSUMER_MODULES:
            self.enter_context(patch.object(module, "clamp_gaussian_output", capture))
            for name in ("neg_identity_transform", "neg_log_transform"):
                if hasattr(module, name):
                    self.enter_context(patch.object(module, name, wrap_encoder(getattr(module, name))))
        return self


def early(_nominal, domain):
    return domain.min - 0.25


def late(_nominal, domain):
    return domain.max + 0.25


def reference(_nominal, domain):
    return domain.max + 0.75


def missed(_nominal, domain):
    return domain.max + domain.range + 0.25


def close(actual, expected):
    torch.testing.assert_close(actual, torch.as_tensor(expected, dtype=DTYPE).expand_as(actual),
                               rtol=2e-12, atol=2e-12)


def independent_duration(a, b):
    """Paper equation evaluated without either production duration helper."""
    assert a.observation_deadline == b.observation_deadline
    deadline = a.observation_deadline
    return a.fired * (deadline - a.time) - b.fired * (deadline - b.time)


def verify_requested_boundary_example():
    """Nominal endpoint 2, observation cutoff 3: 2.5/2.9 arrive, 3.1 misses."""
    raw = torch.tensor([2.5, 2.9, 3.1], dtype=DTYPE)
    domain = TimeBounds(0.0, 2.0)
    for margin, expected in ((1.0, [2.5, 2.9, None]), (3.0, [2.5, 2.9, 3.1])):
        event = noise._sample_gaussian_spike_time(
            torch.full_like(raw, 2.0), time_std=0.0, time_mean=raw - 2.0,
            domain=domain, deadline_margin=margin,
            generator=torch.Generator().manual_seed(1),
        )
        observed = [float(t) if fired else None for t, fired in zip(event.time, event.fired)]
        assert observed == expected, (margin, observed)
        assert event.observation_deadline == 2.0 + margin
        print({"nominal_deadline": 2.0, "additional_margin": margin,
               "observation_deadline": event.observation_deadline,
               "raw": raw.tolist(), "observed": observed}, flush=True)

    # Run the same concrete case through both production encoder decorators. Only
    # the random draw is controlled; sampling, masking and metadata remain real.
    noise.set_gaussian_time_noise(enabled=True, time_std_fraction=0.1,
                                  deadline_margin_std_ratio=5.0, seed=1)
    try:
        for encoder, values, bounds in (
            (neg_identity_transform, torch.zeros(3, dtype=DTYPE), PotentialBounds(-1, 1)),
            (neg_log_transform, torch.full((3,), math.exp(-1), dtype=DTYPE),
             PotentialBounds(math.exp(-2), 1)),
        ):
            with patch.object(noise.torch, "normal", return_value=raw.clone()):
                event = encoder(values, bounds, return_spike_sample=True)
            assert event.domain == domain and event.observation_deadline == 3.0
            assert event.fired.tolist() == [True, True, False]
            assert event.time.tolist() == [2.5, 2.9, 3.0]
        equality = noise._sample_gaussian_spike_time(
            torch.full((2,), 2.0, dtype=DTYPE), time_std=0.0,
            time_mean=torch.tensor([1.0, math.nextafter(3.0, math.inf) - 2.0], dtype=DTYPE),
            domain=domain, deadline_margin=1.0, generator=torch.Generator().manual_seed(1),
        )
        assert equality.fired.tolist() == [True, False]
    finally:
        noise.set_gaussian_time_noise(enabled=False)


def verify_sampler_and_pwm():
    domain = TimeBounds(0.0, 4.0)
    raw = torch.tensor([-0.25, 4.25, 8.0, 8.25], dtype=DTYPE)
    sample = noise._sample_gaussian_spike_time(
        torch.ones_like(raw), domain=domain, time_std=0.0,
        time_mean=raw - 1.0, deadline_margin=4.0,
        generator=torch.Generator().manual_seed(3),
    )
    close(sample.time, [-0.25, 4.25, 8.0, 8.0])
    assert sample.fired.tolist() == [True, True, True, False]
    assert sample.domain == domain and sample.observation_deadline == 8.0

    # Verify the stochastic draw itself, including both outside-window tails.
    nominal = torch.linspace(0.0, 4.0, 4096, dtype=DTYPE)
    raw_random = torch.normal(nominal, 1.0, generator=torch.Generator().manual_seed(19))
    stochastic = noise._sample_gaussian_spike_time(
        nominal, domain=domain, time_std=1.0, deadline_margin=2.0,
        generator=torch.Generator().manual_seed(19),
    )
    assert torch.equal(stochastic.fired, raw_random <= 6.0)
    assert torch.equal(stochastic.time[stochastic.fired], raw_random[stochastic.fired])
    assert (stochastic.time < 0).any() and (stochastic.time[stochastic.fired] > 4).any()

    for a_time, b_time, a_fired, b_fired in product(
        (-0.25, 4.25), (-0.5, 4.75), (False, True), (False, True)
    ):
        a = SpikeSample(torch.tensor([a_time], dtype=DTYPE), domain, torch.tensor([a_fired]), 8.0)
        b = SpikeSample(torch.tensor([b_time], dtype=DTYPE), domain, torch.tensor([b_fired]), 8.0)
        expected = independent_duration(a, b)
        close(signed_pulse_width_duration(a, b, observation_deadline=8.0), expected)
        result, _ = signed_pulse_width_modulation_operator(
            a, domain, b, domain, torch.ones(1, dtype=DTYPE), PotentialBounds(1, 1),
            observation_deadline=8.0,
        )
        close(result, expected)


def verify_affine_and_multiplication():
    bounds = PotentialBounds(-2.0, 2.0)
    x = torch.tensor([[0.25, -0.5]], dtype=DTYPE)
    linear = spiking_ops.SpikingLinear(2, 1).double()
    conv1d = gpt2.SpikingConv1D(1, 2).double()
    conv2d = spiking_ops.SpikingConv2d(1, 1, 1).double()
    for layer in (linear, conv1d, conv2d):
        with torch.no_grad():
            layer.weight.fill_(0.25)
            layer.bias.fill_(0.125)

    for target in (early, late, missed):
        for prefix, layer, value in (
            ("linear", linear, x), ("conv1d", conv1d, x),
            ("conv2d", conv2d, x.reshape(1, 1, 1, 2)),
        ):
            with Trace({f"{prefix}.data": target, f"{prefix}.reference": reference}) as trace:
                output = layer(Potential(value, bounds))
                duration = independent_duration(trace.events[f"{prefix}.data"][0],
                                                trace.events[f"{prefix}.reference"][0])
                if prefix == "linear":
                    expected = torch.nn.functional.linear(duration, layer.weight, layer.bias)
                elif prefix == "conv1d":
                    expected = duration @ layer.weight + layer.bias
                else:
                    expected = torch.nn.functional.conv2d(duration, layer.weight, layer.bias)
                close(trace.outputs[f"{prefix}.output"][0], expected)
                close(output.value, expected.clamp(output.domain.min, output.domain.max))
        with Trace({"multiplication.data": target, "multiplication.reference": reference}) as trace:
            functions.multiplication_operator(x, bounds, x, bounds)
            width = independent_duration(trace.events["multiplication.data"][0],
                                         trace.events["multiplication.reference"][0])
            close(trace.outputs["multiplication.output"][0], x * width)


def verify_exponential_consumers():
    # Asymmetry distinguishes the normalized offset from the centered offset.
    bounds = PotentialBounds(-1.0, 3.0)
    x = torch.tensor([0.25, -0.5], dtype=DTYPE)
    for target, tau, normalized in product((early, late, missed), (0.5, 1.0, 2.0), (False, True)):
        with Trace({"exponential.input": target}) as trace:
            functions.exponential_function(x, bounds, tau_m=tau, normalized=normalized)
            event = trace.events["exponential.input"][0]
            offset = bounds.max if normalized else event.domain.range / 2.0
            # Same result as the manuscript's observation-time decay and fixed gain.
            expected = torch.where(event.fired, torch.exp((event.time - offset) / tau), 0.0)
            close(trace.outputs["exponential.output"][0], expected)
            physical = math.exp((event.observation_deadline - offset) / tau) * torch.exp(
                -(event.observation_deadline - event.time) / tau
            ) * event.fired
            close(expected, physical)

    # External exponential-difference events also keep raw times before conversion
    # into a potential. Its intermediate potential saturation is intentional.
    domain = TimeBounds(0.0, 2.0)
    for a_time, b_time, a_fired, b_fired in product(
        (-0.25, 2.25), (-0.5, 2.75), (False, True), (False, True)
    ):
        a = SpikeSample(torch.tensor([a_time], dtype=DTYPE), domain, torch.tensor([a_fired]), 4.0)
        b = SpikeSample(torch.tensor([b_time], dtype=DTYPE), domain, torch.tensor([b_fired]), 4.0)
        with Trace() as trace:
            spike_to_potential.exponential_difference_operator(a, domain, b, domain, 1.0)
            expected = torch.exp(independent_duration(a, b).clamp(-2.0, 2.0))
            close(trace.outputs["exponential_difference.output"][0], expected)

    for target in (early, late, missed):
        with Trace({"exponential_difference.internal": target}) as trace:
            domain = TimeBounds(0.0, 2.0)
            spike_to_potential.exponential_difference_operator(x.abs(), domain, x.abs() + 0.1, domain, 1.0)
            event = trace.events["exponential_difference.internal"][0]
            expected = torch.where(event.fired, torch.exp(event.time - 2.0), 0.0)
            close(trace.outputs["exponential_difference.output"][0], expected)
        with Trace({"swiglu.exponential_input": target}) as trace:
            functions.swiglu_function(x, bounds, x, bounds, beta=1.0, tau_s=1.0)
            event = trace.events["swiglu.exponential_input"][0]
            expected = torch.where(event.fired, torch.exp(event.time - bounds.max), 0.0)
            close(trace.outputs["swiglu.exponential_output"][0], expected)


def verify_attention_and_layernorm():
    bounds = PotentialBounds(-2.0, 2.0)
    value = torch.tensor([[[[0.25], [-0.5]]]], dtype=DTYPE)
    weights = torch.tensor([[[[0.25, 0.75]]]], dtype=DTYPE)
    for target in (early, late, missed):
        with Trace({"attention.value": target, "attention.value_reference": reference}) as trace:
            attention._gaussian_attention_value_readout(value, weights, bounds, bounds)
            expected = weights @ independent_duration(trace.events["attention.value"][0],
                                                       trace.events["attention.value_reference"][0])
            close(trace.outputs["attention.value_output"][0], expected)

    value = torch.tensor([[-1.0, -0.5, 0.5, 1.0]], dtype=DTYPE)
    for flags in product((False, True), repeat=3):
        layer = _layer(flags)
        _bind(layer, 2.0)
        for target in (early, late, missed):
            with Trace({"layernorm.log_positive": target,
                        "layernorm.log_negative": target,
                        "layernorm.log_sigma": reference}) as trace:
                output = layer(Potential(value, bounds))
                assert torch.isfinite(output.value).all()
                if not any(flags):
                    assert not trace.events
                    close(output.value, _reference(layer, value, 2.0))
                elif flags[1]:
                    sigma = trace.events["layernorm.log_sigma"][0]
                    positive = trace.events["layernorm.log_positive"][0]
                    negative = trace.events["layernorm.log_negative"][0]
                    width_positive = independent_duration(positive, sigma)
                    width_negative = independent_duration(negative, sigma)
                    if flags[2]:
                        width_positive = width_positive.clamp(-sigma.domain.range, sigma.domain.range)
                        width_negative = width_negative.clamp(-sigma.domain.range, sigma.domain.range)
                    expected = (torch.exp(width_positive) * (value > 0)
                                - torch.exp(width_negative) * (value < 0))
                    close(trace.outputs["layernorm.normalized_output"][0], expected)


def verify_composition_routes():
    x = torch.tensor([-0.5, 0.5], dtype=DTYPE)
    domain = PotentialBounds(-2.0, 2.0)
    # Drive raw early or late events through the maintained composed functions.
    # Their intermediate potential clamps remain part of the declared domains.
    for target in (early, late):
        with Trace({"exponential.input": target,
                    "division.numerator": target,
                    "division.denominator": reference,
                    "gelu.cubic.log_positive": target,
                    "gelu.cubic.log_negative": target,
                    "gelu.cubic.log_reference": reference}) as trace:
            for function in (functions.gelu_approximation, functions.gelu_approximation_sigmoid,
                             functions.tanh, functions.softmin_function):
                output, result_domain = function(x, domain)
                assert torch.isfinite(output).all()
                assert (output >= result_domain.min).all() and (output <= result_domain.max).all()
            for site in ("gelu.cubic.log_positive", "gelu.cubic.log_negative",
                         "gelu.cubic.log_reference", "division.numerator", "division.denominator"):
                assert site in trace.events
                for event in trace.events[site]:
                    assert event.fired.all()
                    if site.endswith(("reference", "denominator")) or target is late:
                        assert (event.time > event.domain.max).all()
                    else:
                        assert (event.time < event.domain.min).all()


def verify_regression_sensitivity():
    """Demonstrate that the audit fails if the former timestamp clamp returns."""
    original = noise._sample_gaussian_spike_time

    def clipped_sample(*args, **kwargs):
        event = original(*args, **kwargs)
        return event._replace(time=event.time.clamp(event.domain.min, event.domain.max))

    with patch.object(noise, "_sample_gaussian_spike_time", clipped_sample):
        try:
            verify_requested_boundary_example()
        except AssertionError:
            pass
        else:
            raise AssertionError("boundary audit failed to detect restored timestamp clamp")


def main():
    torch.set_num_threads(1)
    try:
        for verify in (verify_requested_boundary_example, verify_sampler_and_pwm, verify_affine_and_multiplication,
                       verify_exponential_consumers, verify_attention_and_layernorm,
                       verify_composition_routes, verify_regression_sensitivity):
            verify()
            print(f"PASS {verify.__name__}", flush=True)
    finally:
        noise.set_gaussian_time_noise(enabled=False)


if __name__ == "__main__":
    main()
