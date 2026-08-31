#!/usr/bin/env python3
"""Dataset-independent regression checks for Gaussian spike-time noise."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import math
from pathlib import Path
import subprocess
import sys

import torch

# Make direct ``python scripts/verification/...`` execution resolve repository
# modules without requiring an editable install or caller-provided PYTHONPATH.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from utils.transforms.noise import (
    _broadcast_gaussian_time_inputs,
    _sample_gaussian_spike_time,
    clamp_gaussian_output,
    clear_gaussian_noise_stats,
    gaussian_deadline_miss_probability,
    get_gaussian_noise_stats,
    get_gaussian_time_noise,
    set_gaussian_time_noise,
)
from utils.transforms.potential_to_spike import (
    neg_identity_transform,
    neg_linear_transform,
    neg_log_transform,
)
from utils.transforms.functions import (
    division_function,
    exponential_function,
    gelu_approximation_sigmoid,
    multiplication_operator,
    softmin_function,
    swiglu_function,
    tanh,
)
from utils.transforms.spike_to_potential import (
    exp_operator,
    exponential_difference_operator,
    normalized_exp_operator,
)
from utils.transforms.types import (
    ClosedBounds,
    Potential,
    PotentialBounds,
    SpikeSample,
    TimeBounds,
    check_domain,
)
from utils.transformers.models.spiking_gpt2.modeling_spiking_gpt2 import SpikingConv1D
from utils.transformers.models.spiking_ops import (
    SpikingConv2d,
    SpikingLayerNorm,
    SpikingLinear,
)
from utils.transformers.integrations.spiking_sdpa_attention import (
    _gaussian_attention_value_readout,
    attention_output_bounds,
    spiking_scaled_dot_product_attention,
)


def verify_immutable_memoized_bounds() -> None:
    """Verify immutable domains and memoized fixed attention rails.

    Static potential and time envelopes must not be widened after construction.
    Attention additionally reuses one bounds object for each ``(theta, S_max)``
    configuration so repeated forward calls do not allocate equivalent metadata.

    Raises:
        AssertionError: If endpoint mutation succeeds, equal attention
            configurations do not share an object, or distinct configurations are
            incorrectly aliased.
    """
    # PotentialBounds and TimeBounds inherit the frozen endpoint contract from
    # ClosedBounds. Attempt both endpoint names so either generated setter regressing
    # to a mutable dataclass is detected immediately.
    potential_domain = PotentialBounds(-2.0, 2.0)
    time_domain = TimeBounds(0.0, 4.0)
    for domain, endpoint, replacement in (
        (potential_domain, "min", -3.0),
        (time_domain, "max", 5.0),
    ):
        try:
            setattr(domain, endpoint, replacement)
        except FrozenInstanceError:
            pass
        else:
            raise AssertionError("ClosedBounds endpoints must be immutable")

    # Repeated calls with one normalized configuration must return the exact cached
    # object. A different maximum source length must retain a separate physical rail.
    first = attention_output_bounds(2.0, 5)
    repeated = attention_output_bounds(2.0, 5)
    distinct = attention_output_bounds(2.0, 6)
    assert repeated is first
    assert distinct is not first
    assert first == PotentialBounds(-10.0, 10.0)
    assert distinct == PotentialBounds(-12.0, 12.0)

    # Gaussian seed selects only the sampled physical event stream. Run the same
    # multiplication under two replicas with enough events to make an identical
    # random output negligibly likely, while retaining one caller-derived fixed rail.
    drive = torch.linspace(-1.5, 1.5, 128)
    encoded = torch.linspace(1.5, -1.5, 128)
    operand_domain = PotentialBounds(-2.0, 2.0)
    set_gaussian_time_noise(enabled=True, time_std=0.5, seed=41)
    first_output, first_domain = multiplication_operator(
        drive,
        operand_domain,
        encoded,
        operand_domain,
        theta=2.0,
    )
    set_gaussian_time_noise(enabled=True, time_std=0.5, seed=42)
    second_output, second_domain = multiplication_operator(
        drive,
        operand_domain,
        encoded,
        operand_domain,
        theta=2.0,
    )

    # Different samples must not affect any metadata endpoint. Disable shared noise
    # after the check so following groups start from their own explicit replica state.
    assert not torch.equal(first_output, second_output)
    assert first_domain == second_domain == PotentialBounds(-4.0, 4.0)
    set_gaussian_time_noise(enabled=False)


# @lat: [[evaluation#Evaluation and Verification#Gaussian Spike-Time Verification#Closed-Domain Verification]]
def verify_closed_bounds_validation() -> None:
    """Verify central endpoint validation and optimization-safe domain checks.

    Every physical interval must reject non-real, non-finite, and reversed endpoints
    when it is constructed. Tensor membership failures must raise an explicit
    exception even when Python removes ``assert`` statements under ``-O``.

    Raises:
        AssertionError: If malformed bounds or out-of-domain tensors are accepted.
    """
    for bounds_type in (ClosedBounds, PotentialBounds, TimeBounds):
        assert bounds_type(1.0, 1.0).range == 0.0
        for endpoints, error_type, message in (
            ((False, 1.0), TypeError, "real scalar"),
            (("0", 1.0), TypeError, "real scalar"),
            ((0.0, float("inf")), ValueError, "finite"),
            ((float("nan"), 1.0), ValueError, "finite"),
            ((2.0, 1.0), ValueError, "min <= max"),
        ):
            try:
                bounds_type(*endpoints)
            except error_type as exc:
                assert message in str(exc)
            else:
                raise AssertionError(
                    f"{bounds_type.__name__} accepted invalid endpoints {endpoints}"
                )

    @check_domain
    def identity(input_value: torch.Tensor, domain: PotentialBounds) -> torch.Tensor:
        return input_value

    valid_domain = PotentialBounds(-1.0, 1.0)
    valid = torch.tensor([-1.0, 0.0, 1.0])
    assert identity(valid, valid_domain) is valid
    for invalid in (
        torch.tensor([-1.01, 0.0]),
        torch.tensor([0.0, 1.01]),
        torch.tensor([0.0, float("nan")]),
    ):
        try:
            identity(invalid, valid_domain)
        except ValueError as exc:
            assert "must be within the specified domain" in str(exc)
        else:
            raise AssertionError("check_domain accepted an out-of-domain tensor")

    optimized_check = """
import torch
from utils.transforms.types import PotentialBounds, check_domain

@check_domain
def identity(input_value, domain):
    return input_value

try:
    identity(torch.tensor([2.0]), PotentialBounds(-1.0, 1.0))
except ValueError:
    pass
else:
    raise SystemExit('optimized check_domain accepted an invalid tensor')
"""
    completed = subprocess.run(
        [sys.executable, "-O", "-c", optimized_check],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            "check_domain failed under optimized Python:\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )


# @lat: [[evaluation#Evaluation and Verification#Gaussian Spike-Time Verification]]
def verify_broadcast_gaussian_time_inputs() -> None:
    """Verify scalar/tensor broadcasting while preserving dtype and device.

    The Gaussian sampler and analytic deadline calculation share this private
    normalization boundary. A regression here could silently change parameter
    alignment, promote precision, or move distribution inputs off the nominal
    spike tensor's device before either public calculation begins.

    Raises:
        AssertionError: If broadcasting, values, dtype, or device differ from the
            contract shared by the sampled and analytic Gaussian paths.
    """
    # Cross-broadcast a column of nominal times with a row of means. This forces
    # both tensor operands to expand rather than merely accepting equal shapes.
    nominal_input = torch.tensor([[0.25], [1.75]], dtype=torch.float64)
    mean_input = torch.tensor([[0.0, 0.5, 1.0]], dtype=torch.float32)
    domain = TimeBounds(0.0, 4.0)
    nominal, mean, std = _broadcast_gaussian_time_inputs(
        nominal_input,
        time_mean=mean_input,
        time_std=0.125,
        domain=domain,
    )

    # Every result must follow the nominal tensor's shape-independent physical
    # representation: same floating dtype and same device after broadcasting.
    assert nominal.shape == mean.shape == std.shape == (2, 3)
    assert nominal.dtype == mean.dtype == std.dtype == nominal_input.dtype
    assert nominal.device == mean.device == std.device == nominal_input.device

    # Check values explicitly so an accidental dimension swap cannot pass shape,
    # dtype, and device assertions while associating means with the wrong events.
    assert torch.equal(nominal, nominal_input.expand(2, 3))
    assert torch.equal(mean, mean_input.to(torch.float64).expand(2, 3))
    assert torch.equal(std, torch.full((2, 3), 0.125, dtype=torch.float64))

    # Reverse the scalar/tensor roles to cover a scalar nominal event expanded by
    # a vector standard deviation, including the deterministic zero-noise entry.
    scalar_nominal, scalar_mean, vector_std = _broadcast_gaussian_time_inputs(
        torch.tensor(2.0, dtype=torch.float32),
        time_mean=0.25,
        time_std=torch.tensor([0.0, 0.5], dtype=torch.float64),
        domain=domain,
    )
    assert torch.equal(scalar_nominal, torch.tensor([2.0, 2.0]))
    assert torch.equal(scalar_mean, torch.tensor([0.25, 0.25]))
    assert torch.equal(vector_std, torch.tensor([0.0, 0.5]))


def verify_gaussian_time_input_validation() -> None:
    """Verify rejection of malformed Gaussian timing inputs and domains.

    Validation is shared by analytic miss probabilities and sampled event times,
    so each invalid case must fail before either path performs Gaussian arithmetic
    or consumes random state. Exception messages are checked to distinguish the
    intended contract failure from an incidental downstream PyTorch error.

    Raises:
        AssertionError: If an invalid case is accepted, raises the wrong exception,
            or reports a reason unrelated to the violated input contract.
    """
    # Keep one valid baseline explicit so each case mutates only the field named by
    # its label. This prevents overlapping faults from masking a missing check.
    valid_nominal = torch.tensor([0.0, 2.0, 4.0], dtype=torch.float32)
    valid_domain = TimeBounds(0.0, 4.0)

    # Type and domain failures are listed separately from numeric failures because
    # they must be rejected before tensor broadcasting touches their contents.
    invalid_cases = [
        (
            "integer nominal time",
            TypeError,
            "floating-point",
            torch.tensor([1, 2]),
            0.0,
            1.0,
            valid_domain,
        ),
        (
            "wrong domain type",
            TypeError,
            "TimeBounds",
            valid_nominal,
            0.0,
            1.0,
            object(),
        ),
    ]

    # Non-finite distribution inputs, negative scale, and nominal codewords outside
    # the declared interval each exercise a distinct numeric validation branch.
    invalid_cases.extend(
        [
            (
                "non-finite nominal time",
                ValueError,
                "must be finite",
                torch.tensor([float("nan")]),
                0.0,
                1.0,
                valid_domain,
            ),
            (
                "non-finite mean",
                ValueError,
                "must be finite",
                valid_nominal,
                float("-inf"),
                1.0,
                valid_domain,
            ),
            (
                "non-finite standard deviation",
                ValueError,
                "must be finite",
                valid_nominal,
                0.0,
                float("nan"),
                valid_domain,
            ),
            (
                "negative standard deviation",
                ValueError,
                "non-negative",
                valid_nominal,
                0.0,
                -0.25,
                valid_domain,
            ),
            (
                "nominal time outside domain",
                ValueError,
                "within the declared time domain",
                torch.tensor([-0.01, 1.0]),
                0.0,
                1.0,
                valid_domain,
            ),
        ]
    )

    # Verify both exception class and diagnostic text for every table entry. A
    # different exception is wrapped with its label to keep failures actionable.
    for label, expected_type, message, nominal, mean, std, domain in invalid_cases:
        try:
            _broadcast_gaussian_time_inputs(
                nominal,
                time_mean=mean,
                time_std=std,
                domain=domain,
            )
        except expected_type as error:
            assert message in str(error), (
                f"{label} reported an unexpected diagnostic: {error}"
            )
        except Exception as error:
            raise AssertionError(
                f"{label} raised {type(error).__name__}, expected "
                f"{expected_type.__name__}"
            ) from error
        else:
            raise AssertionError(f"{label} was accepted")

    # Both code-window endpoints are legal nominal events, and zero standard
    # deviation is the deterministic Gaussian limit rather than an invalid scale.
    nominal, mean, std = _broadcast_gaussian_time_inputs(
        valid_nominal,
        time_mean=0.0,
        time_std=0.0,
        domain=valid_domain,
    )
    assert torch.equal(nominal, valid_nominal)
    assert bool((mean == 0.0).all() and (std == 0.0).all())


def verify_gaussian_sampler_rng_contract() -> None:
    """Verify seeded replay, sequential RNG advance, and zero-noise stability.

    A replica seed must identify an entire advancing sample stream rather than a
    separately reseeded layer call. Conversely, the deterministic zero-standard-
    deviation path must leave that stream untouched so parity checks cannot shift
    later stochastic events merely by executing an exact-noise operator first.

    Raises:
        AssertionError: If equal seeds diverge, sequential calls fail to advance
            state, or a zero-standard-deviation call consumes generator state.
    """
    # Use timestamps away from both rails so this function isolates RNG behavior;
    # explicit early-arrival and deadline classification belong to the next check.
    nominal = torch.linspace(0.5, 3.5, steps=128, dtype=torch.float64)
    domain = TimeBounds(0.0, 4.0)

    # Two independently constructed generators with the same seed represent two
    # replicas that must produce identical complete sample sequences.
    generator = torch.Generator(device=nominal.device).manual_seed(314159)
    replay_generator = torch.Generator(device=nominal.device).manual_seed(314159)
    initial_state = generator.get_state().clone()
    assert torch.equal(initial_state, replay_generator.get_state())

    # The first nonzero-noise call must consume state, and replaying it from the
    # second generator must reproduce both stored times and delivery masks exactly.
    first = _sample_gaussian_spike_time(
        nominal,
        time_std=0.2,
        time_mean=-0.05,
        domain=domain,
        generator=generator,
    )
    replay_first = _sample_gaussian_spike_time(
        nominal,
        time_std=0.2,
        time_mean=-0.05,
        domain=domain,
        generator=replay_generator,
    )
    first_state = generator.get_state().clone()
    assert not torch.equal(initial_state, first_state)
    assert torch.equal(first_state, replay_generator.get_state())
    assert torch.equal(first.time, replay_first.time)
    assert torch.equal(first.fired, replay_first.fired)

    # A second call continues from the advanced state. Matching the second replay
    # proves reproducibility applies to the stream sequence, not only its first draw.
    second = _sample_gaussian_spike_time(
        nominal,
        time_std=0.2,
        time_mean=-0.05,
        domain=domain,
        generator=generator,
    )
    replay_second = _sample_gaussian_spike_time(
        nominal,
        time_std=0.2,
        time_mean=-0.05,
        domain=domain,
        generator=replay_generator,
    )
    second_state = generator.get_state().clone()
    assert not torch.equal(first_state, second_state)
    assert torch.equal(second_state, replay_generator.get_state())
    assert torch.equal(second.time, replay_second.time)
    assert torch.equal(second.fired, replay_second.fired)

    # The all-zero scale branch performs no random draw. Its additive mean still
    # shifts time deterministically, demonstrating that RNG avoidance is not an
    # accidental consequence of returning the nominal tensor unchanged.
    zero_generator = torch.Generator(device=nominal.device).manual_seed(271828)
    zero_state = zero_generator.get_state().clone()
    zero_sample = _sample_gaussian_spike_time(
        nominal,
        time_std=0.0,
        time_mean=0.125,
        domain=domain,
        generator=zero_generator,
    )
    assert torch.equal(zero_state, zero_generator.get_state())
    assert torch.equal(zero_sample.time, nominal + 0.125)
    assert bool(zero_sample.fired.all())
    assert zero_sample.time.dtype == nominal.dtype
    assert zero_sample.time.device == nominal.device


def verify_gaussian_sampler_deadline_contract() -> None:
    """Verify start clamping, inclusive deadline delivery, and strict misses.

    The sampler stores every result as a finite in-domain tensor, but the
    ``fired`` mask must preserve the physical distinction between an event at the
    observation deadline and an event that arrived too late. Deterministic mean
    offsets create exact boundary cases without relying on random draws.

    Raises:
        AssertionError: If an early event is marked missed, deadline equality is
            rejected, or a late event is stored without a false delivery mask.
    """
    # The three raw timestamps become -0.75, 4.0, and 4.25. They deliberately
    # exercise the interval start, exact deadline, and strict post-deadline cases.
    nominal = torch.tensor([0.25, 3.0, 3.5], dtype=torch.float64)
    time_mean = torch.tensor([-1.0, 1.0, 0.75], dtype=torch.float64)
    domain = TimeBounds(0.0, 4.0)
    generator = torch.Generator(device=nominal.device).manual_seed(161803)

    # Zero standard deviation makes the constructed raw timestamps exact while
    # still entering the same sampler that production event-aware encoders use.
    sample = _sample_gaussian_spike_time(
        nominal,
        time_std=0.0,
        time_mean=time_mean,
        domain=domain,
        generator=generator,
    )

    # An event earlier than the modeled interval is observable from its start. Its
    # stored time is clamped upward, but it remains a physically delivered event.
    assert sample.time[0].item() == domain.min
    assert sample.fired[0].item() is True

    # The deadline is inclusive by contract. A raw timestamp equal to TimeBounds.max
    # must therefore remain distinguishable from the late event beside it.
    assert sample.time[1].item() == domain.max
    assert sample.fired[1].item() is True

    # A strict exceedance is the only miss. It stores the deadline as a finite
    # carrier, so downstream code must consult fired rather than infer from time.
    assert sample.time[2].item() == domain.max
    assert sample.fired[2].item() is False
    assert torch.isfinite(sample.time).all()
    assert bool(((sample.time >= domain.min) & (sample.time <= domain.max)).all())
    assert sample.domain == domain


def verify_gaussian_deadline_probability() -> None:
    """Compare analytic deadline-miss probabilities with seeded sampling.

    The analytic helper and event sampler must describe the same strict Gaussian
    tail beyond ``TimeBounds.max``. Three parameter sets cover low, intermediate,
    and symmetric miss probabilities, while a six-standard-error tolerance makes
    the seeded empirical check robust to ordinary binomial sampling variation.

    Raises:
        AssertionError: If deterministic zero-scale probabilities violate the
            inclusive deadline or empirical miss rates disagree with the formula.
    """
    # These shifted nominal times produce tails near 2.3%, 25%, and 50%, avoiding
    # a comparison that only probes the numerically easiest center of the Gaussian.
    nominal = torch.tensor([2.0, 3.5, 3.9], dtype=torch.float64)
    time_mean = torch.tensor([0.0, 0.0, 0.1], dtype=torch.float64)
    time_std = torch.tensor([1.0, 0.75, 0.5], dtype=torch.float64)
    domain = TimeBounds(0.0, 4.0)
    analytic = gaussian_deadline_miss_probability(
        nominal,
        time_std=time_std,
        time_mean=time_mean,
        domain=domain,
    )

    # Expand each parameter set across an independent sample axis. Broadcasting
    # preserves the three conditions while one generator advances through all draws.
    sample_count = 200_000
    generator = torch.Generator(device=nominal.device).manual_seed(424242)
    sample = _sample_gaussian_spike_time(
        nominal[:, None].expand(-1, sample_count),
        time_std=time_std[:, None],
        time_mean=time_mean[:, None],
        domain=domain,
        generator=generator,
    )
    empirical = (~sample.fired).to(torch.float64).mean(dim=1)

    # Estimate each binomial standard error from the analytic probability. Six
    # standard errors plus a small floating allowance gives a deterministic,
    # non-flaky threshold while still detecting material distribution drift.
    standard_error = torch.sqrt(
        analytic * (1.0 - analytic) / float(sample_count)
    )
    tolerance = 6.0 * standard_error + 5e-4
    error = torch.abs(empirical - analytic)
    assert bool((error <= tolerance).all()), (
        f"analytic={analytic.tolist()}, empirical={empirical.tolist()}, "
        f"tolerance={tolerance.tolist()}"
    )

    # Zero standard deviation has an exact probability rather than a limiting
    # approximation. Equality is delivered, strict exceedance is missed, and an
    # event nominally at the deadline remains delivered with zero mean.
    deterministic_probability = gaussian_deadline_miss_probability(
        torch.tensor([3.0, 3.0, 4.0], dtype=torch.float64),
        time_std=0.0,
        time_mean=torch.tensor([1.0, 1.1, 0.0], dtype=torch.float64),
        domain=domain,
    )
    assert torch.equal(
        deterministic_probability,
        torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
    )


def verify_exponential_time_constant_scaling() -> None:
    """Verify non-unit exponential scales and invalid-scale rejection.

    Logarithmic encoders multiply temporal differences by ``tau_s``; exponential
    decoders must divide by the same scale so division remains X/Y instead of
    ``(X/Y)**tau_s``. This regression covers the primitive normalized decoder,
    direct Gaussian exponential, exponential difference, division, softmin,
    fixed-shape activations, SwiGLU, and full-spiking LayerNorm at three scales. It
    also checks dtype-level endpoint
    failures and RNG non-consumption.

    Raises:
        AssertionError: If a non-unit scale changes an algebraic result, Gaussian
            zero-noise parity fails, invalid bounds pass, or rejection advances RNG.
    """
    time_value = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
    time_domain = TimeBounds(-1.0, 1.0)
    potential_value = torch.tensor([-0.75, 0.0, 1.0, 2.0], dtype=torch.float64)
    potential_domain = PotentialBounds(-1.0, 3.0)
    layernorm_baseline: torch.Tensor | None = None

    # The primitive must apply exactly the same 1/tau slope to payload and declared
    # endpoints. Testing float64 equality here catches a silently ignored argument
    # independently of every composed operator.
    for tau_s in (0.5, 1.0, 2.0):
        # Deadline-relative decay must apply the same time constant while keeping
        # its endpoint at one. This primitive is tested separately because it owns
        # dtype-level positive-underflow rejection for wide observation windows.
        decay_input = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        decay_domain = TimeBounds(0.0, 1.0)
        decay, decay_bounds = exp_operator(
            decay_input,
            decay_domain,
            tau_m=tau_s,
        )
        assert torch.equal(
            decay,
            torch.exp(-(float(decay_domain.max) - decay_input) / tau_s),
        )
        assert math.isclose(
            float(decay_bounds.min),
            math.exp(-float(decay_domain.range) / tau_s),
            rel_tol=1e-15,
        )
        assert decay_bounds.max == 1.0

        # Negative-log encoding must scale both payload times and the observation
        # deadline by tau_s. Its endpoint values also lock the inclusive zero/deadline
        # mapping used by division and LayerNorm.
        log_input = torch.tensor([0.1, 1.0, 10.0], dtype=torch.float64)
        log_domain = PotentialBounds(0.1, 10.0)
        set_gaussian_time_noise(enabled=False)
        log_time, log_bounds = neg_log_transform(
            log_input,
            log_domain,
            tau_s=tau_s,
        )
        expected_log_time = tau_s * (
            torch.log(log_input.new_tensor(float(log_domain.max)))
            - torch.log(log_input)
        )
        assert torch.allclose(log_time, expected_log_time)
        assert log_bounds.min == 0.0
        assert math.isclose(
            float(log_bounds.max),
            tau_s
            * (
                math.log(float(log_domain.max))
                - math.log(float(log_domain.min))
            ),
            rel_tol=1e-15,
        )

        # The encoder must not materialize V_max/V in the payload dtype. This
        # float32 domain has finite endpoints and finite logarithms, but its endpoint
        # ratio overflows before ``torch.log`` if division is performed first.
        wide_log_input = torch.tensor([1.0e-30], dtype=torch.float32)
        wide_log_domain = PotentialBounds(1.0e-30, 1.0e30)
        wide_log_time, _ = neg_log_transform(
            wide_log_input,
            wide_log_domain,
            tau_s=tau_s,
        )
        expected_wide_log_time = tau_s * (
            torch.log(wide_log_input.new_tensor(float(wide_log_domain.max)))
            - torch.log(wide_log_input)
        )
        assert torch.isfinite(wide_log_time).all()
        assert torch.equal(wide_log_time, expected_wide_log_time)

        # Endpoint logs also avoid overflowing the Python scalar ratio used to
        # construct the observation deadline. Both extreme float64 rails remain
        # representable even though their quotient does not.
        extreme_log_input = torch.tensor([1.0e-300], dtype=torch.float64)
        extreme_log_domain = PotentialBounds(1.0e-300, 1.0e300)
        extreme_log_time, extreme_log_bounds = neg_log_transform(
            extreme_log_input,
            extreme_log_domain,
            tau_s=tau_s,
        )
        expected_extreme_deadline = tau_s * (
            math.log(float(extreme_log_domain.max))
            - math.log(float(extreme_log_domain.min))
        )
        assert torch.isfinite(extreme_log_time).all()
        assert math.isfinite(float(extreme_log_bounds.max))
        assert math.isclose(
            float(extreme_log_time.item()),
            expected_extreme_deadline,
            rel_tol=1e-15,
        )
        assert math.isclose(
            float(extreme_log_bounds.max),
            expected_extreme_deadline,
            rel_tol=1e-15,
        )

        primitive, primitive_domain = normalized_exp_operator(
            time_value,
            time_domain,
            tau_m=tau_s,
        )
        assert torch.equal(primitive, torch.exp(time_value / tau_s))
        assert math.isclose(
            float(primitive_domain.min),
            math.exp(float(time_domain.min) / tau_s),
            rel_tol=1e-15,
        )
        assert math.isclose(
            float(primitive_domain.max),
            math.exp(float(time_domain.max) / tau_s),
            rel_tol=1e-15,
        )

        # Both public exponential modes must preserve deterministic/Gaussian
        # zero-noise parity on an asymmetric domain, which exposes incorrect use of
        # half-window offsets in the normalized current-gain path.
        for normalized in (True, False):
            set_gaussian_time_noise(enabled=False)
            deterministic_exp, deterministic_domain = exponential_function(
                potential_value,
                potential_domain,
                tau_m=tau_s,
                normalized=normalized,
            )
            set_gaussian_time_noise(enabled=True, time_std=0.0, seed=1701)
            gaussian_exp, gaussian_domain = exponential_function(
                potential_value,
                potential_domain,
                tau_m=tau_s,
                normalized=normalized,
            )
            assert torch.allclose(
                gaussian_exp,
                deterministic_exp,
                atol=1e-12,
                rtol=1e-12,
            )
            assert gaussian_domain.min == 0.0
            assert math.isclose(
                float(gaussian_domain.max),
                float(deterministic_domain.max),
                rel_tol=1e-12,
            )

        # Offset cancellation must occur before exponentiation. The final float32
        # response exp(-x) is representable on [-80, 80], while the obsolete
        # intermediate exp(t) would evaluate exp(160) and overflow before scaling.
        cancellation_input = torch.tensor([-80.0, 0.0, 80.0], dtype=torch.float32)
        cancellation_domain = PotentialBounds(-80.0, 80.0)
        set_gaussian_time_noise(enabled=False)
        cancellation_result, cancellation_bounds = exponential_function(
            cancellation_input,
            cancellation_domain,
            tau_m=1.0,
            normalized=True,
        )
        assert torch.isfinite(cancellation_result).all()
        assert torch.equal(cancellation_result, torch.exp(-cancellation_input))
        assert cancellation_bounds.min == torch.exp(
            torch.tensor(-80.0, dtype=torch.float32)
        ).item()
        assert cancellation_bounds.max == torch.exp(
            torch.tensor(80.0, dtype=torch.float32)
        ).item()

        # A direct time difference must decode as exp((t_B-t_A)/tau_s) in both
        # execution modes. This is the exact stage that previously omitted division
        # by tau_s after its internal negative-identity re-encoding.
        t_A = torch.tensor([0.5, 1.5], dtype=torch.float64)
        t_B = torch.tensor([1.0, 1.0], dtype=torch.float64)
        shared_domain = TimeBounds(0.0, 2.0)
        expected_difference = torch.exp((t_B - t_A) / tau_s)
        set_gaussian_time_noise(enabled=False)
        deterministic_difference, _ = exponential_difference_operator(
            t_A,
            shared_domain,
            t_B,
            shared_domain,
            tau_s=tau_s,
        )
        set_gaussian_time_noise(enabled=True, time_std=0.0, seed=1702)
        gaussian_difference, _ = exponential_difference_operator(
            t_A,
            shared_domain,
            t_B,
            shared_domain,
            tau_s=tau_s,
        )
        assert torch.allclose(deterministic_difference, expected_difference)
        assert torch.allclose(gaussian_difference, expected_difference)

        # Log-encoder scale must cancel completely through exponential difference.
        # Division therefore remains the same X/Y for every positive tau_s rather
        # than acquiring a scale-dependent power.
        numerator = torch.tensor([0.2, 1.0, 5.0], dtype=torch.float64)
        denominator = torch.tensor([1.0, 2.0, 10.0], dtype=torch.float64)
        ratio_domain = PotentialBounds(0.1, 10.0)
        set_gaussian_time_noise(enabled=False)
        deterministic_ratio, _ = division_function(
            numerator,
            denominator,
            ratio_domain,
            tau_s=tau_s,
        )
        set_gaussian_time_noise(enabled=True, time_std=0.0, seed=1703)
        gaussian_ratio, _ = division_function(
            numerator,
            denominator,
            ratio_domain,
            tau_s=tau_s,
        )
        expected_ratio = numerator / denominator
        assert torch.allclose(deterministic_ratio, expected_ratio)
        assert torch.allclose(gaussian_ratio, expected_ratio)

        # Softmin must retain its physical temperature: unnormalized exponentials
        # depend on tau_s, while division cancels its own log/decode scale exactly.
        scores = torch.tensor(
            [[-2.0, -1.0, 0.0, 1.0, 2.0], [1.5, -0.5, 0.25, -1.25, 2.0]],
            dtype=torch.float64,
        )
        score_domain = PotentialBounds(-2.0, 2.0)
        expected_softmin = torch.softmax(-scores / tau_s, dim=-1)
        set_gaussian_time_noise(enabled=False)
        deterministic_softmin, _ = softmin_function(
            scores,
            score_domain,
            tau=tau_s,
        )
        set_gaussian_time_noise(enabled=True, time_std=0.0, seed=1705)
        gaussian_softmin, _ = softmin_function(
            scores,
            score_domain,
            tau=tau_s,
        )
        assert torch.allclose(deterministic_softmin, expected_softmin)
        assert torch.allclose(gaussian_softmin, expected_softmin)

        # SwiGLU pre-scales its gate input by tau_s so the physical time
        # constant changes latency but not sigmoid(beta*u). An asymmetric u domain
        # verifies fixed bias cancellation at every tested scale.
        u = torch.tensor([-0.75, 0.0, 1.0, 2.0], dtype=torch.float64)
        v = torch.tensor([1.0, -0.5, 2.0, 0.25], dtype=torch.float64)
        domain_u = PotentialBounds(-1.0, 3.0)
        domain_v = PotentialBounds(-2.0, 2.0)
        beta = 0.7
        expected_swiglu = v * u * torch.sigmoid(beta * u)
        set_gaussian_time_noise(enabled=False)
        deterministic_swiglu, _ = swiglu_function(
            u,
            domain_u,
            v,
            domain_v,
            beta=beta,
            tau_s=tau_s,
            theta=8.0,
        )
        set_gaussian_time_noise(enabled=True, time_std=0.0, seed=1706)
        gaussian_swiglu, _ = swiglu_function(
            u,
            domain_u,
            v,
            domain_v,
            beta=beta,
            tau_s=tau_s,
            theta=8.0,
        )
        assert torch.allclose(deterministic_swiglu, expected_swiglu)
        assert torch.allclose(gaussian_swiglu, expected_swiglu)

        # GELU-sigmoid and tanh are pretrained activation definitions, not
        # temperature controls. Their input slopes cancel tau_s so only physical
        # latency changes across the three tested scales.
        activation = torch.tensor([-1.5, -0.25, 0.0, 0.75, 1.5], dtype=torch.float64)
        activation_domain = PotentialBounds(-2.0, 2.0)
        expected_gelu = activation * torch.sigmoid(1.702 * activation)
        expected_tanh = torch.tanh(activation)
        set_gaussian_time_noise(enabled=False)
        deterministic_gelu, _ = gelu_approximation_sigmoid(
            activation, activation_domain, tau_s=tau_s, theta=8.0
        )
        deterministic_tanh, _ = tanh(
            activation, activation_domain, tau_s=tau_s, theta=8.0
        )
        set_gaussian_time_noise(enabled=True, time_std=0.0, seed=1707)
        gaussian_gelu, _ = gelu_approximation_sigmoid(
            activation, activation_domain, tau_s=tau_s, theta=8.0
        )
        gaussian_tanh, _ = tanh(
            activation, activation_domain, tau_s=tau_s, theta=8.0
        )
        assert torch.allclose(deterministic_gelu, expected_gelu)
        assert torch.allclose(gaussian_gelu, expected_gelu)
        assert torch.allclose(deterministic_tanh, expected_tanh)
        assert torch.allclose(gaussian_tanh, expected_tanh)

        # LayerNorm log times scale with tau_s and exponential-difference decoding
        # divides by it, so the complete normalized activation is scale-invariant.
        layernorm_value = torch.tensor(
            [[-1.5, -0.25, 0.75, 1.0], [0.5, -1.0, 1.5, -0.5]],
            dtype=torch.float64,
        )
        layernorm = SpikingLayerNorm(
            4,
            eps=1.0e-5,
            theta=4.0,
            tau_s=tau_s,
            clip_margin=0.1,
            use_spiking_mul=True,
            use_spiking_log=True,
            use_spiking_expdiff=True,
        ).to(dtype=torch.float64)
        with torch.no_grad():
            layernorm.weight.copy_(
                torch.tensor([1.0, 0.8, 1.2, 0.5], dtype=torch.float64)
            )
            layernorm.bias.copy_(
                torch.tensor([0.1, -0.2, 0.05, 0.3], dtype=torch.float64)
            )
        layernorm_input = Potential(
            layernorm_value,
            PotentialBounds(-2.0, 2.0),
        )
        set_gaussian_time_noise(enabled=False)
        deterministic_layernorm = layernorm(layernorm_input)
        set_gaussian_time_noise(enabled=True, time_std=0.0, seed=1707)
        gaussian_layernorm = layernorm(layernorm_input)
        assert torch.allclose(
            gaussian_layernorm.value,
            deterministic_layernorm.value,
        )
        if layernorm_baseline is None:
            layernorm_baseline = deterministic_layernorm.value.detach().clone()
        else:
            assert torch.allclose(deterministic_layernorm.value, layernorm_baseline)

    # Endpoint overflow and positive-domain underflow must fail in the carrier dtype,
    # even though Python float could still represent one of those wider values.
    for invalid_domain in (TimeBounds(0.0, 100.0), TimeBounds(-200.0, 0.0)):
        try:
            normalized_exp_operator(
                torch.tensor([0.0], dtype=torch.float32),
                invalid_domain,
                tau_m=1.0,
            )
        except ValueError as error:
            assert "finite and strictly positive" in str(error)
        else:
            raise AssertionError(f"accepted unrepresentable domain {invalid_domain}")

    # Deadline-relative decay rejects a window whose earliest response becomes
    # indistinguishable from reset zero in float32.
    try:
        exp_operator(
            torch.tensor([0.0], dtype=torch.float32),
            TimeBounds(0.0, 200.0),
            tau_m=1.0,
        )
    except ValueError as error:
        assert "strictly positive" in str(error)
    else:
        raise AssertionError("accepted underflowing exponential-decay window")

    # Invalid Gaussian scales are rejected before the internal encoder. Snapshot the
    # generator to prove validation does not consume a sample or shift later events.
    set_gaussian_time_noise(enabled=True, time_std=1.0, seed=1704)
    try:
        generator = get_gaussian_time_noise().generator
        assert generator is not None
        state_before = generator.get_state().clone()
        try:
            neg_log_transform(
                torch.tensor([1.0], dtype=torch.float64),
                PotentialBounds(0.1, 10.0),
                tau_s=0.0,
                return_spike_sample=True,
                noise_site="verification.invalid_log_scale",
            )
        except ValueError as error:
            assert "finite and positive" in str(error)
        else:
            raise AssertionError("accepted zero negative-log tau_s")
        assert torch.equal(state_before, generator.get_state())

        try:
            exponential_difference_operator(
                torch.tensor([0.5], dtype=torch.float64),
                TimeBounds(0.0, 2.0),
                torch.tensor([1.0], dtype=torch.float64),
                TimeBounds(0.0, 2.0),
                tau_s=0.0,
            )
        except ValueError as error:
            assert "finite and positive" in str(error)
        else:
            raise AssertionError("accepted zero exponential-difference tau_s")
        assert torch.equal(state_before, generator.get_state())
    finally:
        set_gaussian_time_noise(enabled=False)

    # Domain errors are explicit runtime validation rather than optimization-sensitive
    # assertions, and they apply equally when Gaussian sampling is disabled.
    try:
        neg_log_transform(
            torch.tensor([1.0], dtype=torch.float64),
            PotentialBounds(0.0, 10.0),
            tau_s=1.0,
        )
    except ValueError as error:
        assert "strictly positive minimum" in str(error)
    else:
        raise AssertionError("accepted non-positive negative-log domain")


def verify_gaussian_encoder_boundary() -> None:
    """Verify production encoder dispatch, event parity, misses, and counters.

    The decorated identity encoder must preserve its deterministic tuple contract
    when noise is disabled, expose ``SpikeSample`` only for an enabled event-aware
    request, and attribute the exact sampled delivery mask to the requested site.
    Forced zero-scale mean offsets make parity and miss counts exact.

    Raises:
        AssertionError: If tuple behavior changes, event requests bypass Gaussian
            configuration, or per-site event and miss counters diverge from output.
    """
    potential = torch.tensor([0.0, 1.0, 4.0], dtype=torch.float64)
    domain = PotentialBounds(0.0, 4.0)

    # Begin from an explicitly disabled process-wide configuration. The ordinary
    # encoder result must remain the negative-identity codeword and declared window.
    set_gaussian_time_noise(enabled=False)
    try:
        # A non-identity window verifies the affine encoder independently of its
        # convenience wrapper and locks both endpoint directions exactly.
        linear_time, linear_domain = neg_linear_transform(
            potential,
            domain,
            window_length=2.0,
        )
        assert torch.equal(
            linear_time,
            torch.tensor([2.0, 1.5, 0.0], dtype=torch.float64),
        )
        assert linear_domain == TimeBounds(0.0, 2.0)

        deterministic_time, time_domain = neg_identity_transform(potential, domain)
        assert torch.equal(
            deterministic_time,
            torch.tensor([4.0, 3.0, 0.0], dtype=torch.float64),
        )
        assert time_domain == TimeBounds(0.0, 4.0)

        # An event-aware return type without an enabled replica is a configuration
        # error; silently manufacturing an all-fired mask would hide caller misuse.
        try:
            neg_identity_transform(
                potential,
                domain,
                return_spike_sample=True,
                noise_site="verification.disabled",
            )
        except RuntimeError as error:
            assert "requires enabled Gaussian time noise" in str(error)
        else:
            raise AssertionError("disabled event-aware encoder request was accepted")

        # Zero scale enters the physical event path without perturbing timestamps.
        # The endpoint at the deadline is delivered because equality is inclusive.
        set_gaussian_time_noise(enabled=True, time_std=0.0, seed=101)
        generator = get_gaussian_time_noise().generator
        assert generator is not None
        state_before = generator.get_state().clone()
        try:
            neg_linear_transform(
                potential,
                domain,
                window_length=0.0,
                return_spike_sample=True,
                noise_site="verification.invalid_linear_window",
            )
        except ValueError as error:
            assert "finite and positive" in str(error)
        else:
            raise AssertionError("accepted zero negative-linear window")
        assert torch.equal(state_before, generator.get_state())

        parity_sample = neg_identity_transform(
            potential,
            domain,
            return_spike_sample=True,
            noise_site="verification.encoder_parity",
        )
        assert isinstance(parity_sample, SpikeSample)
        assert torch.equal(parity_sample.time, deterministic_time)
        assert bool(parity_sample.fired.all())

        # Encoder statistics must count the same elements and delivery mask returned
        # to the consumer, while output saturation counters remain untouched here.
        parity_counts = get_gaussian_noise_stats()["verification.encoder_parity"]
        assert parity_counts == {
            "events": 3,
            "misses": 0,
            "outputs": 0,
            "output_underflows": 0,
            "output_overflows": 0,
        }

        # Reconfiguration starts a fresh measurement interval. A positive mean
        # shifts the deadline codeword late while the opening codeword still fires.
        set_gaussian_time_noise(
            enabled=True,
            time_std=0.0,
            time_mean=0.5,
            seed=202,
        )
        miss_sample = neg_identity_transform(
            torch.tensor([0.0, 4.0], dtype=torch.float64),
            domain,
            return_spike_sample=True,
            noise_site="verification.encoder_miss",
        )
        assert isinstance(miss_sample, SpikeSample)
        assert torch.equal(miss_sample.time, torch.tensor([4.0, 0.5]))
        assert torch.equal(miss_sample.fired, torch.tensor([False, True]))
        miss_counts = get_gaussian_noise_stats()["verification.encoder_miss"]
        assert miss_counts["events"] == 2
        assert miss_counts["misses"] == 1

        # Finite Python values can still overflow or underflow float32. The encoder
        # must reject both cases instead of returning an invalid tensor beside a
        # superficially finite TimeBounds declaration.
        for invalid_window in (1.0e300, 1.0e-300):
            try:
                neg_linear_transform(
                    torch.tensor([0.0], dtype=torch.float32),
                    PotentialBounds(0.0, 1.0),
                    window_length=invalid_window,
                )
            except ValueError as error:
                assert "input tensor dtype" in str(error)
            else:
                raise AssertionError(
                    f"accepted dtype-unrepresentable window {invalid_window}"
                )

        # Equal rails pass the outer payload-membership check for an equal payload,
        # so the encoder itself must explicitly reject the zero-width mapping.
        try:
            neg_linear_transform(
                torch.tensor([1.0], dtype=torch.float64),
                PotentialBounds(1.0, 1.0),
            )
        except ValueError as error:
            assert "strictly ordered" in str(error)
        else:
            raise AssertionError("accepted zero-width negative-linear domain")
    finally:
        # Restore the process-wide singleton even if an assertion fails so importing
        # this verifier from a larger test process cannot leak an enabled replica.
        set_gaussian_time_noise(enabled=False)


def verify_gaussian_statistics_contract() -> None:
    """Verify output saturation counters, detached snapshots, and safe clearing.

    Rail statistics must observe the raw readout before clamping, use strict
    inequalities at representable endpoints, and accumulate independently from
    event counters. Public snapshots must not expose live mutable dictionaries,
    while clearing measurements must preserve the configured replica and RNG state.

    Raises:
        AssertionError: If clamping, counter accumulation, snapshot isolation, or
            clear-without-reconfiguration behavior violates the statistics contract.
    """
    domain = PotentialBounds(-1.0, 1.0)
    raw_output = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    site = "verification.output_clamp"

    # Rail enforcement is unconditional, but a disabled Gaussian replica must not
    # create instrumentation sites or alter the empty measurement interval.
    set_gaussian_time_noise(enabled=False)
    try:
        disabled_clamp = clamp_gaussian_output(
            raw_output,
            domain,
            site=site,
            name="disabled_output",
        )
        assert torch.equal(
            disabled_clamp,
            torch.tensor([-1.0, -1.0, 0.0, 1.0, 1.0]),
        )
        assert get_gaussian_noise_stats() == {}

        # Enable one replica and record raw outputs. Values exactly on either rail
        # remain representable; only strict excursions increment saturation counts.
        set_gaussian_time_noise(enabled=True, time_std=0.25, seed=303)
        enabled_clamp = clamp_gaussian_output(
            raw_output,
            domain,
            site=site,
            name="enabled_output",
        )
        assert torch.equal(enabled_clamp, disabled_clamp)
        counts = get_gaussian_noise_stats()[site]
        assert counts == {
            "events": 0,
            "misses": 0,
            "outputs": 5,
            "output_underflows": 1,
            "output_overflows": 1,
        }

        # A second call at the same site must accumulate into the fixed schema rather
        # than replace its mapping or mix the output denominator with event totals.
        clamp_gaussian_output(
            torch.tensor([-3.0, 0.0, 3.0]),
            domain,
            site=site,
            name="repeated_output",
        )
        accumulated = get_gaussian_noise_stats()[site]
        assert accumulated["outputs"] == 8
        assert accumulated["output_underflows"] == 2
        assert accumulated["output_overflows"] == 2
        assert accumulated["events"] == accumulated["misses"] == 0

        # Mutate both levels of a public snapshot. A fresh read must retain the live
        # counts and must not inherit sites inserted only into the detached copy.
        snapshot = get_gaussian_noise_stats()
        snapshot[site]["outputs"] = 999
        snapshot["snapshot_only"] = snapshot[site].copy()
        fresh_snapshot = get_gaussian_noise_stats()
        assert fresh_snapshot[site]["outputs"] == 8
        assert "snapshot_only" not in fresh_snapshot

        # Clearing statistics starts a new measurement interval without replacing
        # the active configuration, generator object, or its current random state.
        config_before = get_gaussian_time_noise()
        generator_before = config_before.generator
        assert generator_before is not None
        state_before = generator_before.get_state().clone()
        clear_gaussian_noise_stats()
        config_after = get_gaussian_time_noise()
        assert get_gaussian_noise_stats() == {}
        assert config_after is config_before
        assert config_after.generator is generator_before
        assert torch.equal(config_after.generator.get_state(), state_before)
    finally:
        # Leave no enabled global replica or counters for subsequent verifier calls.
        set_gaussian_time_noise(enabled=False)


def verify_gaussian_multiplication_operator() -> None:
    """Verify multiplication parity, factor-specific rails, misses, and saturation.

    Multiplication treats the encoded operand and one scalar zero codeword as two
    causal signed-PWM rails sharing one observation deadline. The checks force each
    event to miss in isolation, validate the resulting differential potential, and
    confirm that stochastic excursions retain ideal product rails with pre-clamp
    statistics.

    Raises:
        AssertionError: If deterministic parity, missing-event physics, output
            bounds, per-site counters, or rail saturation behavior regresses.
    """
    domain = PotentialBounds(-2.0, 2.0)
    theta = 2.0

    # Establish the public operator's deterministic value and ideal product rails.
    # This is the reference contract the event-aware zero-noise path must preserve.
    drive = torch.tensor([1.5, -1.0], dtype=torch.float64)
    operand = torch.tensor([1.0, -0.5], dtype=torch.float64)
    set_gaussian_time_noise(enabled=False)
    try:
        deterministic, deterministic_domain = multiplication_operator(
            drive,
            domain,
            operand,
            domain,
            theta,
        )
        assert torch.equal(deterministic, drive * operand)
        assert deterministic_domain == PotentialBounds(-4.0, 4.0)

        # A fixed coefficient still uses the full identity-encoder time window, but
        # its ideal product rail must retain the declared singleton factor instead of
        # acquiring a spurious full-theta multiplier.
        fixed_factor = torch.full_like(drive, 0.25)
        fixed_factor_domain = PotentialBounds(0.25, 0.25)
        fixed_deterministic, fixed_domain = multiplication_operator(
            drive,
            domain,
            fixed_factor,
            fixed_factor_domain,
            theta,
        )
        assert torch.equal(fixed_deterministic, drive * fixed_factor)
        assert fixed_domain == PotentialBounds(-0.5, 0.5)

        # Zero timing scale still enters both decorated event encoders. Its physical
        # readout and declared rails must exactly match the deterministic operator.
        set_gaussian_time_noise(enabled=True, time_std=0.0, seed=401)
        zero_noise, zero_noise_domain = multiplication_operator(
            drive,
            domain,
            operand,
            domain,
            theta,
        )
        assert torch.equal(zero_noise, deterministic)
        assert zero_noise_domain == deterministic_domain

        # Exercise the private Gaussian helper with the same narrow factor contract.
        # Zero timing noise preserves the fixed-coefficient value and rail exactly,
        # proving that deterministic and event-aware dispatch share one bound rule.
        fixed_zero_noise, fixed_zero_noise_domain = multiplication_operator(
            drive,
            domain,
            fixed_factor,
            fixed_factor_domain,
            theta,
        )
        assert torch.equal(fixed_zero_noise, fixed_deterministic)
        assert fixed_zero_noise_domain == fixed_domain

        # With B=-theta the data codeword is nominally at the deadline. A small
        # positive mean misses only that rail, while the delivered reference rail
        # contributes -V * (T_obs - t_reference) to the differential readout.
        set_gaussian_time_noise(
            enabled=True,
            time_std=0.0,
            time_mean=0.5,
            seed=402,
        )
        data_miss, data_miss_domain = multiplication_operator(
            torch.tensor([1.5], dtype=torch.float64),
            domain,
            torch.tensor([-theta], dtype=torch.float64),
            domain,
            theta,
        )
        assert torch.equal(
            data_miss,
            torch.tensor([-2.25], dtype=torch.float64),
        )
        assert data_miss_domain == deterministic_domain
        data_stats = get_gaussian_noise_stats()
        assert data_stats["multiplication.data"]["misses"] == 1
        assert data_stats["multiplication.reference"]["misses"] == 0

        # With B=+theta the data codeword starts at zero. A larger positive mean
        # keeps that rail on time but pushes the scalar reference beyond the deadline,
        # leaving +V * (T_obs - t_data) on the differential readout.
        set_gaussian_time_noise(
            enabled=True,
            time_std=0.0,
            time_mean=2.5,
            seed=403,
        )
        reference_miss, reference_miss_domain = multiplication_operator(
            torch.tensor([1.5], dtype=torch.float64),
            domain,
            torch.tensor([theta], dtype=torch.float64),
            domain,
            theta,
        )
        assert torch.equal(
            reference_miss,
            torch.tensor([2.25], dtype=torch.float64),
        )
        assert reference_miss_domain == deterministic_domain
        reference_stats = get_gaussian_noise_stats()
        assert reference_stats["multiplication.data"]["misses"] == 0
        assert reference_stats["multiplication.reference"]["misses"] == 1

        # This seeded high-variance case produces an early opening and a missing
        # reference. The raw duration exceeds the ideal factor rail, so the output
        # clamps at +4 while recording exactly one pre-clamp overflow.
        set_gaussian_time_noise(enabled=True, time_std=10.0, seed=4)
        saturated, saturated_domain = multiplication_operator(
            torch.tensor([2.0], dtype=torch.float64),
            domain,
            torch.tensor([0.0], dtype=torch.float64),
            domain,
            theta,
        )
        assert torch.equal(saturated, torch.tensor([4.0], dtype=torch.float64))
        assert saturated_domain == deterministic_domain
        saturation_stats = get_gaussian_noise_stats()
        assert saturation_stats["multiplication.output"] == {
            "events": 0,
            "misses": 0,
            "outputs": 1,
            "output_underflows": 0,
            "output_overflows": 1,
        }
    finally:
        # Restore process-wide state for the next independently reviewed operator.
        set_gaussian_time_noise(enabled=False)


def verify_gaussian_exponential_function() -> None:
    """Verify exponential parity, early delivery, reset misses, and rails.

    The exponential composition has one opening event and no closing reference.
    A delivered event decodes its finite stored time, while an input miss leaves
    the response at reset zero. Consequently, Gaussian output rails extend the
    deterministic lower bound to zero but retain the same maximum response.

    Raises:
        AssertionError: If zero-noise values, early-event clamping, reset behavior,
            event statistics, or the extended Gaussian output envelope regresses.
    """
    domain = PotentialBounds(-2.0, 2.0)
    input_value = torch.tensor([-2.0, 0.0, 2.0], dtype=torch.float64)

    # The symmetric identity code followed by normalized exponential decoding
    # evaluates exp(-x) at tau_m=1, including both temporal-window endpoints.
    set_gaussian_time_noise(enabled=False)
    try:
        deterministic, deterministic_domain = exponential_function(
            input_value,
            domain,
            tau_m=1.0,
            normalized=True,
        )
        expected = torch.exp(-input_value)
        assert torch.allclose(deterministic, expected)
        expected_min = torch.exp(torch.tensor(-2.0, dtype=torch.float64)).item()
        expected_max = torch.exp(torch.tensor(2.0, dtype=torch.float64)).item()
        assert abs(float(deterministic_domain.min) - expected_min) < 1e-12
        assert abs(float(deterministic_domain.max) - expected_max) < 1e-12

        # Zero timing scale enters the event-aware implementation without changing
        # values. Its lower rail becomes zero because later noisy calls can miss and
        # physically leave the exponential state at reset.
        set_gaussian_time_noise(enabled=True, time_std=0.0, seed=501)
        zero_noise, zero_noise_domain = exponential_function(
            input_value,
            domain,
            tau_m=1.0,
            normalized=True,
        )
        assert torch.allclose(zero_noise, deterministic)
        assert zero_noise_domain.min == 0.0
        assert abs(
            float(zero_noise_domain.max) - float(deterministic_domain.max)
        ) < 1e-12
        zero_stats = get_gaussian_noise_stats()
        assert zero_stats["exponential.input"]["events"] == input_value.numel()
        assert zero_stats["exponential.input"]["misses"] == 0

        # The x=+2 codeword is nominally at time zero. A negative mean moves it
        # before the interval, where storage clamps to the start and remains fired;
        # decoding therefore returns the Gaussian path's minimum positive response.
        set_gaussian_time_noise(
            enabled=True,
            time_std=0.0,
            time_mean=-0.5,
            seed=502,
        )
        early, early_domain = exponential_function(
            torch.tensor([2.0], dtype=torch.float64),
            domain,
            tau_m=1.0,
            normalized=True,
        )
        assert torch.allclose(
            early,
            torch.exp(torch.tensor([-2.0], dtype=torch.float64)),
        )
        assert early_domain == zero_noise_domain
        early_stats = get_gaussian_noise_stats()
        assert early_stats["exponential.input"]["misses"] == 0

        # The x=-2 codeword is nominally at the deadline. A positive mean makes the
        # sole opening event miss, so the physical exponential response stays zero.
        set_gaussian_time_noise(
            enabled=True,
            time_std=0.0,
            time_mean=0.5,
            seed=503,
        )
        missed, missed_domain = exponential_function(
            torch.tensor([-2.0], dtype=torch.float64),
            domain,
            tau_m=1.0,
            normalized=True,
        )
        assert torch.equal(missed, torch.zeros_like(missed))
        assert missed_domain == zero_noise_domain
        missed_stats = get_gaussian_noise_stats()
        assert missed_stats["exponential.input"]["events"] == 1
        assert missed_stats["exponential.input"]["misses"] == 1

        # Finite delivered times are clamped before exponentiation and misses reset
        # to zero, so this operator cannot exceed its constructed rails. The output
        # counter must still record its denominator with no saturation events.
        assert missed_stats["exponential.output"] == {
            "events": 0,
            "misses": 0,
            "outputs": 1,
            "output_underflows": 0,
            "output_overflows": 0,
        }
    finally:
        # Restore process-wide state before the next operator verification runs.
        set_gaussian_time_noise(enabled=False)


def verify_gaussian_exponential_difference_operator() -> None:
    """Verify exponential-difference parity and its three missing-event stages.

    The operator first reads two causal signed-PWM rails into an intermediate
    potential, then re-encodes that value through an internal exponential event.
    Either external event can miss independently while the other rail remains at
    observation time, and an internal miss resets the final response.

    Raises:
        AssertionError: If zero-noise parity, observation-time interval physics,
            internal reset behavior, output rails, or site counters regress.
    """
    time_domain = TimeBounds(0.0, 4.0)
    opening_time = torch.tensor([1.0, 3.0], dtype=torch.float64)
    closing_time = torch.tensor([2.0, 1.0], dtype=torch.float64)

    # Plain tensors represent delivered events. The deterministic reference is the
    # direct temporal ratio exp(t_B - t_A) with ideal exponential rails.
    set_gaussian_time_noise(enabled=False)
    try:
        deterministic, deterministic_domain = exponential_difference_operator(
            opening_time,
            time_domain,
            closing_time,
            time_domain,
            tau_s=1.0,
        )
        expected = torch.exp(closing_time - opening_time)
        assert torch.allclose(deterministic, expected)

        # With zero scale, tensor inputs are wrapped as delivered events and only
        # the internal decorated encoder is sampled. Values remain identical while
        # the physical lower rail expands to include internal-event reset zero.
        set_gaussian_time_noise(enabled=True, time_std=0.0, seed=601)
        zero_noise, zero_noise_domain = exponential_difference_operator(
            opening_time,
            time_domain,
            closing_time,
            time_domain,
            tau_s=1.0,
        )
        assert torch.allclose(zero_noise, deterministic)
        assert zero_noise_domain.min == 0.0
        assert abs(
            float(zero_noise_domain.max) - float(deterministic_domain.max)
        ) < 1e-12
        zero_stats = get_gaussian_noise_stats()
        assert zero_stats["exponential_difference.internal"]["events"] == 2
        assert zero_stats["exponential_difference.internal"]["misses"] == 0

        # A missing A event leaves only the delivered B rail. Under the fixed -1
        # drive this produces intermediate +(T_obs-t_B)=+2 and response exp(-2).
        set_gaussian_time_noise(enabled=True, time_std=0.0, seed=602)
        opening_miss = SpikeSample(
            time=torch.tensor([4.0], dtype=torch.float64),
            domain=time_domain,
            fired=torch.tensor([False]),
        )
        delivered_close = SpikeSample(
            time=torch.tensor([2.0], dtype=torch.float64),
            domain=time_domain,
            fired=torch.tensor([True]),
        )
        opening_reset, opening_reset_domain = exponential_difference_operator(
            opening_miss,
            time_domain,
            delivered_close,
            time_domain,
            tau_s=1.0,
        )
        assert torch.allclose(
            opening_reset,
            torch.exp(torch.tensor([-2.0], dtype=torch.float64)),
        )
        assert opening_reset_domain == zero_noise_domain

        # A missing B event leaves only the delivered A rail. Under the fixed -1
        # drive this yields intermediate -(T_obs-t_A)=-3 and response exp(3).
        set_gaussian_time_noise(enabled=True, time_std=0.0, seed=603)
        delivered_open = SpikeSample(
            time=torch.tensor([1.0], dtype=torch.float64),
            domain=time_domain,
            fired=torch.tensor([True]),
        )
        closing_miss = SpikeSample(
            time=torch.tensor([4.0], dtype=torch.float64),
            domain=time_domain,
            fired=torch.tensor([False]),
        )
        deadline_readout, deadline_readout_domain = exponential_difference_operator(
            delivered_open,
            time_domain,
            closing_miss,
            time_domain,
            tau_s=1.0,
        )
        assert torch.allclose(
            deadline_readout,
            torch.exp(torch.tensor([3.0], dtype=torch.float64)),
        )
        assert deadline_readout_domain == zero_noise_domain

        # The tensor pair t_A=0, t_B=4 produces the lower intermediate rail -4,
        # whose internal codeword is nominally at deadline 8. A positive mean makes
        # only that internal event miss, forcing the final exponential response to zero.
        set_gaussian_time_noise(
            enabled=True,
            time_std=0.0,
            time_mean=0.5,
            seed=604,
        )
        internal_miss, internal_miss_domain = exponential_difference_operator(
            torch.tensor([0.0], dtype=torch.float64),
            time_domain,
            torch.tensor([4.0], dtype=torch.float64),
            time_domain,
            tau_s=1.0,
        )
        assert torch.equal(internal_miss, torch.zeros_like(internal_miss))
        assert internal_miss_domain == zero_noise_domain
        internal_stats = get_gaussian_noise_stats()
        assert internal_stats["exponential_difference.internal"]["events"] == 1
        assert internal_stats["exponential_difference.internal"]["misses"] == 1

        # Delivered internal times are clamped to their finite interval and misses
        # reset to zero, so the constructed output envelope contains every response.
        assert internal_stats["exponential_difference.output"] == {
            "events": 0,
            "misses": 0,
            "outputs": 1,
            "output_underflows": 0,
            "output_overflows": 0,
        }
    finally:
        # Restore global state before the next composed operator verification.
        set_gaussian_time_noise(enabled=False)


def verify_gaussian_division_function() -> None:
    """Verify constrained division without restricting generic exponential difference.

    Division independently log-encodes numerator and denominator, then delegates
    their complete event records to the unrestricted exponential-difference primitive.
    The public wrapper must return one noise-independent ``[0, 1]`` domain, preserve
    exact deterministic and zero-noise ratios, count raw Gaussian overflow before
    clamping, and retain reset zero when the internal exponential event misses.

    Raises:
        AssertionError: If ratio values, public bounds, reset/deadline propagation,
            saturation accounting, site attribution, or unrestricted exponential-
            difference behavior regresses.
    """
    domain = PotentialBounds(0.1, 10.0)
    expected_domain = PotentialBounds(0.0, 1.0)
    numerator = torch.tensor([0.2, 1.0, 5.0], dtype=torch.float64)
    denominator = torch.tensor([1.0, 2.0, 10.0], dtype=torch.float64)

    # The deterministic shared-log-domain construction must cancel its fixed offset,
    # reproduce X/Y, and expose the constrained public rail rather than the generic
    # exponential ratio [domain.min/domain.max, domain.max/domain.min].
    set_gaussian_time_noise(enabled=False)
    try:
        deterministic, deterministic_domain = division_function(
            numerator,
            denominator,
            domain,
            tau_s=1.0,
        )
        assert torch.allclose(deterministic, numerator / denominator)
        assert deterministic_domain == expected_domain

        # Zero scale samples both external log events and the internal exponential
        # event without changing values. Public metadata remains identical across
        # noise modes, and representable ratios create no saturation.
        set_gaussian_time_noise(enabled=True, time_std=0.0, seed=701)
        zero_noise, zero_noise_domain = division_function(
            numerator,
            denominator,
            domain,
            tau_s=1.0,
        )
        assert torch.allclose(zero_noise, deterministic)
        assert zero_noise_domain == expected_domain
        zero_stats = get_gaussian_noise_stats()
        assert zero_stats["division.numerator"]["misses"] == 0
        assert zero_stats["division.denominator"]["misses"] == 0
        assert zero_stats["exponential_difference.internal"]["misses"] == 0
        assert zero_stats["division.output"] == {
            "events": 0,
            "misses": 0,
            "outputs": numerator.numel(),
            "output_underflows": 0,
            "output_overflows": 0,
        }

        # X at the positive-domain floor encodes exactly at the log deadline while
        # Y at the ceiling encodes at zero. Mean 0.5 misses only the numerator rail;
        # the remaining physical trajectory stays below one and therefore must not
        # be confused with the opposite one-sided miss that overflows the public rail.
        set_gaussian_time_noise(
            enabled=True,
            time_std=0.0,
            time_mean=0.5,
            seed=702,
        )
        numerator_miss, numerator_miss_domain = division_function(
            torch.tensor([0.1], dtype=torch.float64),
            torch.tensor([10.0], dtype=torch.float64),
            domain,
            tau_s=1.0,
        )
        assert torch.allclose(
            numerator_miss,
            torch.exp(
                torch.tensor(
                    [1.0 - math.log(100.0)],
                    dtype=torch.float64,
                )
            ),
        )
        assert numerator_miss_domain == expected_domain
        numerator_stats = get_gaussian_noise_stats()
        assert numerator_stats["division.numerator"]["misses"] == 1
        assert numerator_stats["division.denominator"]["misses"] == 0
        assert numerator_stats["exponential_difference.internal"]["misses"] == 0
        assert numerator_stats["division.output"]["outputs"] == 1
        assert numerator_stats["division.output"]["output_underflows"] == 0
        assert numerator_stats["division.output"]["output_overflows"] == 0

        # Equal operands have equal nominal log times, so independent draws can
        # isolate the denominator. Seed 9 yields an on-time numerator, late closing
        # event, and on-time internal event. The raw result exceeds one, must count
        # exactly one overflow, and must be delivered on the public upper rail.
        equal_value = torch.tensor([1.0], dtype=torch.float64)
        set_gaussian_time_noise(enabled=True, time_std=5.0, seed=9)
        denominator_miss, denominator_miss_domain = division_function(
            equal_value,
            equal_value,
            domain,
            tau_s=1.0,
        )
        denominator_stats = get_gaussian_noise_stats()
        assert denominator_stats["division.numerator"]["misses"] == 0
        assert denominator_stats["division.denominator"]["misses"] == 1
        assert denominator_stats["exponential_difference.internal"]["misses"] == 0
        assert denominator_miss_domain == expected_domain
        assert torch.equal(denominator_miss, torch.ones_like(denominator_miss))
        assert denominator_stats["division.output"] == {
            "events": 0,
            "misses": 0,
            "outputs": 1,
            "output_underflows": 0,
            "output_overflows": 1,
        }

        # Seed 4 keeps both log events on time but misses the internal re-encoding.
        # That stage owns the final exponential response, so its miss must remain
        # reset zero inside the same public rail and must not count as saturation.
        set_gaussian_time_noise(enabled=True, time_std=5.0, seed=4)
        internal_miss, internal_miss_domain = division_function(
            equal_value,
            equal_value,
            domain,
            tau_s=1.0,
        )
        internal_stats = get_gaussian_noise_stats()
        assert internal_stats["division.numerator"]["misses"] == 0
        assert internal_stats["division.denominator"]["misses"] == 0
        assert internal_stats["exponential_difference.internal"]["misses"] == 1
        assert torch.equal(internal_miss, torch.zeros_like(internal_miss))
        assert internal_miss_domain == expected_domain
        assert internal_stats["exponential_difference.output"]["output_underflows"] == 0
        assert internal_stats["exponential_difference.output"]["output_overflows"] == 0
        assert internal_stats["division.output"] == {
            "events": 0,
            "misses": 0,
            "outputs": 1,
            "output_underflows": 0,
            "output_overflows": 0,
        }

        # The [0, 1] restriction belongs only to division_function. A direct
        # exponential-difference call retains the reverse event ordering and a value
        # above one, which the signed dual-rail LayerNorm path requires.
        set_gaussian_time_noise(enabled=False)
        unrestricted, unrestricted_domain = exponential_difference_operator(
            torch.tensor([0.0], dtype=torch.float64),
            TimeBounds(0.0, 2.0),
            torch.tensor([2.0], dtype=torch.float64),
            TimeBounds(0.0, 2.0),
            tau_s=1.0,
        )
        assert unrestricted.item() > 1.0
        assert unrestricted_domain.max > 1.0
    finally:
        # Restore global state before the normalization-operator regression.
        set_gaussian_time_noise(enabled=False)


def verify_gaussian_tanh_function() -> None:
    """Verify tanh parity, structural rails, and excursion clamping.

    The maintained tanh composition scales its input, evaluates a negative
    exponential, divides one by the one-plus-exponential response, and maps that
    gate onto ``[-1, 1]``. This regression verifies exact deterministic and
    zero-noise values, noise-mode-independent structural metadata, and pre-clamp
    saturation accounting under a deterministic positive timing shift.

    Raises:
        AssertionError: If analytic parity, event topology, structural bounds,
            saturation accounting, or final clamping regresses.
    """
    value = torch.tensor(
        [-2.0, -1.0, 0.0, 1.0, 2.0],
        dtype=torch.float64,
    )
    domain = PotentialBounds(-2.0, 2.0)
    expected = torch.tanh(value)
    expected_domain = PotentialBounds(-1.0, 1.0)

    # Noise-disabled evaluation fixes the mathematical reference. The returned
    # metadata must use tanh's structural range rather than a linearly transformed
    # version of the generic division interval.
    set_gaussian_time_noise(enabled=False)
    try:
        deterministic, deterministic_domain = tanh(
            value,
            domain,
            tau_s=1.0,
            theta=4.0,
        )
        assert torch.allclose(
            deterministic,
            expected,
            atol=1.0e-12,
            rtol=1.0e-12,
        )
        assert deterministic_domain == expected_domain

        # Zero standard deviation traverses multiplication, exponential, division,
        # and the public tanh clamp without perturbing any carrier. It must preserve
        # both values and rails while counting every final activation exactly once.
        set_gaussian_time_noise(enabled=True, time_std=0.0, seed=851)
        zero_noise, zero_noise_domain = tanh(
            value,
            domain,
            tau_s=1.0,
            theta=4.0,
        )
        assert torch.allclose(
            zero_noise,
            deterministic,
            atol=1.0e-12,
            rtol=1.0e-12,
        )
        assert zero_noise_domain == expected_domain
        zero_stats = get_gaussian_noise_stats()
        assert zero_stats["multiplication.data"]["events"] == value.numel()
        assert zero_stats["multiplication.reference"]["events"] == 1
        assert zero_stats["exponential.input"]["events"] == value.numel()
        assert zero_stats["division.numerator"]["events"] == value.numel()
        assert zero_stats["division.denominator"]["events"] == value.numel()
        assert zero_stats["tanh.output"]["outputs"] == value.numel()
        assert zero_stats["tanh.output"]["output_underflows"] == 0
        assert zero_stats["tanh.output"]["output_overflows"] == 0

        # A positive shift creates one-sided observation-time trajectories whose raw
        # mapped values exceed tanh's upper rail. Saturation may be recorded by a
        # tightened division first or by tanh itself, but it must remain observable.
        set_gaussian_time_noise(
            enabled=True,
            time_std=0.0,
            time_mean=0.5,
            seed=852,
        )
        shifted, shifted_domain = tanh(
            value,
            domain,
            tau_s=1.0,
            theta=4.0,
        )
        shifted_stats = get_gaussian_noise_stats()
        assert shifted_domain == expected_domain
        assert shifted_stats["tanh.output"]["outputs"] == value.numel()
        division_output_stats = shifted_stats.get(
            "division.output",
            {
                "output_overflows": 0,
                "output_underflows": 0,
            },
        )
        structural_saturations = (
            shifted_stats["tanh.output"]["output_overflows"]
            + shifted_stats["tanh.output"]["output_underflows"]
            + division_output_stats["output_overflows"]
            + division_output_stats["output_underflows"]
        )
        assert structural_saturations > 0

        # The returned activation must remain a finite carrier in tanh's fixed rail
        # regardless of which nested event produced the pre-clamp excursion.
        assert torch.isfinite(shifted).all()
        assert bool(
            (
                (shifted >= expected_domain.min)
                & (shifted <= expected_domain.max)
            ).all()
        )
    finally:
        # Restore process-wide state before the sigmoid-GELU regression.
        set_gaussian_time_noise(enabled=False)


def verify_gaussian_sigmoid_gelu_function() -> None:
    """Verify sigmoid-GELU parity and its fixed gate contract.

    The sigmoid approximation composes ``x * sigmoid(1.702*x)``. The internal
    normalized ratio must use ``[0, 1]`` before the final multiplication so the
    GELU output interval depends only on the declared input range, not on a generic
    exponential division window. The regression covers exact values, event counts,
    gate saturation, and finite rail-clamped output.

    Raises:
        AssertionError: If analytic parity, fixed gate-derived output bounds,
            nested event topology, saturation accounting, or clamping regresses.
    """
    value = torch.tensor(
        [-2.0, -1.0, 0.0, 1.0, 2.0],
        dtype=torch.float64,
    )
    domain = PotentialBounds(-2.0, 2.0)
    expected = value * torch.sigmoid(1.702 * value)

    # Multiplying the declared signed input by a gate in [0,1] yields [-2,2].
    # This expected domain is reconstructed independently of the production
    # division metadata and therefore detects any reintroduced generic gate rail.
    gate_domain = PotentialBounds(0.0, 1.0)
    product_candidates = (
        domain.min * gate_domain.min,
        domain.min * gate_domain.max,
        domain.max * gate_domain.min,
        domain.max * gate_domain.max,
    )
    expected_domain = PotentialBounds(
        min(product_candidates),
        max(product_candidates),
    )

    # Establish the deterministic composed reference and its structural gate-derived
    # output interval before any process-wide Gaussian state is enabled.
    set_gaussian_time_noise(enabled=False)
    try:
        deterministic, deterministic_domain = gelu_approximation_sigmoid(
            value,
            domain,
            tau_s=1.0,
            theta=8.0,
        )
        assert torch.allclose(
            deterministic,
            expected,
            atol=1.0e-12,
            rtol=1.0e-12,
        )
        assert deterministic_domain == expected_domain

        # Zero-noise event-aware execution must preserve the complete composition.
        # Two multiplication calls scale and gate the input; one gate output counter
        # is recorded per activation without underflow or overflow.
        set_gaussian_time_noise(enabled=True, time_std=0.0, seed=861)
        zero_noise, zero_noise_domain = gelu_approximation_sigmoid(
            value,
            domain,
            tau_s=1.0,
            theta=8.0,
        )
        assert torch.allclose(
            zero_noise,
            deterministic,
            atol=1.0e-12,
            rtol=1.0e-12,
        )
        assert zero_noise_domain == expected_domain
        zero_stats = get_gaussian_noise_stats()
        assert zero_stats["multiplication.data"]["events"] == 2 * value.numel()
        assert zero_stats["multiplication.reference"]["events"] == 2
        assert zero_stats["exponential.input"]["events"] == value.numel()
        assert zero_stats["division.numerator"]["events"] == value.numel()
        assert zero_stats["division.denominator"]["events"] == value.numel()
        assert zero_stats["gelu_sigmoid.gate"]["outputs"] == value.numel()
        assert zero_stats["gelu_sigmoid.gate"]["output_underflows"] == 0
        assert zero_stats["gelu_sigmoid.gate"]["output_overflows"] == 0

        # The selected deterministic timing shift produces gate values above one in
        # the current physical composition. Accept saturation at division or gate so
        # a future tighter division contract does not invalidate this public check.
        set_gaussian_time_noise(
            enabled=True,
            time_std=0.0,
            time_mean=0.5,
            seed=862,
        )
        shifted, shifted_domain = gelu_approximation_sigmoid(
            value,
            domain,
            tau_s=1.0,
            theta=8.0,
        )
        shifted_stats = get_gaussian_noise_stats()
        assert shifted_domain == expected_domain
        assert shifted_stats["gelu_sigmoid.gate"]["outputs"] == value.numel()
        division_output_stats = shifted_stats.get(
            "division.output",
            {
                "output_overflows": 0,
                "output_underflows": 0,
            },
        )
        structural_saturations = (
            shifted_stats["gelu_sigmoid.gate"]["output_overflows"]
            + shifted_stats["gelu_sigmoid.gate"]["output_underflows"]
            + division_output_stats["output_overflows"]
            + division_output_stats["output_underflows"]
        )
        assert structural_saturations > 0

        # The final multiplication consumes the fixed gate and returns a finite value
        # inside the independently reconstructed product interval.
        assert torch.isfinite(shifted).all()
        assert bool(
            (
                (shifted >= expected_domain.min)
                & (shifted <= expected_domain.max)
            ).all()
        )
    finally:
        # Restore process-wide state before softmin verification.
        set_gaussian_time_noise(enabled=False)


def verify_gaussian_softmin_function() -> None:
    """Verify softmin normalization, structural rails, and nested accounting.

    Softmin passes individual exponentials and their reduction through one shared
    logarithmic division domain. The regression includes endpoint-heavy rows that
    expose an invalid denominator-only lower bound, then checks that zero Gaussian
    scale preserves the deterministic composition and its probability mass. Both
    noise modes must return the structural normalized-weight interval ``[0, 1]``.
    A forced-late case verifies exact nested event counts, pre-clamp saturation
    accounting, and finite rail-bounded observation-time outputs.

    Raises:
        AssertionError: If dense parity, zero-noise parity, normalization, shared
            rail propagation, structural bounds, saturation accounting,
            forced-miss accounting, or finite clamping regresses.
    """
    domain = PotentialBounds(-2.0, 2.0)
    scores = torch.tensor(
        [
            [-2.0, -1.0, 0.0, 1.0, 2.0],
            [1.5, -0.5, 0.25, -1.25, 2.0],
        ],
        dtype=torch.float64,
    )
    expected = torch.softmax(-scores, dim=-1)
    expected_domain = PotentialBounds(0.0, 1.0)

    # Endpoint-heavy scores force the smallest individual exponential below the
    # denominator's N-scaled minimum. Exact dense parity therefore proves that the
    # shared log domain contains both numerator and reduced-denominator operands.
    set_gaussian_time_noise(enabled=False)
    try:
        deterministic, deterministic_domain = softmin_function(scores, domain)
        assert torch.allclose(deterministic, expected, atol=1e-12, rtol=1e-12)
        assert deterministic_domain == expected_domain
        assert torch.allclose(
            deterministic.sum(dim=-1),
            torch.ones(scores.size(0), dtype=scores.dtype),
            atol=1e-12,
            rtol=1e-12,
        )

        # Zero standard deviation still exercises every event-aware encoder and
        # decoder. It must preserve values while returning exactly the same public
        # structural rails as deterministic execution, independent of the wider
        # ratio metadata used inside the composed division operator.
        set_gaussian_time_noise(enabled=True, time_std=0.0, seed=801)
        zero_noise, zero_noise_domain = softmin_function(scores, domain)
        assert torch.allclose(zero_noise, deterministic, atol=1e-12, rtol=1e-12)
        assert zero_noise_domain == expected_domain
        assert deterministic_domain == zero_noise_domain
        zero_stats = get_gaussian_noise_stats()
        assert zero_stats["exponential.input"]["events"] == scores.numel()
        assert zero_stats["division.numerator"]["events"] == scores.numel()
        assert zero_stats["division.denominator"]["events"] == scores.size(0)
        assert zero_stats["exponential_difference.internal"]["events"] == scores.numel()
        assert zero_stats["softmin.output"]["outputs"] == scores.numel()
        assert zero_stats["softmin.output"]["output_underflows"] == 0
        assert zero_stats["softmin.output"]["output_overflows"] == 0
        assert all(site_stats["misses"] == 0 for site_stats in zero_stats.values())

        # A deterministic positive timing shift larger than every involved window
        # forces the external exponential and division events past their deadlines.
        # Missed-event physics need not preserve a probability sum, so verify only
        # finite rail-bounded readout and the precise sampling topology here.
        set_gaussian_time_noise(
            enabled=True,
            time_std=0.0,
            time_mean=5.0,
            seed=802,
        )
        forced_late, forced_late_domain = softmin_function(scores, domain)
        forced_stats = get_gaussian_noise_stats()
        assert forced_stats["exponential.input"]["misses"] == scores.numel()
        assert forced_stats["division.numerator"]["misses"] == scores.numel()
        assert forced_stats["division.denominator"]["misses"] == scores.size(0)
        assert forced_late_domain == expected_domain

        # Every final weight participates in the softmin saturation denominator.
        # The current composition records forced-late overflows at this final site;
        # a tightened division may clamp and count the same physical excursion one
        # level earlier, so accept either site while requiring observable saturation.
        assert forced_stats["softmin.output"]["outputs"] == scores.numel()
        division_output_stats = forced_stats.get(
            "division.output",
            {
                "output_overflows": 0,
                "output_underflows": 0,
            },
        )
        structural_saturations = (
            forced_stats["softmin.output"]["output_overflows"]
            + forced_stats["softmin.output"]["output_underflows"]
            + division_output_stats["output_overflows"]
            + division_output_stats["output_underflows"]
        )
        assert structural_saturations > 0

        # Observation-time trajectories and the final rail clamp must always return
        # finite weights in [0,1], even when upstream events never arrive and the
        # unnormalized observation-time trajectory no longer sums to probability one.
        assert torch.isfinite(forced_late).all()
        assert bool(
            (
                (forced_late >= forced_late_domain.min)
                & (forced_late <= forced_late_domain.max)
            ).all()
        )
    finally:
        # Restore process-wide state before the next composed-operator regression.
        set_gaussian_time_noise(enabled=False)


def verify_gaussian_swiglu_function() -> None:
    """Verify current-bias cancellation and the fixed SwiGLU gate contract.

    The activation must reproduce ``v*u*sigmoid(beta*u)`` at the default temporal
    scale even when the declared ``u`` domain is asymmetric. That case exposes any
    uncancelled identity-encoder offset directly. The regression also fixes the
    number of sampled events across the exponential, division, and two multiplication
    stages. It also verifies the structural ``[0,1]`` sigmoid gate, forced gate
    saturation, and all-late reset propagation on finite final rails.

    Raises:
        AssertionError: If analytic parity, zero-noise parity, gate bounds,
            saturation accounting, event topology, forced-miss reset behavior,
            or finite output bounds regress.
    """
    u = torch.tensor([-0.75, 0.0, 1.0, 2.0], dtype=torch.float64)
    v = torch.tensor([1.0, -0.5, 2.0, 0.25], dtype=torch.float64)
    domain_u = PotentialBounds(-1.0, 3.0)
    domain_v = PotentialBounds(-2.0, 2.0)
    beta = 0.7
    expected = v * u * torch.sigmoid(beta * u)

    # Reconstruct both product domains from the fixed gate interval. The first
    # multiplication maps u*gate into [-1,3]; multiplying that by v in [-2,2]
    # produces the final module-wide interval [-6,6].
    gate_domain = PotentialBounds(0.0, 1.0)
    swish_candidates = (
        domain_u.min * gate_domain.min,
        domain_u.min * gate_domain.max,
        domain_u.max * gate_domain.min,
        domain_u.max * gate_domain.max,
    )
    expected_swish_domain = PotentialBounds(
        min(swish_candidates),
        max(swish_candidates),
    )
    output_candidates = (
        domain_v.min * expected_swish_domain.min,
        domain_v.min * expected_swish_domain.max,
        domain_v.max * expected_swish_domain.min,
        domain_v.max * expected_swish_domain.max,
    )
    expected_domain = PotentialBounds(
        min(output_candidates),
        max(output_candidates),
    )

    # An asymmetric input domain makes the encoded temporal offset nonzero and
    # distinct from half the code-window width. Dense parity therefore verifies the
    # fixed current gain, rather than accidentally passing through domain symmetry.
    set_gaussian_time_noise(enabled=False)
    try:
        deterministic, deterministic_domain = swiglu_function(
            u,
            domain_u,
            v,
            domain_v,
            beta=beta,
            tau_s=1.0,
            theta=8.0,
        )
        assert torch.allclose(deterministic, expected, atol=1e-12, rtol=1e-12)
        assert deterministic_domain == expected_domain

        # Zero Gaussian scale traverses the event-aware implementation without
        # perturbing any carrier. It must match both the corrected deterministic
        # values and output rails while sampling every nested encoder exactly once.
        set_gaussian_time_noise(enabled=True, time_std=0.0, seed=901)
        zero_noise, zero_noise_domain = swiglu_function(
            u,
            domain_u,
            v,
            domain_v,
            beta=beta,
            tau_s=1.0,
            theta=8.0,
        )
        assert torch.allclose(zero_noise, deterministic, atol=1e-12, rtol=1e-12)
        assert zero_noise_domain == expected_domain
        zero_stats = get_gaussian_noise_stats()
        assert zero_stats["swiglu.exponential_input"]["events"] == u.numel()
        assert zero_stats["division.numerator"]["events"] == u.numel()
        assert zero_stats["division.denominator"]["events"] == u.numel()
        assert zero_stats["exponential_difference.internal"]["events"] == u.numel()
        assert zero_stats["swiglu.gate"]["outputs"] == u.numel()
        assert zero_stats["swiglu.gate"]["output_underflows"] == 0
        assert zero_stats["swiglu.gate"]["output_overflows"] == 0
        assert zero_stats["multiplication.data"]["events"] == 2 * u.numel()
        assert zero_stats["multiplication.reference"]["events"] == 2
        assert all(site_stats["misses"] == 0 for site_stats in zero_stats.values())

        # A moderate positive shift creates an observation-time sigmoid excursion
        # while retaining finite nested carriers. Saturation may occur in a tightened
        # division or at the explicit SwiGLU gate, but it cannot widen final metadata.
        set_gaussian_time_noise(
            enabled=True,
            time_std=0.0,
            time_mean=0.5,
            seed=902,
        )
        gate_shifted, gate_shifted_domain = swiglu_function(
            u,
            domain_u,
            v,
            domain_v,
            beta=beta,
            tau_s=1.0,
            theta=8.0,
        )
        gate_stats = get_gaussian_noise_stats()
        assert gate_shifted_domain == expected_domain
        assert gate_stats["swiglu.gate"]["outputs"] == u.numel()
        division_output_stats = gate_stats.get(
            "division.output",
            {
                "output_overflows": 0,
                "output_underflows": 0,
            },
        )
        gate_saturations = (
            gate_stats["swiglu.gate"]["output_overflows"]
            + gate_stats["swiglu.gate"]["output_underflows"]
            + division_output_stats["output_overflows"]
            + division_output_stats["output_underflows"]
        )
        assert gate_saturations > 0
        assert torch.isfinite(gate_shifted).all()
        assert bool(
            (
                (gate_shifted >= expected_domain.min)
                & (gate_shifted <= expected_domain.max)
            ).all()
        )

        # A shift beyond every code window forces the direct exponential event,
        # division events, internal re-encoding, and both multiplication stages to
        # miss. Bias cancellation must not turn the missed exponential reset nonzero.
        set_gaussian_time_noise(
            enabled=True,
            time_std=0.0,
            time_mean=20.0,
            seed=903,
        )
        forced_late, forced_late_domain = swiglu_function(
            u,
            domain_u,
            v,
            domain_v,
            beta=beta,
            tau_s=1.0,
            theta=8.0,
        )
        forced_stats = get_gaussian_noise_stats()
        assert forced_stats["swiglu.exponential_input"]["misses"] == u.numel()
        assert forced_stats["division.numerator"]["misses"] == u.numel()
        assert forced_stats["division.denominator"]["misses"] == u.numel()
        assert forced_stats["exponential_difference.internal"]["misses"] == u.numel()
        assert forced_stats["multiplication.data"]["misses"] == 2 * u.numel()
        assert forced_stats["multiplication.reference"]["misses"] == 2

        # With every opening and reference event absent, both multiplication stages
        # remain at reset. The returned carrier must still use the unchanged finite
        # deterministic rails so later operators can consume it normally.
        assert torch.equal(forced_late, torch.zeros_like(forced_late))
        assert forced_late_domain == expected_domain
        assert torch.isfinite(forced_late).all()
        assert bool(
            (
                (forced_late >= forced_late_domain.min)
                & (forced_late <= forced_late_domain.max)
            ).all()
        )
    finally:
        # Restore process-wide state before model-layer regression checks begin.
        set_gaussian_time_noise(enabled=False)


def verify_gaussian_spiking_linear() -> None:
    """Verify frozen affine bounds plus isolated signed-PWM event misses.

    A spiking linear layer samples one data event per input element and one scalar
    zero-reference event for the complete invocation. These events supply two causal
    signed-PWM rails. Endpoint-valued inputs and deterministic mean shifts isolate
    each miss class so the regression checks the exact differential duration passed
    to the learned affine map rather than merely asserting finite noisy outputs.

    Raises:
        AssertionError: If dense parity, output-specific frozen bounds, mutation
            rejection, event counts, miss readout, or output finiteness regresses.
    """
    layer = SpikingLinear(3, 2, bias=True, theta=2.0, dtype=torch.float64)
    with torch.no_grad():
        layer.weight.copy_(
            torch.tensor(
                [[0.5, -0.25, 0.75], [-1.0, 0.4, 0.2]],
                dtype=torch.float64,
            )
        )
        layer.bias.copy_(torch.tensor([0.1, -0.2], dtype=torch.float64))
    value = torch.tensor(
        [[-1.5, 0.25, 1.75], [2.0, -2.0, 0.5]],
        dtype=torch.float64,
    )
    potential = Potential(value, PotentialBounds(-2.0, 2.0))

    # Noise-free PWM durations equal the clamped activation, so the converted layer
    # must match the dense affine reference and establish the ideal output rails used
    # by every later noisy case.
    set_gaussian_time_noise(enabled=False)
    try:
        deterministic = layer(potential)
        expected = torch.nn.functional.linear(value, layer.weight, layer.bias)
        assert torch.allclose(deterministic.value, expected, atol=1e-12, rtol=1e-12)

        # Reconstruct the output-specific safety rail independently. Each row has
        # radius theta*sum(abs(weight)), then its own bias translates both endpoints.
        linear_radius = layer.theta * layer.weight.detach().abs().sum(dim=1)
        expected_domain = PotentialBounds(
            (layer.bias.detach() - linear_radius).min().item(),
            (layer.bias.detach() + linear_radius).max().item(),
        )
        assert deterministic.domain == expected_domain

        # A second fixed domain proves the affine adapter consumes upstream metadata
        # instead of silently replacing it with [-theta, theta]. Exact interval
        # arithmetic must retain the asymmetric endpoint selected by each weight sign.
        asymmetric_domain = PotentialBounds(-1.0, 2.0)
        asymmetric_value = value.clamp(
            asymmetric_domain.min,
            asymmetric_domain.max,
        )
        asymmetric = layer(Potential(asymmetric_value, asymmetric_domain))
        assert torch.allclose(
            asymmetric.value,
            torch.nn.functional.linear(asymmetric_value, layer.weight, layer.bias),
            atol=1e-12,
            rtol=1e-12,
        )
        lower_terms = torch.minimum(
            layer.weight.detach() * asymmetric_domain.min,
            layer.weight.detach() * asymmetric_domain.max,
        )
        upper_terms = torch.maximum(
            layer.weight.detach() * asymmetric_domain.min,
            layer.weight.detach() * asymmetric_domain.max,
        )
        expected_asymmetric_domain = PotentialBounds(
            (lower_terms.sum(dim=1) + layer.bias.detach()).min().item(),
            (upper_terms.sum(dim=1) + layer.bias.detach()).max().item(),
        )
        assert asymmetric.domain == expected_asymmetric_domain
        assert layer.freeze_parameter_bounds(asymmetric_domain) is asymmetric.domain

        # Zero scale enters the event-aware implementation without changing either
        # data or scalar reference times. Verify exact value/domain parity and that
        # the reference is sampled once rather than once per batch or feature.
        set_gaussian_time_noise(enabled=True, time_std=0.0, seed=1001)
        zero_noise = layer(potential)
        zero_stats = get_gaussian_noise_stats()
        assert torch.allclose(zero_noise.value, deterministic.value)
        assert zero_noise.domain == deterministic.domain
        assert zero_stats["linear.data"]["events"] == value.numel()
        assert zero_stats["linear.reference"]["events"] == 1
        assert zero_stats["linear.data"]["misses"] == 0
        assert zero_stats["linear.reference"]["misses"] == 0

        # At the lower input rail, nominal data events already equal the deadline.
        # A small positive shift misses every data rail while the midpoint reference
        # still arrives, leaving a signed duration of -1.5 for every input feature.
        lower_value = torch.full((2, 3), -2.0, dtype=torch.float64)
        set_gaussian_time_noise(
            enabled=True,
            time_std=0.0,
            time_mean=0.5,
            seed=1002,
        )
        data_miss = layer(Potential(lower_value, potential.domain))
        data_stats = get_gaussian_noise_stats()
        expected_data_miss = torch.nn.functional.linear(
            torch.full_like(lower_value, -1.5),
            layer.weight,
            layer.bias,
        )
        assert torch.allclose(data_miss.value, expected_data_miss)
        assert data_stats["linear.data"]["misses"] == lower_value.numel()
        assert data_stats["linear.reference"]["misses"] == 0

        # At the upper rail, a 2.5 shift leaves every data event at time 2.5 but
        # pushes the zero reference beyond deadline 4. The physical duration is
        # therefore 4-2.5=1.5 for every input feature before applying the weights.
        upper_value = torch.full((2, 3), 2.0, dtype=torch.float64)
        set_gaussian_time_noise(
            enabled=True,
            time_std=0.0,
            time_mean=2.5,
            seed=1003,
        )
        reference_miss = layer(Potential(upper_value, potential.domain))
        reference_stats = get_gaussian_noise_stats()
        expected_reference = torch.nn.functional.linear(
            torch.full_like(upper_value, 1.5),
            layer.weight,
            layer.bias,
        )
        assert torch.allclose(reference_miss.value, expected_reference)
        assert reference_miss.domain == deterministic.domain
        assert reference_stats["linear.data"]["misses"] == 0
        assert reference_stats["linear.reference"]["misses"] == 1
        assert torch.isfinite(reference_miss.value).all()

        # A standard in-place parameter update after first use must invalidate the
        # frozen rail. Explicit refresh is required before a new inference regime.
        with torch.no_grad():
            layer.bias.add_(0.01)
        try:
            layer(potential)
        except RuntimeError:
            pass
        else:
            raise AssertionError("SpikingLinear accepted stale frozen bounds")
        assert (
            layer.freeze_parameter_bounds(potential.domain, refresh=True)
            != expected_domain
        )
    finally:
        # Restore process-wide state before the convolutional adapter regression.
        set_gaussian_time_noise(enabled=False)


def verify_gaussian_spiking_conv2d() -> None:
    """Verify frozen convolution bounds and signed-PWM miss trajectories.

    The Gaussian convolution contracts signed sampled pulse widths through PyTorch's
    grouped convolution kernel, and the deterministic path uses the same optimized
    reduction. A padded spatial example verifies zero-potential padding, one scalar
    reference event, output-channel absolute-sum rails, and mutation invalidation.

    Raises:
        AssertionError: If dense or zero-noise parity, frozen bounds, mutation
            rejection, event counts, miss readout, padding, or finiteness regresses.
    """
    layer = SpikingConv2d(
        1,
        2,
        kernel_size=2,
        stride=1,
        padding=1,
        bias=True,
        theta=2.0,
        dtype=torch.float64,
    )
    with torch.no_grad():
        layer.weight.copy_(
            torch.tensor(
                [
                    [[[0.5, -0.25], [0.75, 0.1]]],
                    [[[-0.4, 0.2], [0.3, -0.6]]],
                ],
                dtype=torch.float64,
            )
        )
        layer.bias.copy_(torch.tensor([0.15, -0.05], dtype=torch.float64))
    value = torch.tensor(
        [[[[ -1.5, 0.25, 1.75], [2.0, -2.0, 0.5], [1.0, -0.75, 1.5]]]],
        dtype=torch.float64,
    )
    potential = Potential(value, PotentialBounds(-2.0, 2.0))

    # Noise-free unfolded PWM must equal dense convolution on the same clamped input,
    # including zeros outside the image rather than encoded lower-rail padding.
    set_gaussian_time_noise(enabled=False)
    try:
        deterministic = layer(potential)
        expected = torch.nn.functional.conv2d(
            value,
            layer.weight,
            layer.bias,
            layer.stride,
            layer.padding,
            layer.dilation,
            layer.groups,
        )
        assert torch.allclose(deterministic.value, expected, atol=1e-12, rtol=1e-12)

        # Each output channel owns one kernel absolute-sum radius. Padding cannot
        # enlarge this full-receptive-field safety rail, and bias shifts endpoints.
        conv_radius = layer.theta * layer.weight.detach().abs().sum(dim=(1, 2, 3))
        expected_domain = PotentialBounds(
            (layer.bias.detach() - conv_radius).min().item(),
            (layer.bias.detach() + conv_radius).max().item(),
        )
        assert deterministic.domain == expected_domain

        # An asymmetric fixed rail exercises sign-aware interval arithmetic and
        # proves convolution consumes upstream calibration metadata. Zero remains in
        # the interval, so ordinary spatial padding retains its physical meaning.
        asymmetric_domain = PotentialBounds(-1.0, 2.0)
        asymmetric_value = value.clamp(
            asymmetric_domain.min,
            asymmetric_domain.max,
        )
        asymmetric = layer(Potential(asymmetric_value, asymmetric_domain))
        expected_asymmetric = torch.nn.functional.conv2d(
            asymmetric_value,
            layer.weight,
            layer.bias,
            layer.stride,
            layer.padding,
            layer.dilation,
            layer.groups,
        )
        assert torch.allclose(
            asymmetric.value,
            expected_asymmetric,
            atol=1e-12,
            rtol=1e-12,
        )
        lower_terms = torch.minimum(
            layer.weight.detach() * asymmetric_domain.min,
            layer.weight.detach() * asymmetric_domain.max,
        )
        upper_terms = torch.maximum(
            layer.weight.detach() * asymmetric_domain.min,
            layer.weight.detach() * asymmetric_domain.max,
        )
        expected_asymmetric_domain = PotentialBounds(
            (lower_terms.sum(dim=(1, 2, 3)) + layer.bias.detach()).min().item(),
            (upper_terms.sum(dim=(1, 2, 3)) + layer.bias.detach()).max().item(),
        )
        assert asymmetric.domain == expected_asymmetric_domain
        assert layer.freeze_parameter_bounds(asymmetric_domain) is asymmetric.domain

        # Zero scale samples every spatial activation once and one reference once.
        # Its direct duration convolution must preserve both values and propagated
        # fan-in rails from the explicit deterministic implementation.
        set_gaussian_time_noise(enabled=True, time_std=0.0, seed=1101)
        zero_noise = layer(potential)
        zero_stats = get_gaussian_noise_stats()
        assert torch.allclose(zero_noise.value, deterministic.value)
        assert zero_noise.domain == deterministic.domain
        assert zero_stats["conv2d.data"]["events"] == value.numel()
        assert zero_stats["conv2d.reference"]["events"] == 1
        assert zero_stats["conv2d.data"]["misses"] == 0
        assert zero_stats["conv2d.reference"]["misses"] == 0

        # Lower-rail data events shifted past the deadline leave their rails at reset.
        # The delivered reference rail supplies signed width -1.5 at every real input
        # location; convolution padding must still supply zero outside the image.
        lower_value = torch.full_like(value, -2.0)
        set_gaussian_time_noise(
            enabled=True,
            time_std=0.0,
            time_mean=0.5,
            seed=1102,
        )
        data_miss = layer(Potential(lower_value, potential.domain))
        data_stats = get_gaussian_noise_stats()
        expected_data_miss = torch.nn.functional.conv2d(
            torch.full_like(lower_value, -1.5),
            layer.weight,
            layer.bias,
            layer.stride,
            layer.padding,
            layer.dilation,
            layer.groups,
        )
        assert torch.allclose(data_miss.value, expected_data_miss)
        assert data_stats["conv2d.data"]["misses"] == lower_value.numel()
        assert data_stats["conv2d.reference"]["misses"] == 0

        # Upper-rail data shifted to 2.5 still fire, while the shared midpoint
        # reference misses deadline 4. Every real input location contributes duration
        # 1.5; ordinary zero padding must remain zero around that duration tensor.
        upper_value = torch.full_like(value, 2.0)
        set_gaussian_time_noise(
            enabled=True,
            time_std=0.0,
            time_mean=2.5,
            seed=1103,
        )
        reference_miss = layer(Potential(upper_value, potential.domain))
        reference_stats = get_gaussian_noise_stats()
        expected_reference = torch.nn.functional.conv2d(
            torch.full_like(upper_value, 1.5),
            layer.weight,
            layer.bias,
            layer.stride,
            layer.padding,
            layer.dilation,
            layer.groups,
        )
        assert torch.allclose(reference_miss.value, expected_reference)
        assert reference_miss.domain == deterministic.domain
        assert reference_stats["conv2d.data"]["misses"] == 0
        assert reference_stats["conv2d.reference"]["misses"] == 1
        assert torch.isfinite(reference_miss.value).all()

        # Cache reuse must not hide a later static perturbation. Refresh explicitly
        # establishes the only supported transition to a new parameter regime.
        with torch.no_grad():
            layer.weight.add_(0.01)
        try:
            layer(potential)
        except RuntimeError:
            pass
        else:
            raise AssertionError("SpikingConv2d accepted stale frozen bounds")
        assert (
            layer.freeze_parameter_bounds(potential.domain, refresh=True)
            != expected_domain
        )
    finally:
        # Restore process-wide state before the GPT-2 projection regression.
        set_gaussian_time_noise(enabled=False)


def verify_gaussian_spiking_conv1d() -> None:
    """Verify GPT-2 Conv1D frozen bounds and signed-PWM readout.

    Hugging Face Conv1D stores its matrix as ``[in_features, out_features]`` and
    applies it over the final dimension of arbitrary leading shapes. The regression
    verifies that Gaussian signed-pulse-width contraction preserves this convention,
    freezes output-column absolute-sum rails, samples one call-wide reference, rejects
    stale parameter versions, and follows symmetric one-sided-miss equations.

    Raises:
        AssertionError: If transposed affine parity, leading-shape preservation,
            frozen bounds, mutation rejection, event counts, misses, or finiteness
            regresses.
    """
    layer = SpikingConv1D(2, 3, theta=2.0).to(dtype=torch.float64)
    with torch.no_grad():
        layer.weight.copy_(
            torch.tensor(
                [[0.5, -0.25], [0.75, 0.4], [-1.0, 0.2]],
                dtype=torch.float64,
            )
        )
        layer.bias.copy_(torch.tensor([0.1, -0.2], dtype=torch.float64))
    value = torch.tensor(
        [
            [[-1.5, 0.25, 1.75], [2.0, -2.0, 0.5]],
            [[1.0, -0.75, 1.5], [-0.5, 1.25, -1.0]],
        ],
        dtype=torch.float64,
    )
    potential = Potential(value, PotentialBounds(-2.0, 2.0))

    # The explicit PWM path must reduce the first matrix dimension exactly like
    # ``value @ weight`` while retaining the two leading batch/token dimensions.
    set_gaussian_time_noise(enabled=False)
    try:
        deterministic = layer(potential)
        expected = torch.matmul(value, layer.weight) + layer.bias
        assert deterministic.value.shape == value.shape[:-1] + (layer.nf,)
        assert torch.allclose(deterministic.value, expected, atol=1e-12, rtol=1e-12)

        # Conv1D stores fan-in on dimension zero, so each output column's absolute
        # sum defines its safety radius before learned bias translates the endpoints.
        conv1d_radius = layer.theta * layer.weight.detach().abs().sum(dim=0)
        expected_domain = PotentialBounds(
            (layer.bias.detach() - conv1d_radius).min().item(),
            (layer.bias.detach() + conv1d_radius).max().item(),
        )
        assert deterministic.domain == expected_domain

        # A zero-containing asymmetric calibration rail must pass through the
        # transposed projection without being replaced by ``[-theta, theta]``.
        # Independent interval arithmetic verifies the output metadata as well.
        asymmetric_domain = PotentialBounds(-1.0, 2.0)
        asymmetric_value = value.clamp(
            asymmetric_domain.min,
            asymmetric_domain.max,
        )
        asymmetric = layer(Potential(asymmetric_value, asymmetric_domain))
        expected_asymmetric = (
            torch.matmul(asymmetric_value, layer.weight) + layer.bias
        )
        assert torch.allclose(
            asymmetric.value,
            expected_asymmetric,
            atol=1e-12,
            rtol=1e-12,
        )
        lower_terms = torch.minimum(
            layer.weight.detach() * asymmetric_domain.min,
            layer.weight.detach() * asymmetric_domain.max,
        )
        upper_terms = torch.maximum(
            layer.weight.detach() * asymmetric_domain.min,
            layer.weight.detach() * asymmetric_domain.max,
        )
        expected_asymmetric_domain = PotentialBounds(
            (lower_terms.sum(dim=0) + layer.bias.detach()).min().item(),
            (upper_terms.sum(dim=0) + layer.bias.detach()).max().item(),
        )
        assert asymmetric.domain == expected_asymmetric_domain
        assert layer.freeze_parameter_bounds(asymmetric_domain) is asymmetric.domain

        # Zero scale enters addmm-based Gaussian execution. All data carriers remain
        # exact and the one scalar reference must not be replicated per token.
        set_gaussian_time_noise(enabled=True, time_std=0.0, seed=1201)
        zero_noise = layer(potential)
        zero_stats = get_gaussian_noise_stats()
        assert torch.allclose(zero_noise.value, deterministic.value)
        assert zero_noise.domain == deterministic.domain
        assert zero_stats["conv1d.data"]["events"] == value.numel()
        assert zero_stats["conv1d.reference"]["events"] == 1
        assert zero_stats["conv1d.data"]["misses"] == 0
        assert zero_stats["conv1d.reference"]["misses"] == 0

        # Lower-rail data events shifted beyond the deadline leave their rails at
        # reset. The delivered reference produces signed width -1.5 across every
        # token and feature before the transposed weight contraction.
        lower_value = torch.full_like(value, -2.0)
        set_gaussian_time_noise(
            enabled=True,
            time_std=0.0,
            time_mean=0.5,
            seed=1202,
        )
        data_miss = layer(Potential(lower_value, potential.domain))
        data_stats = get_gaussian_noise_stats()
        expected_data_miss = (
            torch.matmul(torch.full_like(lower_value, -1.5), layer.weight)
            + layer.bias
        )
        assert torch.allclose(data_miss.value, expected_data_miss)
        assert data_stats["conv1d.data"]["misses"] == lower_value.numel()
        assert data_stats["conv1d.reference"]["misses"] == 0

        # Upper-rail data shifted to 2.5 remain delivered while the midpoint scalar
        # reference exceeds deadline 4. The resulting duration 1.5 must contract
        # against the transposed weight without swapping output or input axes.
        upper_value = torch.full_like(value, 2.0)
        set_gaussian_time_noise(
            enabled=True,
            time_std=0.0,
            time_mean=2.5,
            seed=1203,
        )
        reference_miss = layer(Potential(upper_value, potential.domain))
        reference_stats = get_gaussian_noise_stats()
        expected_reference = (
            torch.matmul(torch.full_like(upper_value, 1.5), layer.weight)
            + layer.bias
        )
        assert torch.allclose(reference_miss.value, expected_reference)
        assert reference_miss.domain == deterministic.domain
        assert reference_stats["conv1d.data"]["misses"] == 0
        assert reference_stats["conv1d.reference"]["misses"] == 1
        assert torch.isfinite(reference_miss.value).all()

        # Mutating the transposed projection after its first use must fail until an
        # explicit refresh establishes a coherent new parameter-bound pair.
        with torch.no_grad():
            layer.bias.add_(0.01)
        try:
            layer(potential)
        except RuntimeError:
            pass
        else:
            raise AssertionError("SpikingConv1D accepted stale frozen bounds")
        assert (
            layer.freeze_parameter_bounds(potential.domain, refresh=True)
            != expected_domain
        )
    finally:
        # Restore process-wide state before the LayerNorm regression.
        set_gaussian_time_noise(enabled=False)


# @lat: [[calibration#Layer-wise Calibration#Frozen Execution#Affine Fixed-Domain Consumption]]
def verify_affine_fixed_domain_contracts() -> None:
    """Run the fixed-input-domain regressions for every affine adapter.

    The three detailed checks cover ordinary linear projection, spatial convolution,
    and GPT-2's transposed projection. Together they verify asymmetric upstream rail
    consumption, exact sign-aware interval arithmetic, immutable memoization, shared
    zero-reference PWM parity, and explicit parameter-generation refresh.
    """
    # Linear is the canonical final-dimension affine layout used by ViT and BERT-like
    # adapters; its regression also checks both one-sided Gaussian miss trajectories.
    verify_gaussian_spiking_linear()

    # Conv2d adds grouped receptive-field reduction and zero-potential spatial
    # padding, both of which must preserve the same fixed-domain contract.
    verify_gaussian_spiking_conv2d()

    # GPT-2 Conv1D stores fan-in on dimension zero, so this final regression catches
    # an accidental transpose while enforcing the identical cache and PWM semantics.
    verify_gaussian_spiking_conv1d()


def verify_gaussian_spiking_layernorm() -> None:
    """Verify LayerNorm event semantics and frozen bound contracts.

    LayerNorm can enable or bypass variance multiplication, logarithmic encoding,
    and exponential-difference decoding independently. The regression first proves
    that an entirely dense configuration samples no events even when Gaussian noise
    is globally enabled. It checks the direct exponential ablation against explicitly
    reconstructed signed pulse widths, then checks full-spiking zero-noise parity and
    forces all nested events late to verify the final learned affine reset value.
    Finally, all eight ablation topologies independently reconstruct their expected
    analytic domains, reuse one immutable metadata object across noise modes, and
    reject stale parameter or configuration caches until explicitly refreshed.

    Raises:
        AssertionError: If dense bypass, full-spiking parity, site topology,
            all-miss bias retention, frozen bounds, mutation rejection, refresh,
            or finiteness regress.
    """
    value = torch.tensor(
        [[-1.5, -0.25, 0.75, 1.0], [0.5, -1.0, 1.5, -0.5]],
        dtype=torch.float64,
    )
    potential = Potential(value, PotentialBounds(-2.0, 2.0))
    weight = torch.tensor([1.0, 0.8, 1.2, 0.5], dtype=torch.float64)
    bias = torch.tensor([0.1, -0.2, 0.05, 0.3], dtype=torch.float64)

    # With all operator stages disabled, Gaussian configuration must not invent an
    # injection site. The module remains ordinary pretrained LayerNorm while its
    # metadata uses the finite-feature affine envelope frozen before evaluation.
    dense_layer = SpikingLayerNorm(
        4,
        eps=1.0e-5,
        theta=4.0,
        tau_s=1.0,
        clip_margin=0.1,
        use_spiking_mul=False,
        use_spiking_log=False,
        use_spiking_expdiff=False,
    ).to(dtype=torch.float64)
    with torch.no_grad():
        dense_layer.weight.copy_(weight)
        dense_layer.bias.copy_(bias)

    set_gaussian_time_noise(enabled=True, time_std=2.0, seed=1300)
    try:
        dense_output = dense_layer(potential)
        dense_expected = torch.nn.functional.layer_norm(
            value,
            (4,),
            dense_layer.weight,
            dense_layer.bias,
            dense_layer.eps,
        )
        assert torch.allclose(dense_output.value, dense_expected)
        assert get_gaussian_noise_stats() == {}

        # Keep log encoding enabled while bypassing exponential difference. A fixed
        # positive shift makes half of each residual rail miss while both shared sigma
        # rails arrive, isolating the direct branch's symmetric pulse-width equation.
        direct_exp_layer = SpikingLayerNorm(
            4,
            eps=1.0e-5,
            theta=4.0,
            tau_s=1.0,
            clip_margin=0.1,
            use_spiking_mul=False,
            use_spiking_log=True,
            use_spiking_expdiff=False,
        ).to(dtype=torch.float64)
        with torch.no_grad():
            direct_exp_layer.weight.copy_(weight)
            direct_exp_layer.bias.copy_(bias)

        mean_shift = 0.75
        set_gaussian_time_noise(
            enabled=True,
            time_std=0.0,
            time_mean=mean_shift,
            seed=1301,
        )
        direct_exp_output = direct_exp_layer(potential)
        direct_exp_stats = get_gaussian_noise_stats()

        # Reconstruct the nominal log times and deadline carriers independently from
        # the production branch, then apply d_err-d_sigma before the direct exp call.
        x_err = value - value.mean(dim=-1, keepdim=True)
        magnitude_domain = PotentialBounds(0.0, 3.9)
        x_err_pos_magnitude = magnitude_domain.clamp(x_err.clamp_min(0.0))
        x_err_neg_magnitude = magnitude_domain.clamp((-x_err).clamp_min(0.0))
        domain_err = PotentialBounds(0.1, 3.9)
        x_err_pos = domain_err.clamp(x_err_pos_magnitude)
        x_err_neg = domain_err.clamp(x_err_neg_magnitude)
        positive_active = x_err_pos_magnitude >= domain_err.min
        negative_active = x_err_neg_magnitude >= domain_err.min
        var_x = (
            x_err_pos_magnitude.square() + x_err_neg_magnitude.square()
        ).mean(dim=-1, keepdim=True) + direct_exp_layer.eps
        domain_var = PotentialBounds(domain_err.min ** 2, domain_err.max ** 2)
        var_x = domain_var.clamp(var_x)
        deadline = math.log(domain_err.max / domain_err.min)
        nominal_sigma = 0.5 * torch.log(
            value.new_tensor(domain_err.max ** 2) / var_x
        )
        nominal_pos = torch.log(value.new_tensor(domain_err.max) / x_err_pos)
        nominal_neg = torch.log(value.new_tensor(domain_err.max) / x_err_neg)

        def shifted_width(nominal_time: torch.Tensor) -> torch.Tensor:
            shifted_time = nominal_time + mean_shift
            fired = shifted_time <= deadline
            stored_time = shifted_time.clamp(0.0, deadline)
            return torch.where(
                fired,
                deadline - stored_time,
                torch.zeros_like(stored_time),
            )

        sigma_width = shifted_width(nominal_sigma)
        expected_positive = torch.exp(
            shifted_width(nominal_pos) - sigma_width
        )
        expected_negative = torch.exp(
            shifted_width(nominal_neg) - sigma_width
        )
        expected_result = torch.where(
            positive_active, expected_positive, torch.zeros_like(expected_positive)
        ) - torch.where(
            negative_active, expected_negative, torch.zeros_like(expected_negative)
        )
        expected_direct_exp = weight * expected_result + bias
        assert torch.allclose(direct_exp_output.value, expected_direct_exp)
        assert direct_exp_stats["layernorm.log_sigma"]["misses"] == 0
        assert direct_exp_stats["layernorm.log_positive"]["misses"] == 4
        assert direct_exp_stats["layernorm.log_negative"]["misses"] == 4

        # Enable every spiking stage with identical learned parameters. Zero scale
        # must preserve the established deterministic composition while exposing the
        # precise event multiplicities of two variance products, three log encoders,
        # two exponential differences, and one final learned-weight product.
        full_layer = SpikingLayerNorm(
            4,
            eps=1.0e-5,
            theta=4.0,
            tau_s=1.0,
            clip_margin=0.1,
            use_spiking_mul=True,
            use_spiking_log=True,
            use_spiking_expdiff=True,
        ).to(dtype=torch.float64)
        with torch.no_grad():
            full_layer.weight.copy_(weight)
            full_layer.bias.copy_(bias)

        set_gaussian_time_noise(enabled=False)
        deterministic = full_layer(potential)
        set_gaussian_time_noise(enabled=True, time_std=0.0, seed=1301)
        zero_noise = full_layer(potential)
        zero_stats = get_gaussian_noise_stats()
        assert torch.allclose(zero_noise.value, deterministic.value)
        assert bool(
            (
                (zero_noise.value >= zero_noise.domain.min)
                & (zero_noise.value <= zero_noise.domain.max)
            ).all()
        )
        assert zero_stats["multiplication.data"]["events"] == 20
        assert zero_stats["multiplication.reference"]["events"] == 3
        assert zero_stats["layernorm.log_sigma"]["events"] == 2
        assert zero_stats["layernorm.log_positive"]["events"] == value.numel()
        assert zero_stats["layernorm.log_negative"]["events"] == value.numel()
        assert zero_stats["exponential_difference.internal"]["events"] == 16
        assert all(site_stats["misses"] == 0 for site_stats in zero_stats.values())

        # A constant feature vector has zero centered magnitude on both rails. The
        # encoder floor may represent its logarithmic carriers, but active masks must
        # prevent either carrier from contributing to the normalized output.
        constant_value = torch.full_like(value, 1.25)
        set_gaussian_time_noise(enabled=False)
        constant_output = full_layer(
            Potential(constant_value, PotentialBounds(-2.0, 2.0))
        )
        assert torch.allclose(constant_output.value, bias.expand_as(constant_value))

        # A shift larger than the identity, log, and internal exponential windows
        # prevents every sampled stage from firing. Both final multiplication rails
        # remain at reset, so only the learned LayerNorm bias reaches the output.
        set_gaussian_time_noise(
            enabled=True,
            time_std=0.0,
            time_mean=20.0,
            seed=1302,
        )
        forced_late = full_layer(potential)
        forced_stats = get_gaussian_noise_stats()
        assert forced_stats["multiplication.data"]["misses"] == 20
        assert forced_stats["multiplication.reference"]["misses"] == 3
        assert forced_stats["layernorm.log_sigma"]["misses"] == 2
        assert forced_stats["layernorm.log_positive"]["misses"] == value.numel()
        assert forced_stats["layernorm.log_negative"]["misses"] == value.numel()
        assert forced_stats["exponential_difference.internal"]["misses"] == 16

        # Bias is broadcast over the batch after the reset-valued learned-weight
        # product. Noise never narrows the predeclared output rail to observed values.
        expected_bias = bias.expand_as(forced_late.value)
        assert torch.allclose(forced_late.value, expected_bias)
        assert forced_late.domain == zero_noise.domain
        assert torch.isfinite(forced_late.value).all()

        # Enumerate the complete three-flag topology instead of checking only the
        # dense and fully spiking endpoints. Mixed paths select different physical
        # value computations, but each must freeze metadata before noise sampling.
        for use_spiking_mul in (False, True):
            for use_spiking_log in (False, True):
                for use_spiking_expdiff in (False, True):
                    ablation_layer = SpikingLayerNorm(
                        4,
                        eps=1.0e-5,
                        theta=4.0,
                        tau_s=1.0,
                        clip_margin=0.1,
                        use_spiking_mul=use_spiking_mul,
                        use_spiking_log=use_spiking_log,
                        use_spiking_expdiff=use_spiking_expdiff,
                    ).to(dtype=torch.float64)
                    with torch.no_grad():
                        ablation_layer.weight.copy_(weight)
                        ablation_layer.bias.copy_(bias)

                    # Reconstruct the two learned-parameter domains independently.
                    # These endpoints are checkpoint metadata and must not depend on
                    # the activation batch or on whether Gaussian noise is enabled.
                    expected_weight_domain = PotentialBounds(
                        weight.min().item(),
                        weight.max().item(),
                    )
                    expected_bias_domain = PotentialBounds(
                        bias.min().item(),
                        bias.max().item(),
                    )

                    # Dense population normalization uses sqrt(d-1). Every mixed
                    # dual-rail topology uses sqrt(d), preventing the physical
                    # identity window from inheriting the much wider log-rail ratio.
                    all_dense = not (
                        use_spiking_mul
                        or use_spiking_log
                        or use_spiking_expdiff
                    )
                    if all_dense:
                        result_limit = math.sqrt(value.shape[-1] - 1)
                        effective_weight = weight
                    else:
                        result_limit = math.sqrt(value.shape[-1])
                        effective_weight = (
                            weight.clamp(
                                -ablation_layer.theta,
                                ablation_layer.theta,
                            )
                            if use_spiking_expdiff
                            else weight
                        )

                    # The spiking final multiplication propagates one global gamma
                    # interval, whereas dense and direct branches apply gamma and
                    # beta featurewise. Mirror those distinct mathematical contracts
                    # without reading any production cache or activation extrema.
                    if use_spiking_expdiff and not all_dense:
                        product_candidates = (
                            -result_limit * effective_weight.min().item(),
                            -result_limit * effective_weight.max().item(),
                            result_limit * effective_weight.min().item(),
                            result_limit * effective_weight.max().item(),
                        )
                        expected_output_domain = PotentialBounds(
                            min(product_candidates) + bias.min().item(),
                            max(product_candidates) + bias.max().item(),
                        )
                    else:
                        lower_candidate = (
                            effective_weight * -result_limit + bias
                        )
                        upper_candidate = (
                            effective_weight * result_limit + bias
                        )
                        expected_output_domain = PotentialBounds(
                            torch.minimum(
                                lower_candidate,
                                upper_candidate,
                            ).min().item(),
                            torch.maximum(
                                lower_candidate,
                                upper_candidate,
                            ).max().item(),
                        )

                    # First freeze publishes one tuple containing all three immutable
                    # domains. A second lookup must return that exact tuple rather
                    # than rebuilding equal objects from parameter tensors.
                    frozen_bounds = ablation_layer.freeze_parameter_bounds()
                    assert frozen_bounds[0] == expected_weight_domain
                    assert frozen_bounds[1] == expected_bias_domain
                    assert frozen_bounds[2] == expected_output_domain
                    assert (
                        ablation_layer.freeze_parameter_bounds()
                        is frozen_bounds
                    )

                    # Toggle only process-wide timing noise. Zero standard deviation
                    # must preserve values and both execution paths must attach the
                    # same cached output object, including the event-free dense case.
                    set_gaussian_time_noise(enabled=False)
                    deterministic_ablation = ablation_layer(potential)
                    set_gaussian_time_noise(
                        enabled=True,
                        time_std=0.0,
                        seed=1310,
                    )
                    gaussian_ablation = ablation_layer(potential)
                    assert torch.allclose(
                        gaussian_ablation.value,
                        deterministic_ablation.value,
                    )
                    assert deterministic_ablation.domain is frozen_bounds[2]
                    assert gaussian_ablation.domain is frozen_bounds[2]
                    assert bool(
                        (
                            (deterministic_ablation.value >= frozen_bounds[2].min)
                            & (deterministic_ablation.value <= frozen_bounds[2].max)
                        ).all()
                    )

        # A standard in-place parameter update must invalidate the first frozen
        # regime. Explicit refresh is the only supported transition after checkpoint
        # loading or static perturbation, and it must publish a new output domain.
        mutation_layer = SpikingLayerNorm(
            4,
            eps=1.0e-5,
            theta=4.0,
            tau_s=1.0,
            clip_margin=0.1,
            use_spiking_mul=True,
            use_spiking_log=True,
            use_spiking_expdiff=True,
        ).to(dtype=torch.float64)
        with torch.no_grad():
            mutation_layer.weight.copy_(weight)
            mutation_layer.bias.copy_(bias)
        original_bounds = mutation_layer.freeze_parameter_bounds()
        with torch.no_grad():
            mutation_layer.bias.add_(0.25)
        try:
            mutation_layer(potential)
        except RuntimeError:
            pass
        else:
            raise AssertionError(
                "SpikingLayerNorm accepted stale parameter bounds"
            )
        parameter_bounds = mutation_layer.freeze_parameter_bounds(refresh=True)
        assert parameter_bounds[2] != original_bounds[2]

        # Bound-defining configuration belongs to the same cache identity as gamma
        # and beta. Changing the clip margin must fail closed until refresh publishes
        # a new tuple, even though the finite-feature output endpoints stay equal.
        mutation_layer.clip_margin = 0.2
        try:
            mutation_layer.freeze_parameter_bounds()
        except RuntimeError:
            pass
        else:
            raise AssertionError(
                "SpikingLayerNorm accepted stale configuration bounds"
            )
        configuration_bounds = mutation_layer.freeze_parameter_bounds(
            refresh=True
        )
        assert configuration_bounds is not parameter_bounds
        assert configuration_bounds[2] == parameter_bounds[2]

        # A refreshed deterministic call must consume the newly published object;
        # this also proves that invalidation does not permanently poison the module.
        set_gaussian_time_noise(enabled=False)
        refreshed_output = mutation_layer(potential)
        assert refreshed_output.domain is configuration_bounds[2]
    finally:
        # Restore process-wide state before the attention regression.
        set_gaussian_time_noise(enabled=False)


def verify_gaussian_spiking_attention() -> None:
    """Verify complete attention parity and signed-PWM value integration.

    The end-to-end check covers score multiplication, softmin, and value readout on
    nontrivial query/key/value tensors. Separate calls to the maintained Gaussian
    value helper isolate data and shared-reference misses without allowing noisy
    score events to obscure the expected weighted pulse width. Together they fix
    dense parity, one reference event per call, and symmetric one-sided readout.

    Raises:
        AssertionError: If dense or zero-noise attention parity, value-event counts,
            miss-specific weighted integration, finite rails, or clamping regress.
    """
    query = torch.tensor(
        [[[[0.5, -0.25], [1.0, 0.75]]]],
        dtype=torch.float64,
    )
    key = torch.tensor(
        [[[[0.25, 0.5], [-0.75, 0.2], [0.6, -0.4]]]],
        dtype=torch.float64,
    )
    value = torch.tensor(
        [[[[1.0, -0.5], [-1.5, 0.75], [0.25, 1.25]]]],
        dtype=torch.float64,
    )

    # Configure five source positions even though this request uses only three.
    # The resulting rail must remain tied to configuration rather than request shape.
    source_length_max = 5
    output_domain = attention_output_bounds(
        theta=2.0,
        source_length_max=source_length_max,
    )
    assert output_domain == PotentialBounds(-10.0, 10.0)

    # With all tensors inside the symmetric rail and scores below the softmin cap,
    # the composed operator must equal ordinary scaled dot-product attention.
    set_gaussian_time_noise(enabled=False)
    try:
        deterministic = spiking_scaled_dot_product_attention(
            query,
            key,
            value,
            theta=2.0,
            tau=1.0,
            source_length_max=source_length_max,
        )
        dense_weight = torch.softmax(
            torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1)),
            dim=-1,
        )
        expected = torch.matmul(dense_weight, value)
        assert torch.allclose(deterministic, expected, atol=1e-12, rtol=1e-12)

        # A small theta still sets the outer score ceiling. Masked positions must use
        # the resulting fixed upper endpoint, rather than a larger independent value,
        # and pass the operator's declared-domain validation.
        keep_mask = torch.tensor(
            [[[[True, True, False], [True, False, False]]]],
            dtype=torch.bool,
        )
        masked = spiking_scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=keep_mask,
            theta=2.0,
            tau=1.0,
            source_length_max=source_length_max,
        )
        assert torch.isfinite(masked).all()

        # Zero scale traverses all score and normalization encoders before reaching
        # the value PWM. The final output must remain exact while sampling each value
        # once and a single scalar closing event for the complete attention call.
        set_gaussian_time_noise(enabled=True, time_std=0.0, seed=1401)
        zero_noise = spiking_scaled_dot_product_attention(
            query,
            key,
            value,
            theta=2.0,
            tau=1.0,
            source_length_max=source_length_max,
        )
        zero_stats = get_gaussian_noise_stats()
        assert torch.allclose(zero_noise, deterministic, atol=1e-12, rtol=1e-12)
        assert zero_stats["attention.value"]["events"] == value.numel()
        assert zero_stats["attention.value_reference"]["events"] == 1
        assert zero_stats["attention.value"]["misses"] == 0
        assert zero_stats["attention.value_reference"]["misses"] == 0

        # Drive one score to each finite suppression rail in float32. The shared
        # softmin division domain then spans exp(-2*cap) times the source reduction;
        # both deterministic and event-aware paths must retain a positive carrier
        # instead of underflowing during exponential-difference validation.
        cap_query = torch.tensor([[[[2000.0]]]], dtype=torch.float32)
        cap_key = torch.tensor([[[[2000.0], [-2000.0]]]], dtype=torch.float32)
        cap_value = torch.tensor([[[[1.0], [-1.0]]]], dtype=torch.float32)
        set_gaussian_time_noise(enabled=False)
        deterministic_cap = spiking_scaled_dot_product_attention(
            cap_query,
            cap_key,
            cap_value,
            theta=2000.0,
            tau=1.0,
            source_length_max=2,
        )
        set_gaussian_time_noise(enabled=True, time_std=0.0, seed=1404)
        gaussian_cap = spiking_scaled_dot_product_attention(
            cap_query,
            cap_key,
            cap_value,
            theta=2000.0,
            tau=1.0,
            source_length_max=2,
        )
        assert torch.isfinite(deterministic_cap).all()
        assert torch.isfinite(gaussian_cap).all()
        assert torch.allclose(gaussian_cap, deterministic_cap, atol=1e-6, rtol=1e-6)

        # Fix attention weights explicitly so a lower-rail value shift can isolate
        # data misses. Every value rail stays at reset, while the delivered reference
        # rail supplies signed width -1.5; normalized weights preserve that value.
        fixed_weight = torch.tensor(
            [[[[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]]]],
            dtype=torch.float64,
        )
        lower_value = torch.full_like(value, -2.0)
        value_domain = PotentialBounds(-2.0, 2.0)
        set_gaussian_time_noise(
            enabled=True,
            time_std=0.0,
            time_mean=0.5,
            seed=1402,
        )
        data_miss = _gaussian_attention_value_readout(
            lower_value,
            fixed_weight,
            value_domain,
            output_domain,
        )
        data_stats = get_gaussian_noise_stats()
        expected_data_miss = torch.full_like(data_miss, -1.5)
        assert torch.allclose(data_miss, expected_data_miss)
        assert data_stats["attention.value"]["misses"] == lower_value.numel()
        assert data_stats["attention.value_reference"]["misses"] == 0

        # Upper-rail values shifted to 2.5 still fire, while the shared midpoint
        # reference misses deadline 4. Each value duration is 1.5; normalized fixed
        # weights therefore preserve that same scalar in every output feature.
        upper_value = torch.full_like(value, 2.0)
        set_gaussian_time_noise(
            enabled=True,
            time_std=0.0,
            time_mean=2.5,
            seed=1403,
        )
        reference_miss = _gaussian_attention_value_readout(
            upper_value,
            fixed_weight,
            value_domain,
            output_domain,
        )
        reference_stats = get_gaussian_noise_stats()
        expected_reference = torch.full_like(reference_miss, 1.5)
        assert torch.allclose(reference_miss, expected_reference)
        assert reference_stats["attention.value"]["misses"] == 0
        assert reference_stats["attention.value_reference"]["misses"] == 1

        # The helper uses the configured five-position envelope [-10, 10], not the
        # current three-position request. Both physical readouts remain finite and
        # inside that fixed rail after pre-clamp saturation accounting.
        assert torch.isfinite(data_miss).all()
        assert torch.isfinite(reference_miss).all()
        assert bool(
            (
                (reference_miss >= output_domain.min)
                & (reference_miss <= output_domain.max)
            ).all()
        )
    finally:
        # Leave the shared Gaussian configuration disabled after all regressions.
        set_gaussian_time_noise(enabled=False)


if __name__ == "__main__":
    verify_immutable_memoized_bounds()
    verify_closed_bounds_validation()
    verify_broadcast_gaussian_time_inputs()
    verify_gaussian_time_input_validation()
    verify_gaussian_sampler_rng_contract()
    verify_gaussian_sampler_deadline_contract()
    verify_gaussian_deadline_probability()
    verify_exponential_time_constant_scaling()
    verify_gaussian_encoder_boundary()
    verify_gaussian_statistics_contract()
    verify_gaussian_multiplication_operator()
    verify_gaussian_exponential_function()
    verify_gaussian_exponential_difference_operator()
    verify_gaussian_division_function()
    verify_gaussian_tanh_function()
    verify_gaussian_sigmoid_gelu_function()
    verify_gaussian_softmin_function()
    verify_gaussian_swiglu_function()
    verify_affine_fixed_domain_contracts()
    verify_gaussian_spiking_layernorm()
    verify_gaussian_spiking_attention()
    print("Gaussian verification passed.")
