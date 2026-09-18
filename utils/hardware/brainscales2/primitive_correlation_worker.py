"""Private process worker for causal-correlation primitive acquisition."""

from __future__ import annotations

from pathlib import Path
import pickle
import sys
import textwrap

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from utils.hardware.brainscales2.primitive_backend import PrimitiveHardwareBackend
from utils.hardware.brainscales2.primitive_pynn_worker import _setup_hardware_client


def _correlation_rule_type(pynn):
    class CorrelationReadRule(pynn.PlasticityRule):
        def __init__(self, timer):
            observables = {
                "causal": pynn.PlasticityRule.ObservablePerSynapse(
                    pynn.PlasticityRule.ObservablePerSynapse.Type.uint8,
                    pynn.PlasticityRule.ObservablePerSynapse.LayoutPerRow.packed_active_columns,
                )
            }
            super().__init__(timer=timer, observables=observables)

        def generate_kernel(self):
            return textwrap.dedent(
                r"""
                #include "grenade/vx/ppu/synapse_array_view_handle.h"
                #include "grenade/vx/ppu/neuron_view_handle.h"
                #include "libnux/vx/correlation.h"
                #include "libnux/vx/dls.h"
                #include "libnux/vx/vector_row.h"
                #include "hate/tuple.h"

                using namespace grenade::vx::ppu;
                using namespace libnux::vx;
                extern volatile PPUOnDLS ppu;

                template <size_t N>
                void PLASTICITY_RULE_KERNEL(
                    std::array<SynapseArrayViewHandle, N>& synapses,
                    std::array<NeuronViewHandle, 0>&,
                    Recording& recording)
                {
                    hate::for_each(
                        [&ppu](SynapseArrayViewHandle const& local_synapses,
                               auto& local_recording) {
                            if (local_synapses.hemisphere != ppu) {
                                return;
                            }
                            for (size_t row = 0;
                                 row < local_synapses.rows.size();
                                 ++row) {
                                VectorRowMod8 result;
                                get_causal_correlation(
                                    &result.even.data,
                                    &result.odd.data,
                                    local_synapses.rows[row]);
                                for (size_t column = 0;
                                     column < local_synapses.columns.size();
                                     ++column) {
                                    local_recording[row][column] =
                                        result[local_synapses.columns[column]];
                                }
                            }
                        },
                        synapses,
                        recording.causal);
                    reset_all_correlations();
                }
                """
            )

    return CorrelationReadRule


def _calibrated_chip(pynn, calibration_path: Path):
    from dlens_vx_v3 import lola, sta

    with calibration_path.open("rb") as handle:
        calibration = pickle.load(handle)
    injected = pynn.InjectedConfiguration()
    builder = sta.PlaybackProgramBuilder()
    calibration.apply(builder)
    injected.pre_static_config = builder
    dumper = sta.PlaybackProgramBuilderDumper()
    calibration.apply(dumper)
    chip = sta.convert_to_chip(dumper.done(), lola.Chip())
    for readout_chain in chip.cadc_readout_chains:
        for channels in (
            readout_chain.channels_causal,
            readout_chain.channels_acausal,
        ):
            for channel in channels:
                channel.enable_connect_neuron = False
    return chip, injected


def _decode_correlation(recording, *, periods: int, devices: int) -> torch.Tensor:
    rows = list(recording)
    if len(rows) != 1:
        raise RuntimeError(
            f"expected one plastic synapse row, received {len(rows)}"
        )
    entries = list(rows[0])
    if len(entries) != periods:
        raise RuntimeError(
            f"expected {periods} correlation records, received {len(entries)}"
        )
    decoded = torch.empty((periods, devices), dtype=torch.float64)
    for period, entry in enumerate(entries):
        data = entry.data
        if hasattr(data, "tolist"):
            data = data.tolist()
        values = torch.as_tensor(data, dtype=torch.float64).reshape(-1)
        if values.numel() != devices:
            raise RuntimeError(
                f"correlation record has {values.numel()} values; expected {devices}"
            )
        decoded[period] = values
    return decoded


def _point_orders(
    *, repeats: int, point_count: int, seed: int, trial_start: int
) -> list[list[int]]:
    """Return deterministic per-trial acquisition permutations."""
    orders: list[list[int]] = []
    for trial in range(repeats):
        generator = torch.Generator().manual_seed(
            seed + 104729 * (trial_start + trial + 1)
        )
        orders.append(
            torch.randperm(point_count, generator=generator).tolist()
        )
    return orders


def _restore_point_order(
    scheduled: torch.Tensor, point_orders: list[list[int]]
) -> torch.Tensor:
    """Restore canonical point indices after scheduled acquisition."""
    if scheduled.ndim != 3 or scheduled.shape[0] != len(point_orders):
        raise ValueError("scheduled correlation tensor has incompatible shape")
    restored = torch.empty_like(scheduled)
    for trial, point_order in enumerate(point_orders):
        if sorted(point_order) != list(range(scheduled.shape[1])):
            raise ValueError("correlation point order is not a permutation")
        for acquisition_point, point in enumerate(point_order):
            restored[trial, point] = scheduled[trial, acquisition_point]
    return restored


def _run(request: dict) -> dict:
    import pynn_brainscales.brainscales2 as pynn

    config = request["config"]
    differences = request["differences"].to(torch.float64)
    repeats = int(request["repeats"])
    trial_start = int(request["trial_start"])
    point_count = int(differences.numel())
    guard_ms = config.psi_ed_trial_guard_s * 1.0e3
    post_ms = config.psi_ed_post_time_s * 1.0e3
    readout_ms = config.psi_ed_readout_time_s * 1.0e3
    periods = 1 + 2 * repeats * point_count
    pre_times: list[float] = []
    post_times: list[float] = [post_ms]
    point_orders = _point_orders(
        repeats=repeats,
        point_count=point_count,
        seed=config.seed,
        trial_start=trial_start,
    )
    for trial in range(repeats):
        point_order = point_orders[trial]
        for acquisition_point, point in enumerate(point_order):
            difference = float(differences[point])
            pair = trial * point_count + acquisition_point
            quiet_period = 1 + 2 * pair
            stimulated_period = quiet_period + 1
            post_times.extend(
                [
                    quiet_period * guard_ms + post_ms,
                    stimulated_period * guard_ms + post_ms,
                ]
            )
            separation_ms = (
                config.psi_ed_separation_center_s - difference
            ) * 1.0e3
            pre_times.append(
                stimulated_period * guard_ms + post_ms - separation_ms
            )

    chip, injected = _calibrated_chip(
        pynn, config.correlation_calibration_path
    )
    setup_complete = False
    pynn.setup(
        initial_config=chip,
        injected_config=injected,
        neuronPermutation=list(config.physical_coordinates),
    )
    setup_complete = True
    try:
        timer = pynn.Timer(
            start=readout_ms,
            period=guard_ms,
            num_periods=periods,
        )
        rule = _correlation_rule_type(pynn)(timer)
        source_type = pynn.standardmodels.cells.SpikeSourceArray
        pre = pynn.Population(1, source_type(spike_times=pre_times))
        trigger = pynn.Population(
            config.psi_ed_trigger_fan_in,
            source_type(spike_times=post_times),
        )
        neuron = pynn.Population(
            config.device_count,
            pynn.cells.HXNeuron(
                leak_v_leak=config.reset_code_minimum,
                leak_i_bias=config.leak_bias,
                leak_enable_division=True,
                threshold_enable=True,
                threshold_v_threshold=config.threshold_code,
                reset_v_reset=config.reset_code_minimum,
                constant_current_enable=False,
                reset_i_bias=1022,
                reset_enable_multiplication=True,
                refractory_period_refractory_time=255,
                refractory_period_enable_pause=True,
            ),
        )
        neuron.record("spikes")
        projection = pynn.Projection(
            pre,
            neuron,
            pynn.AllToAllConnector(),
            synapse_type=pynn.standardmodels.synapses.PlasticSynapse(
                plasticity_rule=rule,
                weight=config.psi_ed_plastic_weight,
            ),
            receptor_type="excitatory",
        )
        pynn.Projection(
            trigger,
            neuron,
            pynn.AllToAllConnector(),
            synapse_type=pynn.standardmodels.synapses.StaticSynapse(
                weight=config.psi_ed_trigger_weight
            ),
            receptor_type="excitatory",
        )
        pynn.run(periods * guard_ms)
        raw = _decode_correlation(
            projection.get_data("causal"),
            periods=periods,
            devices=config.device_count,
        )
        segment = neuron.get_data("spikes").segments[-1]
        first, count = PrimitiveHardwareBackend._decode_pynn_spikes(
            segment.spiketrains,
            repeats=periods,
            devices=config.device_count,
            window_ms=guard_ms,
        )
        quiet_indices: list[int] = []
        stimulated_indices: list[int] = []
        for pair in range(repeats * point_count):
            quiet_indices.append(1 + 2 * pair)
            stimulated_indices.append(2 + 2 * pair)
        shape = (repeats, point_count, config.device_count)
        quiet_scheduled = raw[quiet_indices].reshape(shape)
        stimulated_scheduled = raw[stimulated_indices].reshape(shape)
        first_scheduled = first[stimulated_indices].reshape(shape)
        count_scheduled = count[stimulated_indices].reshape(shape)
        quiet = _restore_point_order(quiet_scheduled, point_orders)
        stimulated = _restore_point_order(stimulated_scheduled, point_orders)
        first = _restore_point_order(first_scheduled, point_orders)
        count = _restore_point_order(count_scheduled, point_orders)
        identifier = pynn.helper.get_unique_identifier()
        chip_identifier = (
            [str(item) for item in identifier]
            if isinstance(identifier, (tuple, list))
            else [str(identifier)]
        )
        return {
            "quiet": quiet,
            "stimulated": stimulated,
            "first": first,
            "count": count,
            "metadata": {
                "chip_identifier": chip_identifier,
                "periods": periods,
                "discarded_warmup_periods": 1,
                "raw_correlation_range": [0, 255],
                "point_orders": point_orders,
            },
        }
    finally:
        if setup_complete:
            pynn.end()


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: primitive_correlation_worker.py REQUEST RESPONSE"
        )
    request_path = Path(sys.argv[1])
    response_path = Path(sys.argv[2])
    request = torch.load(request_path, map_location="cpu", weights_only=False)
    _setup_hardware_client()
    torch.save(_run(request), response_path)


if __name__ == "__main__":
    main()
