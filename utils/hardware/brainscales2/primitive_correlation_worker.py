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


def _run(request: dict) -> dict:
    import pynn_brainscales.brainscales2 as pynn

    config = request["config"]
    differences = request["differences"].to(torch.float64)
    repeats = int(request["repeats"])
    point_count = int(differences.numel())
    guard_ms = config.psi_ed_trial_guard_s * 1.0e3
    post_ms = config.psi_ed_post_time_s * 1.0e3
    readout_ms = config.psi_ed_readout_time_s * 1.0e3
    periods = 1 + 2 * repeats * point_count
    pre_times: list[float] = []
    post_times: list[float] = [post_ms]
    for trial in range(repeats):
        for point, difference in enumerate(differences.tolist()):
            pair = trial * point_count + point
            quiet_period = 1 + 2 * pair
            stimulated_period = quiet_period + 1
            post_times.extend(
                [
                    quiet_period * guard_ms + post_ms,
                    stimulated_period * guard_ms + post_ms,
                ]
            )
            separation_ms = (
                config.psi_ed_separation_center_s - float(difference)
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
        quiet = raw[quiet_indices].reshape(shape)
        stimulated = raw[stimulated_indices].reshape(shape)
        first = first[stimulated_indices].reshape(shape)
        count = count[stimulated_indices].reshape(shape)
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
