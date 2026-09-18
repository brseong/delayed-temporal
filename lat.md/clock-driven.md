---
lat:
  require-code-mention: true
---
# Clock-Driven TTFS Evaluation

Clock-driven evaluation executes the maintained TTFS operator graph with one configured discrete-time resolution policy while preserving the frozen potential ranges used by the continuous-time baseline.

This path is a deterministic evaluation mode. Gaussian spike-time noise, static mismatch, and learned-parameter perturbations remain disabled so a task-metric change is attributable to the clock alone. It does not claim execution on Loihi hardware.

## Execution Contract

The central temporal primitives own the clock policy rather than individual model adapters. The absolute step width configuration uses one global width; the equal interval count configuration derives a local width from each declared time window.

Potential-to-spike encoders test threshold delivery at every clock edge and place crossings on the first non-earlier edge. PWM readout updates active accumulators once per step, and exponential readout applies one decay or growth update per step.

Affine and attention kernels retain their tensor reductions, but their temporal inputs come from the same explicit PWM loop as the scalar primitive. The implementation keeps one current-state tensor and does not allocate a time-leading history tensor or skip steps with a closed-form expression.

A second clock configuration divides every declared time window into the same requested number of equal time steps. The step width is derived independently from each fixed window.

The encoder observes both endpoints, while PWM and exponential readout execute one state update for every interval. The iteration count therefore remains independent of the physical time-window length.

Clock index recovery accounts for dtype roundoff from both repeated state updates and subtracting a nonzero time window origin. The allowance scales with the represented coordinate magnitude and remains capped at one quarter of a time step, so a materially unaligned duration is rejected.

Durations relative to a deadline or offset are formed by subtracting their integer clock indices and converting the resulting index once. This preserves alignment when two large aligned times cancel to a small duration.

Each completed pulse width accumulation loop converts its accumulated states back to integer clock indices before recombination. This removes only roundoff from repeated addition and does not skip any step.

## Evaluation Contract

The first maintained evaluation uses calibrated ViT-B/16 on the fixed ImageNet-1k validation subset of 5,000 images.

A source-matched frozen calibration table and the selected threshold remain unchanged across the continuous and clock-driven runs. Four contiguous shards cover the fixed 5,000 images exactly once and run on GPUs 4–7; correct and total counts are summed only after coverage validation.

Each shard records the time bin, task counts, prediction digest, encoder rounding, code-window lengths, and executed state-update counts. The aggregate preserves all shard records and reports one accuracy over exactly 5,000 images.

The evaluator prints accuracy to eight decimal places. Result validation compares the serialized value at that precision and stores the exact ratio of correct predictions to evaluated samples.

Completed tag `vit_base_clock_driven_imagenet5k_theta20_float64_v2` used execution source `0c5c394`. The continuous evaluation obtained 4,300/5,000 (86.00%), while clock-driven execution with time bin 1.0 obtained 3/5,000 (0.06%).

This result measures one coarse time bin and does not establish behavior at finer time bins. `verification.json` preserves the complete shard coverage, identities, hashes, and positive encoder, PWM, and exponential update counts.

The maintained fine sweep evaluates one continuous reference and time bins 0.01, 0.02, ..., 0.10 on the first 500 images of the fixed validation ordering. The population is divided into 21 balanced contiguous shards, and every shard contains at most 24 images. Each aggregate is accepted only after exact 500-image coverage and identity validation.

Completed tag `vit_base_clock_driven_imagenet500_theta20_float64_fine_v1` used execution source `ec058e5` and the same frozen calibration table as the continuous reference.

The continuous reference obtained 432/500 (86.4%). Clock time steps 0.01 through 0.10 obtained 85.6%, 86.0%, 84.8%, 84.6%, 81.6%, 84.8%, 81.0%, 76.2%, 76.8%, and 60.4%, respectively.

The observed sequence is not monotone: 0.06 exceeds 0.05 and 0.09 exceeds 0.08. These measurements are retained without filtering.

The authoritative outputs are `summary.csv`, `summary.json`, `raw_shards.csv`, and `verification.json` under `artifacts/logs/clock_driven/vit_base_clock_driven_imagenet500_theta20_float64_fine_v1/`. The generated figure files are under its `figures/` directory.

The verification record confirms 11 conditions, 231 shard runs, exact 500 image coverage per condition, disabled timing noise, one shared calibration identity, and positive explicit state update counts.

The equal step benchmark uses the first 500 images of the fixed validation ordering and evaluates 64, 128, 256, 512, 1024, and 2048 time steps per time window together with one continuous reference.

Every condition uses 21 contiguous shards and the same frozen calibration table, checkpoint, preprocessing, threshold, and disabled noise settings as the completed fine time-step sweep.

For this sweep schedule, the local worker owns 0.01 shards 0 through 17 and shard 19 on devices 3 through 7, while cluster workers own shards 18 and 20. The cluster also owns time bins 0.02 through 0.10. The supervisor stops the local controller after every local shard is complete and imports the two cluster shards only after identity, log, and coverage validation.

The superseded coarse campaign completed its continuous reference and six of eight 0.1 shards before the requested range changed. Its partial records remain preserved and are not combined with the fine sweep.

The default local device policy remains devices 4 through 7. This campaign uses an explicit override that permits devices 0 through 7, while two consecutive idle observations exclude occupied devices before launch. The worker pool is then fixed: devices that become idle later are not admitted because unscheduled external work can reclaim them between polling and launch. Restarting from complete logs is the supported way to change the pool.

The v1 attempt was rejected before producing a time bin result because repeated 0.1 step accumulation caused an aligned duration to fail the strict numerical alignment check. The v2 execution admits only bounded arithmetic drift accumulated by explicit state updates and still rejects a displacement of one quarter of a bin. It shares the earlier calibration table only after confirming that every changed path is unable to affect calibration. Runtime validation substitutes the recorded source revision and the calibration table's recorded ViT evaluator digest, while requiring every other metadata field to match exactly.

Changes inside a shared temporal operator are accepted for calibration reuse only when the complete file patch matches its approved digest and the continuous execution branch remains unchanged. The execution runner also verifies the table's ViT evaluator digest before it enables either compatibility value.

## Verification

The verification cases distinguish clock semantics from ordinary floating-point evaluation.

### Causal Encoder Clocking

An encoded threshold crossing is observed by a sequential edge loop at the first clock edge at or after its continuous time, and its declared deadline is aligned by the same rule.

### Equal Steps in Each Time Window

This verification checks that different physical time-window lengths use the same configured interval count.

Every encoder site must report the requested minimum and maximum window count. Encoder observation executes one more edge than the interval count, while PWM and exponential loops execute exactly the interval count per call.

The evaluator rejects simultaneous absolute and window-relative resolution settings, and final reporting rejects any shard whose encoder or state update counts disagree with its declared setting.

### PWM State Updates

The production signed PWM path performs and counts an explicit per-time-step accumulation of the two causal integration paths.

### Exponential State Updates

The production exponential readout performs and counts repeated per-time-step state updates without using a power shortcut.

### Optimized Model Kernels

ViT affine and attention tensor kernels consume the same clock-driven PWM duration as the scalar primitive and match explicit expected outputs.

### Disabled-Mode Parity

Disabling clock-driven execution preserves the continuous-time encoder, PWM, and exponential results.

### ViT Runtime Isolation

The ViT evaluator accepts a positive global time step only for a spiking backend and rejects simultaneous timing noise or a multi-GPU process.

### Contiguous Evaluation Shards

The configured evaluation population is divided into balanced contiguous half-open ranges with no overlap or omission, and aggregation requires complete ordered coverage of that population.

### Selected Evaluation Shards

A distributed worker may evaluate explicit shard indices without changing the global shard count or contiguous range definition.

Duplicate indices or indices outside the valid range are rejected. Final reporting still requires every shard.

### Completed Sweep Reporting

Final reporting accepts only all 11 conditions, 21 contiguous shards per condition, exact 500 image coverage, matching identities, positive state update counts, and matching generated summaries.

### Composed Encoder Statistics

Result validation requires the identity and logarithmic encoder statistics while preserving additional named encoder sites emitted by composed operators such as GELU.
