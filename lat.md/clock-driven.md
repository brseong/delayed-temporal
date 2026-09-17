---
lat:
  require-code-mention: true
---
# Clock-Driven TTFS Evaluation

Clock-driven evaluation executes the maintained TTFS operator graph on one global discrete clock while preserving the frozen potential ranges used by the continuous-time baseline.

This path is a deterministic evaluation mode. Gaussian spike-time noise, static mismatch, and learned-parameter perturbations remain disabled so a task-metric change is attributable to the clock alone. It does not claim execution on Loihi hardware.

## Execution Contract

The central temporal primitives own the global clock rather than individual model adapters.

Potential-to-spike encoders test threshold delivery at every clock edge and place crossings on the first non-earlier edge. PWM readout updates active accumulators once per step, and exponential readout applies one decay or growth update per step.

Affine and attention kernels retain their tensor reductions, but their temporal inputs come from the same explicit PWM loop as the scalar primitive. The implementation keeps one current-state tensor and does not allocate a time-leading history tensor or skip steps with a closed-form expression.

## Evaluation Contract

The first maintained evaluation uses calibrated ViT-B/16 on the fixed ImageNet-1k validation subset of 5,000 images.

A source-matched frozen calibration table and the selected threshold remain unchanged across the continuous and clock-driven runs. Four contiguous shards cover the fixed 5,000 images exactly once and run on GPUs 4–7; correct and total counts are summed only after coverage validation.

Each shard records the time bin, task counts, prediction digest, encoder rounding, code-window lengths, and executed state-update counts. The aggregate preserves all shard records and reports one accuracy over exactly 5,000 images.

Completed tag `vit_base_clock_driven_imagenet5k_theta20_float64_v2` used execution source `0c5c394`. The continuous evaluation obtained 4,300/5,000 (86.00%), while clock-driven execution with time bin 1.0 obtained 3/5,000 (0.06%).

This result measures one coarse time bin and does not establish behavior at finer time bins. `verification.json` preserves the complete shard coverage, identities, hashes, and positive encoder, PWM, and exponential update counts.

The additional sweep tagged `vit_base_clock_driven_imagenet500_theta20_float64_v1` uses the first 500 images of the same fixed validation ordering. Four contiguous shards cover 125 images each. It reuses the verified calibration table and execution source while evaluating one continuous reference and global time bins 0.1, 0.2, ..., 1.0. Each aggregate is accepted only after exact 500-image coverage and identity validation.

## Verification

The verification cases distinguish clock semantics from ordinary floating-point evaluation.

### Causal Encoder Clocking

An encoded threshold crossing is observed by a sequential edge loop at the first clock edge at or after its continuous time, and its declared deadline is aligned by the same rule.

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

### Composed Encoder Statistics

Result validation requires the identity and logarithmic encoder statistics while preserving additional named encoder sites emitted by composed operators such as GELU.
