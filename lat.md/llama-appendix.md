# Llama 2 7B Appendix Diagnostic

The bounded Llama comparison records conversion fidelity and timing noise sensitivity on two language modeling datasets without promoting a short run to a complete evaluation.

## Frozen Population and Calibration

A calibration from 32 WikiText-2 training texts is reused for the first 128 test texts of WikiText-2 and IMDb. IMDb perplexity is a language modeling diagnostic, not sentiment accuracy.

Both backends use float64, batch size 8, and token length 128. [[scripts/evaluation/error_analysis_llama.py#main]] owns model loading, tokenization, and the source and converted perplexity calculations. The diagnostic shares one frozen calibration across all conditions; it does not use the 5,000-text calibration required for a full comparison.

## Noise Conditions and Result Identity

The converted clean result is the deterministic reference. Each noisy cell scales the separate measured linear and logarithmic timing noise fractions by 0.00001, 0.0001, or 0.001.

[[scripts/experiments/run_llama_appendix_noise.py#main]] validates checkpoint, implementation, calibration, dataset, device, hardware summary, and condition identities before reusing a cell. A summary and TeX table are written only when both datasets have all three multiplier values at seeds 0, 1, and 2. Each noisy cell reports mean perplexity and a 95% Student-$t$ confidence interval; the clean values are deterministic references. [[scripts/verification/verify_llama_appendix_noise.py#main]] checks identity rejection and table generation without running a GPU evaluation.

The generated TeX table shows each noisy mean and interval in one line at three decimal places; the source summary retains full precision.

## Current Result State

All 18 noisy cells (two datasets, three multipliers, seeds 0 to 2) under `artifacts/llama/llama2-7b-hf/two-dataset-quick-20260925/` pass the identity checks, and the summary and table report seeds 0, 1, and 2.

Four IMDb cells at multipliers 0.0001 and 0.001 with seeds 1 and 2 stopped after the first of 16 batches when their processes were terminated externally. They were rerun with the same checkpoint, `calibration32.json`, implementation digest, and replica layout of two devices; the stopped logs are kept beside the results as `*.interrupted-20260925T2244Z.log`.

| Dataset | Source clean | Converted clean | 0.00001 | 0.0001 | 0.001 |
|---|---:|---:|---:|---:|---:|
| WikiText-2 | 15.1839 | 15.1832 | 15.1788 ± 0.0043 | 15.4010 ± 0.3521 | 620.0070 ± 351.5189 |
| IMDb | 13.7912 | 13.7912 | 13.7909 ± 0.0024 | 13.9259 ± 0.2109 | 525.5148 ± 284.3734 |

Noisy cells give mean perplexity ± half the 95% Student-$t$ interval width; values for each seed stay in `appendix_llama_noise_diagnostic.json`. At 0.00001 and 0.0001 the means stay within 1.5% of the converted clean reference, while at 0.001 both datasets exceed 500 with wide seed spread.
