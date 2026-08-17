# Architecture

This project evaluates pretrained Transformers whose dense operations are re-expressed as composable TTFS spiking operators while retaining standard model inputs, outputs, and checkpoints.

## System Boundary

The maintained research path is the operator-based Transformer stack under `utils/transforms` and `utils/transformers`, driven by runners and experiment scripts under `scripts/`.

The repository contains four kinds of code:

- The domain and operator kernel in `utils/transforms`, described by [[domain#Domain Model]] and [[operators#Operator System]].
- Model adapters in `utils/transformers`, described by [[models#Model Integration]].
- Evaluation runners under `scripts/evaluation`, shell orchestration under `scripts/experiments`, and focused programs under `analysis/`, described by [[evaluation#Evaluation and Verification]].
- An older discrete-time Jeffress/correlation research path in `utils/model.py`, `utils/layer.py`, `utils/module.py`, and related helpers.

The installed Hugging Face Transformers and SpikingJelly packages are dependencies, not project-owned architecture. Local model files adapt their public abstractions rather than maintaining a general-purpose fork.

## Layered Design

The main stack separates numeric domains, operator algebra, model integration, and experimental orchestration so each layer can be reasoned about independently.

1. `OpenBounds`, `PotentialBounds`, `TimeBounds`, and `Potential` define the values carried between operations; see [[utils/transforms/types.py#Potential]].
2. Potential-to-time and time-to-potential transforms define TTFS encoders and decoders; see [[domain#TTFS Encoding]].
3. A small temporal integration primitive is composed into multiplication, division, softmin, and nonlinear functions; see [[operators#Composed Functions]].
4. Shared spiking layers replace dense Transformer operations while preserving parameter shapes; see [[utils/transformers/models/spiking_ops.py#SpikingLinear]].
5. ViT, BERT, RoBERTa, and GPT-2 adapters expose Hugging Face-compatible task models; see [[models#Supported Model Families]].
6. Evaluation scripts select a conventional or spiking backend and record fidelity, task metrics, clamping, and robustness; see [[evaluation#Backend Comparison]].

## Main Execution Flow

An evaluation starts from a pretrained checkpoint and converts only the chosen Transformer internals into the project’s operator representation.

The common flow is:

1. An entry point parses backend and ablation settings, loads a dataset and pretrained checkpoint, and registers the optional attention backend.
2. Embeddings remain ordinary tensors until an encoder creates one `Potential` carrying the tensor and its initial bounds.
3. Transformer layers propagate `Potential` values through spiking linear, attention, activation, normalization, and residual operations.
4. The final model unwraps `Potential.value`; task heads and Hugging Face output objects remain conventional.
5. The runner computes accuracy or perplexity and writes diagnostics to W&B, TensorBoard, and optional local artifacts.

ViT’s complete orchestration is implemented by [[scripts/evaluation/error_analysis_vit.py#evaluate_vit_model]]. Text classification and language modeling use parallel flows in [[scripts/evaluation/error_analysis_bert.py#evaluate_bert_model]], [[scripts/evaluation/error_analysis_roberta.py#evaluate_roberta_model]], and [[scripts/evaluation/error_analysis_gpt2.py#evaluate_gpt2_model]].

## Data and Control Planes

Numeric values travel through `Potential` objects, while global experiment configuration controls attention selection, noise injection, and diagnostic instrumentation.

The data plane is explicit: a tensor is paired with bounds, transformed into spike events when an operator requires them, and read back as a bounded potential at $T_{\mathrm{obs}}$. The control plane is less local: Hugging Face configs select modules, while current and legacy noise paths use process-wide settings.

This distinction matters for concurrency. Ordinary model computation is per-instance, but global noise configuration and clamp logging are process-global mutable state; scoped experiments must not assume thread-safe or `DataParallel`-safe configuration changes.

## Legacy Discrete-Time Subsystem

The Jeffress/correlation code is an exploratory timestep simulation path and is not part of the current algebraic Transformer forward pass.

[[utils/model.py#L2Net]] builds a stateful SpikingJelly network over explicit time axes, delay filters, synapse filters, and LIF neurons. [[utils/model.py#AbstractL2Net]] and [[utils/model.py#AbstractExpSubNet]] provide tensor-level surrogates for related distance and exponential-difference experiments.

This path uses datasets and loaders under `utils/datasets.py` and `utils/load.py`. It should be treated as historical research support unless an experiment imports it explicitly; the current `error_analysis_*.py` runners use [[models#Model Integration]] instead.

## Architectural Invariants

The implementation relies on a small set of invariants that must remain visible when operators or model families are extended.

- Signed activations are bounded before encoders that require finite TTFS windows.
- Logarithmic encoding receives a strictly positive declared domain.
- Residual additions combine bounds with interval arithmetic.
- Attention masks suppress positions before normalization rather than relying on a post-softmax correction.
- Pretrained weight and bias tensor shapes remain compatible with the corresponding Hugging Face modules.
- Noise-free operator paths should be assessed separately from stochastic non-idealities; see [[decisions#Separate Deterministic Fidelity from Noise]].
