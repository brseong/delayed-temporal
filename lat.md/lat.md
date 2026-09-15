This directory defines the high-level concepts, business logic, and architecture of this project using markdown. It is managed by [lat.md](https://www.npmjs.com/package/lat.md) — a tool that anchors source code to these definitions. Install the `lat` command with `npm i -g lat.md` and run `lat --help`.

- [[architecture]] — System boundaries, layers, execution flow, and the legacy discrete-time subsystem.
- [[domain]] — Potentials, bounds, TTFS encodings, scale parameters, and finite-window semantics.
- [[operators]] — Primitive temporal integration and the composite Transformer operator vocabulary.
- [[models]] — Hugging Face model-family adapters, checkpoint compatibility, and ablation controls.
- [[decisions]] — Rationale and trade-offs behind the project’s major architectural choices.
- [[noise]] — Gaussian event timing, fixed-deadline potential readout, static non-idealities, and verification.
- [[calibration]] — Deterministic layer-wise collection, immutable ranges, clipping accounting, and strict persistence.
- [[vit-calibration-policy2]] — ViT 범위 전달, LayerNorm 내부 calibration, 구버전 결과 보존과 짧은 실행 검증.
- [[evaluation]] — Experiment entry points, metrics, diagnostics, sweeps, and verification boundaries.
- [[comparison-costs]] — ViT comparison SOP derivation and source audit.
- [[conversion-comparison]] — Four-model calibrated ViT comparison and execution contract.
- [[bounds-audit]] — Complete audit of fixed potential ranges, runtime extrema, calibration cases, and migration formulas.
- [[todo]] — Required manuscript and implementation work, including the only active experiment checklist.
- [[deferred-experiments]] — Optional experiment ideas with explicit promotion conditions and no active checkboxes.
