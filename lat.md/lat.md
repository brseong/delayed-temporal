This directory defines the high-level concepts, business logic, and architecture of this project using markdown. It is managed by [lat.md](https://www.npmjs.com/package/lat.md) — a tool that anchors source code to these definitions. Install the `lat` command with `npm i -g lat.md` and run `lat --help`.

- [[architecture]] — System boundaries, layers, execution flow, and the legacy discrete-time subsystem.
- [[domain]] — Potentials, bounds, TTFS encodings, scale parameters, and finite-window semantics.
- [[operators]] — Primitive temporal integration and the composite Transformer operator vocabulary.
- [[models]] — Hugging Face model-family adapters, checkpoint compatibility, and ablation controls.
- [[decisions]] — Rationale and trade-offs behind the project’s major architectural choices.
- [[noise]] — Gaussian event timing, fixed-deadline potential readout, static non-idealities, and verification.
- [[calibration]] — Deterministic layer-wise collection, immutable ranges, clipping accounting, and strict persistence.
- [[evaluation]] — Experiment entry points, metrics, diagnostics, sweeps, and verification boundaries.
- [[bounds-audit]] — Complete audit of fixed potential ranges, runtime extrema, calibration cases, and migration formulas.
- [[todo]] — Deferred implementation work with explicit completion and validation requirements.
