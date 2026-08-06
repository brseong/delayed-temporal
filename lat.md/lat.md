This directory defines the high-level concepts, business logic, and architecture of this project using markdown. It is managed by [lat.md](https://www.npmjs.com/package/lat.md) — a tool that anchors source code to these definitions. Install the `lat` command with `npm i -g lat.md` and run `lat --help`.

- [[architecture]] — System boundaries, layers, execution flow, and the legacy discrete-time subsystem.
- [[domain]] — Potentials, bounds, TTFS encodings, scale parameters, and finite-window semantics.
- [[operators]] — Primitive temporal integration and the composite Transformer operator vocabulary.
- [[models]] — Hugging Face model-family adapters, checkpoint compatibility, and ablation controls.
- [[decisions]] — Rationale and trade-offs behind the project’s major architectural choices.
- [[noise]] — Computational jitter, drop/insertion, mismatch, injection scope, and interpretation limits.
- [[evaluation]] — Experiment entry points, metrics, diagnostics, sweeps, and verification boundaries.
