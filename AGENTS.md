# Project instructions

This is the canonical instruction file for coding agents working in this repository. Tool-specific instruction files should import this file instead of duplicating its contents.

## Project overview

This repository contains the research code and manuscript for *Biologically Plausible Dual Operators for TTFS-Coded Analog Spiking Transformers*. It converts pretrained Transformer operations into composable time-to-first-spike (TTFS) operators and evaluates deterministic approximation error, task accuracy, operation cost, and robustness to non-idealities. The main workflow evaluates converted pretrained models; it does not train them.

The current manuscript is `paper/neurips_2026.tex`. Supporting review translations, verification notes, and improvement checklists are also kept under `paper/`.

## Environment

This paper's experiments are split across two execution environments: the local GPU host `baekryun` and the UOS UBAI supercomputing cluster reached through the `gate1`/`gate2` login hosts. Several agents work across both, so confirm which host you are on with `hostname` before running anything. Their execution rules are not interchangeable.

Shared expectations:

- Use Python 3.12 as specified by `.python-version`.
- Install dependencies with `pip install -r requirements.txt`. The requirements include editable checkouts of Hugging Face Transformers and a SpikingJelly fork.
- Evaluations may require CUDA, local datasets and checkpoints, W&B credentials, and sufficient GPU memory. Do not assume every full experiment can run in a lightweight development environment.
- Some scripts activate `./venv/bin/activate`, but a repository-local virtual environment is not guaranteed. Plain `python3` is valid when the active environment already has the dependencies.

### baekryun (local GPU host)

`hostname` returns `baekryun-cuda129`. Jobs run directly with no scheduler, so an agent must check device occupancy itself before launching anything.

- Eight NVIDIA RTX A6000 (49 GB) devices are present as indices 0--7. The maintained sweep scripts restrict themselves to GPUs 4--7 through `allowed_gpu_list` and refuse to start on an occupied device.
- The maintained interpreter is `/opt/conda/envs/dt/bin/python` (conda environment `dt`), overridable through `PYTHON_BIN`.
- Pretrained ViT checkpoints and ImageNet data are expected outside the repository under `/data/nas/` by the existing experiment scripts.

### UBAI cluster (gate1/gate2)

The cluster's canonical rules live in `/home1/sizz1997/myubai/wagner2026philtrans/AGENTS.md` on `gate1.hpc`. Follow them whenever work targets the cluster instead of the local GPU host.

- Gate nodes are login nodes. Use them only for SSH and file transfer, `git`, light editing and log inspection, and Slurm commands (`sbatch`, `squeue`, `sinfo`, `scancel`).
- Never run training, inference, builds, or bulk data processing on a gate node. Heavy processes block other users and are terminated by administrators.
- Submit every compute job to Slurm with `sbatch`. Allocate a compute node with `srun --pty bash` even for interactive debugging.
- `enroot` 3.5.0 and the Slurm `pyxis` plugin v0.23.0 exist only on compute nodes; gate nodes have neither.
- `#SBATCH --container-image` is rejected at submission because the gate `sbatch` does not load `pyxis`. Put container options on the `srun` call inside the job script instead.
- Home directories are not mounted inside containers by default. Pass `--container-mounts=$HOME:$HOME` or `--container-mount-home`. The container keeps the user's uid/gid and is not root unless `--container-remap-root` is given.
- The `/enroot` image cache is node-local, so `--container-name` reuse does not survive a different node assignment. Save frequently used images to a squashfs file under the home directory with `--container-save` from inside a submitted job, then reference that path.
- From `baekryun` the login hosts are reachable as `ssh gate1` and `ssh gate2` once `~/.ssh/config` defines them. SSH keys and host configuration are machine-local setup and must never be committed to this repository.

## Repository layout

- `utils/transforms/`: numeric domains, TTFS encoders/decoders, temporal primitives, composed operators, noise configuration, and bounds propagation.
- `utils/transformers/`: shared spiking layers and model adapters for pretrained Transformer families.
- `scripts/evaluation/`: model-family evaluation entry points.
- `scripts/experiments/`: shell wrappers for evaluation grids, ablations, quantile collection, and noise sweeps.
- `scripts/lib/`: shared shell helpers such as GPU allocation.
- `scripts/setup/`: environment and checkpoint preparation utilities.
- `scripts/verification/`: symbolic consistency checks for paper claims.
- `scripts/notebooks/`: exploratory notebooks and figure-generation notebooks.
- `analysis/`: focused local analysis and figure-generation programs.
- `artifacts/`: generated figures, W&B exports, quantiles, and experiment logs. Treat these as outputs, not source code.
- `paper/`: manuscript source, review notes, references, and publication-ready figure assets.
- `lat.md/`: structured architecture, design-decision, domain, and verification documentation.

Vendored or reference implementations such as `TTFSFormer/`, `src/transformers`, `src/spikingjelly`, and the `SpikingBERT` submodule should generally not be modified unless the task explicitly concerns them.

## Common commands

Run experiment wrappers from the repository root:

```bash
bash scripts/setup/convert_vits.sh
bash scripts/experiments/error_analysis_vit.sh
bash scripts/experiments/theta_jitter_analysis_vit.sh
bash scripts/experiments/jitter_analysis_vit.sh
bash scripts/experiments/ablation_gelu_vit.sh
bash scripts/experiments/error_analysis_bert.sh sst2
```

A short ViT evaluation can be launched directly:

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluation/error_analysis_vit.py \
  --experiment_name smoke --model_backend spiking \
  --model_id /data/nas/vit_small_patch16_224.augreg_in21k_ft_in1k \
  --dataset_id imagenet-1k --batch_size 32 --theta 2000 \
  --spiking-layernorm --spiking-mlp --spiking-attention \
  --max_eval_batches 5
```

`--quick_test` limits the dataset, and `--max_eval_batches N` limits evaluation batches. `--model_backend hf` selects the dense baseline. Boolean spiking flags support `--no-...` negation through `BooleanOptionalAction`.

After changing operator definitions or paper operation-count tables, run:

```bash
python3 scripts/verification/verify_sop.py
```

This checker verifies the internal arithmetic of the stated operation-cost model. It does not establish physical circuit feasibility or validate system-level energy assumptions.

## Architecture and design constraints

The maintained path has three layers:

1. `utils/transforms/` defines bounded values and the TTFS operator algebra. `Potential` pairs a tensor with `PotentialBounds`; bounds propagation is part of the operator contract.
2. `utils/transformers/` integrates those operators into pretrained-model-compatible layers and model families.
3. `scripts/evaluation/` and `scripts/experiments/` select backends, configure ablations and noise, run datasets, and record metrics.

Preserve these invariants when making changes:

- Keep tensors and their declared bounds synchronized through every operator.
- Respect finite and positive domains required by TTFS and logarithmic encoders.
- Keep residual bound propagation explicit.
- Preserve pretrained parameter shapes and Hugging Face-compatible task outputs.
- Separate deterministic approximation and clamping effects from stochastic noise effects in code and reporting.
- Treat global noise configuration and clamp logging as mutable process-wide state; do not assume scoped changes are thread-safe or `DataParallel`-safe.
- Match validation effort to the changed layer. Operator changes need reference-value and boundary checks; noise changes need seeded distribution and injection-scope checks; model changes need at least a smoke evaluation when dependencies permit.

The `theta` threshold controls the representable potential interval, and out-of-range values are clamped. Quantile collection writes calibration data under `artifacts/quantiles/`. W&B exports used by notebooks live under `artifacts/wandb/`, generated plots under `artifacts/figures/`, and publication copies under `paper/figures/`.

## Coding conventions

- Run commands from the repository root unless a script documents otherwise.
- Keep new paths consistent with the repository layout above; do not restore generated scripts, notebooks, logs, CSV exports, or figures to the top level.
- Comments and diagnostic output may be Korean or English. Follow the surrounding file's language and style.
- Add type annotations where practical.
- Preserve user-generated artifacts and unrelated worktree changes.

## Paper-review discussions

- When a paper-review or mathematical-verification answer contains several equations, a long derivation, or multiple technical cases, write the detailed material into the relevant Markdown note under `paper/` instead of presenting the full derivation only in chat.
- Use Markdown/LaTeX math syntax (`$...$` and `$$...$$`) for equations. Do not put mathematical expressions in fenced code blocks unless the user explicitly requests plain-text math.
- Keep sequential reviewer-issue verification in `paper/reviewer_technical_verification_notes_ko.md` and improvement actions in `paper/neurips_2026_review_checklist_ko.md` unless the user requests another file.
- In chat, give only a concise conclusion and a clickable link to the detailed note. Continue discussing reviewer issues one at a time.

%% lat:begin %%
# Before starting work

- Run `lat search` to find sections relevant to your task. Read them to understand the design intent before writing code.
- Run `lat expand` on user prompts to expand any `[[refs]]` — this resolves section names to file locations and provides context.

# Post-task checklist (REQUIRED — do not skip)

After EVERY task, before responding to the user:

- [ ] Update `lat.md/` if you added or changed any functionality, architecture, tests, or behavior
- [ ] Run `lat check` — all wiki links and code refs must pass
- [ ] Do not skip these steps. Do not consider your task done until both are complete.

---

# What is lat.md?

This project uses [lat.md](https://www.npmjs.com/package/lat.md) to maintain a structured knowledge graph of its architecture, design decisions, and test specs in the `lat.md/` directory. It is a set of cross-linked markdown files that describe **what** this project does and **why** — the domain concepts, key design decisions, business logic, and test specifications. Use it to ground your work in the actual architecture rather than guessing.

# Commands

```bash
lat locate "Section Name"      # find a section by name (exact, fuzzy)
lat refs "file#Section"        # find what references a section
lat search "natural language"  # semantic search across all sections
lat expand "user prompt text"  # expand [[refs]] to resolved locations
lat check                      # validate all links and code refs
```

Run `lat --help` when in doubt about available commands or options.

If `lat search` fails because no API key is configured, explain to the user that semantic search requires a key provided via `LAT_LLM_KEY` (direct value), `LAT_LLM_KEY_FILE` (path to key file), or `LAT_LLM_KEY_HELPER` (command that prints the key). Supported key prefixes: `sk-...` (OpenAI) or `vck_...` (Vercel). If the user doesn't want to set it up, use `lat locate` for direct lookups instead.

# Syntax primer

- **Section ids**: `lat.md/path/to/file#Heading#SubHeading` — full form uses project-root-relative path (e.g. `lat.md/tests/search#RAG Replay Tests`). Short form uses bare file name when unique (e.g. `search#RAG Replay Tests`, `cli#search#Indexing`).
- **Wiki links**: `[[target]]` or `[[target|alias]]` — cross-references between sections. Can also reference source code: `[[src/foo.ts#myFunction]]`.
- **Source code links**: Wiki links in `lat.md/` files can reference functions, classes, constants, and methods in TypeScript/JavaScript/Python/Rust/Go/C files. Use the full path: `[[src/config.ts#getConfigDir]]`, `[[src/server.ts#App#listen]]` (class method), `[[lib/utils.py#parse_args]]`, `[[src/lib.rs#Greeter#greet]]` (Rust impl method), `[[src/app.go#Greeter#Greet]]` (Go method), `[[src/app.h#Greeter]]` (C struct). `lat check` validates these exist.
- **Code refs**: `// @lat: [[section-id]]` (JS/TS/Rust/Go/C) or `# @lat: [[section-id]]` (Python) — ties source code to concepts

# Test specs

Key tests can be described as sections in `lat.md/` files (e.g. `tests.md`). Add frontmatter to require that every leaf section is referenced by a `// @lat:` or `# @lat:` comment in test code:

```markdown
---
lat:
  require-code-mention: true
---
# Tests

Authentication and authorization test specifications.

## User login

Verify credential validation and error handling for the login endpoint.

### Rejects expired tokens
Tokens past their expiry timestamp are rejected with 401, even if otherwise valid.

### Handles missing password
Login request without a password field returns 400 with a descriptive error.
```

Every section MUST have a description — at least one sentence explaining what the test verifies and why. Empty sections with just a heading are not acceptable. (This is a specific case of the general leading paragraph rule below.)

Each test in code should reference its spec with exactly one comment placed next to the relevant test — not at the top of the file:

```python
# @lat: [[tests#User login#Rejects expired tokens]]
def test_rejects_expired_tokens():
    ...

# @lat: [[tests#User login#Handles missing password]]
def test_handles_missing_password():
    ...
```

Do not duplicate refs. One `@lat:` comment per spec section, placed at the test that covers it. `lat check` will flag any spec section not covered by a code reference, and any code reference pointing to a nonexistent section.

# Section structure

Every section in `lat.md/` **must** have a leading paragraph — at least one sentence immediately after the heading, before any child headings or other block content. The first paragraph must be ≤250 characters (excluding `[[wiki link]]` content). This paragraph serves as the section's overview and is used in search results, command output, and RAG context — keeping it concise guarantees the section's essence is always captured.

```markdown
# Good Section

Brief overview of what this section documents and why it matters.

More detail can go in subsequent paragraphs, code blocks, or lists.

## Child heading

Details about this child topic.
```

```markdown
# Bad Section

## Child heading

Details about this child topic.
```

The second example is invalid because `Bad Section` has no leading paragraph. `lat check` validates this rule and reports errors for missing or overly long leading paragraphs.
%% lat:end %%
