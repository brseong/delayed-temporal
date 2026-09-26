#!/usr/bin/env python3
"""Audit the ICLR 2027 evidence and rebuild every manuscript result artifact."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS_ROOT = REPOSITORY_ROOT / "artifacts"
DEFAULT_PAPER_ROOT = REPOSITORY_ROOT / "paper/iclr_2027"
REPRODUCTION_ROOT = Path("reproduction/iclr_2027")
MAINTAINED_PYTHON = Path("/opt/conda/envs/dt/bin/python")
DEFAULT_PYTHON_BIN = str(
    MAINTAINED_PYTHON if MAINTAINED_PYTHON.is_file() else Path(sys.executable)
)


@dataclass(frozen=True)
class ManuscriptItem:
    """One manuscript result and the code path that owns its evidence."""

    key: str
    manuscript_item: str
    experiment_owner: str
    result_processor: str
    evidence: str
    publication_output: str
    execution_environment: str


ITEMS = (
    ManuscriptItem(
        "conversion",
        "Tables: vision and language conversion fidelity",
        "scripts/experiments/run_poseidon_local_range_paper_campaign.py",
        "scripts/analysis/summarize_local_range_paper_campaign.py",
        "results/paper_end_to_end_local_range_poseidon_v1/table_results.csv",
        "iclr2027_conference_experiment.tex",
        "poseidon1 at the recorded source revision",
    ),
    ManuscriptItem(
        "sop",
        "Table: ViT synaptic operations",
        "scripts/analysis/vit_comparison_costs.py",
        "scripts/verification/verify_vit_comparison_costs.py",
        "checkpoint config.json files",
        "iclr2027_conference_experiment.tex",
        "CPU",
    ),
    ManuscriptItem(
        "clock",
        "Figure: discrete-time simulation",
        "scripts/experiments/run_clock_driven_vit.py",
        "scripts/analysis/plot_clock_discretization.py",
        "logs/clock_driven/*full_conversion_float64_v3",
        "figures/ViT-clock-discretization.pdf",
        "the local host or the cluster at the recorded source revision",
    ),
    ManuscriptItem(
        "model_noise",
        "Figure panel: timing-noise sensitivity across models",
        "scripts/experiments/run_poseidon_screening_median_model_sweep.py; "
        "scripts/experiments/run_screening_median_text_sweep.py",
        "scripts/analysis/summarize_screening_median_text_sweep.py",
        "results/model_scale_screening_median_raw_timestamp_ed_v3; "
        "results/text_screening_median_raw_timestamp_ed_v2",
        "figures/model-timing-noise-current.pdf",
        "the local host and poseidon1 at the recorded source revision",
    ),
    ManuscriptItem(
        "depth_noise",
        "Figure panel: noise in the first K encoder blocks",
        "scripts/experiments/run_vit_bss2_depth_campaign.py",
        "scripts/analysis/summarize_vit_bss2_depth.py",
        "vit-bss2-depth-current-contract-formal-v1",
        "figures/vit-b-depth-timing-noise.pdf",
        "the local host or the cluster at the recorded source revision",
    ),
    ManuscriptItem(
        "appendix_noise",
        "Appendix figure: simulated timing-noise sweeps",
        "scripts/experiments/run_appendix_vit_noise_campaign.py",
        "scripts/analysis/summarize_local_range_paper_campaign.py --noise-only",
        "logs/appendix_vit_base_local_range_raw_timestamp_float64_v2",
        "figures/ViT-noise-eval.pdf",
        "the local host and poseidon1 at the recorded source revision",
    ),
    ManuscriptItem(
        "framework",
        "Figure: potential-time framework schematic",
        "paper/iclr_2027/figures/potential-time-framework.tex",
        "latexmk and pdftoppm",
        "the TikZ source (no experimental data)",
        "figures/potential-time-framework.pdf",
        "CPU",
    ),
)


def artifact_paths(root: Path) -> dict[str, Path]:
    """Return the authoritative completed evidence below one artifact root."""

    depth = (
        root
        / "vit-bss2-depth-current-contract-formal-v1/logs/bss2_vit_depth/"
        "vit_base_bss2_encoder_depth_noise_float64_v1/formal"
    )
    return {
        "table": root
        / "results/paper_end_to_end_local_range_poseidon_v1/table_results.csv",
        "clock_time": root
        / "logs/clock_driven/"
        "vit_base_clock_driven_imagenet500_full_conversion_float64_fine_v3",
        "clock_window": root
        / "logs/clock_driven/"
        "vit_base_clock_driven_window_steps_384_768_imagenet500_"
        "full_conversion_float64_v3",
        "clock_figure": root / "figures/ViT-clock-discretization.pdf",
        "clock_preview": root / "figures/ViT-clock-discretization.png",
        "model_noise": root / "results/model_scale_screening_median_raw_timestamp_ed_v3",
        "text_noise": root / "results/text_screening_median_raw_timestamp_ed_v2",
        "model_noise_figure": root
        / "results/text_screening_median_raw_timestamp_ed_v2/"
        "model_timing_noise_model_comparison.pdf",
        "model_noise_preview": root
        / "results/text_screening_median_raw_timestamp_ed_v2/"
        "model_timing_noise_model_comparison.png",
        "depth": depth,
        "depth_figure": depth / "depth_noise_scaled_sparse_accuracy.pdf",
        "depth_preview": depth / "depth_noise_scaled_sparse_accuracy.png",
        "depth_clean": root
        / "logs/conversion_comparison/"
        "vit_base_depth_current_noise_contract_float64_v1/"
        "vit/imagenet_vit_base/result.json",
        "appendix_manifest": root
        / "logs/appendix_vit_base_local_range_raw_timestamp_float64_v2/manifest.json",
        "appendix_baseline": root
        / "logs/conversion_comparison/"
        "vit_base_appendix_local_range_raw_timestamp_baseline_float64_v2/"
        "vit/imagenet_vit_base",
        "appendix_figure": root
        / "figures/appendix_vit_base_local_range_raw_timestamp_float64_v2/"
        "ViT-noise-eval.pdf",
        "appendix_preview": root
        / "figures/appendix_vit_base_local_range_raw_timestamp_float64_v2/"
        "ViT-noise-eval.png",
        "hardware_summary": root
        / "brainscales2-primitives/20260924T_best_median_screen_summary.json",
    }


def sha256(path: Path) -> str:
    """Hash one evidence or publication file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_paths(paths: Iterable[Path]) -> None:
    """Reject an incomplete evidence tree before running any generator."""

    missing = [path for path in paths if not path.exists()]
    if missing:
        rendered = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"required reproduction inputs are missing:\n{rendered}")


def run(command: list[str], *, cwd: Path = REPOSITORY_ROOT) -> None:
    """Run one visible reproduction command and propagate any failure."""

    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def format_delta(value: float, *, scale: float) -> str:
    """Format one manuscript table difference, normalizing signed zero."""

    rounded = round(value * scale, 2)
    if rounded == 0.0:
        return "0.00"
    return f"{rounded:+.2f}"


def expected_table_fragments(table_path: Path) -> tuple[str, ...]:
    """Derive manuscript result fragments from the completed CSV."""

    with table_path.open(newline="", encoding="utf-8") as handle:
        rows = {row["model"]: row for row in csv.DictReader(handle)}
    expected_models = {
        "cifar10_vit_small",
        "imagenet_vit_small",
        "imagenet_vit_base",
        "imagenet_vit_large",
        "roberta",
        "roberta_large",
        "gpt2",
    }
    if set(rows) != expected_models:
        raise ValueError("conversion table evidence has an unexpected model set")

    fragments: list[str] = []
    for model in (
        "cifar10_vit_small",
        "imagenet_vit_small",
        "imagenet_vit_base",
        "imagenet_vit_large",
    ):
        row = rows[model]
        ann = 100.0 * float(row["ann_metric"])
        snn = 100.0 * float(row["snn_metric"])
        delta = format_delta(float(row["snn_minus_ann"]), scale=100.0)
        fragments.append(f"{ann:.2f} & {snn:.2f} & ${delta}$")
    for model in ("roberta", "roberta_large"):
        row = rows[model]
        ann = 100.0 * float(row["ann_metric"])
        snn = 100.0 * float(row["snn_metric"])
        delta = format_delta(float(row["snn_minus_ann"]), scale=100.0)
        fragments.append(f"{ann:.2f}\\% & {snn:.2f}\\% & ${delta}$")
    row = rows["gpt2"]
    delta = format_delta(float(row["snn_minus_ann"]), scale=1.0)
    if delta == "0.00" and float(row["snn_minus_ann"]) > 0.0:
        delta = "+0.00"
    fragments.append(
        f"{float(row['ann_metric']):.2f}   & {float(row['snn_metric']):.2f}   "
        f"& ${delta}$"
    )
    return tuple(fragments)


def verify_table_binding(table_path: Path, experiment_tex: Path) -> None:
    """Require every generated conversion value to occur in the active manuscript."""

    source = experiment_tex.read_text(encoding="utf-8")
    missing = [item for item in expected_table_fragments(table_path) if item not in source]
    if missing:
        raise ValueError(f"manuscript conversion values differ: {missing}")


def paper_copy_pairs(
    paths: dict[str, Path], paper_root: Path
) -> tuple[tuple[Path, Path], ...]:
    """Pair each completed experimental figure with its active manuscript copy."""

    figures = paper_root / "figures"
    return (
        (paths["clock_figure"], figures / "ViT-clock-discretization.pdf"),
        (paths["model_noise_figure"], figures / "model-timing-noise-current.pdf"),
        (paths["depth_figure"], figures / "vit-b-depth-timing-noise.pdf"),
        (paths["appendix_figure"], figures / "ViT-noise-eval.pdf"),
    )


def verify_paper_copies(paths: dict[str, Path], paper_root: Path) -> None:
    """Reject a paper figure that does not exactly match completed evidence."""

    for evidence, publication in paper_copy_pairs(paths, paper_root):
        require_paths((evidence, publication))
        if sha256(evidence) != sha256(publication):
            raise ValueError(f"publication figure differs from evidence: {publication}")


def verify_internal(artifacts_root: Path, paper_root: Path) -> None:
    """Check the active manuscript against its completed tables and figures."""

    paths = artifact_paths(artifacts_root)
    require_paths(
        (
            paths["table"],
            paths["clock_time"] / "summary.json",
            paths["clock_window"] / "summary.json",
            paths["model_noise"] / "aggregate.csv",
            paths["text_noise"] / "text_aggregate.csv",
            paths["depth"] / "scaled_sparse_summary_manifest.json",
            paths["depth_clean"],
            paths["appendix_manifest"],
            paths["hardware_summary"],
        )
    )
    verify_table_binding(
        paths["table"], paper_root / "iclr2027_conference_experiment.tex"
    )
    verify_paper_copies(paths, paper_root)
    print("Active manuscript tables and experimental figures match completed evidence.")


VERIFIERS = (
    "scripts/verification/verify_local_range_paper_summary.py",
    "scripts/verification/verify_full_calibrated_text_comparison.py",
    "scripts/verification/verify_vit_comparison_costs.py",
    "scripts/verification/verify_sop.py",
    "scripts/verification/verify_clock_driven.py",
    "scripts/verification/verify_appendix_vit_noise_campaign.py",
    "scripts/verification/verify_screening_median_model_sweep.py",
    "scripts/verification/verify_screening_median_text_sweep.py",
    "scripts/verification/verify_vit_bss2_depth_campaign.py",
    "scripts/verification/verify_iclr_2027_reproduction.py",
)


def verify_all(artifacts_root: Path, paper_root: Path, python_bin: str) -> None:
    """Run the focused contracts after checking the active evidence binding."""

    verify_internal(artifacts_root, paper_root)
    environment = {"DELAYED_TEMPORAL_ARTIFACTS_ROOT": str(artifacts_root)}
    for verifier in VERIFIERS:
        command = [python_bin, verifier]
        print("+", " ".join(command), flush=True)
        subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            check=True,
            env={**os.environ, **environment},
        )
    run(["lat", "check"])


def copy_pair(source_prefix: Path, destination_prefix: Path) -> None:
    """Promote one verified PDF and PNG pair into the paper tree."""

    destination_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        source = source_prefix.with_suffix(suffix)
        destination = destination_prefix.with_suffix(suffix)
        require_paths((source,))
        shutil.copy2(source, destination)
        print(f"Promoted {source} -> {destination}")


def verify_generated_pair(generated_prefix: Path, evidence_prefix: Path) -> None:
    """Require a regenerated PDF and PNG pair to match its evidence copy."""

    for suffix in (".pdf", ".png"):
        generated = generated_prefix.with_suffix(suffix)
        evidence = evidence_prefix.with_suffix(suffix)
        require_paths((generated, evidence))
        if sha256(generated) != sha256(evidence):
            raise ValueError(f"regenerated figure differs from evidence: {generated}")


def render_figures(artifacts_root: Path, paper_root: Path, python_bin: str) -> None:
    """Rebuild all generated paper figures from authenticated completed evidence."""

    paths = artifact_paths(artifacts_root)
    verify_internal(artifacts_root, paper_root)
    staging = artifacts_root / REPRODUCTION_ROOT
    staging.mkdir(parents=True, exist_ok=True)

    clock_prefix = staging / "ViT-clock-discretization"
    run(
        [
            python_bin,
            "scripts/analysis/plot_clock_discretization.py",
            "--time-step-root",
            str(paths["clock_time"]),
            "--window-root",
            str(paths["clock_window"]),
            "--output-prefix",
            str(clock_prefix),
            "--output-prefix",
            str(artifacts_root / "figures/ViT-clock-discretization"),
        ]
    )
    verify_generated_pair(
        clock_prefix, artifacts_root / "figures/ViT-clock-discretization"
    )
    copy_pair(clock_prefix, paper_root / "figures/ViT-clock-discretization")

    model_output = staging / "model_noise"
    run(
        [
            python_bin,
            "scripts/analysis/summarize_screening_median_text_sweep.py",
            "--input-root",
            str(paths["text_noise"]),
            "--output-dir",
            str(model_output),
            "--vision-root",
            str(paths["model_noise"]),
        ]
    )
    verify_generated_pair(
        model_output / "model_timing_noise_model_comparison",
        paths["model_noise_figure"].with_suffix(""),
    )
    copy_pair(
        model_output / "model_timing_noise_model_comparison",
        paper_root / "figures/model-timing-noise-current",
    )

    depth_output = staging / "depth_noise"
    depth_command = [
        python_bin,
        "scripts/analysis/summarize_vit_bss2_depth.py",
        "--phase",
        "formal",
        "--input-root",
        str(paths["depth"]),
        "--output-dir",
        str(depth_output),
        "--scaled-sparse-condition",
        "screening-median",
    ]
    for scale in ("0.03", "0.05", "0.075", "0.1"):
        depth_command.extend(("--measured-noise-scale", scale))
    for block_count in ("1", "2", "4", "8", "12"):
        depth_command.extend(("--first-block-count", block_count))
    depth_command.extend(("--clean-reference-result", str(paths["depth_clean"])))
    run(depth_command)
    verify_generated_pair(
        depth_output / "depth_noise_scaled_sparse_accuracy",
        paths["depth_figure"].with_suffix(""),
    )
    copy_pair(
        depth_output / "depth_noise_scaled_sparse_accuracy",
        paper_root / "figures/vit-b-depth-timing-noise",
    )

    appendix_tag = "appendix_vit_base_local_range_raw_timestamp_float64_v2"
    run(
        [
            python_bin,
            "scripts/analysis/summarize_local_range_paper_campaign.py",
            "--artifacts-root",
            str(artifacts_root),
            "--noise-calibration-source",
            str(paths["appendix_baseline"]),
            "--noise-tag",
            "vit_base_appendix_local_range_raw_timestamp_float64_v2",
            "--output-tag",
            appendix_tag,
            "--noise-only",
            "--require-raw-timestamp-contract",
            "--campaign-manifest",
            str(paths["appendix_manifest"]),
        ]
    )
    copy_pair(
        artifacts_root / "figures" / appendix_tag / "ViT-noise-eval",
        paper_root / "figures/ViT-noise-eval",
    )

    render_framework(artifacts_root, paper_root)
    verify_internal(artifacts_root, paper_root)


def render_framework(artifacts_root: Path, paper_root: Path) -> None:
    """Build the TikZ framework figure, which does not contain experiment data."""

    output = artifacts_root / REPRODUCTION_ROOT / "framework"
    output.mkdir(parents=True, exist_ok=True)
    source = paper_root / "figures/potential-time-framework.tex"
    run(
        [
            "latexmk",
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-outdir={output}",
            str(source),
        ]
    )
    prefix = output / "potential-time-framework"
    run(
        [
            "pdftoppm",
            "-png",
            "-r",
            "300",
            "-singlefile",
            str(prefix.with_suffix(".pdf")),
            str(prefix),
        ]
    )
    copy_pair(prefix, paper_root / "figures/potential-time-framework")


def build_paper(artifacts_root: Path, paper_root: Path) -> Path:
    """Compile the manuscript after all publication artifacts are present."""

    output = artifacts_root / REPRODUCTION_ROOT / "paper"
    output.mkdir(parents=True, exist_ok=True)
    run(
        [
            "latexmk",
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-outdir={output}",
            "iclr2027_conference.tex",
        ],
        cwd=paper_root,
    )
    pdf = output / "iclr2027_conference.pdf"
    require_paths((pdf,))
    log = (output / "iclr2027_conference.log").read_text(
        encoding="utf-8", errors="replace"
    )
    forbidden = ("undefined references", "Citation `", "Reference `")
    if any(marker in log for marker in forbidden):
        raise ValueError("the manuscript build contains unresolved references")
    print(f"Built manuscript: {pdf}")
    return pdf


def print_inventory(*, as_json: bool) -> None:
    """Print the manuscript result map without touching the artifact tree."""

    payload = [item.__dict__ for item in ITEMS]
    if as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    headings = ("KEY", "MANUSCRIPT ITEM", "EXPERIMENT OWNER", "EVIDENCE")
    rows = [
        (item.key, item.manuscript_item, item.experiment_owner, item.evidence)
        for item in ITEMS
    ]
    widths = [
        max(len(headings[i]), *(len(row[i]) for row in rows)) for i in range(4)
    ]
    print("  ".join(headings[i].ljust(widths[i]) for i in range(4)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(row[i].ljust(widths[i]) for i in range(4)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "command",
        choices=("inventory", "check", "verify", "render", "build", "all"),
    )
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)
    parser.add_argument("--paper-root", type=Path, default=DEFAULT_PAPER_ROOT)
    parser.add_argument("--python-bin", default=DEFAULT_PYTHON_BIN)
    parser.add_argument("--json", action="store_true", help="Emit inventory as JSON.")
    args = parser.parse_args()
    artifacts_root = args.artifacts_root.resolve(strict=True)
    paper_root = args.paper_root.resolve(strict=True)

    if args.command == "inventory":
        print_inventory(as_json=args.json)
    elif args.command == "check":
        verify_internal(artifacts_root, paper_root)
    elif args.command == "verify":
        verify_all(artifacts_root, paper_root, args.python_bin)
    elif args.command == "render":
        render_figures(artifacts_root, paper_root, args.python_bin)
    elif args.command == "build":
        verify_internal(artifacts_root, paper_root)
        build_paper(artifacts_root, paper_root)
    else:
        verify_all(artifacts_root, paper_root, args.python_bin)
        render_figures(artifacts_root, paper_root, args.python_bin)
        verify_all(artifacts_root, paper_root, args.python_bin)
        build_paper(artifacts_root, paper_root)


if __name__ == "__main__":
    main()
