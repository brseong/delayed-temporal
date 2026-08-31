"""Validate and summarize the GPT-2 floating-point precision controls."""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.transformers.integrations.spiking_sdpa_attention import (
    attention_score_representability_bounds,
)


_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_LOSS = re.compile(rf"^Average Loss: (?P<value>{_FLOAT})$", re.MULTILINE)
_PPL = re.compile(rf"^Perplexity: (?P<value>{_FLOAT})$", re.MULTILINE)
_DTYPE = re.compile(r"^Floating dtype: (?P<value>float(?:32|64))$", re.MULTILINE)
_CONFIG = re.compile(
    rf"^Spiking config - .*?theta:(?P<theta>{_FLOAT}), "
    rf"attention_theta:(?P<attention_theta>{_FLOAT}),",
    re.MULTILINE,
)
_CLAMP = re.compile(
    r"^Clamp\[(?P<site>[^]]+)] values=(?P<values>\d+), "
    r"underflows=(?P<underflows>\d+) .*?overflows=(?P<overflows>\d+)",
    re.MULTILINE,
)


@dataclass(frozen=True)
class ExpectedRun:
    """One required appendix condition and its exact evaluator configuration."""

    name: str
    backend: str
    dtype: str
    attention_theta: float | None


@dataclass(frozen=True)
class Result:
    """One completed task metric plus derived numerical controls."""

    name: str
    backend: str
    dtype: str
    attention_theta: float | None
    timestamp_ulp: float | None
    score_radius: float | None
    loss: float
    perplexity: float
    score_values: int
    score_excursions: int
    attention_payload_values: int
    attention_payload_excursions: int
    log_path: str


def expected_runs() -> list[ExpectedRun]:
    """Return the locked appendix design in presentation order."""
    runs = [
        ExpectedRun("hf_float32", "hf", "float32", None),
        ExpectedRun("wrapper_float32", "spiking-wrapper", "float32", None),
    ]
    runs.extend(
        ExpectedRun(f"float32_attn{theta}", "spiking", "float32", float(theta))
        for theta in (50, 100, 200, 500, 1000, 2000)
    )
    runs.append(ExpectedRun("float64_attn2000", "spiking", "float64", 2000.0))
    return runs


def _last_float(pattern: re.Pattern[str], text: str, label: str, path: Path) -> float:
    matches = list(pattern.finditer(text))
    if not matches:
        raise ValueError(f"missing {label} in {path}")
    return float(matches[-1].group("value"))


def parse_run(expected: ExpectedRun, log_dir: Path) -> Result:
    """Parse one log and reject incomplete or configuration-mismatched output."""
    path = log_dir / f"{expected.name}.log"
    if not path.is_file():
        raise ValueError(f"missing precision-control log: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    if "Traceback (most recent call last)" in text:
        raise ValueError(f"evaluator traceback in {path}")

    dtype_matches = list(_DTYPE.finditer(text))
    if not dtype_matches or dtype_matches[-1].group("value") != expected.dtype:
        raise ValueError(f"dtype mismatch in {path}")
    loss = _last_float(_LOSS, text, "average loss", path)
    perplexity = _last_float(_PPL, text, "perplexity", path)
    if not math.isclose(math.exp(loss), perplexity, rel_tol=5.0e-4):
        raise ValueError(f"loss/perplexity mismatch in {path}")

    timestamp_ulp = score_radius = None
    score_values = score_excursions = 0
    attention_payload_values = attention_payload_excursions = 0
    if expected.attention_theta is not None:
        configs = list(_CONFIG.finditer(text))
        if not configs:
            raise ValueError(f"missing spiking configuration in {path}")
        config = configs[-1]
        if float(config.group("theta")) != 2000.0:
            raise ValueError(f"global theta mismatch in {path}")
        if float(config.group("attention_theta")) != expected.attention_theta:
            raise ValueError(f"attention theta mismatch in {path}")

        torch_dtype = torch.float32 if expected.dtype == "float32" else torch.float64
        theta_tensor = torch.tensor(expected.attention_theta, dtype=torch_dtype)
        timestamp_ulp = float(
            torch.nextafter(theta_tensor, torch.tensor(math.inf, dtype=torch_dtype))
            - theta_tensor
        )
        score_radius = float(
            attention_score_representability_bounds(
                expected.attention_theta,
                1.0,
                128,
                torch_dtype,
            ).max
        )
        for match in _CLAMP.finditer(text):
            site = match.group("site")
            values = int(match.group("values"))
            excursions = int(match.group("underflows")) + int(
                match.group("overflows")
            )
            if site.endswith("/attn_score"):
                score_values += values
                score_excursions += excursions
            if site.rsplit("/", 1)[-1] in {
                "query",
                "key",
                "value",
                "attention_value_output",
                "softmin_weight",
                "division_result",
            }:
                attention_payload_values += values
                attention_payload_excursions += excursions
        if score_values == 0:
            raise ValueError(f"missing attention-score clamp counts in {path}")
        if attention_payload_values == 0:
            raise ValueError(f"missing attention payload clamp counts in {path}")

    return Result(
        name=expected.name,
        backend=expected.backend,
        dtype=expected.dtype,
        attention_theta=expected.attention_theta,
        timestamp_ulp=timestamp_ulp,
        score_radius=score_radius,
        loss=loss,
        perplexity=perplexity,
        score_values=score_values,
        score_excursions=score_excursions,
        attention_payload_values=attention_payload_values,
        attention_payload_excursions=attention_payload_excursions,
        log_path=str(path.resolve()),
    )


def write_csv(results: Sequence[Result], path: Path) -> None:
    """Write auditable raw and derived values without rounding away precision."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "condition",
                "backend",
                "dtype",
                "attention_theta",
                "timestamp_ulp",
                "score_radius",
                "loss",
                "perplexity",
                "score_values",
                "score_excursions",
                "score_excursion_rate",
                "attention_payload_values",
                "attention_payload_excursions",
                "log_path",
            ),
        )
        writer.writeheader()
        for result in results:
            rate = (
                result.score_excursions / result.score_values
                if result.score_values
                else ""
            )
            writer.writerow(
                {
                    "condition": result.name,
                    "backend": result.backend,
                    "dtype": result.dtype,
                    "attention_theta": result.attention_theta or "",
                    "timestamp_ulp": result.timestamp_ulp or "",
                    "score_radius": result.score_radius or "",
                    "loss": result.loss,
                    "perplexity": result.perplexity,
                    "score_values": result.score_values or "",
                    "score_excursions": (
                        result.score_excursions if result.score_values else ""
                    ),
                    "score_excursion_rate": rate,
                    "attention_payload_values": (
                        result.attention_payload_values
                        if result.attention_theta is not None
                        else ""
                    ),
                    "attention_payload_excursions": (
                        result.attention_payload_excursions
                        if result.attention_theta is not None
                        else ""
                    ),
                    "log_path": result.log_path,
                }
            )


def write_markdown(results: Sequence[Result], path: Path) -> None:
    """Write a compact appendix table with reference-relative deltas."""
    by_name = {result.name: result for result in results}
    hf = by_name["hf_float32"]
    wrapper = by_name["wrapper_float32"]
    lines = [
        "# GPT-2 floating-point precision control",
        "",
        "All runs use the complete WikiText-2 test split, batch size 16, sequence "
        "length 128, no calibration, no timing noise, and global $\\theta=2000$.",
        "",
        "| Condition | dtype | Attention $\\theta$ | timestamp ULP | score rail | Loss | PPL | $\\Delta$PPL vs HF | score excursions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        theta = "--" if result.attention_theta is None else f"{result.attention_theta:g}"
        ulp = "--" if result.timestamp_ulp is None else f"{result.timestamp_ulp:.6g}"
        rail = "--" if result.score_radius is None else f"$\\pm${result.score_radius:.6g}"
        if result.score_values:
            rate = result.score_excursions / result.score_values
            excursions = f"{result.score_excursions:,}/{result.score_values:,} ({rate:.6%})"
        else:
            excursions = "--"
        lines.append(
            f"| {result.name} | {result.dtype} | {theta} | {ulp} | {rail} | "
            f"{result.loss:.4f} | {result.perplexity:.4f} | "
            f"{result.perplexity - hf.perplexity:+.4f} | {excursions} |"
        )
    lines.extend(
        [
            "",
            f"The local-wrapper integration gap is {wrapper.perplexity - hf.perplexity:+.4f} PPL. "
            "The float32 sweep holds the softmin execution score rail fixed, so its task trend "
            "tests timestamp subtraction precision independently of tail trimming. "
            "Query, key, value, normalized-weight, division-result, and attention-output "
            "rails record zero excursions at every sweep point. "
            "The float64 endpoint is a broader numerical-reference control because it "
            "also raises the exponent representability ceiling.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse input and output paths for the deterministic summarizer."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    """Validate every required run before emitting either result artifact."""
    args = parse_arguments()
    results = [parse_run(run, args.log_dir) for run in expected_runs()]
    float32_radii = {
        result.score_radius
        for result in results
        if result.backend == "spiking" and result.dtype == "float32"
    }
    if len(float32_radii) != 1:
        raise ValueError("float32 theta sweep does not preserve one score rail")
    if any(
        result.attention_payload_excursions
        for result in results
        if result.backend == "spiking"
    ):
        raise ValueError("attention payload rails have excursions in precision sweep")
    write_csv(results, args.csv)
    write_markdown(results, args.markdown)
    print(f"Wrote {args.csv} and {args.markdown}")


if __name__ == "__main__":
    main()
