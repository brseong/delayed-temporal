"""Prepare a reviewable paper patch from a complete validated comparison bundle.

This program never writes the paper. It prints a unified diff, allowing source
review and terminology checks before a separate authorized application step.
"""

from __future__ import annotations

import argparse
import difflib
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.analysis.summarize_vit_comparison import (
    MODEL_KEYS, comparison_rows, validate_results, verify_publication_bundle,
)


CAPTION = (
    r"Comparison with prior ANN-to-SNN conversions on image classification. "
    r"ANN/SNN are top-1 accuracy (\%); Ops.\ are estimated synaptic operations in $10^9$, "
    r"and Energy is in mJ per inference. Our rows use CIFAR-10 test 10k or fixed "
    r"ImageNet-1k validation 5k. Literature rows retain their own checkpoints and "
    r"evaluation protocols and do not support a controlled ranking of absolute accuracy. "
    r"$^{\dagger}$ SpikeZIP-TF uses quantization-aware training: 16 levels for CIFAR-10 "
    r"and 32 for ImageNet; its ANN values precede that training. SpikeZIP Ops.\ are "
    r"omitted because its power model reports spikes rather than inference SOP. "
    r"$^{\ddagger}$ ViT-B/L energies are derived from reported power times 64\,ms "
    r"in~\citet{you2024spikezip}, not measured device energy. Unsupported costs are "
    r"shown as --. Patch size is 16 throughout."
)
PROTOCOL_BEGIN = "% BEGIN GENERATED VIT COMPARISON PROTOCOL"
PROTOCOL_END = "% END GENERATED VIT COMPARISON PROTOCOL"
PROTOCOL = (
    PROTOCOL_BEGIN + "\n"
    r"\paragraph{Comparison protocol.} "
    r"The four Ours models use the same fixed $\theta=40$, float64 and global time "
    r"constant 1. Each checkpoint uses a newly collected, frozen calibration from "
    r"training seed-0 5k: min/max with 5\% range margin, 2,048 histogram bins and two "
    r"passes. ANN and SNN use identical evaluation examples and preprocessing; "
    r"noise, deadline margin and mismatch are disabled. Our Energy assumes "
    r"0.9\,pJ per SOP. The estimated TTFS mapping includes the whole network and "
    r"classification head; the evaluated classifier itself is dense. The estimate "
    r"excludes memory, control, routing, leakage and analog peripheral costs and "
    r"is not a hardware measurement. The fixed $\theta$ is not claimed to be optimal "
    r"for every checkpoint." + "\n" + PROTOCOL_END
)


def _table_span(source: str) -> tuple[int, int]:
    label = r"\label{tab:conv-acc}"
    if source.count(label) != 1:
        raise ValueError("Expected one comparison table label")
    position = source.index(label)
    start = source.rfind(r"\begin{table}", 0, position)
    end = source.find(r"\end{table}", position)
    if start < 0 or end < 0:
        raise ValueError("Cannot identify the comparison table boundary")
    if source.count(r"\begin{table}", start, end) != 1:
        raise ValueError("Ambiguous comparison table boundary")
    return start, end + len(r"\end{table}")


def _replace_caption(table: str) -> str:
    marker = r"\caption{"
    if table.count(marker) != 1:
        raise ValueError("Expected one comparison caption")
    start = table.index(marker)
    content_start = start + len(marker)
    depth = 1
    for offset in range(content_start, len(table)):
        # Escaped braces in prose do not delimit LaTeX groups.
        if offset and table[offset - 1] == "\\":
            continue
        if table[offset] == "{":
            depth += 1
        elif table[offset] == "}":
            depth -= 1
            if depth == 0:
                return table[:content_start] + CAPTION + table[offset:]
    raise ValueError("Unterminated comparison caption")


def _verify_row_order(table: str, matches: list[re.Match]) -> None:
    """Refuse a rearranged table rather than assign numbers to the wrong model."""
    tabular = table.find(r"\begin{tabular}")
    if tabular < 0:
        raise ValueError("Missing comparison tabular environment")
    seen = []
    for match in matches:
        before = table[tabular:match.start()]
        tasks = list(re.finditer(r"CIFAR-10|ImageNet-1k", before))
        architectures = list(re.finditer(r"ViT-[SBL]", before))
        if not tasks or not architectures:
            raise ValueError("Cannot associate a table row with task and architecture")
        task = "cifar10" if tasks[-1].group() == "CIFAR-10" else "imagenet"
        architecture = {"ViT-S": "small", "ViT-B": "base", "ViT-L": "large"}[architectures[-1].group()]
        seen.append(f"{task}_vit_{architecture}")
    if tuple(seen) != MODEL_KEYS:
        raise ValueError("Comparison table row order or architecture changed")


def _replace_ours(table: str, summary: list[dict]) -> str:
    pattern = re.compile(
        r"(?P<prefix>\\textbf\{Ours\}\s*&\s*Continuous\s*&\s*TTFS\s*&)"
        r"\s*[^&]+&\s*[^&]+&\s*[^&]+&\s*[^&]+?(?P<end>\\\\)"
    )
    matches = list(pattern.finditer(table))
    if len(matches) != 4:
        raise ValueError("Expected exactly four Ours rows")
    _verify_row_order(table, matches)
    by_key = {row["model_key"]: row for row in summary}
    for match, key in reversed(list(zip(matches, MODEL_KEYS, strict=True))):
        row = by_key[key]
        values = (f" {row['ann_accuracy_percent']:.2f} & {row['snn_accuracy_percent']:.2f} & "
                  f"{row['ops_billions']:.2f} & {row['energy_mj']:.1f} ")
        table = table[:match.start()] + match["prefix"] + values + match["end"] + table[match.end():]
    return table


def _correct_spikezip_costs(table: str) -> str:
    pattern = re.compile(
        r"(?P<prefix>SpikeZIP-TF\$\^\{\\dagger\}\$~\\cite\{you2024spikezip\}\s*&)"
        r"(?P<steps>\s*\d+\s*)&(?P<coding>\s*Direct\s*)&"
        r"(?P<ann>\s*[\d.]+\s*)&(?P<snn>\s*[\d.]+\s*)&"
        r"(?P<ops>[^&]+)&(?P<energy>[^&]+?)(?P<end>\\\\)"
    )
    matches = list(pattern.finditer(table))
    if len(matches) != 4:
        raise ValueError("Expected four audited SpikeZIP rows")
    expected = ((32, 99.2, 98.7), (64, 82.34, 81.45),
                (64, 83.75, 82.71), (64, 85.41, 83.82))
    energies = ("--", "--", r"$403.2^{\ddagger}$", r"$1270.4^{\ddagger}$")
    for match, condition, energy in reversed(list(zip(matches, expected, energies, strict=True))):
        actual = int(match["steps"]), float(match["ann"]), float(match["snn"])
        if actual != condition:
            raise ValueError("A literature row differs from the audited source")
        retained = "&".join(match[key] for key in ("steps", "coding", "ann", "snn"))
        replacement = match["prefix"] + retained + "& -- & " + energy + " " + match["end"]
        table = table[:match.start()] + replacement + table[match.end():]
    return table


def prepare_paper_update(source: str, bundle_dir: Path) -> str:
    """Return proposed manuscript text after verifying the complete artifact bundle."""
    provenance = verify_publication_bundle(bundle_dir)
    _, indexed = validate_results(provenance["experiment"], provenance["validated_results"])
    summary, _ = comparison_rows(provenance["experiment"], indexed)
    start, end = _table_span(source)
    table = _replace_caption(_correct_spikezip_costs(_replace_ours(source[start:end], summary)))
    updated = source[:start] + table + source[end:]
    if PROTOCOL_BEGIN in updated or PROTOCOL_END in updated:
        if updated.count(PROTOCOL_BEGIN) != 1 or updated.count(PROTOCOL_END) != 1:
            raise ValueError("Ambiguous generated comparison protocol")
        protocol_start, protocol_end = updated.index(PROTOCOL_BEGIN), updated.index(PROTOCOL_END)
        if protocol_end < protocol_start:
            raise ValueError("Invalid generated comparison protocol boundary")
        updated = updated[:protocol_start] + PROTOCOL + updated[protocol_end + len(PROTOCOL_END):]
    else:
        insertion = start + len(table)
        updated = updated[:insertion] + "\n\n" + PROTOCOL + updated[insertion:]
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--paper-source", required=True, type=Path)
    args = parser.parse_args()
    original = args.paper_source.read_text()
    proposed = prepare_paper_update(original, args.bundle_dir)
    sys.stdout.writelines(difflib.unified_diff(
        original.splitlines(keepends=True), proposed.splitlines(keepends=True),
        fromfile=str(args.paper_source), tofile=str(args.paper_source),
    ))


if __name__ == "__main__":
    main()
