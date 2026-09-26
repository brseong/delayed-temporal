#!/usr/bin/env python3
"""Audit text or added diff lines against a terminology and notation lexicon."""

from __future__ import annotations

import argparse
from bisect import bisect_left
from dataclasses import dataclass
from fnmatch import fnmatch
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Sequence


VALID_DISPOSITIONS = frozenset({"allow", "review", "internal-only", "forbid"})
VALID_SURFACES = frozenset({"manuscript", "internal"})
DEFAULT_INCLUDES = (
    "*.md",
    "*.tex",
    "*.py",
    "*.json",
    "*.csv",
    "*.tsv",
    "*.yaml",
    "*.yml",
)
DEFAULT_EXCLUDES = (
    ".git/**",
    "**/.git/**",
    "*.pdf",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.svg",
)


class LexiconError(ValueError):
    """Raised when a lexicon does not satisfy the supported schema."""


@dataclass(frozen=True)
class Rule:
    rule_id: str
    term: str
    description: str
    disposition: str
    patterns: tuple[re.Pattern[str], ...]
    surfaces: frozenset[str]
    replacement: str | None
    source: str | None


@dataclass(frozen=True)
class CandidateRule:
    rule_id: str
    description: str
    pattern: re.Pattern[str]


@dataclass(frozen=True)
class Segment:
    path: str
    text: str
    start_line: int
    newline_offsets: tuple[int, ...]

    @classmethod
    def create(cls, path: str, text: str, start_line: int = 1) -> "Segment":
        return cls(
            path=path,
            text=text,
            start_line=start_line,
            newline_offsets=tuple(
                index for index, character in enumerate(text) if character == "\n"
            ),
        )

    def position(self, offset: int) -> tuple[int, int]:
        line_index = bisect_left(self.newline_offsets, offset)
        line_start = (
            0 if line_index == 0 else self.newline_offsets[line_index - 1] + 1
        )
        return self.start_line + line_index, offset - line_start + 1


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    column: int
    rule_id: str
    disposition: str
    description: str
    term: str | None = None
    replacement: str | None = None
    source: str | None = None
    matched_text: str | None = None


def _require_string(mapping: dict[str, Any], key: str, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise LexiconError(f"{context}.{key} must be a non-empty string")
    return value


def _compile_pattern(
    pattern: str, *, ignore_case: bool, context: str
) -> re.Pattern[str]:
    flags = re.MULTILINE
    if ignore_case:
        flags |= re.IGNORECASE
    try:
        compiled = re.compile(pattern, flags)
    except re.error as error:
        raise LexiconError(f"{context} has invalid regex: {error}") from error
    if compiled.match("") is not None:
        raise LexiconError(f"{context} must not match an empty string")
    return compiled


def _load_rules(raw_entries: Any) -> tuple[Rule, ...]:
    if not isinstance(raw_entries, list):
        raise LexiconError("entries must be a list")

    rules: list[Rule] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_entries):
        context = f"entries[{index}]"
        if not isinstance(raw, dict):
            raise LexiconError(f"{context} must be an object")

        rule_id = _require_string(raw, "id", context)
        if rule_id in seen_ids:
            raise LexiconError(f"duplicate entry id: {rule_id}")
        seen_ids.add(rule_id)

        term = _require_string(raw, "term", context)
        description = _require_string(raw, "description", context)
        disposition = _require_string(raw, "disposition", context)
        if disposition not in VALID_DISPOSITIONS:
            allowed = ", ".join(sorted(VALID_DISPOSITIONS))
            raise LexiconError(
                f"{context}.disposition must be one of: {allowed}"
            )

        raw_surfaces = raw.get("surfaces", sorted(VALID_SURFACES))
        if (
            not isinstance(raw_surfaces, list)
            or not raw_surfaces
            or not all(isinstance(item, str) for item in raw_surfaces)
        ):
            raise LexiconError(f"{context}.surfaces must be a non-empty string list")
        surfaces = frozenset(raw_surfaces)
        unknown_surfaces = surfaces - VALID_SURFACES
        if unknown_surfaces:
            names = ", ".join(sorted(unknown_surfaces))
            raise LexiconError(f"{context}.surfaces contains unknown values: {names}")

        raw_patterns = raw.get("patterns")
        if (
            not isinstance(raw_patterns, list)
            or not raw_patterns
            or not all(isinstance(item, str) and item for item in raw_patterns)
        ):
            raise LexiconError(f"{context}.patterns must be a non-empty string list")
        ignore_case = raw.get("ignore_case", False)
        if not isinstance(ignore_case, bool):
            raise LexiconError(f"{context}.ignore_case must be boolean")
        patterns = tuple(
            _compile_pattern(
                pattern,
                ignore_case=ignore_case,
                context=f"{context}.patterns[{pattern_index}]",
            )
            for pattern_index, pattern in enumerate(raw_patterns)
        )

        replacement = raw.get("replacement")
        if replacement is not None and not isinstance(replacement, str):
            raise LexiconError(f"{context}.replacement must be a string or null")
        source = raw.get("source")
        if source is not None and not isinstance(source, str):
            raise LexiconError(f"{context}.source must be a string or null")

        rules.append(
            Rule(
                rule_id=rule_id,
                term=term,
                description=description,
                disposition=disposition,
                patterns=patterns,
                surfaces=surfaces,
                replacement=replacement,
                source=source,
            )
        )
    return tuple(rules)


def _load_candidate_rules(raw_rules: Any) -> tuple[CandidateRule, ...]:
    if raw_rules is None:
        return ()
    if not isinstance(raw_rules, list):
        raise LexiconError("candidate_patterns must be a list")

    candidates: list[CandidateRule] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_rules):
        context = f"candidate_patterns[{index}]"
        if not isinstance(raw, dict):
            raise LexiconError(f"{context} must be an object")
        rule_id = _require_string(raw, "id", context)
        if rule_id in seen_ids:
            raise LexiconError(f"duplicate candidate pattern id: {rule_id}")
        seen_ids.add(rule_id)
        description = _require_string(raw, "description", context)
        regex = _require_string(raw, "regex", context)
        ignore_case = raw.get("ignore_case", False)
        if not isinstance(ignore_case, bool):
            raise LexiconError(f"{context}.ignore_case must be boolean")
        candidates.append(
            CandidateRule(
                rule_id=rule_id,
                description=description,
                pattern=_compile_pattern(
                    regex, ignore_case=ignore_case, context=f"{context}.regex"
                ),
            )
        )
    return tuple(candidates)


def load_lexicon(path: Path) -> tuple[tuple[Rule, ...], tuple[CandidateRule, ...]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise LexiconError("lexicon root must be an object")
    if raw.get("schema_version") != 1:
        raise LexiconError("schema_version must equal 1")
    return (
        _load_rules(raw.get("entries")),
        _load_candidate_rules(raw.get("candidate_patterns")),
    )


def _matches_any(path: str, patterns: Sequence[str]) -> bool:
    name = Path(path).name
    return any(fnmatch(path, pattern) or fnmatch(name, pattern) for pattern in patterns)


def _iter_files(
    paths: Sequence[Path],
    includes: Sequence[str],
    excludes: Sequence[str],
) -> Iterable[Path]:
    seen: set[Path] = set()
    for supplied_path in paths:
        if supplied_path.is_file():
            resolved = supplied_path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield supplied_path
            continue
        if not supplied_path.is_dir():
            raise FileNotFoundError(f"input path does not exist: {supplied_path}")

        for candidate in sorted(supplied_path.rglob("*")):
            if not candidate.is_file():
                continue
            relative = candidate.relative_to(supplied_path).as_posix()
            if ".git" in candidate.parts:
                continue
            if not _matches_any(relative, includes):
                continue
            if _matches_any(relative, excludes):
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield candidate


def segments_from_paths(
    paths: Sequence[Path],
    includes: Sequence[str],
    excludes: Sequence[str],
) -> list[Segment]:
    if not paths:
        return [Segment.create("<stdin>", sys.stdin.read())]
    return [
        Segment.create(str(path), path.read_text(encoding="utf-8"))
        for path in _iter_files(paths, includes, excludes)
    ]


def segments_from_unified_diff(text: str) -> list[Segment]:
    segments: list[Segment] = []
    current_path: str | None = None
    new_line: int | None = None
    added_start: int | None = None
    added_lines: list[str] = []
    hunk_pattern = re.compile(r"^@@ .*\+(\d+)(?:,(\d+))? @@")

    def flush() -> None:
        nonlocal added_start, added_lines
        if current_path is not None and added_start is not None and added_lines:
            segments.append(
                Segment.create(current_path, "\n".join(added_lines), added_start)
            )
        added_start = None
        added_lines = []

    for line in text.splitlines():
        if line.startswith("+++ "):
            flush()
            header_path = line[4:].split("\t", 1)[0]
            if header_path.startswith("b/"):
                header_path = header_path[2:]
            current_path = None if header_path == "/dev/null" else header_path
            new_line = None
            continue

        if line.startswith("@@ "):
            flush()
            match = hunk_pattern.match(line)
            if match is None:
                raise ValueError(f"invalid unified diff hunk header: {line}")
            new_line = int(match.group(1))
            continue

        if current_path is None or new_line is None:
            continue
        if line.startswith("+") and not line.startswith("+++"):
            if added_start is None:
                added_start = new_line
            added_lines.append(line[1:])
            new_line += 1
        elif line.startswith("-") and not line.startswith("---"):
            flush()
        elif line.startswith(" "):
            flush()
            new_line += 1
        elif line == "\\ No newline at end of file":
            continue
        else:
            flush()

    flush()
    return segments


def _overlaps(start: int, end: int, spans: Sequence[tuple[int, int]]) -> bool:
    return any(start < known_end and known_start < end for known_start, known_end in spans)


def audit_segments(
    segments: Sequence[Segment],
    rules: Sequence[Rule],
    candidate_rules: Sequence[CandidateRule],
    *,
    surface: str,
    include_candidates: bool,
) -> list[Finding]:
    findings: list[Finding] = []
    seen: set[tuple[str, int, int, str, str]] = set()

    for segment in segments:
        known_spans: list[tuple[int, int]] = []
        for rule in rules:
            for pattern in rule.patterns:
                for match in pattern.finditer(segment.text):
                    known_spans.append(match.span())
                    if rule.disposition == "allow" or surface not in rule.surfaces:
                        continue
                    line, column = segment.position(match.start())
                    key = (
                        segment.path,
                        line,
                        column,
                        rule.rule_id,
                        rule.disposition,
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    findings.append(
                        Finding(
                            path=segment.path,
                            line=line,
                            column=column,
                            rule_id=rule.rule_id,
                            disposition=rule.disposition,
                            description=rule.description,
                            term=rule.term,
                            replacement=rule.replacement,
                            source=rule.source,
                        )
                    )

        if include_candidates:
            for rule in candidate_rules:
                for match in rule.pattern.finditer(segment.text):
                    if _overlaps(match.start(), match.end(), known_spans):
                        continue
                    line, column = segment.position(match.start())
                    key = (segment.path, line, column, rule.rule_id, "candidate")
                    if key in seen:
                        continue
                    seen.add(key)
                    findings.append(
                        Finding(
                            path=segment.path,
                            line=line,
                            column=column,
                            rule_id=rule.rule_id,
                            disposition="candidate",
                            description=rule.description,
                            matched_text=match.group(0),
                        )
                    )

    return sorted(
        findings,
        key=lambda finding: (
            finding.path,
            finding.line,
            finding.column,
            finding.disposition,
            finding.rule_id,
        ),
    )


def _finding_record(finding: Finding) -> dict[str, Any]:
    return {
        "path": finding.path,
        "line": finding.line,
        "column": finding.column,
        "rule_id": finding.rule_id,
        "disposition": finding.disposition,
        "description": finding.description,
        "term": finding.term,
        "replacement": finding.replacement,
        "source": finding.source,
        "matched_text": finding.matched_text,
    }


def print_findings(findings: Sequence[Finding], output_format: str) -> None:
    if output_format == "json":
        print(
            json.dumps(
                [_finding_record(finding) for finding in findings],
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if not findings:
        print("OK: no terminology or notation findings")
        return

    labels = {
        "forbid": "ERROR",
        "internal-only": "ERROR",
        "review": "REVIEW",
        "candidate": "CANDIDATE",
    }
    for finding in findings:
        detail = finding.description
        if finding.term is not None:
            detail += f" Term: {finding.term}."
        if finding.replacement:
            detail += f" Preferred form: {finding.replacement}."
        if finding.source:
            detail += f" Authority: {finding.source}."
        if finding.matched_text is not None:
            detail += f" Candidate: {finding.matched_text!r}."
        label = labels[finding.disposition]
        print(
            f"{label} {finding.path}:{finding.line}:{finding.column} "
            f"[{finding.rule_id}] {detail}"
        )

    blocking = sum(finding.disposition != "candidate" for finding in findings)
    candidates = len(findings) - blocking
    print(f"Summary: {blocking} blocking finding(s), {candidates} candidate(s)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--lexicon", required=True, type=Path)
    parser.add_argument(
        "--surface",
        choices=sorted(VALID_SURFACES),
        default="manuscript",
    )
    parser.add_argument(
        "--diff",
        action="store_true",
        help="Read a unified diff from stdin and inspect added lines only.",
    )
    parser.add_argument(
        "--candidates",
        action="store_true",
        help="Also report heuristic candidates not covered by lexicon entries.",
    )
    parser.add_argument(
        "--strict-candidates",
        action="store_true",
        help="Return exit code 1 when heuristic candidates are present.",
    )
    parser.add_argument(
        "--include",
        action="append",
        help="Glob included while expanding directories; may be repeated.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        help="Glob excluded while expanding directories; may be repeated.",
    )
    parser.add_argument("--format", choices=("text", "json"), default="text")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.diff and args.paths:
        parser.error("--diff reads stdin and cannot be combined with paths")

    includes = tuple(args.include or DEFAULT_INCLUDES)
    excludes = tuple(args.exclude or DEFAULT_EXCLUDES)

    try:
        rules, candidate_rules = load_lexicon(args.lexicon)
        if args.diff:
            segments = segments_from_unified_diff(sys.stdin.read())
        else:
            segments = segments_from_paths(args.paths, includes, excludes)
        findings = audit_segments(
            segments,
            rules,
            candidate_rules,
            surface=args.surface,
            include_candidates=args.candidates or args.strict_candidates,
        )
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError, LexiconError, ValueError) as error:
        print(f"CONFIG ERROR: {error}", file=sys.stderr)
        return 2

    print_findings(findings, args.format)
    has_blocking = any(
        finding.disposition != "candidate" for finding in findings
    )
    has_candidates = any(
        finding.disposition == "candidate" for finding in findings
    )
    return int(has_blocking or (args.strict_candidates and has_candidates))


if __name__ == "__main__":
    raise SystemExit(main())
