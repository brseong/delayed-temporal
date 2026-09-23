"""Reject any executable reintroduction of the removed global range setting."""

from __future__ import annotations

import ast
import contextlib
import io
import inspect
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


FORBIDDEN_NAMES = {"theta", "attention_theta", "selected_theta"}
PRODUCTION_ROOTS = (
    ROOT / "utils" / "transforms",
    ROOT / "utils" / "transformers",
    ROOT / "scripts" / "analysis",
    ROOT / "scripts" / "evaluation",
    ROOT / "scripts" / "experiments",
    ROOT / "scripts" / "setup",
)
def _terminal_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _scan_python(path: Path) -> list[str]:
    """Return executable identifiers that could restore a global range knob."""
    relative = path.relative_to(ROOT).as_posix()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            arguments = (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
            for argument in arguments:
                if argument.arg in FORBIDDEN_NAMES:
                    violations.append(f"{relative}:{argument.lineno}:argument:{argument.arg}")
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            for target in targets:
                for candidate in ast.walk(target):
                    name = _terminal_name(candidate)
                    if name in FORBIDDEN_NAMES:
                        violations.append(f"{relative}:{candidate.lineno}:assignment:{name}")
        elif isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg in FORBIDDEN_NAMES:
                    violations.append(f"{relative}:{keyword.value.lineno}:keyword:{keyword.arg}")
            if _terminal_name(node.func) == "add_argument":
                for argument in node.args:
                    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                        if argument.value == "--theta" or "attention-theta" in argument.value:
                            violations.append(
                                f"{relative}:{argument.lineno}:cli:{argument.value}"
                            )
    return violations


# @lat: [[calibration#Layer-wise Calibration#Global Range Removal]]
def verify_production_surface() -> None:
    """Active code exposes no global range argument, attribute, assignment, or CLI."""
    violations: list[str] = []
    for root in PRODUCTION_ROOTS:
        for path in sorted(root.rglob("*.py")):
            violations.extend(_scan_python(path))
    assert not violations, "global theta was reintroduced:\n" + "\n".join(violations)


def verify_legacy_config_rejection() -> None:
    """All maintained model families fail closed on old serialized settings."""
    # Vendored Transformers emits unrelated auto-docstring diagnostics while
    # importing the local RoBERTa class. Keep this verifier focused on the
    # configuration behavior it actually asserts.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from utils.transformers.models.spiking_bert.configuration_bert import BertConfig
        from utils.transformers.models.spiking_gpt2.configuration_gpt2 import GPT2Config
        from utils.transformers.models.spiking_roberta.configuration_roberta import RobertaConfig
        from utils.transformers.models.spiking_vit.configuration_spiking_vit import ViTConfig

    for config_type, kwargs in (
        (ViTConfig, {"theta": 40.0}),
        (BertConfig, {"theta": 40.0}),
        (RobertaConfig, {"theta": 40.0}),
        (GPT2Config, {"theta": 40.0}),
        (GPT2Config, {"attention_theta": 40.0}),
    ):
        try:
            config_type(**kwargs)
        except TypeError:
            pass
        else:
            raise AssertionError(f"{config_type.__name__} accepted legacy {tuple(kwargs)}")


def verify_operator_signatures() -> None:
    """Public temporal compositions consume declared bounds, never a global scalar."""
    from utils.transforms import functions
    from utils.transformers.integrations import spiking_sdpa_attention
    from utils.transformers.models import spiking_ops

    callables = (
        functions.multiplication_operator,
        functions.scaled_dot_product_function,
        functions.gelu_approximation,
        functions.tanh,
        functions.swiglu_function,
        spiking_sdpa_attention.spiking_scaled_dot_product_attention,
        spiking_ops.SpikingLayerNorm,
        spiking_ops.SpikingLinear,
        spiking_ops.SpikingConv2d,
    )
    for callable_object in callables:
        names = set(inspect.signature(callable_object).parameters)
        assert names.isdisjoint(FORBIDDEN_NAMES), (callable_object, names)


def verify_evaluator_help() -> None:
    """Maintained evaluator parsers reject the former command-line setting."""
    for filename in (
        "error_analysis_vit.py",
        "error_analysis_bert.py",
        "error_analysis_roberta.py",
        "error_analysis_gpt2.py",
    ):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "evaluation" / filename), "--help"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        assert "--theta" not in result.stdout


def verify_maintained_guidance() -> None:
    """Examples and executable notebook cells cannot advertise the retired knob."""
    forbidden_fragments = (
        "--theta", "attention_theta", "mismatch_theta", "selected_theta",
        "wandb-theta-std", "df['theta']", 'df["theta"]',
    )
    violations: list[str] = []
    for path in (
        ROOT / "README.md", ROOT / "AGENTS.md",
        ROOT / "scripts" / "experiments" / "README.md",
    ):
        text = path.read_text(encoding="utf-8")
        for fragment in forbidden_fragments:
            if fragment in text:
                violations.append(f"{path.relative_to(ROOT)}:{fragment}")

    import json

    for path in sorted((ROOT / "scripts" / "notebooks").glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook.get("cells", ())):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", ()))
            for fragment in forbidden_fragments:
                if fragment in source:
                    violations.append(
                        f"{path.relative_to(ROOT)}:cell-{index}:{fragment}"
                    )
    assert not violations, "maintained guidance restores global theta:\n" + "\n".join(violations)


def main() -> None:
    verify_production_surface()
    verify_legacy_config_rejection()
    verify_operator_signatures()
    verify_evaluator_help()
    verify_maintained_guidance()
    print("Global theta removal verification passed")


if __name__ == "__main__":
    main()
