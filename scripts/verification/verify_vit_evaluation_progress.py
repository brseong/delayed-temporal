"""Check ordinary ViT evaluation accuracy logs without model execution."""

from __future__ import annotations

import ast
from contextlib import redirect_stdout
from io import StringIO
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
EVALUATOR = ROOT / "scripts/evaluation/error_analysis_vit.py"
PREFIX = "Evaluation progress — "


def evaluator_tree() -> ast.Module:
    return ast.parse(EVALUATOR.read_text())


def function_node(name: str) -> ast.FunctionDef:
    return next(node for node in evaluator_tree().body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def execute_nodes(nodes: list[ast.stmt], namespace: dict) -> None:
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(EVALUATOR), "exec"), namespace)


def load_helper():
    namespace = {"json": json, "math": math}
    execute_nodes([function_node("log_evaluation_progress")], namespace)
    return namespace["log_evaluation_progress"]


def arguments(**overrides) -> dict:
    values = {
        "experiment_name": "cifar10_test", "backend": "spiking",
        "completed_batches": 1, "total_batches": 3, "correct": 31,
        "evaluated_samples": 32, "expected_samples": 65,
        "elapsed_seconds": 12.34567,
    }
    return {**values, **overrides}


def recorded_progress(**overrides) -> tuple[str, dict]:
    with patch("builtins.print") as printer:
        load_helper()(**arguments(**overrides))
    printer.assert_called_once()
    positional, keywords = printer.call_args
    assert len(positional) == 1 and keywords == {"flush": True}
    line = positional[0]
    assert line.startswith(PREFIX) and "\n" not in line and "\r" not in line
    payload = json.loads(line.removeprefix(PREFIX))
    assert line == PREFIX + json.dumps(payload, sort_keys=True, allow_nan=False)
    return line + "\n", payload


def verify_cumulative_accuracy_and_flush() -> None:
    for backend in ("hf", "spiking"):
        for index, (correct, count) in enumerate(((31, 32), (61, 64), (61, 65)), 1):
            _, payload = recorded_progress(
                backend=backend, completed_batches=index, correct=correct,
                evaluated_samples=count,
            )
            expected = arguments(backend=backend, completed_batches=index,
                                 correct=correct, evaluated_samples=count)
            expected.update(
                status="partial", accuracy=correct / count, elapsed_seconds=12.346,
                estimated_remaining_seconds=round(12.34567 * (65 - count) / count, 3),
            )
            assert payload == expected
            assert 0.0 <= payload["accuracy"] <= 1.0
            if count == 65:
                assert payload["estimated_remaining_seconds"] == 0
                assert payload["status"] == "partial"
    _, zero = recorded_progress(correct=0, elapsed_seconds=0.0)
    assert zero["accuracy"] == zero["estimated_remaining_seconds"] == 0.0


def verify_invalid_progress() -> None:
    bad = [
        {"completed_batches": 0}, {"completed_batches": 4}, {"total_batches": 0},
        {"correct": -1}, {"correct": 33}, {"evaluated_samples": 0},
        {"evaluated_samples": 66}, {"expected_samples": 31},
        {"elapsed_seconds": -0.01}, {"elapsed_seconds": math.inf},
        {"elapsed_seconds": -math.inf}, {"elapsed_seconds": math.nan},
    ]
    for field in ("completed_batches", "total_batches", "correct",
                  "evaluated_samples", "expected_samples"):
        bad.extend(({field: True}, {field: 1.0}))
    helper = load_helper()
    for overrides in bad:
        with patch("builtins.print") as printer:
            try:
                helper(**arguments(**overrides))
            except ValueError:
                pass
            else:
                raise AssertionError(f"Invalid progress accepted: {overrides}")
            printer.assert_not_called()


def verify_batch_limits_and_terminal_output() -> None:
    body = function_node("evaluate_vit_model").body
    start = next(i for i, node in enumerate(body) if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "evaluation_total_batches"
                         for target in node.targets))
    loop = next(node for node in body[start:] if isinstance(node, ast.For))
    setup = body[start:body.index(loop)]

    class Loader:
        def __init__(self, samples: int):
            self.dataset = range(samples)

        def __len__(self):
            return math.ceil(len(self.dataset) / 32)

    for samples, cap, batches, expected in (
        (65, 0, 3, 65), (65, 2, 2, 64), (65, 10, 3, 65), (31, 2, 1, 31),
    ):
        namespace = {
            "dataloader": Loader(samples), "batch_size": 32, "benchmark_enabled": False,
            "args": SimpleNamespace(max_eval_batches=cap),
            "time": SimpleNamespace(monotonic=lambda: 123.0),
        }
        execute_nodes(setup, namespace)
        assert namespace["evaluation_total_batches"] == batches
        assert namespace["evaluation_expected_samples"] == expected
        assert namespace["evaluation_started_at"] == 123.0
    progress_bars = [node for node in ast.walk(loop.iter)
                     if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                     and node.func.id == "tqdm"]
    assert len(progress_bars) == 1
    disable = next(item.value for item in progress_bars[0].keywords if item.arg == "disable")
    for terminal in (False, True):
        actual = eval(compile(ast.Expression(disable), str(EVALUATOR), "eval"),
                      {"sys": SimpleNamespace(stderr=SimpleNamespace(isatty=lambda: terminal))})
        assert actual is not terminal


def verify_loop_with_tracking_disabled() -> None:
    function = function_node("evaluate_vit_model")
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "log_evaluation_progress"]
    assert len(calls) == 1
    call = calls[0]
    for name in ("correct_count", "evaluated_count"):
        updates = [node for node in ast.walk(function) if isinstance(node, ast.AugAssign)
                   and isinstance(node.target, ast.Name) and node.target.id == name]
        assert len(updates) == 1 and updates[0].lineno < call.lineno
    guard = next(node for node in ast.walk(function) if isinstance(node, ast.If)
                 and ast.unparse(node.test) == "measured_batch and evaluated_count"
                 and call in list(ast.walk(node)))
    inner = next(node for node in guard.body if isinstance(node, ast.If)
                 and call in list(ast.walk(node)))
    assert ast.unparse(inner.test) == "not benchmark_enabled"
    assert not any(isinstance(node, ast.Name) and node.id in {"wandb", "tb_writer"}
                   for node in ast.walk(function_node("log_evaluation_progress")))
    for backend in ("hf", "spiking"):
        for benchmark in (False, True):
            output = StringIO()
            namespace = {
                "args": SimpleNamespace(experiment_name="tracking_disabled", tensorboard=False),
                "model_backend": backend, "measured_batch": True,
                "benchmark_enabled": benchmark, "batch_index": 1,
                "correct_count": 61, "evaluated_count": 64,
                "evaluation_total_batches": 3, "evaluation_expected_samples": 65,
                "evaluation_started_at": 100.0,
                "time": SimpleNamespace(monotonic=lambda: 112.0),
                "wandb": SimpleNamespace(log=lambda value: None),
                "log_evaluation_progress": load_helper(),
            }
            with patch.dict(os.environ, {"WANDB_MODE": "disabled"}), redirect_stdout(output):
                execute_nodes([guard], namespace)
            if benchmark:
                assert output.getvalue() == ""
            else:
                lines = output.getvalue().splitlines()
                assert len(lines) == 1 and lines[0].startswith(PREFIX)
                payload = json.loads(lines[0].removeprefix(PREFIX))
                assert payload["accuracy"] == 61 / 64
                assert payload["backend"] == backend and payload["completed_batches"] == 2
                assert payload["expected_samples"] == 65


def verify_final_parser_requires_final_records() -> None:
    from scripts.experiments import vit_comparison as contract
    from scripts.verification.verify_vit_comparison_runner import (
        experiment_fixture, log_fixture,
    )

    with tempfile.TemporaryDirectory(prefix="vit-progress-verification-") as temporary:
        root = Path(temporary)
        experiment = experiment_fixture(root)
        for kind in ("dense", "spiking"):
            task = contract.make_task(experiment, "cifar10_vit_small", kind, 32,
                                      calibration_sha256="c" * 64 if kind == "spiking" else "")
            log = root / task["log_file"]
            log.parent.mkdir(parents=True, exist_ok=True)
            full = log_fixture(experiment, task)
            first, _ = recorded_progress(backend=task["backend"], expected_samples=10000)
            last, _ = recorded_progress(
                backend=task["backend"], completed_batches=313, total_batches=313,
                correct=8000, evaluated_samples=10000, expected_samples=10000,
            )
            progress = first + last
            log.write_text(full)
            expected = contract.parsed_result(task, root)
            log.write_text(progress + full + progress)
            assert contract.parsed_result(task, root) == expected

            final_lines = [line for line in full.splitlines(keepends=True)
                           if line.startswith(("Correct: ", "Evaluated samples: ",
                                               "Prediction SHA256: ", "Accuracy: "))]
            assert len(final_lines) == 4
            partial = full
            for line in final_lines:
                partial = partial.replace(line, "")
            invalid = [progress, partial + progress]
            invalid.extend(full.replace(line, "") + progress for line in final_lines)
            invalid.extend(full + progress + line for line in final_lines)
            for text in invalid:
                log.write_text(text)
                try:
                    contract.parsed_result(task, root)
                except ValueError:
                    pass
                else:
                    raise AssertionError("Incomplete or duplicated final records were accepted")


def main() -> None:
    for verify in (
        verify_cumulative_accuracy_and_flush, verify_invalid_progress,
        verify_batch_limits_and_terminal_output, verify_loop_with_tracking_disabled,
        verify_final_parser_requires_final_records,
    ):
        verify()
        print(f"PASS {verify.__name__}")
    print("ViT evaluation progress: five verification groups passed")


if __name__ == "__main__":
    main()
