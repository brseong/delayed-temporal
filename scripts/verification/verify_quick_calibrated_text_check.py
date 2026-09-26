"""Verify short text comparison commands, admission and strict result parsing on CPU."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments import quick_calibrated_text_check as runner
from scripts.runtime import files as runtime_files


def rejects(function, *args) -> None:
    try:
        function(*args)
    except (ValueError, RuntimeError, FileExistsError):
        return
    raise AssertionError("invalid input was accepted")


def classification_log() -> str:
    rows = [f"Evaluation progress: batch={index}/32 correct={index * 8 - 1} total={index * 8}/256 "
            "accuracy=0.99609375 elapsed_s=1.0 eta_s=0.0" for index in range(1, 33)]
    return "\n".join(["Evaluation dataset fingerprint: fixed-data", *rows,
                      "Correct/total: 255/256", "Prediction SHA256: " + "a" * 64])


def gpt2_log() -> str:
    rows = [json.dumps({"event": "evaluation_progress", "average_loss": 2.0, "perplexity": 7.389056,
                       "loss_aggregation": "mean_of_batch_losses", "batch": index, "total_batches": 32,
                       "valid_loss_batches": index, "evaluated_samples": index * 8}, sort_keys=True)
            for index in range(1, 33)]
    return "\n".join(["Evaluation dataset fingerprint: fixed-data", *rows,
                      "Average Loss: 2.0000", "Perplexity: 7.3891"])


def verify_commands_and_admission() -> None:
    for family in ("bert", "roberta", "gpt2"):
        args = argparse.Namespace(family=family, source_root=Path("/source"), python_bin="python",
                                  model_id="/checkpoint", cache_dir="/cache")
        commands = runner.build_commands(args, Path("/output"))
        assert list(commands) == ["collect", "ann", "snn"]
        for phase, command in commands.items():
            def value(flag):
                return command[command.index(flag) + 1]
            assert value("--dtype") == "float64" and "--theta" not in command
            assert value("--batch_size") == "8" and value("--max_length") == "128"
            assert "--no-tensorboard" in command and "--no-gaussian-time-noise" in command
            assert "--report-clamp-stats" not in command
            assert "--spiking-attention" in command and "--spiking-layernorm" in command
            if phase == "ann":
                assert value("--model_backend") == "hf"
                assert value("--calibration-mode") == "none"
                assert "--calibration-path" not in command
            else:
                assert value("--model_backend") == "spiking"
                assert value("--calibration-samples") == "256"
                assert value("--calibration-seed") == "0"
            if phase != "collect":
                assert value("--max_eval_batches") == "32"
        for gpu in (4, 5, 6, 7):
            runner.validate_gpu_request(gpu)
    for gpu in (0, 1, 2, 3, 8, -1, True, "4"):
        rejects(runner.validate_gpu_request, gpu)
    for filesystem in ("tmpfs", "ramfs", "", "ext4\ntmpfs"):
        rejects(runner.validate_disk_filesystem, filesystem)
    runner.validate_disk_filesystem("ext4")
    commit = "a" * 40
    with patch.object(runner.subprocess, "check_output", side_effect=[commit + "\n", ""]):
        assert runner.check_source(Path("/source"), commit) == commit
    with patch.object(runner.subprocess, "check_output", return_value="b" * 40):
        rejects(runner.check_source, Path("/source"), commit)
    with patch.object(runner.subprocess, "check_output", side_effect=[commit, " M changed.py"]):
        rejects(runner.check_source, Path("/source"), commit)


def verify_result_parsing() -> None:
    for family in ("bert", "roberta"):
        good = classification_log()
        result = runner.parse_evaluation(good, family)
        assert result["correct"] == 255 and result["total"] == 256
        assert result["accuracy"] == 255 / 256
        for bad in (good.replace("255/256", "255/255"), good + "\nCorrect/total: 255/256",
                    good.replace("batch=2/32", "batch=1/32"), good.replace("correct=255", "correct=254")):
            rejects(runner.parse_evaluation, bad, family)
        site_report = "\nCalibration[module/query] values=100, underflows=0 (rate=0), overflows=0 (rate=0)"
        assert runner.parse_evaluation(good + site_report, family, {"module/query"})["calibration_site_count"] == 1
        rejects(runner.parse_evaluation, good, family, {"module/query"})
        rejects(runner.parse_evaluation, good + site_report * 2, family, {"module/query"})
        rejects(runner.parse_evaluation, good + site_report.replace("values=100", "values=0"), family, {"module/query"})
    good = gpt2_log()
    result = runner.parse_evaluation(good, "gpt2")
    assert result["average_loss"] == 2.0
    assert result["evaluation_dataset_fingerprint"] == "fixed-data"
    for bad in (good.replace('"valid_loss_batches": 32', '"valid_loss_batches": 31'),
                good.replace('"average_loss": 2.0', '"average_loss": null'),
                good.replace("Perplexity: 7.3891", ""), good + "\nPerplexity: 7.3891",
                good.replace('"evaluated_samples": 256', '"evaluated_samples": 255'),
                good.replace("Evaluation dataset fingerprint: fixed-data", ""),
                good + "\nEvaluation dataset fingerprint: fixed-data"):
        rejects(runner.parse_evaluation, bad, "gpt2")


def verify_table_parsing() -> None:
    table = {"metadata": {"dtype": "float64", "preprocessing": json.dumps({"subset_samples": 256}),
                          "model_options": [
                              ["operator_backed_output_head_version", 1],
                              ["text_calibration_policy_version", 2],
                          ]},
             "layers": [{"module_name": "module", "tensor_name": "query"}]}
    temporary_root = ROOT / "artifacts/runtime"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="verify-text-pilot-", dir=temporary_root) as directory:
        path = Path(directory) / "table.json"
        runtime_files.new_json(path, table)
        assert runner.calibration_sites(path)[1] == {"module/query"}
        rejects(runtime_files.new_json, path, table)
        for index, bad in enumerate(
            (copy.deepcopy(table), copy.deepcopy(table), copy.deepcopy(table))
        ):
            if index == 0:
                bad["layers"] *= 2
            elif index == 1:
                bad["metadata"]["model_options"] = [["text_calibration_policy_version", 0]]
            else:
                bad["metadata"]["model_options"] = [["text_calibration_policy_version", 2]]
            bad_path = Path(directory) / f"bad-{index}.json"
            runtime_files.new_json(bad_path, bad)
            rejects(runner.calibration_sites, bad_path)


def main() -> None:
    verify_commands_and_admission()
    verify_result_parsing()
    verify_table_parsing()
    print("Short calibrated text comparison verification passed")


if __name__ == "__main__":
    main()
