#!/usr/bin/env python3
"""Run one assigned calibrated ViT evaluation in an allocated UBAI container."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time
from typing import Any


SOURCE_ROOT = Path('/data/delayed-temporal-worktrees/gelu-timeconstant-noise')
SOURCE_COMMIT = '648af9bbbea796bcbc6589449b1f9714e156ebc8'
REPOSITORY_ROOT = Path('/data/delayed-temporal')
WRAPPER_PATH = REPOSITORY_ROOT / 'scripts/analysis/evaluate_calibrated_vit.py'
EXPERIMENT_ROOT = REPOSITORY_ROOT / 'artifacts/logs/noise_scan/vit_base_noise_calibrated_theta40_float64_v1'
DEPLOYMENT_ROOT = Path('/calibrated-deployment')
PYTHON = '/opt/conda/envs/dt/bin/python'


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def check_hash(path: Path, expected: str) -> None:
    if not re.fullmatch(r'[0-9a-f]{64}', expected or ''):
        raise ValueError(f'Invalid SHA-256 for {path}')
    if sha256_file(path) != expected:
        raise ValueError(f'SHA-256 mismatch: {path}')


def absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or '..' in path.parts or any(c in value for c in '\n\r\0,:'):
        raise ValueError(f'Invalid absolute path: {value!r}')
    return path


def runtime_values(deployment: dict[str, Any]) -> list[str]:
    """Return validated bootstrap values without executing shell fragments."""
    runtime = deployment['runtime']
    path_fields = (
        'host_source_root', 'host_repository_root', 'host_assets_root',
        'host_experiment_root', 'host_deployment_root', 'env_archive',
    )
    values = [str(absolute_path(runtime[name])) for name in path_fields]
    for name in ('env_archive_sha256', 'container_image_sha256'):
        if not re.fullmatch(r'[0-9a-f]{64}', runtime[name]):
            raise ValueError(f'Invalid runtime {name}')
    unpacked_bytes = runtime['env_unpacked_bytes']
    if type(unpacked_bytes) is not int or unpacked_bytes <= 0:
        raise ValueError('An unpacked environment size bound is required')
    scratch_bytes = runtime.get('minimum_scratch_bytes', 8 * 1024 ** 3)
    if type(scratch_bytes) is not int or scratch_bytes < 4 * 1024 ** 3:
        raise ValueError('At least 4 GiB of additional scratch space is required')
    values += [runtime['env_archive_sha256'], str(unpacked_bytes),
               str(absolute_path(runtime['container_image'])), runtime['container_image_sha256'],
               str(scratch_bytes)]
    return values


def read_assigned(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle, dialect='excel-tab')
        if not reader.fieldnames or 'run_id' not in reader.fieldnames:
            raise ValueError('Assigned manifest has no run_id column')
        rows = list(reader)
    if len({row['run_id'] for row in rows}) != len(rows):
        raise ValueError('Assigned manifest has duplicate run identifiers')
    return rows


def validate_contract(deployment: dict[str, Any], manifest: Path) -> tuple[dict[str, Any], list[Any], list[dict[str, str]]]:
    """Validate frozen identities before importing the numerical implementation."""
    for name, expected in (
        ('source_root', SOURCE_ROOT), ('wrapper_path', WRAPPER_PATH),
        ('experiment_root', EXPERIMENT_ROOT),
    ):
        if absolute_path(deployment[name]) != expected:
            raise ValueError(f'Canonical {name} must remain unchanged')
    if deployment['state'] not in {'prepared', 'ready'}:
        raise ValueError('Invalid deployment state')
    check_hash(Path(__file__), deployment['worker_sha256'])
    check_hash(Path(__file__).with_name('calibrated_noise_task.sbatch'), deployment['task_script_sha256'])
    for path, field in (
        (EXPERIMENT_ROOT / 'experiment.json', 'experiment_sha256'),
        (EXPERIMENT_ROOT / 'calibration.json', 'calibration_sha256'),
        (EXPERIMENT_ROOT / 'calibration-evidence.json', 'calibration_evidence_sha256'),
        (EXPERIMENT_ROOT / 'manifests/grid.tsv', 'grid_sha256'),
        (manifest, 'assigned_manifest_sha256'),
    ):
        check_hash(path, deployment[field])
    declared_manifest = absolute_path(deployment['assigned_manifest_path'])
    if not declared_manifest.is_relative_to(DEPLOYMENT_ROOT):
        raise ValueError('Assigned manifest must be inside the deployment directory')
    if manifest != declared_manifest:
        check_hash(declared_manifest, deployment['assigned_manifest_sha256'])
    experiment = json.loads((EXPERIMENT_ROOT / 'experiment.json').read_text())
    evidence = json.loads((EXPERIMENT_ROOT / 'calibration-evidence.json').read_text())
    if evidence['experiment_sha256'] != deployment['experiment_sha256']:
        raise ValueError('Calibration evidence refers to another experiment')
    if evidence['calibration_sha256'] != deployment['calibration_sha256'] or evidence['sites'] != 48:
        raise ValueError('Calibration evidence is incomplete or changed')
    table = json.loads((EXPERIMENT_ROOT / 'calibration.json').read_text())
    if len(table['layers']) != 48:
        raise ValueError('Calibration table must contain 48 configured sites')
    required = {
        'source_commit': SOURCE_COMMIT, 'theta': 40, 'precision': 'float64',
        'batch_size': 32, 'runs': 65, 'seeds': [0, 1, 2],
        'calibration_samples': 5000, 'calibration_seed': 0,
        'calibration_bins': 2048, 'calibration_lower_quantile': 0.0,
        'calibration_upper_quantile': 1.0, 'calibration_margin_fraction': 0.05,
        'gelu_cubic_implementation': 'phi_nl_psi_ed', 'gelu_cubic_floor': 1e-5,
        'calibration_evaluator_path': str(WRAPPER_PATH),
    }
    for key, expected in required.items():
        if experiment.get(key) != expected:
            raise ValueError(f'Unsupported experiment setting: {key}')
    for tool in deployment.get('runtime_tools', []):
        tool_path = absolute_path(tool['path'])
        if not tool_path.is_relative_to(DEPLOYMENT_ROOT / 'tools'):
            raise ValueError('Runtime tools must stay inside the deployment tools directory')
        check_hash(tool_path, tool['sha256'])
    head = subprocess.check_output(['git', '-C', str(SOURCE_ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(
        ['git', '-C', str(SOURCE_ROOT), 'status', '--porcelain', '--untracked-files=no'], text=True,
    )
    if head != SOURCE_COMMIT or dirty.strip():
        raise ValueError('Numerical source must have the unchanged frozen HEAD')
    check_hash(WRAPPER_PATH, experiment['calibration_evaluator_sha256'])
    check_hash(SOURCE_ROOT / experiment['evaluator_path'], experiment['evaluator_sha256'])
    sys.path[:0] = [str(SOURCE_ROOT), str(SOURCE_ROOT / 'src/transformers/src'),
                   str(SOURCE_ROOT / 'src/spikingjelly')]
    from scripts.analysis.summarize_sigma_margin_sweep import read_manifest

    specs = read_manifest(EXPERIMENT_ROOT / 'manifests/grid.tsv', require_canonical=False)
    if len(specs) != 65:
        raise ValueError('The complete experiment manifest must contain 65 conditions')
    full = {spec.run_id: spec.row for spec in specs}
    for spec in specs:
        if spec.source_commit != SOURCE_COMMIT or spec.gpu_family != 'rtxa6000':
            raise ValueError('Manifest source or GPU family mismatch')
        if spec.checkpoint_sha256 != experiment['checkpoint_sha256']:
            raise ValueError('Manifest checkpoint identity mismatch')
        if spec.row['checkpoint_path'] != experiment['checkpoint_path']:
            raise ValueError('Manifest checkpoint path mismatch')
        expected_mode = 'none' if spec.backend == 'hf' else 'validate'
        if spec.row['calibration_sha256'] != deployment['calibration_sha256'] or spec.row['calibration_mode'] != expected_mode:
            raise ValueError('Manifest calibration identity mismatch')
    assigned = read_assigned(manifest)
    for row in assigned:
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', row['run_id']):
            raise ValueError('Unsafe assigned run identifier')
        if row != full.get(row['run_id']) or row['backend'] != 'spiking':
            raise ValueError('Assigned condition differs from the original spiking manifest')
    return experiment, specs, assigned


def evaluator_command(row: dict[str, str], experiment: dict[str, Any]) -> list[str]:
    gaussian = ['--no-gaussian-time-noise', '--time-noise-seed', '0']
    if row['stage'] == 'sigma_margin':
        gaussian = ['--gaussian-time-noise', '--time-noise-seed', row['seed']]
    return [
        PYTHON, '-u', str(WRAPPER_PATH), '--source-root', str(SOURCE_ROOT),
        '--calibration-dataset-path', experiment['calibration_dataset_path'],
        '--calibration-dataset-fingerprint', experiment['calibration_dataset_fingerprint'],
        '--gelu-cubic-implementation', 'phi_nl_psi_ed', '--gelu-cubic-floor', '1e-5',
        '--experiment_name', row['run_id'], '--device', 'cuda', '--model_backend', 'spiking',
        '--model_id', row['checkpoint_path'], '--dataset_id', 'imagenet-1k',
        '--evaluation-dataset-path', row['dataset_path'], '--evaluation-split', row['split'],
        '--batch_size', '32', '--quick-test', '--theta', row['theta'], '--precision', 'float64',
        '--calibration-mode', 'validate', '--calibration-path', str(EXPERIMENT_ROOT / 'calibration.json'),
        '--calibration-samples', '5000', '--calibration-seed', '0', '--calibration-bins', '2048',
        '--calibration-lower-quantile', '0', '--calibration-upper-quantile', '1',
        '--calibration-margin-fraction', '0.05', *gaussian,
        '--time-noise-std-frac', row['time_noise_std_frac'], '--time-noise-mean', '0',
        '--time-noise-deadline-margin-std', row['deadline_margin_std'], '--no-mismatch-enabled',
        '--mismatch-theta-std', '0', '--weight-noise-std', '0', '--bias-noise-std', '0',
        '--source-commit', row['source_commit'], '--checkpoint-sha256', row['checkpoint_sha256'],
        '--no-tensorboard', '--report-clamp-stats', '--spiking-layernorm', '--spiking-mlp',
        '--spiking-attention',
    ]


def validate_log(spec: Any, path: Path, calibration_sha: str, deployment_sha: str | None = None) -> Any:
    from scripts.analysis.summarize_adaptive_timing_noise_sweep import validate_gelu_log_contract
    from scripts.analysis.summarize_sigma_margin_sweep import parse_run_log

    temporary_spec = replace(spec, log_file=path.name)
    parsed = parse_run_log(temporary_spec, path.parent)
    validate_gelu_log_contract([temporary_spec], path.parent)
    lines = path.read_text().splitlines()
    expected = f'Calibration identity — mode: validate, sha256: {calibration_sha}'
    if lines.count(expected) != 1:
        raise ValueError('Incomplete or mismatched calibration identity in evaluator log')
    if deployment_sha is not None and lines.count(f'UBAI deployment identity — sha256: {deployment_sha}') != 1:
        raise ValueError('Evaluator log does not match this deployment')
    return parsed


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'{path.name}.partial.{os.getpid()}')
    with temporary.open('w') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def require_single_gpu() -> str:
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not visible or len(visible.split(',')) != 1 or visible in {'-1', 'all', 'none'}:
        raise ValueError('Exactly one Slurm-allocated GPU must be visible')
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Evaluation must run inside a Slurm allocation')
    # The inspection process exits before evaluation; this controller owns no CUDA context.
    probe = subprocess.check_output([
        PYTHON, '-c', 'import json, torch; '
        'print(json.dumps({"count": torch.cuda.device_count(), '
        '"model": torch.cuda.get_device_name(0) if torch.cuda.device_count() else ""}))',
    ], text=True)
    gpu = json.loads(probe)
    if gpu['count'] != 1:
        raise ValueError('The evaluator must see exactly one GPU')
    model = gpu['model']
    if 'RTX A6000' not in model:
        raise ValueError(f'Expected RTX A6000, received {model}')
    return model


def check_python_version(deployment: dict[str, Any]) -> None:
    expected = deployment['runtime'].get('expected_python', '3.12.13')
    if expected != '3.12.13' or platform.python_version() != expected:
        raise ValueError(f'Python 3.12.13 is required, received {platform.python_version()}')


def check_assets(deployment: dict[str, Any]) -> list[dict[str, Any]]:
    """Hash the transported assets on a compute node, never on a login node."""
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Artifact verification must run inside a Slurm allocation')
    from scripts.setup.hash_artifact import artifact_identity

    checked = []
    for asset in deployment.get('assets', []):
        path = absolute_path(asset['path'])
        identity = artifact_identity(path)
        if identity['aggregate_sha256'] != asset['aggregate_sha256']:
            raise ValueError(f'Transported artifact SHA-256 mismatch: {path}')
        if 'bytes' in asset and identity['bytes'] != asset['bytes']:
            raise ValueError(f'Transported artifact size mismatch: {path}')
        checked.append({'path': str(path), 'aggregate_sha256': identity['aggregate_sha256'],
                        'bytes': identity['bytes']})
    return checked


def check_storage(deployment: dict[str, Any]) -> dict[str, Any]:
    """Measure shared project storage without counting nested mounts twice."""
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Storage verification must run inside a Slurm allocation')
    assets_root = REPOSITORY_ROOT / 'artifacts/assets/theta-selection-v1'
    paths = (assets_root, SOURCE_ROOT, REPOSITORY_ROOT, DEPLOYMENT_ROOT, EXPERIMENT_ROOT)
    measured = {}
    for path in paths:
        command = ['du', '-sb']
        if path in (SOURCE_ROOT, REPOSITORY_ROOT):
            # Editable dependency checkouts and artifact mounts are counted separately.
            command += ['--exclude=src', '--exclude=artifacts']
        output = subprocess.check_output([*command, str(path)], text=True)
        measured[str(path)] = int(output.split()[0])
    other = deployment['runtime'].get('existing_other_storage_bytes', 0)
    limit = deployment['runtime'].get('storage_limit_bytes', 60 * 1000 ** 3)
    if type(other) is not int or other < 0 or type(limit) is not int or not 0 < limit <= 60 * 1000 ** 3:
        raise ValueError('Invalid storage accounting contract')
    total = sum(measured.values()) + other
    if total > limit:
        raise ValueError(f'Shared project storage exceeds the limit: {total} > {limit}')
    return {'measured_paths': measured, 'asset_tree_bytes': measured[str(assets_root)],
            'existing_other_storage_bytes': other, 'project_storage_upper_bound_bytes': total,
            'storage_limit_bytes': limit, 'scope': 'shared project files; not the complete account quota'}


def execute(deployment_path: Path, deployment: dict[str, Any], manifest: Path, task_index: int) -> None:
    if deployment['state'] != 'ready':
        raise ValueError('Prepared deployment cannot run evaluations; an explicit work handoff is required')
    check_python_version(deployment)
    experiment, specs, assigned = validate_contract(deployment, manifest)
    if not 0 <= task_index < len(assigned):
        raise ValueError('Array index is outside the assigned manifest')
    row = assigned[task_index]
    spec = next(item for item in specs if item.run_id == row['run_id'])
    if spec.stage == 'sigma_margin':
        clean = next(item for item in specs if item.run_id == 'clean_spiking_baseline')
        clean_path = EXPERIMENT_ROOT / 'logs' / clean.log_file
        check_hash(clean_path, deployment.get('reference_clean_log_sha256', ''))
        clean_result = validate_log(clean, clean_path, deployment['calibration_sha256'])
        if clean_result.accuracy <= 0.01:
            raise ValueError('Clean calibrated validation requires inspection before noise evaluation')
    log_path = EXPERIMENT_ROOT / 'logs' / row['log_file']
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = EXPERIMENT_ROOT / 'locks' / f'{row["run_id"]}.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if log_path.exists():
            validate_log(spec, log_path, deployment['calibration_sha256'])
            print(f'Skipping complete calibrated evaluation: {row["run_id"]}', flush=True)
            return
        require_single_gpu()
        deployment_sha = sha256_file(deployment_path)
        suffix = f'{os.environ["SLURM_JOB_ID"]}.{task_index}.{os.getpid()}'
        partial = log_path.with_name(f'{log_path.name}.partial.{suffix}')
        status_path = EXPERIMENT_ROOT / 'status/ubai' / f'{row["run_id"]}.json'
        environment = os.environ.copy()
        environment.pop('WANDB_API_KEY', None)
        environment.update(WANDB_MODE='disabled', WANDB_SILENT='true', WANDB_CONSOLE='off',
                           HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                           PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
        environment['PYTHONPATH'] = ':'.join((str(SOURCE_ROOT), str(SOURCE_ROOT / 'src/transformers/src'),
                                             str(SOURCE_ROOT / 'src/spikingjelly')))
        child = None
        old_handlers = {}

        def stop(signum: int, _frame: Any) -> None:
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            raise InterruptedError(f'Allocated task received signal {signum}')

        for sig in (signal.SIGTERM, signal.SIGINT):
            old_handlers[sig] = signal.signal(sig, stop)
        status = {'run_id': row['run_id'], 'task_index': task_index, 'job_id': os.environ['SLURM_JOB_ID'],
                  'deployment_sha256': deployment_sha, 'started_at': time.time(), 'partial_log': str(partial)}
        try:
            with partial.open('x', encoding='utf-8') as handle:
                handle.write(f'Slurm identity — job_id: {os.environ["SLURM_JOB_ID"]}, task_id: {task_index}, '
                             f'node: {os.uname().nodename}, gpu_family: {row["gpu_family"]}\n')
                handle.write(f'UBAI deployment identity — sha256: {deployment_sha}\n')
                handle.flush()
                child = subprocess.Popen(evaluator_command(row, experiment), cwd=SOURCE_ROOT,
                                         env=environment, stdout=handle, stderr=subprocess.STDOUT)
                atomic_json(status_path, {**status, 'state': 'running', 'pid': child.pid})
                return_code = child.wait()
                handle.flush()
                os.fsync(handle.fileno())
            if return_code != 0:
                raise RuntimeError(f'Evaluator failed with exit status {return_code}; partial log retained')
            validate_contract(deployment, manifest)
            check_hash(deployment_path, deployment_sha)
            result = validate_log(spec, partial, deployment['calibration_sha256'], deployment_sha)
            if log_path.exists():
                raise FileExistsError('A completed log appeared during evaluation; partial log retained')
            partial.replace(log_path)
            atomic_json(status_path, {**status, 'state': 'complete', 'finished_at': time.time(),
                                     'log_file': str(log_path), 'log_sha256': sha256_file(log_path),
                                     'correct': result.correct, 'samples': result.samples})
            print(f'Completed calibrated evaluation: {row["run_id"]}', flush=True)
        except BaseException as error:
            atomic_json(status_path, {**status, 'state': 'failed', 'finished_at': time.time(), 'error': str(error)})
            raise
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deployment', type=Path, required=True)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--task-index', type=int)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--runtime-values', action='store_true')
    args = parser.parse_args()
    deployment = json.loads(args.deployment.read_text())
    if args.runtime_values:
        check_hash(Path(__file__), deployment['worker_sha256'])
        check_hash(Path(__file__).with_name('calibrated_noise_task.sbatch'), deployment['task_script_sha256'])
        print('\n'.join(runtime_values(deployment)))
        return
    manifest = args.manifest or absolute_path(deployment['assigned_manifest_path'])
    if args.check_only:
        check_python_version(deployment)
        _, specs, assigned = validate_contract(deployment, manifest)
        assets = check_assets(deployment)
        storage = check_storage(deployment)
        receipt = {
            'deployment_sha256': sha256_file(args.deployment), 'python': platform.python_version(),
            'python_executable': sys.executable, 'platform': platform.platform(),
            'total_conditions': len(specs), 'assigned_conditions': len(assigned),
            'assets': assets, 'storage': storage, 'checked_at': time.time(),
        }
        atomic_json(EXPERIMENT_ROOT / 'status/ubai-preparation.json', receipt)
        print('Validated calibrated deployment — ' + json.dumps(receipt, sort_keys=True))
        return
    if args.task_index is None:
        parser.error('--task-index is required for evaluation')
    execute(args.deployment, deployment, manifest, args.task_index)


if __name__ == '__main__':
    main()
