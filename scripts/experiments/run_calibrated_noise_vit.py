"""Run a separately identified calibrated counterpart of the two ViT sweeps."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from decimal import Decimal
import fcntl
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

CODE_ROOT = Path(__file__).resolve().parents[2]
REPO = Path('/data/delayed-temporal')
SOURCE = Path('/data/delayed-temporal-worktrees/gelu-timeconstant-noise')
BASE = REPO / 'artifacts/logs/noise_scan/vit_base_rt_sweep_ratio4_theta40_gelu_synaptic_scaling_float64_v5'
ROOT = REPO / 'artifacts/logs/noise_scan/vit_base_noise_calibrated_theta40_float64_v1'
TRAIN = REPO / 'artifacts/assets/theta-selection-v1/datasets/imagenet_theta_selection_v1/train_seed0_5000'
WRAPPER = REPO / 'scripts/analysis/evaluate_calibrated_vit.py'
PYTHON = Path('/opt/conda/envs/dt/bin/python')
TABLE = ROOT / 'calibration.json'
GPUS = ('4', '5', '6', '7')
sys.path[:0] = [str(SOURCE), str(CODE_ROOT), str(REPO / 'artifacts/runtime')]
import ratio_gpu_runner as gpu_policy
from scripts.experiments import run_sigma_margin_local as local
from scripts.experiments.ubai.build_sigma_margin_manifest import FIELDS
from scripts.analysis.summarize_sigma_margin_sweep import read_manifest, parse_run_log, aggregate_runs, aggregate_sites, write_csv, build_frontier
from scripts.analysis.summarize_adaptive_timing_noise_sweep import validate_gelu_log_contract
from scripts.runtime import files as runtime_files
from scripts.runtime import identity

OUTPUT_LOCK = threading.Lock()


def rows_at(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def serialize(rows, *, calibrated=False):
    fields = list(FIELDS) + (['calibration_sha256', 'calibration_mode'] if calibrated else [])
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, delimiter='\t', lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def state(phase, **details):
    runtime_files.atomic_json(
        ROOT / 'campaign-status.json', {'phase': phase, 'updated_at': time.time(), **details}
    )


def condition(row):
    return (row['stage'], row['backend'], Decimal(row['time_noise_std_frac']),
            Decimal(row['deadline_margin_std']), row['seed'])


def prepare():
    local.require_clean_source(SOURCE)
    previous = json.loads((BASE / 'ratio-grid-13/experiment.json').read_text())
    by_condition = {}
    sources = [BASE / 'ratio-grid-13/manifests/grid.tsv', BASE / 'log-grid-9/manifests/grid.tsv']
    for path in sources:
        for row in rows_at(path):
            row = {k: row[k] for k in FIELDS}
            key = condition(row)
            if key in by_condition and by_condition[key] != row:
                raise ValueError('Shared condition identity mismatch')
            by_condition[key] = row
    rows = list(by_condition.values())
    assert len(rows) == 65 and len({r['run_id'] for r in rows}) == 65
    assert local.validate_source_identity(SOURCE, rows) == previous['source_commit']
    if identity.sha256_file(SOURCE / previous['evaluator_path']) != previous['evaluator_sha256']:
        raise ValueError('Frozen evaluator changed')
    training_identity = identity.artifact_identity(TRAIN)
    if training_identity['aggregate_sha256'] != 'fb4b6b81318ba9d00f8ede2ecde7ae9aaa831d32e38d2c26fa6e1776544cc74b':
        raise ValueError('Training artifact mismatch')
    if identity.artifact_identity(Path(previous['checkpoint_path']))['aggregate_sha256'] != previous['checkpoint_sha256']:
        raise ValueError('Checkpoint mismatch')
    experiment = {
        'tag': ROOT.name, 'source_commit': previous['source_commit'],
        'evaluator_path': previous['evaluator_path'], 'evaluator_sha256': previous['evaluator_sha256'],
        'calibration_evaluator_path': str(WRAPPER),
        'calibration_evaluator_sha256': identity.sha256_file(WRAPPER),
        'driver_sha256': identity.sha256_file(Path(__file__)),
        'gpu_policy_sha256': identity.sha256_file(Path(gpu_policy.__file__)),
        'checkpoint_path': previous['checkpoint_path'], 'checkpoint_sha256': previous['checkpoint_sha256'],
        'calibration_dataset_path': str(TRAIN), 'calibration_dataset_fingerprint': 'cabf903d14d1b1ac',
        'calibration_dataset_sha256': training_identity['aggregate_sha256'],
        'calibration_samples': 5000, 'calibration_seed': 0, 'calibration_bins': 2048,
        'calibration_lower_quantile': 0.0, 'calibration_upper_quantile': 1.0,
        'calibration_margin_fraction': 0.05, 'expected_calibration_sites': 48,
        'theta': 40, 'theta_selection_revalidated': False, 'precision': 'float64', 'batch_size': 32,
        'gelu_cubic_implementation': 'phi_nl_psi_ed', 'gelu_cubic_floor': 1e-5,
        'physical_gpus': list(GPUS), 'runs': 65, 'seeds': [0, 1, 2],
        'source_manifest_sha256': {str(p): identity.sha256_file(p) for p in sources},
        'dense_reference_log_sha256': identity.sha256_file(
            BASE / 'local-assets/logs/dense_reference.log'
        ),
        'paper_promotion_allowed': False,
    }
    runtime_files.immutable(
        ROOT / 'experiment.json', (json.dumps(experiment, indent=2) + '\n').encode()
    )
    runtime_files.immutable(ROOT / 'manifests/planned.tsv', serialize(rows).encode())
    (ROOT / 'logs').mkdir(parents=True, exist_ok=True)
    dense_target = ROOT / 'logs/dense_reference.log'
    if not dense_target.exists():
        shutil.copy2(BASE / 'local-assets/logs/dense_reference.log', dense_target)
    if identity.sha256_file(dense_target) != experiment['dense_reference_log_sha256']:
        raise ValueError('Dense reference changed')
    return rows, experiment


def check_runtime_identity(experiment):
    local.require_clean_source(SOURCE)
    if local.git_head(SOURCE) != experiment['source_commit']:
        raise ValueError('Source HEAD changed')
    for path, expected in [(WRAPPER, experiment['calibration_evaluator_sha256']),
                           (Path(__file__), experiment['driver_sha256']),
                           (Path(gpu_policy.__file__), experiment['gpu_policy_sha256']),
                           (SOURCE / experiment['evaluator_path'], experiment['evaluator_sha256'])]:
        if identity.sha256_file(path) != expected:
            raise ValueError(f'Runtime file changed: {path}')


def scheduler_args(experiment, gpus, status_name):
    return SimpleNamespace(
        manifest=ROOT / 'manifests/grid.tsv', log_dir=ROOT / 'logs',
        runtime_root=REPO / 'artifacts/runtime/calibrated-noise-v1', source_root=SOURCE,
        python_bin=PYTHON, evaluator_script=WRAPPER,
        evaluator_arg=['--source-root', str(SOURCE), '--calibration-dataset-path', str(TRAIN),
                       '--calibration-dataset-fingerprint', experiment['calibration_dataset_fingerprint'],
                       '--gelu-cubic-implementation', 'phi_nl_psi_ed', '--gelu-cubic-floor', '1e-5'],
        gpus=list(gpus), status_json=ROOT / f'status/{status_name}.json', allow_noncanonical_manifest=True,
    )


class CalibratedScheduler(local.LocalScheduler):
    def __init__(self, args, rows, experiment, table_sha):
        super().__init__(args, rows)
        self.experiment = experiment
        self.table_sha = table_sha
        self.specs = {s.run_id: s for s in read_manifest(args.manifest, require_canonical=False)} if args.manifest.exists() else {}

    def command(self, row):
        command = super().command(row)
        command[command.index('--calibration-mode') + 1] = 'validate'
        command += ['--calibration-path', str(TABLE), '--calibration-samples', '5000',
                    '--calibration-seed', '0', '--calibration-bins', '2048',
                    '--calibration-lower-quantile', '0', '--calibration-upper-quantile', '1',
                    '--calibration-margin-fraction', '0.05']
        return command

    def validate(self, run_id):
        spec = self.specs[run_id]
        path = self.args.log_dir / spec.log_file
        if not path.exists():
            return False
        try:
            parse_run_log(spec, self.args.log_dir)
            if spec.backend == 'hf':
                return identity.sha256_file(path) == self.experiment['dense_reference_log_sha256']
            check_runtime_identity(self.experiment)
            if identity.sha256_file(TABLE) != self.table_sha:
                return False
            text = path.read_text()
            expected = f'Calibration identity — mode: validate, sha256: {self.table_sha}'
            if text.count(expected) != 1:
                return False
            validate_gelu_log_contract([spec], self.args.log_dir)
            return True
        except (ValueError, RuntimeError, OSError):
            return False

    def run_one(self, gpu, row):
        check_runtime_identity(self.experiment)
        if identity.sha256_file(TABLE) != self.table_sha:
            raise ValueError('Frozen calibration table changed')
        super().run_one(gpu, row)
        with OUTPUT_LOCK:
            summarize(self, complete=False)


def summarize(scheduler, *, complete):
    runs = []
    for spec in scheduler.specs.values():
        if scheduler.validate(spec.run_id):
            runs.append(parse_run_log(spec, ROOT / 'logs'))
        elif complete:
            raise ValueError(f'Missing complete calibrated result: {spec.run_id}')
    raw = []
    for run in runs:
        row = asdict(run)
        row.pop('sites')
        row['calibration_sha256'] = scheduler.table_sha
        row['calibration_mode'] = 'none' if run.backend == 'hf' else 'validate'
        raw.append(row)
    if raw:
        write_csv(ROOT / 'outputs/raw_runs.csv', raw)
    grouped = {}
    for run in runs:
        if run.stage != 'baseline':
            grouped.setdefault((run.time_noise_std_frac, run.deadline_margin_std), []).append(run)
    ready = [r for r in runs if r.stage == 'baseline']
    for replicas in grouped.values():
        if len(replicas) == 3 and {r.seed for r in replicas} == {0, 1, 2}:
            ready.extend(replicas)
    summary = aggregate_runs(ready)
    if summary:
        write_csv(ROOT / 'outputs/summary.csv', summary)
    sites = aggregate_sites(ready)
    if sites:
        write_csv(ROOT / 'outputs/site_summary.csv', sites)
    runtime_files.atomic_json(ROOT / 'outputs/progress.json', {
        'complete': complete, 'validated_runs': len(runs), 'total_runs': 65,
        'complete_stochastic_cells': sum(1 for s in summary if s['stage'] == 'sigma_margin'),
        'calibration_sha256': scheduler.table_sha,
    })
    if complete:
        for axis in ('rt', 'ratio'):
            selected = [s for s in summary if s['stage'] == 'baseline' or (
                float(s['deadline_margin_std']) == 4 if axis == 'rt'
                else float(s['time_noise_std_frac']) == 1e-5)]
            csv_path = ROOT / f'outputs/{axis}/summary.csv'
            write_csv(csv_path, selected)
            subprocess.run([str(PYTHON), str(REPO / 'artifacts/runtime/plot_noise_curves.py'),
                            '--summary', str(csv_path), '--axis', axis,
                            '--output', str(ROOT / f'outputs/{axis}/accuracy-physical-rate')], check=True)
            if axis == 'ratio':
                runtime_files.atomic_json(
                    ROOT / 'outputs/ratio/frontier.json', build_frontier(selected)
                )


def collect(rows, experiment):
    evidence_path = ROOT / 'calibration-evidence.json'
    if evidence_path.exists():
        evidence = json.loads(evidence_path.read_text())
        if (evidence['experiment_sha256'] != identity.sha256_file(ROOT / 'experiment.json')
                or evidence['calibration_sha256'] != identity.sha256_file(TABLE)):
            raise ValueError('Calibration evidence changed')
        return evidence['calibration_sha256']
    if TABLE.exists():
        raise ValueError('Calibration table exists without validated evidence; inspect before reuse')
    state('waiting', purpose='calibration')
    gpu = gpu_policy.available_gpus()[0]
    check_runtime_identity(experiment)
    args = scheduler_args(experiment, [gpu], 'collect')
    args.runtime_root.mkdir(parents=True, exist_ok=True)
    scheduler = CalibratedScheduler(args, [], experiment, '')
    clean = next(r for r in rows if r['run_id'] == 'clean_spiking_baseline')
    command = scheduler.command(clean)
    command[command.index('--calibration-mode') + 1] = 'collect'
    command[command.index('--experiment_name') + 1] = 'calibration_collect'
    runtime = Path(tempfile.mkdtemp(prefix='collect-', dir=args.runtime_root))
    environment = scheduler.base_environment()
    environment.update(CUDA_VISIBLE_DEVICES=gpu, TMPDIR=str(runtime), XDG_CACHE_HOME=str(runtime / 'cache'))
    log = ROOT / 'logs/calibration_collect.log'
    with log.open('a') as handle:
        child = subprocess.Popen(command, cwd=SOURCE, env=environment, stdout=handle, stderr=subprocess.STDOUT)
        state('collect', gpu=gpu, pid=child.pid, log=str(log))
        code = gpu_policy.physical_wait(child, gpu, pid_reader=gpu_policy.physical_pids)
    if code:
        raise RuntimeError(f'Calibration collection failed: {code}; see {log}')
    from utils.transforms.calibration import load_calibration_table
    table = load_calibration_table(TABLE)
    if len(table.layers) != 48:
        raise ValueError('Expected 48 calibrated sites')
    table_sha = identity.sha256_file(TABLE)
    evidence = {
        'calibration_sha256': table_sha,
        'experiment_sha256': identity.sha256_file(ROOT / 'experiment.json'),
        'collection_log_sha256': identity.sha256_file(log),
        'sites': 48,
        'gpu': gpu,
    }
    runtime_files.immutable(
        evidence_path, (json.dumps(evidence, indent=2) + '\n').encode()
    )
    return table_sha


# @lat: [[evaluation#Evaluation and Verification#Calibrated ViT Noise Comparison]]
def execute(rows, experiment):
    with (ROOT / 'driver.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        table_sha = collect(rows, experiment)
        rows = [{**r, 'calibration_sha256': table_sha,
                 'calibration_mode': 'none' if r['backend'] == 'hf' else 'validate'} for r in rows]
        runtime_files.immutable(
            ROOT / 'manifests/grid.tsv', serialize(rows, calibrated=True).encode()
        )
        local.live_compute_pids = gpu_policy.admission_pids
        local.wait_for_gpu_child = lambda child, gpu: gpu_policy.physical_wait(child, gpu, pid_reader=gpu_policy.physical_pids)
        pool = gpu_policy.available_gpus()
        args = scheduler_args(experiment, pool, 'clean')
        clean_rows = [r for r in rows if r['stage'] == 'baseline']
        clean = CalibratedScheduler(args, clean_rows, experiment, table_sha)
        state('clean', gpus=pool)
        signal.signal(signal.SIGTERM, lambda *_: clean.terminate())
        signal.signal(signal.SIGINT, lambda *_: clean.terminate())
        clean.execute()
        clean_run = parse_run_log(clean.specs['clean_spiking_baseline'], ROOT / 'logs')
        if clean_run.accuracy <= .01:
            raise RuntimeError('Calibrated clean accuracy collapsed; inspect before noise sweep')
        pool = gpu_policy.available_gpus()
        sweep = CalibratedScheduler(scheduler_args(experiment, pool, 'sweep'), rows, experiment, table_sha)
        signal.signal(signal.SIGTERM, lambda *_: sweep.terminate())
        signal.signal(signal.SIGINT, lambda *_: sweep.terminate())
        state('sweep', gpus=pool, clean_accuracy=clean_run.accuracy)
        sweep.execute()
        summarize(sweep, complete=True)
        state('complete', runs=65, calibration_sha256=table_sha)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    rows, experiment = prepare()
    print('Prepared 65 conditions; dense reference only is reused.', flush=True)
    if not args.prepare_only:
        try:
            execute(rows, experiment)
        except BaseException as error:
            state('failed', error=f'{type(error).__name__}: {error}')
            raise


if __name__ == '__main__':
    main()
