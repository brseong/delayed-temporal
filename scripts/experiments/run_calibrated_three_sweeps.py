"""Coordinate calibrated theta selection and noise sweeps in seed order across two hosts."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import time
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.experiments.calibrated_three_sweeps import (
    TAG, RangeInsufficient, confirm_selection, make_task, make_tasks, rt_grid,
    select_theta, theta_grid, validate_experiment, validate_result, validate_task,
)
from scripts.runtime import files as runtime_files
from scripts.runtime import identity
from scripts.runtime import local_gpu
from scripts.runtime import slurm

LOCAL_GPUS = (4, 5, 6, 7)
SUPPORTED_LOCAL_GPUS = tuple(range(8))
CPU_GPU_ORDER = (4, 5, 6, 7, 0, 1, 2, 3)
LOCAL_WORKER = 'scripts/experiments/run_calibrated_three_sweep_local_task.py'
REBALANCE_MIN_WAIT_SECONDS = 60
TEMPORARY_GPU_SOURCE = '36615ab4390f9817e3af0e4c4a6f840fc6bd57ee'
UBAI_RESOURCE_POLICY = {'max_gpus': 12, 'max_running_jobs': 10, 'max_submitted_jobs': 20,
                        'experiments_per_paired_job': 2}
PAIR_FILES = ('scripts/experiments/ubai/run_calibrated_three_sweep_pair.py',
              'scripts/experiments/ubai/calibrated_three_sweep_pair.sbatch')
ASSETS = Path('/data/delayed-temporal/artifacts/assets/theta-selection-v1')
REMOTE_BASE = '/home1/sizz1997/myubai'
MAIN_TASKS = 80


class NeedsAttention(RuntimeError):
    """An integrity failure or predeclared scientific gate needs user direction."""


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def initialize(root: Path, source: Path, python_bin: str) -> dict:
    """Freeze source and actual local artifact hashes before creating assignments."""
    from scripts.experiments.run_calibrated_three_sweep_task import check_source
    if socket.gethostname() != 'baekryun-cuda129':
        raise ValueError('Initialization must run on the local GPU host')
    if root.name != TAG:
        raise ValueError('The campaign tag is fixed')
    if (root / 'experiment.json').exists():
        experiment = read_json(root / 'experiment.json')
        validate_experiment(experiment)
        check_source(experiment)
        return experiment
    source = source.resolve()
    experiment = {
        'tag': TAG, 'source_root': str(source),
        'source_commit': subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
        'python_bin': python_bin, 'precision': 'float64', 'batch_size': 32,
        'output_bounds_version': 3, 'seeds': [0, 1, 2], 'tau_s': 1.0,
        'runtime_root': f'/data/delayed-temporal/artifacts/runtime/{TAG}',
        'calibration_dataset_fingerprint': 'cabf903d14d1b1ac',
        'dataset_fingerprint': '260dc8e69ecaea24',
        'theta_values': list(theta_grid()), 'rt_values': list(rt_grid()),
        'ratio_values': [0, 1, 2, 2.5, 3, 3.5, 4, 5, 6],
        'evaluation_count': 71, 'calibration_count': 9, 'environment_smoke_count': 4,
        'selection_tolerance_correct': 25, 'selection_upper_gain_correct': 5,
        'local_gpu_ids': list(LOCAL_GPUS), 'ubai_partitions': ['gpu4', 'gpu5'],
        'max_ubai_tasks': 8, 'account_max_running_jobs': 10,
        'account_max_submitted_jobs': 20, 'account_max_gpus': 12,
        'tracking': 'disabled', 'full_validation': False, 'paper_promotion': False,
    }
    for prefix, relative in (
        ('checkpoint', 'checkpoints/vit_base_patch16_224.augreg2_in21k_ft_in1k'),
        ('calibration_dataset', 'datasets/imagenet_theta_selection_v1/train_seed0_5000'),
        ('dataset', 'datasets/imagenet_theta_selection_v1/validation_50000'),
    ):
        path = ASSETS / relative
        artifact = identity.artifact_identity(path)
        experiment[prefix + '_path'] = str(path)
        experiment[prefix + '_sha256'] = artifact['aggregate_sha256']
    for prefix, relative in (
        ('evaluator', 'scripts/evaluation/error_analysis_vit.py'),
        ('calibration_evaluator', 'scripts/analysis/evaluate_calibrated_vit.py'),
        ('gelu_evaluator', 'scripts/analysis/gelu_cubic_phi_nl_vit.py'),
    ):
        experiment[prefix + '_path'] = relative
        experiment[prefix + '_sha256'] = identity.sha256_file(source / relative)
    experiment['runtime_sha256'] = {relative: identity.sha256_file(source / relative) for relative in (
        'scripts/experiments/run_calibrated_three_sweeps.py',
        'scripts/experiments/run_calibrated_three_sweep_task.py',
        'scripts/experiments/calibrated_three_sweeps.py',
        'scripts/analysis/summarize_calibrated_three_sweeps.py',
        'scripts/experiments/ubai/prepare_calibrated_three_sweeps_ubai.py',
        'scripts/experiments/ubai/calibrated_three_sweep_task.sbatch',
        'scripts/experiments/ubai/calibrated_three_sweep_prep.sbatch',
        'scripts/experiments/ubai/calibrated_git.sh',
        'scripts/runtime/files.py',
        'scripts/runtime/identity.py',
        'scripts/runtime/local_gpu.py',
        'scripts/runtime/slurm.py',
    )}
    experiment['dependency_sha256'] = {
        package: identity.package_source_identity(source / 'src' / package / subtree)[0]
        for package, subtree in (('transformers', 'src'), ('spikingjelly', 'spikingjelly'))
    }
    validate_experiment(experiment)
    check_source(experiment)
    runtime_files.immutable_json(root / 'experiment.json', experiment)
    return experiment


def controller_identity(experiment: dict[str, Any]) -> dict[str, Any]:
    head = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    if dirty:
        raise ValueError('The controller must run from a clean tracked checkout')
    # Only scheduling may differ from the frozen experiment implementation.
    frozen_runtime = experiment['runtime_sha256']
    for relative in ('scripts/experiments/calibrated_three_sweeps.py',
                     'scripts/experiments/run_calibrated_three_sweep_task.py',
                     'scripts/analysis/summarize_calibrated_three_sweeps.py',
                     'scripts/experiments/ubai/prepare_calibrated_three_sweeps_ubai.py'):
        if identity.sha256_file(REPO / relative) != experiment['runtime_sha256'][relative]:
            raise ValueError(f'Controller import differs from the frozen experiment: {relative}')
    shared_runtime = (
        'scripts/runtime/environment.py', 'scripts/runtime/files.py',
        'scripts/runtime/identity.py', 'scripts/runtime/local_gpu.py',
        'scripts/runtime/slurm.py',
    )
    for relative in shared_runtime:
        if relative in frozen_runtime and identity.sha256_file(REPO / relative) != frozen_runtime[relative]:
            raise ValueError(f'Controller runtime differs from the frozen experiment: {relative}')
    return {'source_commit': head, 'source_root': str(REPO),
            'controller_sha256': identity.sha256_file(Path(__file__)),
            'evaluator_source_commit': experiment['source_commit'],
            'gpu_admission_policy': local_gpu.DEFAULT_ADMISSION_POLICY,
            'local_gpu_ids': list(LOCAL_GPUS),
            'supported_local_gpu_ids': list(SUPPORTED_LOCAL_GPUS),
            'shared_runtime_sha256': {relative: identity.sha256_file(REPO / relative)
                                      for relative in shared_runtime},
            'local_worker_sha256': identity.sha256_file(REPO / LOCAL_WORKER),
            'rebalance_min_wait_seconds': REBALANCE_MIN_WAIT_SECONDS,
            'rebalance_account': 'uos',
            'rebalance_sha256': identity.sha256_file(REPO / 'scripts/experiments/calibrated_three_sweep_rebalance.py'),
            'ubai_resource_policy': UBAI_RESOURCE_POLICY,
            'pair_runtime_sha256': {relative: identity.sha256_file(REPO / relative) for relative in PAIR_FILES}}


def default_host(task: dict, ordinal: int) -> str:
    if task.get('host_label'):
        return task['host_label']
    index = task['theta_index'] if task['kind'].startswith('theta') or task['kind'] == 'collect' else ordinal
    return 'local' if index % 3 == 0 else 'ubai'


def quota_available(queue: list[dict], campaign_prefix: str, gpus_per_job: int = 1) -> int:
    if gpus_per_job not in (1, 2):
        raise ValueError('A Slurm job must reserve one or two GPUs')
    own = [row for row in queue if row['name'].startswith(campaign_prefix)]
    # Reserve for submitted jobs as well, so queued work cannot exceed account limits later.
    return max(0, min((12 - sum(row['gpus'] for row in own)) // gpus_per_job,
                      20 - len(queue), 10 - len(queue),
                      (12 - sum(row['gpus'] for row in queue)) // gpus_per_job))


def pair_tasks_compatible(tasks: list[dict]) -> bool:
    if len(tasks) != 2 or len({task['run_id'] for task in tasks}) != 2:
        return False
    phases = {'collect': 'calibration', 'smoke_clean': 'smoke', 'smoke_noise': 'smoke',
              'theta_train': 'theta', 'theta_validation': 'theta', 'dense': 'theta',
              'theta_replay': 'replay', 'noise': 'noise'}
    if any(task.get('host_label') not in (None, 'ubai') or task['kind'] not in phases for task in tasks):
        return False
    if len({phases[task['kind']] for task in tasks}) != 1:
        return False
    return tasks[0]['kind'] != 'noise' or tasks[0]['seed'] == tasks[1]['seed']


def seed_range_reason(results: list[dict], clean_accuracy: float) -> str | None:
    rows = [r for r in results if r['kind'] == 'noise' and r['seed'] == 0
            and r['deadline_margin_std'] == 4.0]
    if len(rows) != 9 or {r['time_noise_std_frac'] for r in rows} != set(rt_grid()):
        raise ValueError('Nine complete seed 0 timing noise results are required')
    if all(r['accuracy'] >= clean_accuracy - .01 for r in rows):
        return 'All nine timing noise points are within one percentage point of clean accuracy'
    if all(r['accuracy'] <= .01 for r in rows):
        return 'All nine timing noise points have accuracy at most one percent'
    return None


def assert_seed_barrier(seed: int, results: list[dict], experiment: dict, theta: float, table_hash: str) -> None:
    for earlier in range(seed):
        expected = {r['run_id'] for r in make_tasks(experiment, 'noise', selected_theta=theta,
                                                  seed=earlier, calibration_sha256=table_hash)}
        rows = [r for r in results if r['kind'] == 'noise' and r['seed'] == earlier]
        actual = {r['run_id'] for r in rows}
        if len(rows) != len(actual) or actual != expected:
            raise ValueError(f'Previous seed {earlier} is not complete')


class Controller:
    def __init__(self, root: Path, *, poll_seconds: float = 30, max_attempts: int = 3,
                 temporary_local_gpus: bool = False):
        self.root = root.resolve()
        self.experiment = read_json(root / 'experiment.json')
        validate_experiment(self.experiment)
        if temporary_local_gpus and (self.experiment['source_commit'] != TEMPORARY_GPU_SOURCE
                                     or self.root.name != TAG):
            raise ValueError('Temporary GPU permission is restricted to the current frozen campaign')
        from scripts.experiments.run_calibrated_three_sweep_task import check_source
        check_source(self.experiment)
        if socket.gethostname() != 'baekryun-cuda129':
            raise ValueError('The central controller runs only on the local GPU host')
        self.source = Path(self.experiment['source_root'])
        self.remote_root = f'{REMOTE_BASE}/delayed-temporal-experiments/{TAG}'
        self.remote_source = f'{REMOTE_BASE}/delayed-temporal-main'
        self.prefix = 'c3-' + self.experiment['source_commit'][:7] + '-'
        self.poll_seconds = max(2, poll_seconds)
        self.max_attempts = max_attempts
        self.local_gpus = SUPPORTED_LOCAL_GPUS if temporary_local_gpus else LOCAL_GPUS
        self.rebalance_account = 'uos'
        self.state_path = root / 'assignments.json'
        self.state = read_json(self.state_path) if self.state_path.exists() else {
            'experiment_sha256': identity.json_sha256(self.experiment), 'tasks': {}, 'phase': 'prepared'}
        if self.state['experiment_sha256'] != identity.json_sha256(self.experiment):
            raise ValueError('Assignment experiment identity mismatch')
        self.children: dict[str, subprocess.Popen] = {}
        self.gpu_locks: dict[str, Any] = {}
        self.prep_verified = False
        self.pair_controller_verified = False
        self.root.joinpath('worker_logs').mkdir(parents=True, exist_ok=True)
        identity = controller_identity(self.experiment)
        runtime_files.immutable_json(
            self.root / 'controllers' / (identity['source_commit'] + '.json'), identity
        )
        self.state['controller_identity'] = identity
        self.state['active_local_gpu_ids'] = list(self.local_gpus)
        self.state['temporary_local_gpus'] = temporary_local_gpus
        self.remote_controller = f"{REMOTE_BASE}/delayed-temporal-controllers/{identity['source_commit']}"
        self.event('controller_started', **identity, active_local_gpu_ids=list(self.local_gpus),
                   temporary_local_gpus=temporary_local_gpus)

    def save(self) -> None:
        self.state['updated_at_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        runtime_files.atomic_json(self.state_path, self.state)

    def event(self, kind: str, **fields: Any) -> None:
        event = {'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'event': kind, **fields}
        with (self.root / 'events.jsonl').open('a') as handle:
            handle.write(json.dumps(event, sort_keys=True, allow_nan=False) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        print(json.dumps(event, sort_keys=True), flush=True)

    def remote(self, arguments: list[str], *, timeout: int = 40) -> str:
        return subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                                        'gate1', shlex.join(arguments)], text=True, timeout=timeout)

    def transfer(self, files: list[str], *, pull: bool) -> None:
        for name in files:
            if Path(name).is_absolute() or '..' in Path(name).parts or '\n' in name:
                raise ValueError('Unsafe transfer path')
        local, remote = str(self.root) + '/', 'gate1:' + self.remote_root + '/'
        source, target = (remote, local) if pull else (local, remote)
        subprocess.run(['rsync', '-a', '--files-from=-', '--ignore-missing-args', source, target],
                       input='\n'.join(files) + '\n', text=True, check=True, timeout=90)

    def check_preparation(self) -> bool:
        if self.prep_verified:
            return True
        if self.remote(['test', '-f', self.remote_root + '/ubai/prep-result.json']) == '':
            self.transfer(['ubai/prep-result.json'], pull=True)
        path = self.root / 'ubai/prep-result.json'
        if not path.exists():
            return False
        value = read_json(path)
        if (value.get('state') != 'verified' or value.get('python_version') != '3.12.13'
                or value.get('source_commit') != self.experiment['source_commit']
                or value.get('deployment_sha256') != identity.sha256_file(self.root / 'ubai/deployment.json')):
            raise NeedsAttention('Slurm preparation identity failed')
        self.prep_verified = True
        self.event('ubai_preparation_verified', job_id=value['job_id'])
        return True

    def completed(self, task: dict) -> dict | None:
        path = self.root / task['result_file']
        if not path.exists():
            return None
        result = read_json(path)
        validate_result(task, result, self.experiment, self.root)
        return result

    def results(self) -> list[dict]:
        result = []
        for path in sorted(self.root.glob('tasks/*.json')):
            row = self.completed(read_json(path))
            if row is not None:
                result.append(row)
        return result

    def prepare_task(self, task: dict, ordinal: int, force_host: str | None = None) -> None:
        validate_task(task, self.experiment)
        runtime_files.immutable_json(self.root / 'tasks' / (task['run_id'] + '.json'), task)
        assignment = self.state['tasks'].setdefault(task['run_id'], {
            'status': 'pending', 'attempt': 0, 'preferred_host': force_host or default_host(task, ordinal),
            'fixed_host': force_host or task.get('host_label'),
            'task_sha256': identity.json_sha256(task)})
        if assignment['task_sha256'] != identity.json_sha256(task):
            raise ValueError('Task assignment identity changed')

    def start_local(self, task: dict, gpu: int) -> bool:
        if gpu not in self.local_gpus:
            raise ValueError('Local GPU is not allowed')
        if any(row.get('host') == 'local' and row.get('gpu') == gpu
               and row['status'] in {'starting', 'running'} for row in self.state['tasks'].values()):
            return False
        shared_locks = Path('/data/delayed-temporal/artifacts/runtime/gpu-locks')
        shared_locks.mkdir(parents=True, exist_ok=True)
        lock = (shared_locks / f'gpu-{gpu}.lock').open('a')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            return False
        try:
            admission = local_gpu.gpu_activity(gpu_ids=self.local_gpus)[gpu]
        except (subprocess.SubprocessError, OSError, ValueError) as exc:
            self.state['local_wait_reason'] = str(exc)
            lock.close()
            return False
        if not local_gpu.gpu_available(admission):
            lock.close()
            return False
        available_cpus = sorted(os.sched_getaffinity(0))
        offset = CPU_GPU_ORDER.index(gpu) * 4
        if offset + 4 > len(available_cpus):
            self.state['local_wait_reason'] = 'Insufficient separate CPU cores for the local GPU slot'
            lock.close()
            return False
        assigned_cpus = available_cpus[offset:offset + 4]
        row = self.state['tasks'][task['run_id']]
        row.update(status='starting', host='local', gpu=gpu, attempt=row['attempt'] + 1,
                   gpu_admission=admission)
        self.save()
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4',
                           MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', WANDB_MODE='disabled',
                           WANDB_DISABLED='true', PYTHONDONTWRITEBYTECODE='1')
        environment.pop('WANDB_API_KEY', None)
        command = [self.experiment['python_bin'], '-u', str(REPO / LOCAL_WORKER),
                   '--experiment', str(self.root / 'experiment.json'), '--task', str(self.root / 'tasks' / (task['run_id'] + '.json')),
                   '--output-root', str(self.root), '--host-label', 'local']
        if self.local_gpus != LOCAL_GPUS:
            command.append('--temporary-local-gpus')
        with (self.root / 'worker_logs' / f"{task['run_id']}.attempt-{row['attempt']}.log").open('a') as output:
            process = subprocess.Popen(command, cwd=self.source, env=environment, stdout=output,
                                       stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock.fileno(),))
        os.sched_setaffinity(process.pid, assigned_cpus)
        self.children[task['run_id']] = process
        self.gpu_locks[task['run_id']] = lock
        row.update(status='running', pid=process.pid, started_at=time.time(), cpu_ids=assigned_cpus,
                   local_worker_path=str(REPO / LOCAL_WORKER))
        self.save()
        self.event('task_started', run_id=task['run_id'], host='local', gpu=gpu,
                   attempt=row['attempt'], gpu_admission=admission)
        return True

    def start_remote(self, task: dict, queue: list[dict]) -> None:
        row = self.state['tasks'][task['run_id']]
        if row.get('rebalance_target') == 'local':
            raise ValueError('A condition returned to local execution cannot be resubmitted remotely')
        name = self.prefix + task['run_id']
        # Recover submission if SSH disconnected after sbatch accepted it.
        existing = [q for q in queue if q['name'] == name]
        if len(existing) > 1:
            raise NeedsAttention('Duplicate Slurm task names require inspection')
        if existing:
            row.update(status='running', host='ubai', job_id=existing[0]['job_id'])
            self.save()
            return
        files = ['experiment.json', 'tasks/' + task['run_id'] + '.json']
        if task['calibration_file'] and task['kind'] != 'collect':
            files.append(task['calibration_file'])
        self.transfer(files, pull=False)
        row.update(status='submitting', host='ubai', attempt=row['attempt'] + 1, slurm_name=name)
        row.pop('pair_id', None)
        self.save()
        exports = 'ALL,EXPERIMENT_SOURCE=' + self.remote_source + ',EXPERIMENT_DEPLOYMENT=' + self.remote_root + '/ubai/deployment.json,TASK_ID=' + task['run_id']
        response = self.remote(['sbatch', '--parsable', '--job-name=' + name, '--export=' + exports,
                                '--output=' + self.remote_root + '/slurm/%x-%j.out',
                                '--error=' + self.remote_root + '/slurm/%x-%j.err',
                                self.remote_source + '/scripts/experiments/ubai/calibrated_three_sweep_task.sbatch'])
        job_id = response.strip().split(';')[0]
        if not job_id.isdecimal():
            raise NeedsAttention('Unexpected Slurm submission response')
        row.update(status='running', job_id=job_id, started_at=time.time())
        self.save()
        self.event('task_started', run_id=task['run_id'], host='ubai', job_id=job_id, attempt=row['attempt'])

    def start_remote_pair(self, tasks: list[dict], queue: list[dict]) -> None:
        if not pair_tasks_compatible(tasks):
            raise ValueError('Paired experiments must be distinct and belong to the same stage')
        if any(self.state['tasks'][task['run_id']]['status'] != 'pending'
               or self.state['tasks'][task['run_id']].get('fixed_host') == 'local'
               or self.state['tasks'][task['run_id']].get('rebalance_target') == 'local' for task in tasks):
            raise ValueError('Only pending cluster experiments can be paired')
        if quota_available(queue, self.prefix, 2) < 1:
            raise ValueError('The paired job exceeds the available Slurm allocation')
        controller_record = self.state['controller_identity']
        if not self.pair_controller_verified:
            head = self.remote(['git', '-C', self.remote_controller, 'rev-parse', 'HEAD']).strip()
            if head != controller_record['source_commit']:
                raise NeedsAttention('The remote paired controller checkout is not synchronized')
            self.pair_controller_verified = True
        entries = [{'run_id': task['run_id'],
                    'task_sha256': identity.sha256_file(self.root / 'tasks' / (task['run_id'] + '.json'))}
                   for task in tasks]
        attempts = [self.state['tasks'][task['run_id']]['attempt'] + 1 for task in tasks]
        pair_id = 'pair-' + identity.json_sha256({
            'tasks': entries,
            'attempts': attempts,
            'controller_commit': controller_record['source_commit'],
        })[:24]
        pair = {'format_version': 1, 'pair_id': pair_id,
                'experiment_sha256': identity.sha256_file(self.root / 'experiment.json'),
                'deployment_sha256': identity.sha256_file(self.root / 'ubai/deployment.json'),
                'source_commit': self.experiment['source_commit'],
                'controller_commit': controller_record['source_commit'],
                'controller_sha256': controller_record['pair_runtime_sha256'], 'tasks': entries}
        runtime_files.immutable_json(self.root / 'pairs' / (pair_id + '.json'), pair)
        files = ['experiment.json', 'pairs/' + pair_id + '.json']
        for task in tasks:
            files.append('tasks/' + task['run_id'] + '.json')
            if task['calibration_file'] and task['kind'] != 'collect':
                files.append(task['calibration_file'])
        self.transfer(list(dict.fromkeys(files)), pull=False)
        name = self.prefix + pair_id
        if any(row['name'] == name for row in queue):
            raise NeedsAttention('A pending pair already has a Slurm submission')
        for task, attempt in zip(tasks, attempts):
            self.state['tasks'][task['run_id']].update(
                status='submitting', host='ubai', attempt=attempt, pair_id=pair_id, slurm_name=name)
        self.save()
        exports = ','.join(('ALL', 'EXPERIMENT_SOURCE=' + self.remote_source,
                            'EXPERIMENT_DEPLOYMENT=' + self.remote_root + '/ubai/deployment.json',
                            'CONTROLLER_SOURCE=' + self.remote_controller,
                            'CONTROLLER_COMMIT=' + controller_record['source_commit'],
                            'PAIR_MANIFEST=' + self.remote_root + '/pairs/' + pair_id + '.json'))
        response = self.remote(['sbatch', '--parsable', '--job-name=' + name, '--export=' + exports,
                                '--output=' + self.remote_root + '/slurm/%x-%j.out',
                                '--error=' + self.remote_root + '/slurm/%x-%j.err',
                                self.remote_controller + '/' + PAIR_FILES[1]])
        job_id = response.strip().split(';')[0]
        if not job_id.isdecimal():
            raise NeedsAttention('Unexpected paired Slurm submission response')
        for task in tasks:
            row = self.state['tasks'][task['run_id']]
            row.update(status='running', job_id=job_id, started_at=time.time())
        self.save()
        self.event('paired_job_started', pair_id=pair_id, job_id=job_id,
                   run_ids=[task['run_id'] for task in tasks], gpus=2)

    def schedule_remote(self, tasks: list[dict], queue: list[dict]) -> list[dict]:
        pending = [task for task in tasks if self.state['tasks'][task['run_id']]['status'] == 'pending'
                   and self.state['tasks'][task['run_id']].get('fixed_host') != 'local'
                   and self.state['tasks'][task['run_id']].get('rebalance_target') != 'local']
        submitted = []
        while pending and quota_available(queue, self.prefix):
            first = pending[0]
            peers = sorted(pending[1:], key=lambda task: task['kind'] != first['kind'])
            peer = next((task for task in peers if pair_tasks_compatible([first, task])), None)
            if peer is not None and quota_available(queue, self.prefix, 2):
                batch = [first, peer]
                self.start_remote_pair(batch, queue)
            else:
                batch = [first]
                self.start_remote(first, queue)
            row = self.state['tasks'][first['run_id']]
            if not any(item['job_id'] == row['job_id'] for item in queue):
                queue.append({'job_id': row['job_id'], 'name': row.get('slurm_name', self.prefix + first['run_id']),
                              'state': 'PENDING', 'gpus': len(batch)})
            for task in batch:
                pending.remove(task)
            submitted.extend(batch)
        return submitted

    def finished_local(self, task: dict, row: dict) -> bool:
        process = self.children.get(task['run_id'])
        if process is not None:
            return process.poll() is not None
        pid = row.get('pid')
        if not pid:
            # A crash between Popen and saving its pid cannot be treated as no work.
            raise NeedsAttention('Local launch was interrupted before its PID was recorded')
        path = Path(f'/proc/{pid}/cmdline')
        if not path.exists():
            return True
        if not path.read_bytes():
            # An adopted child may briefly be a zombie before its new parent reaps it.
            try:
                status = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0]
            except FileNotFoundError:
                return True
            if status == 'Z':
                return True
        if not self.local_process_matches(task['run_id'], row):
            raise NeedsAttention('Stored local PID no longer matches the assigned task')
        return False

    def local_process_matches(self, run_id: str, row: dict) -> bool:
        try:
            arguments = Path(f"/proc/{row['pid']}/cmdline").read_bytes().decode().split('\0')
        except (FileNotFoundError, ProcessLookupError, UnicodeDecodeError):
            return False
        workers = {str(self.source / 'scripts/experiments/run_calibrated_three_sweep_task.py'),
                   row.get('local_worker_path', str(REPO / LOCAL_WORKER))}
        if not workers.intersection(arguments):
            return False
        for flag, value in (('--experiment', str(self.root / 'experiment.json')),
                            ('--task', str(self.root / 'tasks' / (run_id + '.json'))),
                            ('--output-root', str(self.root)), ('--host-label', 'local')):
            if arguments.count(flag) != 1:
                return False
            index = arguments.index(flag)
            if index + 1 >= len(arguments) or arguments[index + 1] != value:
                return False
        return True

    def rebalance_pending(self, tasks: list[dict], queue: list[dict], local_slots: int) -> int:
        from scripts.experiments.calibrated_three_sweep_rebalance import rebalance_pending
        return rebalance_pending(self, tasks, queue, local_slots)

    def poll_rebalance(self, task: dict, queue: list[dict] | None) -> dict | None:
        from scripts.experiments.calibrated_three_sweep_rebalance import poll_rebalance
        return poll_rebalance(self, task, queue)

    def poll_task(self, task: dict, queue: list[dict] | None) -> dict | None:
        row = self.state['tasks'][task['run_id']]
        if row['status'] == 'cancelling_for_local':
            return self.poll_rebalance(task, queue)
        if row['status'] in {'pending', 'complete'}:
            result = self.completed(task)
            if row['status'] == 'complete' and result is None:
                raise NeedsAttention('A previously completed result is missing')
            return result
        if row['host'] == 'local':
            finished = self.finished_local(task, row)
        else:
            if queue is None:
                return None
            if row['status'] == 'submitting':
                # Do not resubmit an ambiguous launch automatically.
                existing = [q for q in queue if q['name'] == row.get('slurm_name', self.prefix + task['run_id'])]
                if len(existing) != 1:
                    raise NeedsAttention('Ambiguous Slurm submission requires accounting inspection')
                row.update(status='running', job_id=existing[0]['job_id'])
                self.save()
            finished = not any(q['job_id'] == row['job_id'] for q in queue)
            if finished:
                accounting = self.remote(['sacct', '-n', '-X', '-j', row['job_id'], '--format=State', '--parsable2']).strip()
                terminal = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'PREEMPTED', 'NODE_FAIL', 'BOOT_FAIL', 'DEADLINE'}
                states = [line.split('|')[0].split()[0].rstrip('+') for line in accounting.splitlines() if line.strip()]
                if not states or any(state not in terminal for state in states):
                    return None
                files = [task['result_file'], task['log_file']]
                if task['kind'] == 'collect':
                    files.append(task['calibration_file'])
                self.transfer(files, pull=True)
        if not finished:
            return None
        if task['run_id'] in self.gpu_locks:
            self.gpu_locks.pop(task['run_id']).close()
        result = self.completed(task)
        if result is None:
            self.event('task_failed', run_id=task['run_id'], host=row['host'], attempt=row['attempt'])
            if row['attempt'] >= self.max_attempts:
                raise NeedsAttention(f"Task {task['run_id']} failed {row['attempt']} times; logs preserved")
            row['status'] = 'pending'
        else:
            row.update(status='complete', elapsed_seconds=result['elapsed_seconds'], finished_at=time.time())
            self.event('task_completed', run_id=task['run_id'], host=row['host'],
                       correct=result.get('correct'), samples=result.get('samples'),
                       elapsed_seconds=result['elapsed_seconds'])
        self.save()
        return result

    def report(self, *, seed: int | None = None, final: bool = False) -> dict:
        from scripts.analysis.summarize_calibrated_three_sweeps import summarize
        return summarize(self.root, snapshot_seed=seed, require_complete=final)

    def estimate_seconds(self, task: dict, host: str) -> float:
        defaults = {'collect': 1500, 'noise': 1980, 'smoke_noise': 80, 'smoke_clean': 45, 'dense': 384}
        measured = [row['elapsed_seconds'] for run_id, row in self.state['tasks'].items()
                    if row['status'] == 'complete' and row.get('host') == host
                    and 'elapsed_seconds' in row
                    and read_json(self.root / 'tasks' / (run_id + '.json'))['kind'] == task['kind']]
        return statistics.median(measured) if measured else defaults.get(task['kind'], 1140)

    def stop_owned(self) -> None:
        for run_id, row in self.state['tasks'].items():
            if row['status'] not in {'starting', 'running', 'submitting', 'cancelling_for_local'}:
                continue
            if row.get('host') == 'local' and row.get('pid'):
                if self.local_process_matches(run_id, row):
                    os.kill(row['pid'], signal.SIGTERM)
            elif row.get('host') == 'ubai' and row.get('job_id'):
                try:
                    current = slurm.parse_queue(self.remote(['squeue', '-h', '-r', '-j', row['job_id'], '-o', '%i|%T|%j|%b']))
                    if any(q['name'] == row.get('slurm_name', self.prefix + run_id) for q in current):
                        self.remote(['scancel', row['job_id']])
                except (subprocess.SubprocessError, OSError) as exc:
                    self.event('stop_requires_retry', run_id=run_id, reason=str(exc))

    def run_tasks(self, phase: str, tasks: list[dict], *, force_hosts: dict[str, str] | None = None) -> list[dict]:
        for ordinal, task in enumerate(tasks):
            self.prepare_task(task, ordinal, (force_hosts or {}).get(task['run_id']))
        runtime_files.immutable_json(self.root / 'phases' / (phase + '.json'), {
            'phase': phase, 'experiment_sha256': identity.json_sha256(self.experiment),
            'tasks': [
                {'run_id': task['run_id'], 'task_sha256': identity.json_sha256(task)}
                for task in tasks
            ],
        })
        self.state['phase'] = phase
        self.state.pop('reason', None)
        self.save()
        done = {}
        last_summary_count = -1
        while len(done) < len(tasks):
            queue, remote_slots = None, 0
            try:
                queue = slurm.parse_queue(self.remote(['squeue', '-h', '-r', '-u', 'sizz1997', '-o', '%i|%T|%j|%b']))
                if self.check_preparation():
                    remote_slots = quota_available(queue, self.prefix)
                    self.state.pop('remote_wait_reason', None)
            except (subprocess.SubprocessError, OSError) as exc:
                self.state['remote_wait_reason'] = str(exc)
            for task in tasks:
                if task['run_id'] not in done:
                    try:
                        result = self.poll_task(task, queue)
                    except (subprocess.SubprocessError, OSError) as exc:
                        self.state['remote_wait_reason'] = str(exc)
                        continue
                    if result is not None:
                        done[task['run_id']] = result
                        self.state['tasks'][task['run_id']]['status'] = 'complete'
            if len(done) != last_summary_count:
                self.report()
                last_summary_count = len(done)
            try:
                activity = local_gpu.gpu_activity(gpu_ids=self.local_gpus)
                self.state['local_gpu_activity'] = activity
                self.state.pop('local_wait_reason', None)
            except (subprocess.SubprocessError, OSError, ValueError) as exc:
                # Missing telemetry prevents new local work, not existing cluster work.
                activity = {}
                self.state['local_wait_reason'] = str(exc)
            reserved = {r['gpu'] for r in self.state['tasks'].values()
                        if r['status'] in {'starting', 'running'} and r.get('host') == 'local'}
            local_free = [
                gpu for gpu in self.local_gpus
                if gpu in activity and local_gpu.gpu_available(activity[gpu]) and gpu not in reserved
            ]
            if queue is not None and local_free:
                try:
                    returned = sum(self.state['tasks'][task['run_id']]['status'] == 'pending'
                                   and self.state['tasks'][task['run_id']].get('rebalance_target') == 'local'
                                   for task in tasks)
                    self.rebalance_pending(tasks, queue, max(0, len(local_free) - returned))
                except (subprocess.SubprocessError, OSError) as exc:
                    self.state['remote_wait_reason'] = str(exc)
            moving = sum(self.state['tasks'][task['run_id']]['status'] == 'cancelling_for_local' for task in tasks)
            # Reserve newly free local slots until guarded cancellation is confirmed.
            local_free = local_free[min(moving, len(local_free)):]
            pending = [task for task in tasks if self.state['tasks'][task['run_id']]['status'] == 'pending']
            # Start each host's planned share first; only unsubmitted tasks can move.
            for allow_move in (False, True):
                remote_pending = []
                for task in list(pending):
                    row = self.state['tasks'][task['run_id']]
                    preferred = row['preferred_host']
                    fixed = row.get('fixed_host') or row.get('rebalance_target')
                    host = fixed or preferred
                    if allow_move and not fixed:
                        if host == 'local' and not local_free and remote_slots:
                            host = 'ubai'
                        elif host == 'ubai' and not remote_slots and local_free:
                            host = 'local'
                    if local_free and not fixed and queue is not None and any(q['state'] == 'PENDING' for q in queue):
                        host = 'local'
                    if host == 'local' and local_free:
                        gpu = local_free.pop(0)
                        if self.start_local(task, gpu):
                            pending.remove(task)
                    elif host == 'ubai' and remote_slots and queue is not None:
                        remote_pending.append(task)
                if remote_pending and queue is not None:
                    try:
                        self.schedule_remote(remote_pending, queue)
                        remote_slots = quota_available(queue, self.prefix)
                    except (subprocess.SubprocessError, OSError) as exc:
                        self.state['remote_wait_reason'] = str(exc)
                        # Submitting state is deliberately not returned to pending.
                        remote_slots = 0
                    pending = [task for task in pending
                               if self.state['tasks'][task['run_id']]['status'] == 'pending']
            self.state.update(phase_completed=len(done), phase_total=len(tasks))
            self.save()
            if len(done) < len(tasks):
                time.sleep(self.poll_seconds)
        return [done[task['run_id']] for task in tasks]

    def run(self) -> None:
        collections = [make_task(self.experiment, 'collect', theta_index=i) for i in range(9)]
        self.run_tasks('calibration', collections)
        table_hashes = {
            i: identity.sha256_file(self.root / f'calibration/theta_{i:02d}.json')
            for i in range(9)
        }
        smokes = [make_task(self.experiment, kind, theta_index=4, host_label=host,
                           seed=0 if kind == 'smoke_noise' else None,
                           rt=1e-5 if kind == 'smoke_noise' else 0,
                           ratio=4 if kind == 'smoke_noise' else 0,
                           calibration_sha256=table_hashes[4], expected_samples=160)
                  for kind in ('smoke_clean', 'smoke_noise') for host in ('local', 'ubai')]
        smoke_results = self.run_tasks('environment-smoke', smokes)
        for kind in ('smoke_clean', 'smoke_noise'):
            pair = [row for row in smoke_results if row['kind'] == kind]
            if len({(r['correct'], r['prediction_sha256']) for r in pair}) != 1:
                raise NeedsAttention(f'Environment prediction mismatch for {kind}')
        self.event('environment_predictions_match')
        theta_results = self.run_tasks('theta-evaluation', [
            *[make_task(self.experiment, kind, theta_index=i, calibration_sha256=table_hashes[i])
              for kind in ('theta_train', 'theta_validation') for i in range(9)],
            make_task(self.experiment, 'dense')])
        training = [row for row in theta_results if row['kind'] == 'theta_train']
        validated = [row for row in theta_results if row['kind'] == 'theta_validation']
        selection = select_theta(training)
        runtime_files.immutable_json(self.root / 'training-selection.json', selection)
        index = selection['theta_index']
        original_host = next(r['host_label'] for r in training if r['theta_index'] == index)
        replay = make_task(self.experiment, 'theta_replay', theta_index=index, calibration_sha256=table_hashes[index])
        confirmation = self.run_tasks('theta-replay', [replay],
            force_hosts={replay['run_id']: 'ubai' if original_host == 'local' else 'local'})
        replay_result = next(row for row in confirmation if row['kind'] == 'theta_replay')
        confirmed = confirm_selection(selection, validated, replay_result, training)
        runtime_files.immutable_json(self.root / 'selection.json', confirmed)
        self.event('theta_confirmed', theta=confirmed['theta'], correct=confirmed['validation_correct'])
        clean_accuracy = confirmed['validation_correct'] / 5000
        for seed in range(3):
            assert_seed_barrier(seed, self.results(), self.experiment, confirmed['theta'], table_hashes[index])
            tasks = make_tasks(self.experiment, 'noise', selected_theta=confirmed['theta'], seed=seed,
                               calibration_sha256=table_hashes[index])
            results = self.run_tasks(f'noise-seed-{seed}', tasks)
            self.report(seed=seed, final=seed == 2)
            self.event('seed_completed', seed=seed, snapshot=f'outputs/seed-{seed}', completed_conditions=17)
            if seed == 0:
                reason = seed_range_reason(results, clean_accuracy)
                if reason:
                    raise NeedsAttention(reason + '; request a new range before seed one')
        self.state['phase'] = 'complete'
        if self.state.get('temporary_local_gpus'):
            self.local_gpus = LOCAL_GPUS
            self.state.update(active_local_gpu_ids=list(LOCAL_GPUS), temporary_local_gpus=False)
            self.event('temporary_local_gpu_permission_ended', local_gpu_ids=list(LOCAL_GPUS))
        self.save()
        self.event('campaign_completed', evaluations=71, calibration_collections=9)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-root', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, default=Path('/data/delayed-temporal-worktrees/calibrated-three-sweeps'))
    parser.add_argument('--python-bin', default='/opt/conda/envs/dt/bin/python')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--initialize', action='store_true')
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--status', action='store_true')
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--temporary-local-gpus', action='store_true',
                        help='Temporarily permit local devices 0–3 for this frozen campaign only')
    args = parser.parse_args()
    if args.temporary_local_gpus and not args.run:
        parser.error('--temporary-local-gpus is only valid with --run')
    root = args.experiment_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.initialize:
        experiment = initialize(root, args.source_root, args.python_bin)
        print(json.dumps({'tag': TAG, 'source_commit': experiment['source_commit'], 'state': 'prepared'}))
        return
    if args.status:
        print(json.dumps(read_json(root / 'assignments.json'), indent=2))
        return
    with (root / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        controller = Controller(root, poll_seconds=args.poll_seconds,
                                temporary_local_gpus=args.temporary_local_gpus)
        def interrupted(signum: int, _frame: Any) -> None:
            raise NeedsAttention(f'Controller interrupted by signal {signum}; partial outputs preserved')
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        try:
            controller.run()
        except Exception as exc:
            controller.stop_owned()
            controller.state.update(phase='needs_attention', reason=str(exc))
            controller.save()
            controller.event('needs_attention', reason=str(exc))
            raise


if __name__ == '__main__':
    main()
