"""Coordinate calibrated theta selection and noise sweeps in seed order across two hosts."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.experiments.calibrated_three_sweeps import (
    TAG, RangeInsufficient, confirm_selection, make_task, make_tasks, rt_grid,
    select_theta, sha256_file, task_sha256, theta_grid, validate_experiment,
    validate_result, validate_task, write_immutable_json,
)

LOCAL_GPUS = (4, 5, 6, 7)
ASSETS = Path('/data/delayed-temporal/artifacts/assets/theta-selection-v1')
REMOTE_BASE = '/home1/sizz1997/myubai'
MAIN_TASKS = 80


class NeedsAttention(RuntimeError):
    """An integrity failure or predeclared scientific gate needs user direction."""


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f'Refusing a symlink: {path}')
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w') as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def initialize(root: Path, source: Path, python_bin: str) -> dict:
    """Freeze source and actual local artifact hashes before creating assignments."""
    from scripts.setup.hash_artifact import artifact_identity
    from scripts.experiments.ubai.prepare_calibrated_three_sweeps_ubai import package_source_identity
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
        identity = artifact_identity(path)
        experiment[prefix + '_path'] = str(path)
        experiment[prefix + '_sha256'] = identity['aggregate_sha256']
    for prefix, relative in (
        ('evaluator', 'scripts/evaluation/error_analysis_vit.py'),
        ('calibration_evaluator', 'scripts/analysis/evaluate_calibrated_vit.py'),
        ('gelu_evaluator', 'scripts/analysis/gelu_cubic_phi_nl_vit.py'),
    ):
        experiment[prefix + '_path'] = relative
        experiment[prefix + '_sha256'] = sha256_file(source / relative)
    experiment['runtime_sha256'] = {relative: sha256_file(source / relative) for relative in (
        'scripts/experiments/run_calibrated_three_sweeps.py',
        'scripts/experiments/run_calibrated_three_sweep_task.py',
        'scripts/experiments/calibrated_three_sweeps.py',
        'scripts/analysis/summarize_calibrated_three_sweeps.py',
        'scripts/experiments/ubai/prepare_calibrated_three_sweeps_ubai.py',
        'scripts/experiments/ubai/calibrated_three_sweep_task.sbatch',
        'scripts/experiments/ubai/calibrated_three_sweep_prep.sbatch',
        'scripts/experiments/ubai/calibrated_git.sh',
    )}
    experiment['dependency_sha256'] = {
        package: package_source_identity(source / 'src' / package / subtree)[0]
        for package, subtree in (('transformers', 'src'), ('spikingjelly', 'spikingjelly'))
    }
    validate_experiment(experiment)
    check_source(experiment)
    write_immutable_json(root / 'experiment.json', experiment)
    return experiment


def parse_gpu_occupancy(devices: str, applications: str) -> dict[int, set[int]]:
    uuids = {}
    for line in devices.strip().splitlines():
        index, uuid = [part.strip() for part in line.split(',')]
        uuids[uuid] = int(index)
    if not set(LOCAL_GPUS).issubset(uuids.values()):
        raise ValueError('Required local GPU indices are missing')
    occupied = {index: set() for index in LOCAL_GPUS}
    for line in applications.strip().splitlines():
        if not line.strip():
            continue
        uuid, pid = [part.strip() for part in line.split(',')]
        if uuid not in uuids or not pid.isdecimal():
            raise ValueError('Incomplete GPU occupancy information')
        if uuids[uuid] in occupied:
            # Host PIDs can belong to another container and need not exist in /proc here.
            occupied[uuids[uuid]].add(int(pid))
    return occupied


def gpu_occupancy() -> dict[int, set[int]]:
    def query(fields: str, kind: str) -> str:
        return subprocess.check_output(['nvidia-smi', f'--query-{kind}={fields}',
                                        '--format=csv,noheader,nounits'], text=True, timeout=15)
    return parse_gpu_occupancy(query('index,uuid', 'gpu'), query('gpu_uuid,pid', 'compute-apps'))


def default_host(task: dict, ordinal: int) -> str:
    if task.get('host_label'):
        return task['host_label']
    index = task['theta_index'] if task['kind'].startswith('theta') or task['kind'] == 'collect' else ordinal
    return 'local' if index % 3 == 0 else 'ubai'


def parse_queue(text: str) -> list[dict]:
    rows = []
    for line in text.strip().splitlines():
        job, state, name, resources = line.split('|', 3)
        matches = re.findall(r'gpu(?::[^:,]+)?:([0-9]+)', resources)
        rows.append({'job_id': job.strip(), 'state': state.strip(), 'name': name.strip(),
                     'gpus': sum(map(int, matches))})
    return rows


def quota_available(queue: list[dict], campaign_prefix: str) -> int:
    own = [row for row in queue if row['name'].startswith(campaign_prefix)]
    running = [row for row in queue if row['state'] not in {'PENDING', 'CONFIGURING'}]
    # Reserve for submitted jobs as well, so queued work cannot exceed account limits later.
    return max(0, min(8 - len(own), 20 - len(queue), 10 - len(queue),
                      12 - sum(row['gpus'] for row in queue), 10 - len(running)))


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
    def __init__(self, root: Path, *, poll_seconds: float = 30, max_attempts: int = 3):
        self.root = root.resolve()
        self.experiment = read_json(root / 'experiment.json')
        validate_experiment(self.experiment)
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
        self.state_path = root / 'assignments.json'
        self.state = read_json(self.state_path) if self.state_path.exists() else {
            'experiment_sha256': task_sha256(self.experiment), 'tasks': {}, 'phase': 'prepared'}
        if self.state['experiment_sha256'] != task_sha256(self.experiment):
            raise ValueError('Assignment experiment identity mismatch')
        self.children: dict[str, subprocess.Popen] = {}
        self.gpu_locks: dict[str, Any] = {}
        self.prep_verified = False
        self.root.joinpath('worker_logs').mkdir(parents=True, exist_ok=True)

    def save(self) -> None:
        self.state['updated_at_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        atomic_json(self.state_path, self.state)

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
                or value.get('deployment_sha256') != sha256_file(self.root / 'ubai/deployment.json')):
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
        write_immutable_json(self.root / 'tasks' / (task['run_id'] + '.json'), task)
        assignment = self.state['tasks'].setdefault(task['run_id'], {
            'status': 'pending', 'attempt': 0, 'preferred_host': force_host or default_host(task, ordinal),
            'fixed_host': force_host or task.get('host_label'), 'task_sha256': task_sha256(task)})
        if assignment['task_sha256'] != task_sha256(task):
            raise ValueError('Task assignment identity changed')

    def start_local(self, task: dict, gpu: int) -> bool:
        if gpu not in LOCAL_GPUS:
            raise ValueError('Local GPU is not allowed')
        shared_locks = Path('/data/delayed-temporal/artifacts/runtime/gpu-locks')
        shared_locks.mkdir(parents=True, exist_ok=True)
        lock = (shared_locks / f'gpu-{gpu}.lock').open('a')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            return False
        if gpu_occupancy()[gpu]:
            lock.close()
            return False
        row = self.state['tasks'][task['run_id']]
        row.update(status='starting', host='local', gpu=gpu, attempt=row['attempt'] + 1)
        self.save()
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4',
                           MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', WANDB_MODE='disabled',
                           WANDB_DISABLED='true', PYTHONDONTWRITEBYTECODE='1')
        environment.pop('WANDB_API_KEY', None)
        command = [self.experiment['python_bin'], '-u', str(self.source / 'scripts/experiments/run_calibrated_three_sweep_task.py'),
                   '--experiment', str(self.root / 'experiment.json'), '--task', str(self.root / 'tasks' / (task['run_id'] + '.json')),
                   '--output-root', str(self.root), '--host-label', 'local']
        with (self.root / 'worker_logs' / f"{task['run_id']}.attempt-{row['attempt']}.log").open('a') as output:
            process = subprocess.Popen(command, cwd=self.source, env=environment, stdout=output,
                                       stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock.fileno(),))
        available_cpus = sorted(os.sched_getaffinity(0))
        offset = LOCAL_GPUS.index(gpu) * 4
        assigned_cpus = [available_cpus[(offset + n) % len(available_cpus)] for n in range(4)]
        os.sched_setaffinity(process.pid, assigned_cpus)
        self.children[task['run_id']] = process
        self.gpu_locks[task['run_id']] = lock
        row.update(status='running', pid=process.pid, started_at=time.time(), cpu_ids=assigned_cpus)
        self.save()
        self.event('task_started', run_id=task['run_id'], host='local', gpu=gpu, attempt=row['attempt'])
        return True

    def start_remote(self, task: dict, queue: list[dict]) -> None:
        row = self.state['tasks'][task['run_id']]
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
        row.update(status='submitting', host='ubai', attempt=row['attempt'] + 1)
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
        command = path.read_bytes().replace(b'\0', b' ').decode(errors='replace')
        if 'run_calibrated_three_sweep_task.py' not in command or task['run_id'] not in command:
            raise NeedsAttention('Stored local PID no longer matches the assigned task')
        return False

    def poll_task(self, task: dict, queue: list[dict] | None) -> dict | None:
        row = self.state['tasks'][task['run_id']]
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
                existing = [q for q in queue if q['name'] == self.prefix + task['run_id']]
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
            if row['status'] not in {'starting', 'running', 'submitting'}:
                continue
            if row.get('host') == 'local' and row.get('pid'):
                command_path = Path(f"/proc/{row['pid']}/cmdline")
                if command_path.exists():
                    command = command_path.read_bytes().replace(b'\0', b' ').decode(errors='replace')
                    if 'run_calibrated_three_sweep_task.py' in command and run_id in command:
                        os.kill(row['pid'], signal.SIGTERM)
            elif row.get('host') == 'ubai' and row.get('job_id'):
                try:
                    current = parse_queue(self.remote(['squeue', '-h', '-r', '-j', row['job_id'], '-o', '%i|%T|%j|%b']))
                    if any(q['name'] == self.prefix + run_id for q in current):
                        self.remote(['scancel', row['job_id']])
                except (subprocess.SubprocessError, OSError) as exc:
                    self.event('stop_requires_retry', run_id=run_id, reason=str(exc))

    def run_tasks(self, phase: str, tasks: list[dict], *, force_hosts: dict[str, str] | None = None) -> list[dict]:
        for ordinal, task in enumerate(tasks):
            self.prepare_task(task, ordinal, (force_hosts or {}).get(task['run_id']))
        write_immutable_json(self.root / 'phases' / (phase + '.json'), {
            'phase': phase, 'experiment_sha256': task_sha256(self.experiment),
            'tasks': [{'run_id': task['run_id'], 'task_sha256': task_sha256(task)} for task in tasks]})
        self.state['phase'] = phase
        self.save()
        done = {}
        last_summary_count = -1
        while len(done) < len(tasks):
            queue, remote_slots = None, 0
            try:
                queue = parse_queue(self.remote(['squeue', '-h', '-r', '-u', 'sizz1997', '-o', '%i|%T|%j|%b']))
                if self.check_preparation():
                    remote_slots = quota_available(queue, self.prefix)
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
            occupied = gpu_occupancy()
            reserved = {r['gpu'] for r in self.state['tasks'].values()
                        if r['status'] in {'starting', 'running'} and r.get('host') == 'local'}
            local_free = [g for g in LOCAL_GPUS if not occupied[g] and g not in reserved]
            pending = [task for task in tasks if self.state['tasks'][task['run_id']]['status'] == 'pending']
            # Start each host's planned share first; only unsubmitted tasks can move.
            for allow_move in (False, True):
                for task in list(pending):
                    row = self.state['tasks'][task['run_id']]
                    preferred = row['preferred_host']
                    host = preferred
                    if allow_move and not row.get('fixed_host'):
                        if host == 'local' and not local_free and remote_slots:
                            host = 'ubai'
                        elif host == 'ubai' and not remote_slots and local_free:
                            host = 'local'
                    if local_free and remote_slots and not row.get('fixed_host'):
                        other = 'ubai' if host == 'local' else 'local'
                        if self.estimate_seconds(task, other) < .95 * self.estimate_seconds(task, host):
                            host = other
                    if host == 'local' and local_free:
                        gpu = local_free.pop(0)
                        if self.start_local(task, gpu):
                            pending.remove(task)
                    elif host == 'ubai' and remote_slots and queue is not None:
                        try:
                            self.start_remote(task, queue)
                        except (subprocess.SubprocessError, OSError) as exc:
                            self.state['remote_wait_reason'] = str(exc)
                            # Submitting state is deliberately not returned to pending.
                            remote_slots = 0
                            continue
                        remote_slots -= 1
                        pending.remove(task)
            self.state.update(phase_completed=len(done), phase_total=len(tasks))
            self.save()
            if len(done) < len(tasks):
                time.sleep(self.poll_seconds)
        return [done[task['run_id']] for task in tasks]

    def run(self) -> None:
        collections = [make_task(self.experiment, 'collect', theta_index=i) for i in range(9)]
        self.run_tasks('calibration', collections)
        table_hashes = {i: sha256_file(self.root / f'calibration/theta_{i:02d}.json') for i in range(9)}
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
        write_immutable_json(self.root / 'training-selection.json', selection)
        index = selection['theta_index']
        original_host = next(r['host_label'] for r in training if r['theta_index'] == index)
        replay = make_task(self.experiment, 'theta_replay', theta_index=index, calibration_sha256=table_hashes[index])
        confirmation = self.run_tasks('theta-replay', [replay],
            force_hosts={replay['run_id']: 'ubai' if original_host == 'local' else 'local'})
        replay_result = next(row for row in confirmation if row['kind'] == 'theta_replay')
        confirmed = confirm_selection(selection, validated, replay_result, training)
        write_immutable_json(self.root / 'selection.json', confirmed)
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
    args = parser.parse_args()
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
        controller = Controller(root, poll_seconds=args.poll_seconds)
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
