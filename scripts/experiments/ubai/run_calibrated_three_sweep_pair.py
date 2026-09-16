#!/usr/bin/env python3
"""Run two independent frozen evaluations with one shared Slurm runtime."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any
import uuid

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.runtime import environment as runtime_environment
from scripts.runtime import files as runtime_files
from scripts.runtime import identity


HELPER = 'scripts/experiments/ubai/prepare_calibrated_three_sweeps_ubai.py'
WORKER = 'scripts/experiments/run_calibrated_three_sweep_task.py'
CONTRACT = 'scripts/experiments/calibrated_three_sweeps.py'
PAIR_SCRIPT = 'scripts/experiments/ubai/run_calibrated_three_sweep_pair.py'
PAIR_BATCH = 'scripts/experiments/ubai/calibrated_three_sweep_pair.sbatch'


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f'Cannot load the frozen runtime module: {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def gpu_tokens(value: str) -> list[str]:
    tokens = value.split(',')
    normalized = [str(int(token)) if token.isdecimal() else token.casefold() for token in tokens]
    if (len(tokens) != 2 or len(set(normalized)) != 2
            or any(not re.fullmatch(r'(?:[0-9]+|GPU-[A-Za-z0-9-]+)', token) for token in tokens)):
        raise ValueError('The paired allocation must expose exactly two distinct GPU devices')
    return tokens


def probe_allocated_devices(visible: str) -> list[str]:
    """Resolve CUDA-visible ordinals to actual device UUIDs in a short-lived child."""
    gpu_tokens(visible)
    environment = dict(os.environ)
    environment['CUDA_VISIBLE_DEVICES'] = visible
    script = (
        'import json,torch; count=torch.cuda.device_count(); '
        'devices=[{"name":torch.cuda.get_device_properties(i).name,'
        '"uuid":str(torch.cuda.get_device_properties(i).uuid)} for i in range(count)]; '
        'print(json.dumps({"count":count,"devices":devices}))'
    )
    result = json.loads(subprocess.check_output(
        ['/opt/conda/envs/dt/bin/python', '-c', script], text=True, env=environment, timeout=60,
    ))
    if result.get('count') != 2 or len(result.get('devices', [])) != 2:
        raise ValueError('The container must expose exactly two allocated CUDA devices')
    devices = []
    for device in result['devices']:
        if 'RTX A6000' not in device.get('name', ''):
            raise ValueError('Both paired devices must be RTX A6000 GPUs')
        identifier = str(device.get('uuid', ''))
        if identifier.startswith('GPU-'):
            identifier = identifier[4:]
        try:
            devices.append('GPU-' + str(uuid.UUID(identifier)))
        except ValueError as error:
            raise ValueError('The allocated device UUID cannot be resolved safely') from error
    if len(set(devices)) != 2:
        raise ValueError('The paired allocation resolved to duplicate physical devices')
    return devices


def controller_identity(root: Path, commit: str, expected_files: dict[str, str]) -> None:
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('An exact controller commit is required')
    if set(expected_files) != {PAIR_SCRIPT, PAIR_BATCH}:
        raise ValueError('Both paired controller files must have frozen hashes')
    environment = dict(os.environ)
    environment.pop('GIT_DIR', None)
    environment.pop('GIT_WORK_TREE', None)
    head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                   text=True, env=environment).strip()
    dirty = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=no'],
                                    text=True, env=environment).strip()
    if head != commit or dirty:
        raise ValueError('Controller checkout differs from its frozen commit')
    for relative, digest in expected_files.items():
        if (not re.fullmatch(r'[0-9a-f]{64}', digest)
                or identity.sha256_file(root / relative) != digest):
            raise ValueError(f'Paired controller file checksum mismatch: {relative}')


def logical_phase(task: dict[str, Any]) -> tuple[str, int | None]:
    kind = task['kind']
    if kind in {'smoke_clean', 'smoke_noise'}:
        return 'smoke', None
    if kind in {'theta_train', 'theta_validation', 'dense'}:
        return 'theta', None
    if kind == 'noise':
        return 'noise', task['seed']
    if kind in {'collect', 'theta_replay'}:
        return kind, None
    raise ValueError('Unknown paired execution stage')


def validate_pair(pair_path: Path, experiment_root: Path, deployment_path: Path,
                  source: Path, controller: Path, commit: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pair = json.loads(pair_path.read_text())
    if pair.get('format_version') != 1 or not re.fullmatch(r'[A-Za-z0-9_.-]{1,100}', pair.get('pair_id', '')):
        raise ValueError('Invalid paired task manifest')
    if pair_path.name != pair['pair_id'] + '.json' or pair_path.parent != experiment_root / 'pairs':
        raise ValueError('Paired manifest must belong to its experiment pairs directory')
    # These are raw file hashes, not reserialized JSON object hashes.
    for path, field in ((experiment_root / 'experiment.json', 'experiment_sha256'),
                        (deployment_path, 'deployment_sha256')):
        if pair.get(field) != identity.sha256_file(path):
            raise ValueError(f'Paired manifest identity mismatch: {field}')
    experiment = json.loads((experiment_root / 'experiment.json').read_text())
    if pair.get('source_commit') != experiment['source_commit'] or pair.get('controller_commit') != commit:
        raise ValueError('Paired manifest source or controller commit differs')
    controller_identity(controller, commit, pair.get('controller_sha256', {}))
    entries = pair.get('tasks', [])
    if (len(entries) != 2 or len({row.get('run_id') for row in entries}) != 2
            or any(not re.fullmatch(r'[A-Za-z0-9_.-]+', row.get('run_id', '')) for row in entries)):
        raise ValueError('A pair requires exactly two distinct task identifiers')
    contract = load_module(source / CONTRACT, '_frozen_pair_contract')
    contract.validate_experiment(experiment)
    tasks = []
    for entry in entries:
        task_path = experiment_root / 'tasks' / (entry['run_id'] + '.json')
        if entry.get('task_sha256') != identity.sha256_file(task_path):
            raise ValueError('Paired task raw file checksum mismatch')
        task = json.loads(task_path.read_text())
        contract.validate_task(task, experiment)
        if task['run_id'] != entry['run_id'] or task.get('host_label') not in {None, 'ubai'}:
            raise ValueError('Paired task identifier or environment differs')
        tasks.append(task)
    if any(task['kind'] == 'noise' for task in tasks):
        if not all(task['kind'] == 'noise' for task in tasks) or len({task['seed'] for task in tasks}) != 1:
            raise ValueError('Paired noise tasks must stay within the same seed stage')
    if any(task['kind'] == 'collect' for task in tasks):
        paths = [task.get('calibration_file') for task in tasks]
        if paths[0] and paths[0] == paths[1]:
            raise ValueError('A collection cannot run beside a consumer of the same table')
    if logical_phase(tasks[0]) != logical_phase(tasks[1]):
        raise ValueError('Paired tasks must remain in the same logical stage')
    return pair, tasks


def reserve_second_scratch(helper: Any, base: Path, job_id: str, pair_id: str) -> Path:
    """Use a second legacy-compatible reservation without extracting another environment."""
    with helper.runtime_lock(base):
        live = helper.sweep_old_runtime(base, job_id)
        reservation = 8 * helper.GIB
        required = sum(item['scratch_bytes'] for item in live) + reservation
        if helper.shutil.disk_usage(base).free < required:
            raise ValueError('Insufficient disk space for the second paired scratch reservation')
        path = Path(tempfile.mkdtemp(prefix=f'{helper.RUNTIME_PREFIX}{job_id}-{pair_id}-scratch-', dir=base))
        owner = {'uid': os.getuid(), 'job_id': job_id, 'task_id': pair_id + '-scratch',
                 'scratch_bytes': reservation}
        try:
            runtime_files.immutable(path / '.owner.json', runtime_files.json_bytes(owner))
            runtime_files.atomic_json(
                base / '.calibrated-three-sweep-reservations.json',
                live + [{'path': str(path), **owner}],
            )
        except BaseException:
            if (path.is_dir() and not path.is_symlink() and path.parent == base
                    and path.stat().st_uid == os.getuid()
                    and path.name.startswith(f'{helper.RUNTIME_PREFIX}{job_id}-')):
                helper.shutil.rmtree(path)
            raise
        return path


def admit_pair_runtime(helper: Any, base: Path, job_id: str, pair_id: str,
                       configuration: dict[str, Any]) -> tuple[Path, Path]:
    runtime = helper.admit_runtime(base, job_id, pair_id, configuration)
    try:
        second = reserve_second_scratch(helper, base, job_id, pair_id)
    except BaseException:
        helper.release_runtime(base, runtime, job_id)
        raise
    return runtime, second


def pair_container_command(helper: Any, deployment: dict[str, Any], runtime: Path,
                           pair_id: str, controller: Path, commit: str) -> list[str]:
    command = helper.container_command(deployment, runtime, pair_id, check_only=False)
    command = ['--gres=gpu:2' if arg == '--gres=gpu:1' else
               '--cpus-per-task=8' if arg == '--cpus-per-task=4' else arg for arg in command]
    command.insert(1, '--cpu-bind=cores')
    for index, argument in enumerate(command):
        if argument.startswith('--container-mounts='):
            command[index] += f',{controller}:{controller}:ro,{runtime}/scratch:{runtime}/scratch'
            break
    python_index = command.index('/opt/conda/envs/dt/bin/python')
    experiment = Path(deployment['experiment_root'])
    return command[:python_index + 1] + [str(controller / PAIR_SCRIPT), '--inside',
        '--source', deployment['source_root'], '--deployment', '/three-sweep-deployment/deployment.json',
        '--pair', str(experiment / 'pairs' / (pair_id + '.json')), '--controller', str(controller),
        '--controller-commit', commit]


def reap_children(children: list[subprocess.Popen]) -> None:
    for child in children:
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for child in children:
        try:
            child.wait(timeout=25)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()


def run_workers(source: Path, experiment: Path, tasks: list[dict[str, Any]], scratch: Path,
                visible: str) -> list[int]:
    devices = gpu_tokens(visible)
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < 8:
        raise ValueError('The paired process requires eight allocated CPU cores')
    children = []
    def stop(signum: int, _frame: Any) -> None:
        raise InterruptedError(f'Paired evaluation interrupted by signal {signum}')
    previous = {signum: signal.signal(signum, stop) for signum in (signal.SIGTERM, signal.SIGINT)}
    try:
        for index, task in enumerate(tasks):
            task_scratch = scratch / f'worker-{index}'
            task_scratch.mkdir(mode=0o700)
            (task_scratch / 'xdg-runtime').mkdir(mode=0o700)
            environment = runtime_environment.worker_environment(
                dict(os.environ), task_scratch, devices[index]
            )
            affinity = set(cpus[4 * index:4 * index + 4])
            command = ['/opt/conda/envs/dt/bin/python', str(source / WORKER),
                       '--experiment', str(experiment / 'experiment.json'),
                       '--task', str(experiment / 'tasks' / (task['run_id'] + '.json')),
                       '--output-root', str(experiment), '--host-label', 'ubai']
            children.append(subprocess.Popen(command, env=environment, start_new_session=True,
                                              preexec_fn=lambda selected=affinity: os.sched_setaffinity(0, selected)))
        # A scientific or technical failure in one condition does not cancel its peer.
        return [child.wait() for child in children]
    finally:
        for signum in previous:
            signal.signal(signum, signal.SIG_IGN)
        try:
            reap_children(children)
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)


def inside(source: Path, deployment_path: Path, pair_path: Path,
           controller: Path, commit: str) -> int:
    deployment = json.loads(deployment_path.read_text())
    if identity.sha256_file(source / HELPER) != deployment['script_sha256'][HELPER]:
        raise ValueError('Frozen helper content differs')
    helper = load_module(source / HELPER, '_frozen_pair_helper_inside')
    helper.require_compute_allocation()
    helper.validate_deployment(deployment)
    if source != Path(deployment['source_root']):
        raise ValueError('Container source path differs from the frozen deployment')
    experiment = Path(deployment['experiment_root'])
    pair, tasks = validate_pair(pair_path, experiment, deployment_path, source, controller, commit)
    scratch = Path(os.environ['TMPDIR'])
    if scratch == Path('/tmp') or scratch.is_relative_to('/tmp'):
        raise ValueError('Paired worker scratch cannot use /tmp')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    devices = probe_allocated_devices(visible)
    print(json.dumps({'pair_id': pair['pair_id'], 'visible_input': visible,
                      'gpu_uuids': devices}, sort_keys=True), flush=True)
    started = time.time_ns()
    codes = run_workers(source, experiment, tasks, scratch, ','.join(devices))
    record = {'pair_id': pair['pair_id'], 'pair_manifest_sha256': identity.sha256_file(pair_path),
              'controller_commit': commit, 'controller_sha256': pair['controller_sha256'],
              'source_commit': pair['source_commit'], 'experiment_sha256': pair['experiment_sha256'],
              'controller_source': str(controller), 'allocation_gpus': 2, 'cpus_per_worker': 4,
              'visible_input': visible, 'gpu_uuids': devices,
              'job_id': os.environ['SLURM_JOB_ID'],
              'tasks': [{'run_id': task['run_id'], 'exit_code': code} for task, code in zip(tasks, codes)]}
    runtime_files.immutable(
        experiment / 'pair-runs' / pair['pair_id'] / f'{os.environ["SLURM_JOB_ID"]}-{started}.json',
        runtime_files.json_bytes(record),
    )
    return 0 if codes == [0, 0] else 1


def host() -> None:
    source = Path(os.environ['EXPERIMENT_SOURCE'])
    controller = Path(os.environ['CONTROLLER_SOURCE'])
    deployment_path = Path(os.environ['EXPERIMENT_DEPLOYMENT'])
    pair_path = Path(os.environ['PAIR_MANIFEST'])
    commit = os.environ['CONTROLLER_COMMIT']
    deployment = json.loads(deployment_path.read_text())
    if identity.sha256_file(source / HELPER) != deployment['script_sha256'][HELPER]:
        raise ValueError('Frozen bootstrap helper content differs')
    helper = load_module(source / HELPER, '_frozen_pair_helper_host')
    job_id = helper.require_compute_allocation()
    for value in (str(source), str(controller), str(deployment_path), str(pair_path)):
        runtime_files.absolute_path(value)
    if (os.environ.get('SLURM_CPUS_PER_TASK') != '8' or os.environ.get('SLURM_NTASKS', '1') != '1'
            or os.environ.get('SLURM_MEM_PER_NODE') != '131072'
            or os.environ.get('SLURM_GPUS_ON_NODE', '2') != '2'
            or os.environ.get('SLURM_JOB_NUM_NODES', '1') != '1'
            or os.environ.get('SLURM_JOB_PARTITION') not in {'gpu4', 'gpu5'}):
        raise ValueError('Paired tasks require two RTX A6000 GPUs, eight CPUs, and 128 GiB memory')
    gpu_tokens(os.environ.get('CUDA_VISIBLE_DEVICES', ''))
    helper.validate_deployment(deployment)
    if source.resolve() != Path(deployment['runtime']['host_source_root']).resolve():
        raise ValueError('Host source path differs from the frozen deployment')
    helper.verify_files(deployment, deployment_path)
    helper.verify_preparation(deployment, deployment_path)
    experiment = Path(deployment['runtime']['host_experiment_root'])
    pair, _tasks = validate_pair(pair_path, experiment, deployment_path, source, controller, commit)
    base = Path('/enroot') / str(os.getuid()) / 'data'
    runtime, reservation = admit_pair_runtime(helper, base, job_id, pair['pair_id'], deployment['runtime'])
    try:
        scratch = runtime / 'scratch'
        (scratch / 'xdg-runtime').mkdir(mode=0o700)
        environment = dict(os.environ)
        for key in ('WANDB_API_KEY', 'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN'):
            environment.pop(key, None)
        for key in ('TMPDIR', 'TMP', 'TEMP'):
            environment[key] = str(scratch)
        for key, name in (('ENROOT_RUNTIME_PATH', 'enroot-runtime'),
                          ('ENROOT_DATA_PATH', 'enroot-data'), ('ENROOT_CACHE_PATH', 'enroot-cache')):
            (scratch / name).mkdir(mode=0o700)
            environment[key] = str(scratch / name)
        environment['NVIDIA_VISIBLE_DEVICES'] = environment['CUDA_VISIBLE_DEVICES']
        environment['NVIDIA_DRIVER_CAPABILITIES'] = 'compute,utility'
        status = helper.run_and_reap(pair_container_command(helper, deployment, runtime, pair['pair_id'],
                                                            controller, commit), environment)
        if status:
            raise RuntimeError(f'Paired allocation completed with exit status {status}')
    finally:
        previous = {signum: signal.signal(signum, signal.SIG_IGN) for signum in (signal.SIGTERM, signal.SIGINT)}
        try:
            try:
                helper.release_runtime(base, reservation, job_id)
            finally:
                helper.release_runtime(base, runtime, job_id)
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--run-host', action='store_true')
    action.add_argument('--inside', action='store_true')
    parser.add_argument('--source', type=Path)
    parser.add_argument('--deployment', type=Path)
    parser.add_argument('--pair', type=Path)
    parser.add_argument('--controller', type=Path)
    parser.add_argument('--controller-commit')
    args = parser.parse_args()
    previous = {}
    def stop(signum: int, _frame: Any) -> None:
        raise InterruptedError(f'Paired runtime interrupted by signal {signum}')
    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, stop)
        if args.run_host:
            host()
        else:
            if not all((args.source, args.deployment, args.pair, args.controller, args.controller_commit)):
                parser.error('--inside requires source, deployment, pair, and controller identities')
            raise SystemExit(inside(args.source, args.deployment, args.pair, args.controller, args.controller_commit))
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == '__main__':
    main()
