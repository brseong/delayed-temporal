#!/usr/bin/env python3
"""Prepare and run calibrated three sweep tasks in disk-backed Slurm containers."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
from typing import Any, Iterator


REPO = Path(__file__).resolve().parents[3]
CANONICAL_REPO = Path('/data/delayed-temporal')
CANONICAL_SOURCE = Path('/data/delayed-temporal-worktrees/calibrated-three-sweeps')
CANONICAL_ASSETS = CANONICAL_REPO / 'artifacts/assets/theta-selection-v1'
DEPLOYMENT_MOUNT = Path('/three-sweep-deployment')
ARCHIVE_SHA256 = '3ac55cc182fa8f0671110c45c29cca9bb7e04e7d05c2bf6239b17be75ebc6762'
GIB = 1024 ** 3
RUNTIME_PREFIX = 'calibrated-three-sweep-'
TERMINAL_STATES = {
    'COMPLETED', 'CANCELLED', 'FAILED', 'TIMEOUT', 'OUT_OF_MEMORY', 'PREEMPTED',
    'BOOT_FAIL', 'NODE_FAIL', 'REVOKED', 'DEADLINE',
}
DISK_FILESYSTEMS = {'ext2', 'ext3', 'ext4', 'xfs', 'btrfs', 'zfs'}
PACKAGE_SUBTREES = {'transformers': 'src', 'spikingjelly': 'spikingjelly'}
IGNORED_SOURCE_DIRECTORIES = {'.git', '__pycache__', '.pytest_cache'}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def checked_hash(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value):
        raise ValueError('Invalid SHA-256 identity')
    return value


def absolute_path(value: Any) -> Path:
    if not isinstance(value, str):
        raise ValueError('An absolute path is required')
    path = Path(value)
    if not path.is_absolute() or '..' in path.parts or any(c in value for c in '\n\r\0,:'):
        raise ValueError(f'Unsafe absolute path: {value!r}')
    return path


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()


def immutable(path: Path, content: bytes) -> None:
    if path.exists():
        if path.is_symlink() or path.read_bytes() != content:
            raise ValueError(f'Refusing to replace a different preparation file: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        handle.write(content)


def atomic_json(path: Path, value: Any) -> None:
    if path.is_symlink():
        raise ValueError(f'Refusing a symlink: {path}')
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as handle:
            handle.write(json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def source_identity(source: Path, expected: str) -> None:
    if not re.fullmatch(r'[0-9a-f]{40}', expected):
        raise ValueError('An exact source commit is required')
    head = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(
        ['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'], text=True,
    ).strip()
    if head != expected or dirty:
        raise ValueError('Source HEAD or tracked files differ from the frozen deployment')


def artifact_records(path: Path) -> tuple[str, list[dict[str, Any]]]:
    """Use the same aggregate hash as scripts/setup/hash_artifact.py."""
    if path.is_file():
        files, root = [path], path.parent
    elif path.is_dir():
        files, root = sorted(item for item in path.rglob('*') if item.is_file()), path
    else:
        raise FileNotFoundError(path)
    if not files:
        raise ValueError(f'Empty artifact: {path}')
    aggregate = hashlib.sha256()
    records = []
    for item in files:
        stat = item.stat()
        relative = item.relative_to(root).as_posix()
        digest = sha256(item)
        aggregate.update(f'{relative}\0{stat.st_size}\0{digest}\n'.encode())
        records.append({'path': str(item), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                        'sha256': digest})
    return aggregate.hexdigest(), records


def package_source_files(path: Path) -> list[Path]:
    """Enumerate importable source trees without transient caches or Git metadata."""
    if not path.is_dir():
        raise ValueError(f'Package source must be a directory: {path}')
    files = []
    for current, directories, names in os.walk(path, followlinks=False):
        root = Path(current)
        directories[:] = sorted(name for name in directories if name not in IGNORED_SOURCE_DIRECTORIES)
        if any((root / name).is_symlink() for name in directories):
            raise ValueError('Package source directory symlinks require an explicit source root')
        files.extend(root / name for name in names
                     if name not in IGNORED_SOURCE_DIRECTORIES and not name.endswith('.pyc'))
    return sorted(files)


def package_source_identity(path: Path) -> tuple[str, list[dict[str, Any]]]:
    """Hash relative source names, sizes, and bytes, independently of installation path."""
    files = package_source_files(path)
    if not files:
        raise ValueError(f'Empty package source: {path}')
    aggregate = hashlib.sha256()
    records = []
    for item in files:
        stat = item.stat()
        relative = item.relative_to(path).as_posix()
        digest = sha256(item)
        aggregate.update(f'{relative}\0{stat.st_size}\0{digest}\n'.encode())
        records.append({'path': str(item), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                        'sha256': digest})
    return aggregate.hexdigest(), records


def asset_paths(experiment: dict[str, Any]) -> list[dict[str, str]]:
    result = []
    for path_field, hash_field in (
        ('checkpoint_path', 'checkpoint_sha256'),
        ('calibration_dataset_path', 'calibration_dataset_sha256'),
        ('dataset_path', 'dataset_sha256'),
    ):
        path = absolute_path(experiment[path_field])
        if not path.is_relative_to(CANONICAL_ASSETS):
            raise ValueError(f'Artifact must use the shared asset mount: {path}')
        result.append({'path': str(path), 'aggregate_sha256': checked_hash(experiment[hash_field])})
    return result


def prepare(root: Path, output: Path, *, image: Path, host_base: Path,
            host_source: Path | None = None, host_git_common_dir: Path | None = None,
            archive_sha256: str = ARCHIVE_SHA256) -> dict[str, Any]:
    """Write deployment metadata and private Git tools; do not allocate work."""
    root = root.resolve()
    experiment_path = root / 'experiment.json'
    experiment = json.loads(experiment_path.read_text())
    source = absolute_path(experiment.get('source_root', str(CANONICAL_SOURCE)))
    source_identity(source, experiment['source_commit'])
    tag = experiment.get('tag', root.name)
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', tag):
        raise ValueError('Unsafe experiment tag')
    if root.name != tag:
        raise ValueError('Experiment directory and tag differ')
    assets = asset_paths(experiment)
    dependencies = experiment.get('dependency_sha256')
    if not isinstance(dependencies, dict) or set(dependencies) != set(PACKAGE_SUBTREES):
        raise ValueError('Both editable package source hashes are required')
    dependency_sources = [
        {'name': name, 'path': str(CANONICAL_ASSETS / 'source-checkouts' / name / subtree),
         'aggregate_sha256': checked_hash(dependencies[name])}
        for name, subtree in PACKAGE_SUBTREES.items()
    ]
    host_assets = host_base / 'delayed-temporal-assets/theta-selection-v1'
    host_experiment = host_base / 'delayed-temporal-experiments' / tag
    runtime_tools = []
    for original, relative in (
        (source / 'scripts/experiments/ubai/calibrated_git.sh', 'tools/git'),
        (Path('/usr/bin/git'), 'tools/git.bin'),
        (Path('/lib/x86_64-linux-gnu/libpcre2-8.so.0'), 'tools/lib/libpcre2-8.so.0'),
        (Path('/lib/x86_64-linux-gnu/libz.so.1'), 'tools/lib/libz.so.1'),
    ):
        immutable(output / relative, original.read_bytes())
        if relative in {'tools/git', 'tools/git.bin'}:
            (output / relative).chmod(0o755)
        runtime_tools.append({'path': relative, 'sha256': sha256(output / relative)})
    script_hashes = {}
    for name in ('calibrated_three_sweep_task.sbatch', 'calibrated_three_sweep_prep.sbatch',
                 'prepare_calibrated_three_sweeps_ubai.py'):
        script_hashes['scripts/experiments/ubai/' + name] = sha256(source / 'scripts/experiments/ubai' / name)
    script_hashes['scripts/experiments/run_calibrated_three_sweep_task.py'] = sha256(
        source / 'scripts/experiments/run_calibrated_three_sweep_task.py')
    deployment = {
        'format_version': 1, 'state': 'prepared', 'tag': tag,
        'source_root': str(source), 'repository_root': str(CANONICAL_REPO),
        'assets_root': str(CANONICAL_ASSETS),
        'source_commit': experiment['source_commit'],
        'experiment_root': str(CANONICAL_REPO / 'artifacts/logs/noise_scan' / tag),
        'experiment_sha256': sha256(experiment_path),
        'script_sha256': script_hashes, 'runtime_tools': runtime_tools, 'assets': assets,
        'dependency_sources': dependency_sources,
        'assignment_required': True, 'paper_promotion_allowed': False,
        'limits': {'max_running_jobs': 10, 'max_submitted_jobs': 20, 'max_gpus': 12,
                   'max_campaign_tasks': 8, 'gpu_per_job': 1, 'cpus_per_job': 4,
                   'memory_gib_per_job': 64, 'partitions': ['gpu4', 'gpu5']},
        'runtime': {
            'host_source_root': str(host_source or host_base / 'delayed-temporal-main'),
            'host_assets_root': str(host_assets), 'host_experiment_root': str(host_experiment),
            'host_deployment_root': str(host_experiment / 'ubai'),
            'host_git_metadata_paths': [str(host_git_common_dir)] if host_git_common_dir else [],
            'env_archive': str(host_assets / 'runtime/dt-environment.tar.zst'),
            'env_archive_sha256': checked_hash(archive_sha256),
            'env_unpacked_bytes': 96 * GIB, 'minimum_scratch_bytes': 8 * GIB,
            'expected_python': '3.12.13',
            'container_image': str(host_assets / 'runtime/ubuntu-24.04.sqsh'),
            'container_image_sha256': sha256(image),
        },
    }
    validate_deployment(deployment)
    immutable(output / 'deployment.json', json_bytes(deployment))
    return deployment


def validate_deployment(deployment: dict[str, Any]) -> None:
    if deployment.get('format_version') != 1 or deployment.get('state') != 'prepared':
        raise ValueError('Unsupported deployment')
    for key in ('source_root', 'repository_root', 'assets_root', 'experiment_root'):
        absolute_path(deployment[key])
    runtime = deployment['runtime']
    for key in ('host_source_root', 'host_assets_root', 'host_experiment_root',
                'host_deployment_root', 'env_archive', 'container_image'):
        absolute_path(runtime[key])
    for key in ('env_archive_sha256', 'container_image_sha256'):
        checked_hash(runtime[key])
    for metadata_path in runtime['host_git_metadata_paths']:
        path = absolute_path(metadata_path)
        if path.name != '.git':
            raise ValueError('A Git common metadata directory is required')
    if runtime.get('expected_python') != '3.12.13':
        raise ValueError('The portable Python version must remain 3.12.13')
    checked_hash(deployment['experiment_sha256'])
    for key, expected in (('env_unpacked_bytes', 96 * GIB), ('minimum_scratch_bytes', 8 * GIB)):
        if type(runtime[key]) is not int or runtime[key] != expected:
            raise ValueError(f'Unexpected disk reservation: {key}')
    required_limits = {'max_running_jobs': 10, 'max_submitted_jobs': 20, 'max_gpus': 12,
                       'max_campaign_tasks': 8, 'gpu_per_job': 1, 'cpus_per_job': 4,
                       'memory_gib_per_job': 64, 'partitions': ['gpu4', 'gpu5']}
    if deployment.get('limits') != required_limits:
        raise ValueError('Unsupported Slurm resource limits')
    for relative, digest in deployment['script_sha256'].items():
        if Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise ValueError('Unsafe script path')
        checked_hash(digest)
    for item in deployment['runtime_tools']:
        if not item['path'].startswith('tools/') or '..' in Path(item['path']).parts:
            raise ValueError('Unsafe runtime tool path')
        checked_hash(item['sha256'])
    for item in deployment['assets']:
        path = absolute_path(item['path'])
        if not path.is_relative_to(absolute_path(deployment['assets_root'])):
            raise ValueError('Asset lies outside its read-only mount')
        checked_hash(item['aggregate_sha256'])
    dependencies = deployment.get('dependency_sources', [])
    if len(dependencies) != 2 or {item.get('name') for item in dependencies} != set(PACKAGE_SUBTREES):
        raise ValueError('Both editable package source roots are required')
    for item in dependencies:
        expected = absolute_path(deployment['assets_root']) / 'source-checkouts' / item['name'] / PACKAGE_SUBTREES[item['name']]
        if absolute_path(item['path']) != expected:
            raise ValueError('Editable source must use its declared importable subtree')
        checked_hash(item['aggregate_sha256'])


def require_compute_allocation() -> str:
    job = os.environ.get('SLURM_JOB_ID', '')
    if not re.fullmatch(r'[0-9]+', job) or socket.gethostname().split('.')[0] in {'gate1', 'gate2'}:
        raise ValueError('This operation requires a Slurm compute allocation')
    return job


def verify_files(deployment: dict[str, Any], deployment_path: Path) -> None:
    runtime = deployment['runtime']
    if deployment_path.parent.resolve() != absolute_path(runtime['host_deployment_root']).resolve():
        raise ValueError('Deployment was not installed at its declared host path')
    source = absolute_path(runtime['host_source_root'])
    source_identity(source, deployment['source_commit'])
    git_dir = Path(subprocess.check_output(
        ['git', '-C', str(source), 'rev-parse', '--absolute-git-dir'], text=True,
    ).strip()).resolve()
    allowed_metadata = [source.resolve(), *(absolute_path(path).resolve()
                                          for path in runtime['host_git_metadata_paths'])]
    if not any(git_dir.is_relative_to(root) for root in allowed_metadata):
        raise ValueError('External Git metadata has no declared container mount')
    paths = [(source / relative, digest) for relative, digest in deployment['script_sha256'].items()]
    paths += [(deployment_path.parent / item['path'], item['sha256'])
              for item in deployment['runtime_tools']]
    paths.append((absolute_path(runtime['host_experiment_root']) / 'experiment.json',
                  deployment['experiment_sha256']))
    paths += [(absolute_path(runtime[key]), runtime[key + '_sha256'])
              for key in ('env_archive', 'container_image')]
    for path, expected in paths:
        if sha256(path) != expected:
            raise ValueError(f'Checksum mismatch: {path}')


def disk_directory(base: Path) -> None:
    base.mkdir(parents=True, exist_ok=True)
    if base.resolve() != base or base.is_symlink() or base.stat().st_uid != os.getuid():
        raise ValueError('Runtime directory must be an owned, non-symlink absolute path')
    filesystem = subprocess.check_output(['findmnt', '-n', '-T', str(base), '-o', 'FSTYPE'], text=True).strip()
    if filesystem not in DISK_FILESYSTEMS:
        raise ValueError(f'Refusing an unknown or memory-backed runtime filesystem: {filesystem!r}')


@contextlib.contextmanager
def runtime_lock(base: Path) -> Iterator[None]:
    descriptor = os.open(base / '.calibrated-three-sweep.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        if os.fstat(descriptor).st_uid != os.getuid():
            raise ValueError('Runtime lock is owned by another user')
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def terminal_job(job_id: str) -> bool:
    if not re.fullmatch(r'[0-9]+', job_id):
        raise ValueError('Invalid Slurm job identifier')
    queue = subprocess.run(['squeue', '--noheader', '--jobs', job_id, '--format=%i|%T|%u'],
                           capture_output=True, text=True, timeout=30, check=False)
    if queue.returncode and 'Invalid job id specified' not in queue.stderr:
        raise ValueError('Slurm queue status is unavailable; runtime cleanup is refused')
    if queue.stdout.strip():
        return False
    account = subprocess.run(
        ['sacct', '--noheader', '--allocations', '--jobs', job_id,
         '--format=JobIDRaw,State,UID', '--parsable2'],
        capture_output=True, text=True, timeout=30, check=True,
    )
    matches = [line.split('|') for line in account.stdout.splitlines()
               if line.split('|')[0].strip() == job_id]
    if len(matches) != 1 or len(matches[0]) != 3:
        raise ValueError(f'No unique Slurm terminal record for runtime owner {job_id}')
    _, state, uid = (part.strip() for part in matches[0])
    if uid != str(os.getuid()):
        raise ValueError('Slurm runtime owner differs from the current user')
    return state.split()[0].rstrip('+') in TERMINAL_STATES


def owned_runtime(base: Path, path: Path) -> dict[str, Any]:
    if path.parent != base or not path.name.startswith(RUNTIME_PREFIX) or path.is_symlink():
        raise ValueError(f'Unsafe runtime cleanup target: {path}')
    if path.resolve() != path or not path.is_dir() or path.stat().st_uid != os.getuid():
        raise ValueError(f'Runtime directory is not owned by this user: {path}')
    owner_path = path / '.owner.json'
    if owner_path.is_symlink() or owner_path.stat().st_uid != os.getuid():
        raise ValueError('Invalid runtime ownership record')
    owner = json.loads(owner_path.read_text())
    if owner.get('uid') != os.getuid() or not re.fullmatch(r'[0-9]+', owner.get('job_id', '')):
        raise ValueError('Invalid runtime ownership record')
    if not path.name.startswith(RUNTIME_PREFIX + owner['job_id'] + '-'):
        raise ValueError('Runtime name and recorded job differ')
    if owner.get('scratch_bytes') != 8 * GIB:
        raise ValueError('Unknown runtime scratch reservation')
    return owner


def sweep_old_runtime(base: Path, current_job: str) -> list[dict[str, Any]]:
    """Inspect every owned reservation; never infer termination from its age."""
    live = []
    for path in sorted(base.glob(RUNTIME_PREFIX + '*')):
        owner = owned_runtime(base, path)
        if owner['job_id'] != current_job and terminal_job(owner['job_id']):
            shutil.rmtree(path)
        else:
            live.append({'path': str(path), **owner})
    return live


def admit_runtime(base: Path, job_id: str, task_id: str, runtime: dict[str, Any]) -> Path:
    disk_directory(base)
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', task_id):
        raise ValueError('Unsafe task identifier')
    with runtime_lock(base):
        live = sweep_old_runtime(base, job_id)
        # df already includes extracted environments; outstanding scratch allocations
        # are also charged in full so concurrent tasks cannot spend the same space.
        reserved = sum(item['scratch_bytes'] for item in live)
        required = runtime['env_unpacked_bytes'] + runtime['minimum_scratch_bytes'] + reserved
        if shutil.disk_usage(base).free < required:
            raise ValueError('Insufficient disk space for concurrent task reservations')
        path = Path(tempfile.mkdtemp(prefix=f'{RUNTIME_PREFIX}{job_id}-{task_id}-', dir=base))
        owner = {'uid': os.getuid(), 'job_id': job_id, 'task_id': task_id,
                 'scratch_bytes': runtime['minimum_scratch_bytes']}
        immutable(path / '.owner.json', json_bytes(owner))
        atomic_json(base / '.calibrated-three-sweep-reservations.json', live + [{'path': str(path), **owner}])
        try:
            (path / 'scratch').mkdir(mode=0o700)
            extract_environment(absolute_path(runtime['env_archive']), path, runtime['env_unpacked_bytes'])
        except BaseException:
            owned_runtime(base, path)
            shutil.rmtree(path)
            atomic_json(base / '.calibrated-three-sweep-reservations.json', live)
            raise
        return path


def extract_environment(archive: Path, target: Path, maximum_bytes: int) -> None:
    decompressor = subprocess.Popen(['zstd', '--decompress', '--stdout', str(archive)], stdout=subprocess.PIPE)
    try:
        unpacker = subprocess.Popen(['tar', '--no-same-owner', '-xf', '-', '-C', str(target)], stdin=decompressor.stdout)
        assert decompressor.stdout is not None
        decompressor.stdout.close()
        try:
            unpacked_status = unpacker.wait()
            decompressed_status = decompressor.wait()
        except BaseException:
            for child in (unpacker, decompressor):
                if child.poll() is None:
                    child.terminate()
            for child in (unpacker, decompressor):
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            raise
        if unpacked_status or decompressed_status:
            raise RuntimeError('Portable environment extraction failed')
    finally:
        if decompressor.poll() is None:
            decompressor.kill()
            decompressor.wait()
    used = int(subprocess.check_output(['du', '-s', '-B1', str(target)], text=True).split()[0])
    if used > maximum_bytes or not os.access(target / 'dt/bin/python', os.X_OK):
        raise ValueError('Extracted environment exceeds its bound or has no interpreter')


def release_runtime(base: Path, path: Path, job_id: str) -> None:
    with runtime_lock(base):
        owner = owned_runtime(base, path)
        if owner['job_id'] != job_id:
            raise ValueError('Refusing cleanup of another task')
        shutil.rmtree(path)
        live = [{'path': str(other), **owned_runtime(base, other)}
                for other in sorted(base.glob(RUNTIME_PREFIX + '*'))]
        atomic_json(base / '.calibrated-three-sweep-reservations.json', live)


def host_asset_path(deployment: dict[str, Any], canonical: str) -> Path:
    relative = absolute_path(canonical).relative_to(absolute_path(deployment['assets_root']))
    return absolute_path(deployment['runtime']['host_assets_root']) / relative


def verify_assets(deployment: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for asset in deployment['assets']:
        path = host_asset_path(deployment, asset['path'])
        digest, files = artifact_records(path)
        if digest != asset['aggregate_sha256']:
            raise ValueError(f'Artifact checksum mismatch: {path}')
        records.extend(files)
    return records


def verify_dependency_sources(deployment: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for source in deployment['dependency_sources']:
        path = host_asset_path(deployment, source['path'])
        digest, files = package_source_identity(path)
        if digest != source['aggregate_sha256']:
            raise ValueError(f'Editable package source checksum mismatch: {path}')
        records.extend(files)
    return records


def verify_preparation(deployment: dict[str, Any], deployment_path: Path) -> None:
    report_path = deployment_path.parent / 'prep-result.json'
    report = json.loads(report_path.read_text())
    if report.get('state') != 'verified' or report.get('deployment_sha256') != sha256(deployment_path):
        raise ValueError('A matching successful CPU preparation is required')
    if (not report.get('asset_files') or not report.get('dependency_files')
            or report.get('python_version') != '3.12.13'):
        raise ValueError('CPU preparation is incomplete')
    expected_roots = [host_asset_path(deployment, item['path']) for item in deployment['assets']]
    expected_files = {str(path) for root in expected_roots
                      for path in ([root] if root.is_file() else root.rglob('*')) if path.is_file()}
    if expected_files != {record['path'] for record in report['asset_files']}:
        raise ValueError('Artifact file membership changed after CPU preparation')
    for record in report['asset_files']:
        path = absolute_path(record['path'])
        if not any(path == root or path.is_relative_to(root) for root in expected_roots):
            raise ValueError('Preparation record lies outside the asset mounts')
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (record['bytes'], record['mtime_ns']):
            raise ValueError(f'Artifact changed after CPU checksum verification: {path}')
    dependency_roots = [host_asset_path(deployment, item['path']) for item in deployment['dependency_sources']]
    expected_dependency_files = {str(path) for root in dependency_roots for path in package_source_files(root)}
    if expected_dependency_files != {record['path'] for record in report['dependency_files']}:
        raise ValueError('Editable package file membership changed after CPU preparation')
    for record in report['dependency_files']:
        path = absolute_path(record['path'])
        if not any(path.is_relative_to(root) for root in dependency_roots):
            raise ValueError('Preparation record lies outside the editable source mounts')
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (record['bytes'], record['mtime_ns']):
            raise ValueError(f'Editable package source changed after CPU checksum verification: {path}')


def task_identifier() -> str:
    task_id = os.environ.get('TASK_ID')
    table = os.environ.get('TASK_IDS_FILE')
    if bool(task_id) == bool(table):
        raise ValueError('Provide exactly one of TASK_ID and TASK_IDS_FILE')
    if table:
        index = os.environ.get('SLURM_ARRAY_TASK_ID', '')
        if not re.fullmatch(r'[0-9]+', index):
            raise ValueError('An array index is required with TASK_IDS_FILE')
        rows = absolute_path(table).read_text().splitlines()
        if len(rows) != len(set(rows)) or not rows:
            raise ValueError('Task list is empty or contains duplicate identifiers')
        if int(index) >= len(rows):
            raise ValueError('Array index is outside the task list')
        task_id = rows[int(index)]
    if not task_id or not re.fullmatch(r'[A-Za-z0-9_.-]+', task_id):
        raise ValueError('Unsafe task identifier')
    return task_id


def container_command(deployment: dict[str, Any], runtime_path: Path,
                      task_id: str, *, check_only: bool) -> list[str]:
    runtime = deployment['runtime']
    source = absolute_path(deployment['source_root'])
    repository = absolute_path(deployment['repository_root'])
    assets = absolute_path(deployment['assets_root'])
    experiment = absolute_path(deployment['experiment_root'])
    host_source, host_assets = runtime['host_source_root'], runtime['host_assets_root']
    mounts = [f'{host_source}:{source}:ro', f'{host_source}:{repository}:ro',
              f'{host_assets}:{assets}:ro',
              f'{runtime["host_experiment_root"]}:{experiment}',
              f'{runtime["host_deployment_root"]}:{DEPLOYMENT_MOUNT}:ro',
              f'{runtime_path}/dt:/opt/conda/envs/dt:ro']
    mounts.extend(f'{path}:{path}:ro' for path in runtime['host_git_metadata_paths'])
    for checkout, target in (('transformers', 'src/transformers'), ('spikingjelly', 'src/spikingjelly')):
        for root in dict.fromkeys((source, repository)):
            mounts.append(f'{host_assets}/source-checkouts/{checkout}:{root}/{target}:ro')
    for target in ('/tmp', '/var/tmp', '/work-tmp'):
        mounts.append(f'{runtime_path}/scratch:{target}')
    environment = [
        f'PATH={DEPLOYMENT_MOUNT}/tools:/opt/conda/envs/dt/bin:/usr/bin:/bin',
        'TMPDIR=/work-tmp', 'TEMP=/work-tmp', 'TMP=/work-tmp',
        'XDG_CACHE_HOME=/work-tmp/cache', 'XDG_RUNTIME_DIR=/work-tmp/xdg-runtime',
        'HF_HOME=/work-tmp/huggingface', 'HF_DATASETS_CACHE=/work-tmp/huggingface/datasets',
        'HUGGINGFACE_HUB_CACHE=/work-tmp/huggingface/hub', 'HF_MODULES_CACHE=/work-tmp/huggingface/modules',
        'TORCH_HOME=/work-tmp/torch', 'TORCHINDUCTOR_CACHE_DIR=/work-tmp/torchinductor',
        'TRITON_CACHE_DIR=/work-tmp/triton', 'CUDA_CACHE_PATH=/work-tmp/cuda',
        'CUPY_CACHE_DIR=/work-tmp/cupy', 'NUMBA_CACHE_DIR=/work-tmp/numba', 'PIP_CACHE_DIR=/work-tmp/pip',
        'MPLCONFIGDIR=/work-tmp/matplotlib', 'PYTHONDONTWRITEBYTECODE=1', 'PYTHONUNBUFFERED=1',
        f'PYTHONPATH={source}:{source}/src/transformers/src:{source}/src/spikingjelly',
        f'GIT_WORK_TREE={source}',
        'WANDB_MODE=disabled', 'WANDB_DISABLED=true', 'WANDB_DIR=/work-tmp/wandb',
        'WANDB_SILENT=true', 'WANDB_CONSOLE=off',
        'HF_HUB_OFFLINE=1', 'HF_DATASETS_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1',
        'OMP_NUM_THREADS=4', 'MKL_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4', 'NUMEXPR_NUM_THREADS=4',
    ]
    if check_only:
        probe = (
            'import platform,torch,numpy,transformers,datasets,spikingjelly; '
            'assert platform.python_version()=="3.12.13",platform.python_version(); '
            'print("THREE_SWEEP_RUNTIME_VERIFIED "+platform.python_version())'
        )
        arguments = ['-c', probe]
    else:
        arguments = [str(source / 'scripts/experiments/run_calibrated_three_sweep_task.py'),
                     '--experiment', str(experiment / 'experiment.json'),
                     '--task', str(experiment / 'tasks' / (task_id + '.json')),
                     '--output-root', str(experiment), '--host-label', 'ubai']
    return ['srun', '--ntasks=1', '--cpus-per-task=4', '--gres=none' if check_only else '--gres=gpu:1',
            '--kill-on-bad-exit=1', '--container-image=' + runtime['container_image'],
            '--container-mounts=' + ','.join(mounts), '--container-workdir=' + str(source),
            '/usr/bin/env', '-u', 'WANDB_API_KEY', *environment,
            '/opt/conda/envs/dt/bin/python', *arguments]


def run_and_reap(command: list[str], environment: dict[str, str]) -> int:
    """Do not allow the caller to remove runtime storage before srun is reaped."""
    child = subprocess.Popen(command, env=environment, start_new_session=True)
    previous = {}
    def interrupted(signum: int, _frame: Any) -> None:
        raise InterruptedError(f'Interrupted by signal {signum}')
    for signum in (signal.SIGTERM, signal.SIGINT):
        previous[signum] = signal.signal(signum, interrupted)
    try:
        return child.wait()
    except BaseException:
        for signum in previous:
            signal.signal(signum, signal.SIG_IGN)
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        raise
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _execute_allocated(deployment_path: Path, *, check_only: bool) -> None:
    job_id = require_compute_allocation()
    deployment = json.loads(deployment_path.read_text())
    validate_deployment(deployment)
    verify_files(deployment, deployment_path)
    if os.environ.get('SLURM_CPUS_PER_TASK') != '4' or os.environ.get('SLURM_NTASKS', '1') != '1':
        raise ValueError('Each task requires exactly four CPUs and one Slurm task')
    if os.environ.get('SLURM_MEM_PER_NODE') != '65536':
        raise ValueError('Each task requires exactly 64 GiB of host memory')
    if check_only:
        task_id = 'prep'
        assets = verify_assets(deployment)
        dependencies = verify_dependency_sources(deployment)
    else:
        task_id = task_identifier()
        gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if not gpu or ',' in gpu or gpu.lower() in {'-1', 'all', 'none', 'void'}:
            raise ValueError('Exactly one Slurm GPU is required')
        if os.environ.get('SLURM_JOB_PARTITION') not in {'gpu4', 'gpu5'}:
            raise ValueError('Only the RTX A6000 partitions gpu4 and gpu5 are supported')
        verify_preparation(deployment, deployment_path)
        task_path = absolute_path(deployment['runtime']['host_experiment_root']) / 'tasks' / (task_id + '.json')
        task = json.loads(task_path.read_text())
        if task.get('run_id') != task_id:
            raise ValueError('Task file and run identifier differ')
        # The controller owns assignment; the worker validates scientific identity.
        assets = []
        dependencies = []
    base = Path('/enroot') / str(os.getuid()) / 'data'
    runtime = admit_runtime(base, job_id, task_id, deployment['runtime'])
    try:
        scratch = runtime / 'scratch'
        (scratch / 'xdg-runtime').mkdir(mode=0o700)
        environment = dict(os.environ)
        environment.pop('WANDB_API_KEY', None)
        for key in ('TMPDIR', 'TEMP', 'TMP'):
            environment[key] = str(scratch)
        for key, name in (('ENROOT_RUNTIME_PATH', 'enroot-runtime'),
                          ('ENROOT_DATA_PATH', 'enroot-data'), ('ENROOT_CACHE_PATH', 'enroot-cache')):
            (scratch / name).mkdir(mode=0o700)
            environment[key] = str(scratch / name)
        environment['NVIDIA_VISIBLE_DEVICES'] = 'void' if check_only else environment['CUDA_VISIBLE_DEVICES']
        environment['NVIDIA_DRIVER_CAPABILITIES'] = 'compute,utility'
        result = run_and_reap(container_command(deployment, runtime, task_id, check_only=check_only), environment)
        if result:
            raise RuntimeError(f'Container task failed with exit status {result}')
        if check_only:
            atomic_json(deployment_path.parent / 'prep-result.json', {
                'state': 'verified', 'deployment_sha256': sha256(deployment_path),
                'source_commit': deployment['source_commit'], 'python_version': '3.12.13',
                'asset_files': assets, 'dependency_files': dependencies,
                'job_id': job_id, 'node': socket.gethostname(),
            })
    finally:
        previous = {signum: signal.signal(signum, signal.SIG_IGN)
                    for signum in (signal.SIGTERM, signal.SIGINT)}
        try:
            release_runtime(base, runtime, job_id)
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)


def execute(deployment_path: Path, *, check_only: bool) -> None:
    previous = {}
    def interrupted(signum: int, _frame: Any) -> None:
        raise InterruptedError(f'Interrupted by signal {signum}')
    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, interrupted)
        _execute_allocated(deployment_path, check_only=check_only)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument('--prepare', action='store_true')
    operation.add_argument('--check-only', action='store_true')
    operation.add_argument('--run-task', action='store_true')
    parser.add_argument('--experiment-root', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--container-image', type=Path)
    parser.add_argument('--host-base', type=Path, default=Path('/home1/sizz1997/myubai'))
    parser.add_argument('--host-source', type=Path)
    parser.add_argument('--host-git-common-dir', type=Path)
    parser.add_argument('--environment-sha256', default=ARCHIVE_SHA256)
    parser.add_argument('--deployment', type=Path)
    args = parser.parse_args()
    if args.prepare:
        if not all((args.experiment_root, args.output, args.container_image)):
            parser.error('--prepare requires --experiment-root, --output, and --container-image')
        value = prepare(args.experiment_root, args.output, image=args.container_image,
                        host_base=args.host_base, host_source=args.host_source,
                        host_git_common_dir=args.host_git_common_dir,
                        archive_sha256=args.environment_sha256)
        print(json.dumps({'deployment': str(args.output / 'deployment.json'),
                          'state': value['state'], 'jobs_submitted': 0}, sort_keys=True))
    else:
        if args.deployment is None:
            parser.error('--deployment is required')
        execute(args.deployment, check_only=args.check_only)


if __name__ == '__main__':
    main()
