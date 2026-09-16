"""Verify calibrated three sweep UBAI preparation and disk safety without jobs."""
from __future__ import annotations

from contextlib import ExitStack
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments.ubai import prepare_calibrated_three_sweeps_ubai as ubai
from scripts.runtime import files as runtime_files
from scripts.runtime import identity


class UBAISafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        parent = ROOT / 'artifacts/runtime'
        parent.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix='three-sweep-ubai-test-', dir=parent)
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.source = self.base / 'source'
        self.source.mkdir()
        self.assets = self.base / 'assets'
        self.assets.mkdir()
        self.experiment_root = self.base / 'new-campaign'
        self.experiment_root.mkdir()
        self.output = self.base / 'deployment'
        self.image = self.base / 'image.sqsh'
        self.image.write_bytes(b'container fixture')
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(ubai, 'CANONICAL_ASSETS', self.assets))
        self.experiment = {'tag': self.experiment_root.name, 'source_root': str(self.source),
                           'source_commit': 'a' * 40, 'dependency_sha256': {}}
        for name, subtree in ubai.PACKAGE_SUBTREES.items():
            package = self.assets / 'source-checkouts' / name / subtree
            package.mkdir(parents=True)
            (package / '__init__.py').write_bytes(f'# {name} package fixture\n'.encode())
            self.experiment['dependency_sha256'][name] = identity.package_source_identity(package)[0]
        for directory, path_key, hash_key in (
            ('checkpoint', 'checkpoint_path', 'checkpoint_sha256'),
            ('training', 'calibration_dataset_path', 'calibration_dataset_sha256'),
            ('validation', 'dataset_path', 'dataset_sha256'),
        ):
            path = self.assets / directory
            path.mkdir()
            (path / 'payload').write_bytes(directory.encode())
            self.experiment[path_key] = str(path)
            self.experiment[hash_key] = identity.artifact_identity(path)['aggregate_sha256']
        (self.experiment_root / 'experiment.json').write_bytes(runtime_files.json_bytes(self.experiment))
        for relative in (
            'scripts/experiments/ubai/calibrated_three_sweep_task.sbatch',
            'scripts/experiments/ubai/calibrated_three_sweep_prep.sbatch',
            'scripts/experiments/ubai/prepare_calibrated_three_sweeps_ubai.py',
            'scripts/experiments/run_calibrated_three_sweep_task.py',
            'scripts/experiments/ubai/calibrated_git.sh',
            'scripts/runtime/files.py',
            'scripts/runtime/identity.py',
            'scripts/runtime/local_gpu.py',
        ):
            path = self.source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'# frozen script fixture\n')

    def deployment(self) -> dict:
        with patch.object(ubai, 'source_identity'):
            return ubai.prepare(self.experiment_root, self.output, image=self.image,
                                host_base=self.base / 'host',
                                host_git_common_dir=self.base / 'git-common/.git')

    def owner(self, base: Path, job: str, name: str = 'task') -> Path:
        path = base / f'{ubai.RUNTIME_PREFIX}{job}-{name}-fixture'
        path.mkdir(parents=True)
        (path / '.owner.json').write_bytes(runtime_files.json_bytes({
            'uid': os.getuid(), 'job_id': job, 'task_id': name, 'scratch_bytes': 8 * ubai.GIB,
        }))
        return path

    # @lat: [[evaluation#Evaluation and Verification#Calibrated Three Sweep Distribution]]
    def test_preparation_is_immutable_and_does_not_submit(self) -> None:
        with patch.object(subprocess, 'Popen', side_effect=AssertionError('must not submit')):
            first = self.deployment()
            self.assertEqual(first, self.deployment())
        self.assertEqual(first['state'], 'prepared')
        self.assertTrue(first['assignment_required'])
        self.assertEqual(first['source_commit'], self.experiment['source_commit'])
        self.assertEqual({item['name']: item['aggregate_sha256'] for item in first['dependency_sources']},
                         self.experiment['dependency_sha256'])
        self.assertEqual(first['limits']['max_campaign_tasks'], 8)
        self.assertEqual(first['limits']['max_running_jobs'], 10)
        self.assertEqual(first['limits']['max_submitted_jobs'], 20)
        self.assertEqual(first['limits']['max_gpus'], 12)
        self.assertEqual(first['runtime']['env_unpacked_bytes'], 96 * ubai.GIB)
        self.assertEqual(first['runtime']['minimum_scratch_bytes'], 8 * ubai.GIB)
        (self.experiment_root / 'experiment.json').write_bytes(
            runtime_files.json_bytes(dict(self.experiment, changed=True))
        )
        with self.assertRaisesRegex(ValueError, 'different preparation'):
            self.deployment()

    def test_artifact_hash_matches_shared_identity_and_detects_change(self) -> None:
        path = self.assets / 'training'
        aggregate, records = identity.artifact_records(path)
        self.assertEqual(aggregate, identity.artifact_identity(path)['aggregate_sha256'])
        self.assertEqual(records[0]['bytes'], len(b'training'))
        (path / 'payload').write_bytes(b'changed')
        self.assertNotEqual(identity.artifact_records(path)[0], aggregate)

    def test_package_identity_is_location_independent_and_ignores_only_transient_files(self) -> None:
        package = self.assets / 'source-checkouts/transformers/src'
        original, records = identity.package_source_identity(package)
        for relative in ('.git/config', '__pycache__/bytecode.pyc', '.pytest_cache/cache',
                         'nested/__pycache__/bytecode.pyc', 'loose.pyc'):
            path = package / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'ignored transient content')
        self.assertEqual(original, identity.package_source_identity(package)[0])
        self.assertEqual([item['path'] for item in records],
                         [item['path'] for item in identity.package_source_identity(package)[1]])
        relocated = self.base / 'relocated'
        relocated.mkdir()
        (relocated / '__init__.py').write_bytes((package / '__init__.py').read_bytes())
        self.assertEqual(original, identity.package_source_identity(relocated)[0])
        (package / 'new_module.py').write_bytes(b'import math\n')
        self.assertNotEqual(original, identity.package_source_identity(package)[0])

    def test_missing_wrong_or_unselected_dependency_sources_are_rejected(self) -> None:
        value = self.deployment()
        value['runtime']['host_assets_root'] = str(self.assets)
        self.assertEqual(len(ubai.verify_dependency_sources(value)), 2)
        changed = copy.deepcopy(value)
        changed['dependency_sources'][0]['path'] = str(self.assets / 'source-checkouts/transformers')
        with self.assertRaises(ValueError):
            ubai.validate_deployment(changed)
        changed = copy.deepcopy(value)
        changed['dependency_sources'].pop()
        with self.assertRaises(ValueError):
            ubai.validate_deployment(changed)
        (self.assets / 'source-checkouts/spikingjelly/spikingjelly/__init__.py').write_bytes(b'changed code')
        with self.assertRaisesRegex(ValueError, 'source checksum mismatch'):
            ubai.verify_dependency_sources(value)

    def test_deployment_rejects_bad_paths_hashes_resources_and_runtime_versions(self) -> None:
        value = self.deployment()
        for section, field, invalid in (
            ('runtime', 'env_archive', '/tmp/../unsafe'),
            ('runtime', 'env_archive', '/data/source:rw'),
            ('runtime', 'env_archive_sha256', 'bad'),
            ('runtime', 'expected_python', '3.11.0'),
            ('runtime', 'env_unpacked_bytes', 95 * ubai.GIB),
            ('runtime', 'minimum_scratch_bytes', 0),
            ('limits', 'gpu_per_job', 2),
            ('limits', 'max_campaign_tasks', 9),
            ('limits', 'partitions', ['gpu1']),
        ):
            changed = copy.deepcopy(value)
            changed[section][field] = invalid
            with self.assertRaises(ValueError, msg=field):
                ubai.validate_deployment(changed)

    def test_frozen_source_rejects_wrong_head_or_tracked_modifications(self) -> None:
        for head, dirty in (('b' * 40 + '\n', ''), ('a' * 40 + '\n', ' M changed.py\n')):
            with patch.object(subprocess, 'check_output', side_effect=[head, dirty]):
                with self.assertRaises(ValueError):
                    ubai.source_identity(self.source, 'a' * 40)
        with patch.object(subprocess, 'check_output', side_effect=['a' * 40 + '\n', '']):
            ubai.source_identity(self.source, 'a' * 40)

    def test_bootstrap_checks_source_assets_and_experiment_before_extraction(self) -> None:
        value = self.deployment()
        archive = self.base / 'environment.tar.zst'
        archive.write_bytes(b'archive fixture')
        runtime = value['runtime']
        runtime.update(host_source_root=str(self.source), host_experiment_root=str(self.experiment_root),
                       host_deployment_root=str(self.output), env_archive=str(archive),
                       env_archive_sha256=identity.sha256_file(archive), container_image=str(self.image))
        deployment_path = self.output / 'deployment.json'
        deployment_path.write_bytes(runtime_files.json_bytes(value))
        with patch.object(ubai, 'source_identity'), \
             patch.object(subprocess, 'check_output', return_value=str(self.source / '.git') + '\n'):
            ubai.verify_files(value, deployment_path)
            for path in (archive, self.image, self.experiment_root / 'experiment.json',
                         self.source / 'scripts/experiments/run_calibrated_three_sweep_task.py'):
                original = path.read_bytes()
                path.write_bytes(original + b'changed')
                with self.assertRaisesRegex(ValueError, 'Checksum mismatch'):
                    ubai.verify_files(value, deployment_path)
                path.write_bytes(original)
        with patch.object(ubai, 'source_identity'), \
             patch.object(subprocess, 'check_output', return_value='/undeclared/common/.git/worktree\n'):
            with self.assertRaisesRegex(ValueError, 'no declared container mount'):
                ubai.verify_files(value, deployment_path)

    def test_container_uses_readonly_inputs_disk_tmp_and_one_gpu(self) -> None:
        value = self.deployment()
        runtime = self.base / 'runtime'
        command = ubai.container_command(value, runtime, 'theta_01_train', check_only=False)
        self.assertEqual(command.count('--gres=gpu:1'), 1)
        self.assertIn('--cpus-per-task=4', command)
        mounts = next(part.split('=', 1)[1].split(',') for part in command if part.startswith('--container-mounts='))
        self.assertIn(f'{runtime}/scratch:/tmp', mounts)
        self.assertIn(f'{runtime}/scratch:/var/tmp', mounts)
        self.assertIn(f'{runtime}/scratch:/work-tmp', mounts)
        self.assertTrue(any(part.endswith(':' + value['source_root'] + ':ro') for part in mounts))
        for path in value['runtime']['host_git_metadata_paths']:
            self.assertIn(f'{path}:{path}:ro', mounts)
        self.assertIn('GIT_WORK_TREE=' + value['source_root'], command)
        self.assertIn('WANDB_MODE=disabled', command)
        self.assertIn('WANDB_API_KEY', command)
        self.assertIn('TMPDIR=/work-tmp', command)
        self.assertEqual(command[-2:], ['--host-label', 'ubai'])
        self.assertTrue(command[command.index('--experiment') + 1].endswith('/experiment.json'))
        self.assertTrue(any(part.endswith('tasks/theta_01_train.json') for part in command))
        self.assertNotIn('--gres=gpu:1', ubai.container_command(value, runtime, 'prep', check_only=True))

    def test_slurm_headers_and_gate_execution_rejection(self) -> None:
        task = (ROOT / 'scripts/experiments/ubai/calibrated_three_sweep_task.sbatch').read_text()
        prep = (ROOT / 'scripts/experiments/ubai/calibrated_three_sweep_prep.sbatch').read_text()
        for line in ('#SBATCH --gres=gpu:1', '#SBATCH --cpus-per-task=4', '#SBATCH --mem=64G',
                     '#SBATCH --partition=gpu4,gpu5', '#SBATCH --time=03:00:00'):
            self.assertIn(line, task)
        self.assertNotIn('#SBATCH --container-image', task)
        self.assertNotIn('#SBATCH --gres=gpu', prep)
        self.assertIn('#SBATCH --partition=cpu1', prep)
        with patch.dict(os.environ, {'SLURM_JOB_ID': '123'}, clear=True), \
             patch.object(ubai.socket, 'gethostname', return_value='gate1.hpc'):
            with self.assertRaises(ValueError):
                ubai.require_compute_allocation()
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                ubai.require_compute_allocation()

    def test_disk_filesystem_rejects_ram_and_unknown_types(self) -> None:
        base = self.base / 'runtime-data'
        for filesystem in ('tmpfs', 'ramfs', '', 'overlay', 'nfs'):
            with patch.object(subprocess, 'check_output', return_value=filesystem + '\n'):
                with self.assertRaises(ValueError, msg=filesystem):
                    ubai.disk_directory(base)
        with patch.object(subprocess, 'check_output', return_value='xfs\n'):
            ubai.disk_directory(base)

    def test_concurrent_scratch_reservations_are_not_spent_twice(self) -> None:
        value = self.deployment()['runtime']
        base = self.base / 'admission'
        old = self.owner(base, '123')
        with patch.object(ubai, 'disk_directory'), \
             patch.object(ubai, 'terminal_job', return_value=False), \
             patch.object(ubai.shutil, 'disk_usage', return_value=SimpleNamespace(free=111 * ubai.GIB)), \
             patch.object(ubai, 'extract_environment') as extract:
            with self.assertRaisesRegex(ValueError, 'concurrent task reservations'):
                ubai.admit_runtime(base, '456', 'new', value)
            extract.assert_not_called()
        with patch.object(ubai, 'disk_directory'), \
             patch.object(ubai, 'terminal_job', return_value=False), \
             patch.object(ubai.shutil, 'disk_usage', return_value=SimpleNamespace(free=112 * ubai.GIB)), \
             patch.object(ubai, 'extract_environment'):
            path = ubai.admit_runtime(base, '456', 'new', value)
        self.assertTrue(old.is_dir())
        self.assertEqual(len(json.loads((base / '.calibrated-three-sweep-reservations.json').read_text())), 2)
        ubai.release_runtime(base, path, '456')
        self.assertFalse(path.exists())
        self.assertTrue(old.exists())

    def test_extraction_failure_releases_only_its_reservation(self) -> None:
        base = self.base / 'extraction'
        base.mkdir()
        with patch.object(ubai, 'disk_directory'), \
             patch.object(ubai.shutil, 'disk_usage', return_value=SimpleNamespace(free=200 * ubai.GIB)), \
             patch.object(ubai, 'extract_environment', side_effect=RuntimeError('failed extraction')):
            with self.assertRaises(RuntimeError):
                ubai.admit_runtime(base, '456', 'new', self.deployment()['runtime'])
        self.assertFalse(list(base.glob(ubai.RUNTIME_PREFIX + '*')))
        self.assertEqual(json.loads((base / '.calibrated-three-sweep-reservations.json').read_text()), [])

    def test_cleanup_requires_exact_owner_and_terminal_slurm_state(self) -> None:
        base = self.base / 'cleanup'
        old = self.owner(base, '123')
        current = self.owner(base, '456')
        with patch.object(ubai, 'terminal_job', return_value=False):
            self.assertEqual(len(ubai.sweep_old_runtime(base, '456')), 2)
        with patch.object(ubai, 'terminal_job', side_effect=ValueError('Slurm unavailable')):
            with self.assertRaises(ValueError):
                ubai.sweep_old_runtime(base, '456')
        self.assertTrue(old.exists())
        with self.assertRaises(ValueError):
            ubai.release_runtime(base, current, '999')
        with patch.object(ubai, 'terminal_job', return_value=True):
            self.assertEqual(len(ubai.sweep_old_runtime(base, '456')), 1)
        self.assertFalse(old.exists())
        self.assertTrue(current.exists())
        unrelated = self.base / 'unrelated'
        unrelated.mkdir()
        link = base / (ubai.RUNTIME_PREFIX + '789-symlink')
        link.symlink_to(unrelated)
        with self.assertRaises(ValueError):
            ubai.owned_runtime(base, link)
        self.assertTrue(unrelated.exists())

    def test_terminal_job_uses_queue_and_accounting_not_age(self) -> None:
        active = SimpleNamespace(returncode=0, stdout='123|RUNNING|user\n', stderr='')
        empty = SimpleNamespace(returncode=0, stdout='', stderr='')
        terminal = SimpleNamespace(returncode=0, stdout=f'123|COMPLETED|{os.getuid()}\n', stderr='')
        with patch.object(subprocess, 'run', return_value=active) as command:
            self.assertFalse(ubai.terminal_job('123'))
            self.assertEqual(command.call_count, 1)
        with patch.object(subprocess, 'run', side_effect=[empty, terminal]):
            self.assertTrue(ubai.terminal_job('123'))
        with patch.object(subprocess, 'run', side_effect=[empty, empty]):
            with self.assertRaises(ValueError):
                ubai.terminal_job('123')
        inaccessible = SimpleNamespace(returncode=1, stdout='', stderr='controller unavailable')
        with patch.object(subprocess, 'run', return_value=inaccessible):
            with self.assertRaises(ValueError):
                ubai.terminal_job('123')
        unknown = SimpleNamespace(returncode=1, stdout='', stderr='Invalid job id specified')
        with patch.object(subprocess, 'run', side_effect=[unknown, terminal]):
            self.assertTrue(ubai.terminal_job('123'))

    def test_tasks_reject_duplicate_unsafe_or_unassigned_identifiers(self) -> None:
        table = self.base / 'tasks.txt'
        table.write_text('theta_00_collect\ntheta_01_collect\n')
        with patch.dict(os.environ, {'TASK_IDS_FILE': str(table), 'SLURM_ARRAY_TASK_ID': '1'}, clear=True):
            self.assertEqual(ubai.task_identifier(), 'theta_01_collect')
        for environment in ({}, {'TASK_ID': '../escape'}, {'TASK_ID': 'a', 'TASK_IDS_FILE': str(table)},
                            {'TASK_IDS_FILE': str(table), 'SLURM_ARRAY_TASK_ID': '2'}):
            with patch.dict(os.environ, environment, clear=True):
                with self.assertRaises(ValueError):
                    ubai.task_identifier()
        table.write_text('same\nsame\n')
        with patch.dict(os.environ, {'TASK_IDS_FILE': str(table), 'SLURM_ARRAY_TASK_ID': '0'}, clear=True):
            with self.assertRaises(ValueError):
                ubai.task_identifier()

    def test_stale_preparation_cannot_reuse_mutated_assets(self) -> None:
        value = self.deployment()
        value['runtime']['host_assets_root'] = str(self.assets)
        deployment_path = self.output / 'deployment.json'
        deployment_path.write_bytes(runtime_files.json_bytes(value))
        records = ubai.verify_assets(value)
        report = {'state': 'verified', 'deployment_sha256': identity.sha256_file(deployment_path),
                  'python_version': '3.12.13', 'asset_files': records,
                  'dependency_files': ubai.verify_dependency_sources(value)}
        (self.output / 'prep-result.json').write_bytes(runtime_files.json_bytes(report))
        ubai.verify_preparation(value, deployment_path)
        (self.assets / 'training/extra').write_bytes(b'new file')
        with self.assertRaises(ValueError):
            ubai.verify_preparation(value, deployment_path)
        (self.assets / 'training/extra').unlink()
        (self.assets / 'training/payload').write_bytes(b'changed bytes')
        with self.assertRaises(ValueError):
            ubai.verify_preparation(value, deployment_path)

    def test_post_preparation_dependency_membership_and_content_changes_are_rejected(self) -> None:
        value = self.deployment()
        value['runtime']['host_assets_root'] = str(self.assets)
        deployment_path = self.output / 'deployment.json'
        deployment_path.write_bytes(runtime_files.json_bytes(value))
        report = {'state': 'verified', 'deployment_sha256': identity.sha256_file(deployment_path),
                  'python_version': '3.12.13', 'asset_files': ubai.verify_assets(value),
                  'dependency_files': ubai.verify_dependency_sources(value)}
        (self.output / 'prep-result.json').write_bytes(runtime_files.json_bytes(report))
        package = self.assets / 'source-checkouts/transformers/src'
        cache = package / '__pycache__'
        cache.mkdir()
        (cache / 'ignored.pyc').write_bytes(b'new cache')
        ubai.verify_preparation(value, deployment_path)
        new_source = package / 'extra.py'
        new_source.write_bytes(b'new source')
        with self.assertRaisesRegex(ValueError, 'file membership changed'):
            ubai.verify_preparation(value, deployment_path)
        new_source.unlink()
        (package / '__init__.py').write_bytes(b'changed source')
        with self.assertRaisesRegex(ValueError, 'source changed'):
            ubai.verify_preparation(value, deployment_path)

    def test_child_is_terminated_and_reaped_before_return_on_interrupt(self) -> None:
        child = Mock()
        child.pid = 456
        child.wait.side_effect = [InterruptedError('stop'), subprocess.TimeoutExpired('srun', 30), 0]
        child.poll.return_value = None
        with patch.object(subprocess, 'Popen', return_value=child), patch.object(os, 'killpg') as terminate:
            with self.assertRaises(InterruptedError):
                ubai.run_and_reap(['srun', 'fixture'], {})
        self.assertEqual(terminate.call_args_list[0].args, (456, signal.SIGTERM))
        self.assertEqual(terminate.call_args_list[1].args, (456, signal.SIGKILL))
        self.assertEqual(child.wait.call_count, 3)

    def test_signal_handler_also_covers_checksum_and_extraction(self) -> None:
        before = signal.getsignal(signal.SIGTERM)
        def signal_during_bootstrap(_path: Path, *, check_only: bool) -> None:
            handler = signal.getsignal(signal.SIGTERM)
            self.assertTrue(callable(handler))
            handler(signal.SIGTERM, None)
        with patch.object(ubai, '_execute_allocated', side_effect=signal_during_bootstrap):
            with self.assertRaises(InterruptedError):
                ubai.execute(self.base / 'deployment.json', check_only=True)
        self.assertEqual(signal.getsignal(signal.SIGTERM), before)


if __name__ == '__main__':
    unittest.main()
