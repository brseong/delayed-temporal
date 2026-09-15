"""Verify paired Slurm execution without allocating GPUs or submitting jobs."""
from __future__ import annotations

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
from scripts.experiments.ubai import run_calibrated_three_sweep_pair as pair
from scripts.experiments.ubai import prepare_calibrated_three_sweeps_ubai as helper
from scripts.experiments.calibrated_three_sweeps import make_task, make_tasks
from scripts.verification.verify_calibrated_three_sweep_contract import experiment_fixture


class PairTests(unittest.TestCase):
    def setUp(self) -> None:
        parent = ROOT / 'artifacts/runtime'
        parent.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix='three-sweep-pair-test-', dir=parent)
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base / 'experiment'
        (self.root / 'pairs').mkdir(parents=True)
        (self.root / 'tasks').mkdir()
        self.experiment = experiment_fixture()
        self.deployment = self.base / 'deployment.json'
        self.deployment.write_text('{}\n')
        self.write(self.root / 'experiment.json', self.experiment)
        self.tasks = make_tasks(self.experiment, 'noise', selected_theta=40.0, seed=0,
                                calibration_sha256='2' * 64)[:2]
        self.manifest = {'format_version': 1, 'pair_id': 'test-pair',
                         'experiment_sha256': pair.sha256(self.root / 'experiment.json'),
                         'deployment_sha256': pair.sha256(self.deployment),
                         'source_commit': self.experiment['source_commit'], 'controller_commit': 'b' * 40,
                         'controller_sha256': {relative: pair.sha256(ROOT / relative)
                                               for relative in (pair.PAIR_SCRIPT, pair.PAIR_BATCH)}, 'tasks': []}
        self.path = self.root / 'pairs/test-pair.json'
        self.refresh()

    @staticmethod
    def write(path: Path, value: dict) -> None:
        path.write_text(json.dumps(value, indent=2) + '\n')

    def refresh(self) -> None:
        self.manifest['tasks'] = []
        for task in self.tasks:
            path = self.root / 'tasks' / (task['run_id'] + '.json')
            self.write(path, task)
            self.manifest['tasks'].append({'run_id': task['run_id'], 'task_sha256': pair.sha256(path)})
        self.write(self.path, self.manifest)

    def validate(self) -> tuple[dict, list]:
        with patch.object(pair, 'controller_identity'):
            return pair.validate_pair(self.path, self.root, self.deployment, ROOT, ROOT, 'b' * 40)

    # @lat: [[evaluation#Evaluation and Verification#Calibrated Three Sweep Paired Jobs]]
    def test_resource_contract_is_two_independent_evaluations(self) -> None:
        script = (ROOT / pair.PAIR_BATCH).read_text()
        for line in ('#SBATCH --gres=gpu:2', '#SBATCH --cpus-per-task=8', '#SBATCH --mem=128G',
                     '#SBATCH --ntasks=1', '#SBATCH --nodes=1', '#SBATCH --partition=gpu4,gpu5'):
            self.assertIn(line, script)
        self.assertNotIn('#SBATCH --container-image', script)
        self.assertEqual(pair.gpu_tokens('0,1'), ['0', '1'])
        self.assertEqual(pair.gpu_tokens('GPU-abcd,GPU-efgh'), ['GPU-abcd', 'GPU-efgh'])
        for invalid in ('', '0', '0,0', '0,00', '0,1,2', 'all', '0, 1', '-1,1'):
            with self.assertRaises(ValueError, msg=invalid):
                pair.gpu_tokens(invalid)

    def test_pair_preserves_raw_task_and_experiment_identity(self) -> None:
        value, tasks = self.validate()
        self.assertEqual(value, self.manifest)
        self.assertEqual(tasks, self.tasks)
        for path in (self.root / 'experiment.json', self.deployment,
                     self.root / 'tasks' / (self.tasks[0]['run_id'] + '.json')):
            original = path.read_bytes()
            path.write_bytes(original + b' ')
            with self.assertRaises(ValueError):
                self.validate()
            path.write_bytes(original)
        original = copy.deepcopy(self.manifest)
        for field in ('source_commit', 'controller_commit'):
            self.manifest[field] = 'f' * 40
            self.write(self.path, self.manifest)
            with self.assertRaises(ValueError):
                self.validate()
            self.manifest = copy.deepcopy(original)

    def test_probe_resolves_visible_devices_to_distinct_physical_uuids(self) -> None:
        devices = [{'name': 'NVIDIA RTX A6000', 'uuid': value} for value in (
            'GPU-12345678-1234-1234-1234-123456789abc', '23456789-2345-2345-2345-23456789abcd')]
        result = {'count': 2, 'devices': devices}
        with patch.object(subprocess, 'check_output', return_value=json.dumps(result)) as probe:
            actual = pair.probe_allocated_devices('3,5')
        self.assertEqual(actual, ['GPU-12345678-1234-1234-1234-123456789abc',
                                  'GPU-23456789-2345-2345-2345-23456789abcd'])
        self.assertEqual(probe.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '3,5')
        self.assertIn('torch.cuda.get_device_properties', probe.call_args.args[0][-1])
        for invalid in (
            {'count': 1, 'devices': devices[:1]},
            {'count': 2, 'devices': [devices[0], devices[0]]},
            {'count': 2, 'devices': [devices[0], dict(devices[1], name='NVIDIA A10')]},
            {'count': 2, 'devices': [devices[0], dict(devices[1], uuid='unavailable')]},
        ):
            with patch.object(subprocess, 'check_output', return_value=json.dumps(invalid)):
                with self.assertRaises(ValueError):
                    pair.probe_allocated_devices('0,1')

    def test_pair_rejects_duplicates_cross_seed_and_table_dependency(self) -> None:
        self.tasks = [self.tasks[0], self.tasks[0]]
        self.refresh()
        with self.assertRaises(ValueError):
            self.validate()
        self.tasks = [make_tasks(self.experiment, 'noise', selected_theta=40.0, seed=seed,
                                  calibration_sha256='2' * 64)[0] for seed in (0, 1)]
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'same seed'):
            self.validate()
        self.tasks = [make_task(self.experiment, 'collect'),
                      make_task(self.experiment, 'theta_train', calibration_sha256='2' * 64)]
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'same table'):
            self.validate()

    def test_clean_and_noisy_smoke_can_share_a_pair(self) -> None:
        self.tasks = [make_tasks(self.experiment, phase, host_label='ubai',
                                 calibration_sha256='2' * 64)[0]
                      for phase in ('smoke_clean', 'smoke_noise')]
        self.refresh()
        self.assertEqual(self.validate()[1], self.tasks)

    def test_logical_phase_groups_do_not_cross_stage_boundaries(self) -> None:
        self.tasks = [make_task(self.experiment, 'collect', theta_index=0),
                      make_task(self.experiment, 'theta_train', theta_index=1, calibration_sha256='2' * 64)]
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'same logical stage'):
            self.validate()
        self.tasks = [make_task(self.experiment, 'theta_validation', calibration_sha256='2' * 64),
                      make_task(self.experiment, 'dense')]
        self.tasks[1]['host_label'] = None
        self.refresh()
        self.assertEqual(self.validate()[1], self.tasks)

    def test_controller_git_check_is_separate_from_evaluator_git_environment(self) -> None:
        with patch.dict(os.environ, {'GIT_WORK_TREE': '/frozen/evaluator', 'GIT_DIR': '/frozen/git'}), \
             patch.object(subprocess, 'check_output', side_effect=['b' * 40 + '\n', '']) as command:
            pair.controller_identity(ROOT, 'b' * 40, self.manifest['controller_sha256'])
            self.assertNotIn('GIT_WORK_TREE', command.call_args_list[0].kwargs['env'])
            self.assertNotIn('GIT_DIR', command.call_args_list[0].kwargs['env'])
        with patch.object(subprocess, 'check_output', side_effect=['c' * 40 + '\n', '']):
            with self.assertRaises(ValueError):
                pair.controller_identity(ROOT, 'b' * 40, self.manifest['controller_sha256'])
        hashes = dict(self.manifest['controller_sha256'], **{pair.PAIR_SCRIPT: '0' * 64})
        with patch.object(subprocess, 'check_output', side_effect=['b' * 40 + '\n', '']):
            with self.assertRaises(ValueError):
                pair.controller_identity(ROOT, 'b' * 40, hashes)

    def test_shared_environment_extracted_once_and_second_scratch_reserved(self) -> None:
        base = self.base / 'data'
        base.mkdir()
        configuration = {'env_unpacked_bytes': 96 * helper.GIB, 'minimum_scratch_bytes': 8 * helper.GIB,
                         'env_archive': str(self.base / 'unused-archive')}
        with patch.object(helper, 'disk_directory'), patch.object(helper, 'extract_environment') as extract, \
             patch.object(helper.shutil, 'disk_usage', return_value=SimpleNamespace(free=200 * helper.GIB)):
            runtime, reservation = pair.admit_pair_runtime(helper, base, '123', 'pair1', configuration)
        self.assertEqual(extract.call_count, 1)
        owners = [helper.owned_runtime(base, path) for path in (runtime, reservation)]
        self.assertEqual(sum(owner['scratch_bytes'] for owner in owners), 16 * helper.GIB)
        self.assertEqual({owner['job_id'] for owner in owners}, {'123'})
        helper.release_runtime(base, reservation, '123')
        helper.release_runtime(base, runtime, '123')
        self.assertFalse(runtime.exists())
        self.assertFalse(reservation.exists())

    def test_failed_second_reservation_releases_first_environment(self) -> None:
        base = self.base / 'data'
        base.mkdir()
        configuration = {'env_unpacked_bytes': 96 * helper.GIB, 'minimum_scratch_bytes': 8 * helper.GIB,
                         'env_archive': str(self.base / 'unused-archive')}
        with patch.object(helper, 'disk_directory'), patch.object(helper, 'extract_environment') as extract, \
             patch.object(helper.shutil, 'disk_usage', side_effect=[SimpleNamespace(free=200 * helper.GIB),
                                                                  SimpleNamespace(free=15 * helper.GIB)]):
            with self.assertRaisesRegex(ValueError, 'second paired scratch'):
                pair.admit_pair_runtime(helper, base, '123', 'pair1', configuration)
        self.assertEqual(extract.call_count, 1)
        self.assertFalse(list(base.glob(helper.RUNTIME_PREFIX + '*')))

    def test_container_reuses_old_mounts_and_adds_controller_and_shared_scratch(self) -> None:
        command = ['srun', '--gres=gpu:1', '--cpus-per-task=4', '--container-mounts=/inputs:/inputs:ro',
                   '/usr/bin/env', 'TMPDIR=/work-tmp', '/opt/conda/envs/dt/bin/python', 'old-worker']
        deployment = {'source_root': '/source', 'experiment_root': '/experiment'}
        with patch.object(helper, 'container_command', return_value=command):
            actual = pair.pair_container_command(helper, deployment, Path('/disk/runtime'), 'pair1',
                                                  Path('/controller'), 'b' * 40)
        self.assertIn('--gres=gpu:2', actual)
        self.assertIn('--cpus-per-task=8', actual)
        self.assertIn('--cpu-bind=cores', actual)
        self.assertNotIn('old-worker', actual)
        mounts = next(item for item in actual if item.startswith('--container-mounts='))
        self.assertIn('/inputs:/inputs:ro', mounts)
        self.assertIn('/controller:/controller:ro', mounts)
        self.assertIn('/disk/runtime/scratch:/disk/runtime/scratch', mounts)
        self.assertIn('--inside', actual)

    def test_workers_have_distinct_gpus_cpu_sets_caches_and_independent_statuses(self) -> None:
        scratch = self.base / 'scratch'
        scratch.mkdir()
        children = [Mock(), Mock()]
        children[0].wait.return_value = 1
        children[0].poll.return_value = 1
        children[1].wait.return_value = 0
        children[1].poll.return_value = 0
        with patch.object(os, 'sched_getaffinity', return_value=set(range(8))), \
             patch.object(subprocess, 'Popen', side_effect=children) as launch, \
             patch.dict(os.environ, {'WANDB_API_KEY': 'must-not-leak'}):
            statuses = pair.run_workers(ROOT, self.root, self.tasks, scratch, '3,5')
        self.assertEqual(statuses, [1, 0])
        self.assertEqual(launch.call_count, 2)
        self.assertEqual([call.kwargs['env']['CUDA_VISIBLE_DEVICES'] for call in launch.call_args_list], ['3', '5'])
        with patch.object(os, 'sched_setaffinity') as affinity:
            for call in launch.call_args_list:
                call.kwargs['preexec_fn']()
            self.assertEqual(affinity.call_args_list[0].args, (0, {0, 1, 2, 3}))
            self.assertEqual(affinity.call_args_list[1].args, (0, {4, 5, 6, 7}))
        for index, call in enumerate(launch.call_args_list):
            environment = call.kwargs['env']
            self.assertNotIn('WANDB_API_KEY', environment)
            self.assertEqual(environment['OMP_NUM_THREADS'], '4')
            self.assertEqual(environment['TMPDIR'], str(scratch / f'worker-{index}'))
            self.assertTrue(environment['TORCH_HOME'].startswith(environment['TMPDIR']))
        self.assertEqual(children[1].wait.call_count, 2)

    def test_worker_launcher_does_not_overwrite_existing_completed_results(self) -> None:
        results = self.root / 'results'
        results.mkdir()
        completed = results / (self.tasks[0]['run_id'] + '.json')
        completed.write_bytes(b'validated result preserved by frozen worker')
        before = pair.sha256(completed)
        scratch = self.base / 'scratch'
        scratch.mkdir()
        child = Mock()
        child.wait.return_value = 0
        child.poll.return_value = 0
        with patch.object(os, 'sched_getaffinity', return_value=set(range(8))), \
             patch.object(subprocess, 'Popen', return_value=child):
            self.assertEqual(pair.run_workers(ROOT, self.root, self.tasks, scratch, '0,1'), [0, 0])
        self.assertEqual(pair.sha256(completed), before)

    def test_reaping_terminates_and_waits_for_both_children(self) -> None:
        children = [Mock(pid=123), Mock(pid=456)]
        for child in children:
            child.poll.return_value = None
        children[0].wait.side_effect = [subprocess.TimeoutExpired('worker', 25), 0]
        children[1].wait.return_value = 0
        with patch.object(os, 'killpg') as kill:
            pair.reap_children(children)
        self.assertEqual(kill.call_args_list[0].args, (123, signal.SIGTERM))
        self.assertEqual(kill.call_args_list[1].args, (456, signal.SIGTERM))
        self.assertEqual(kill.call_args_list[2].args, (123, signal.SIGKILL))
        self.assertEqual(children[0].wait.call_count, 2)
        self.assertEqual(children[1].wait.call_count, 1)

    def test_reaping_peer_continues_if_first_child_exits_during_signal(self) -> None:
        children = [Mock(pid=123), Mock(pid=456)]
        for child in children:
            child.poll.return_value = None
            child.wait.return_value = 0
        with patch.object(os, 'killpg', side_effect=[ProcessLookupError(), None]) as kill:
            pair.reap_children(children)
        self.assertEqual(kill.call_args_list[1].args, (456, signal.SIGTERM))
        self.assertEqual([child.wait.call_count for child in children], [1, 1])


if __name__ == '__main__':
    unittest.main()
