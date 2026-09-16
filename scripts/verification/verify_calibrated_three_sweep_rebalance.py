"""Test queued job reassignment without contacting Slurm or using a GPU."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments import calibrated_three_sweep_rebalance as rebalance
from scripts.experiments.calibrated_three_sweeps import make_tasks
from scripts.runtime import identity
from scripts.verification.verify_calibrated_three_sweep_contract import experiment_fixture


class FakeController:
    def __init__(self, root: Path, *, count: int = 2, paired: bool = True):
        self.root = root
        self.prefix = 'c3-test-'
        self.rebalance_account = 'uos'
        self.tasks = make_tasks(experiment_fixture(), 'noise', selected_theta=40.0, seed=1,
                                calibration_sha256='2' * 64)[:count]
        self.state = {'tasks': {}, 'phase': 'noise-seed-1'}
        self.jobs = {}
        self.queue = []
        self.actions = []
        self.snapshots = []
        self.events = []
        self.transfers = []
        self.outputs = {}
        self.normal_polls = []
        self.accounting = 'CANCELLED by 1275|Unknown|0|\n'
        self.run_on_hold = False
        self.fail_after_hold = False
        self.hold_effective = True
        (self.root / 'tasks').mkdir(parents=True)
        for index, task in enumerate(self.tasks):
            number = index // 2 if paired else index
            job_id = str(1000 + number)
            name = self.prefix + 'job-' + job_id
            row = {'status': 'running', 'host': 'ubai', 'preferred_host': 'ubai',
                   'fixed_host': None, 'attempt': 2, 'started_at': 100.0,
                   'job_id': job_id, 'slurm_name': name,
                   'task_sha256': identity.json_sha256(task)}
            if paired:
                row['pair_id'] = 'pair-' + job_id
            self.state['tasks'][task['run_id']] = row
            (self.root / 'tasks' / (task['run_id'] + '.json')).write_text(json.dumps(task))
            if job_id not in self.jobs:
                self.jobs[job_id] = {'JobId': job_id, 'JobName': name, 'UserId': 'sizz1997(1275)',
                                     'Account': 'uos', 'JobState': 'PENDING',
                                     'Reason': 'Priority', 'Priority': '9000'}
                self.queue.append({'job_id': job_id, 'name': name, 'state': 'PENDING',
                                   'gpus': 2 if paired else 1})

    def save(self):
        self.snapshots.append(copy.deepcopy(self.state))

    def event(self, kind, **fields):
        self.events.append({'kind': kind, **fields})

    def remote(self, arguments, **kwargs):
        self.actions.append(list(arguments))
        if arguments[:3] == ['scontrol', 'show', 'job']:
            return ' '.join(f'{key}={value}' for key, value in self.jobs[arguments[-1]].items())
        if arguments[:2] == ['scontrol', 'hold']:
            job_id = arguments[-1]
            saved = self.snapshots[-1]['tasks']
            assert all(row['status'] == 'cancelling_for_local' and row['rebalance_hold_requested']
                       for row in saved.values() if row.get('job_id') == job_id)
            if self.hold_effective:
                self.jobs[job_id].update(Priority='0', Reason='JobHeldUser')
            if self.run_on_hold:
                self.jobs[job_id]['JobState'] = 'RUNNING'
            if self.fail_after_hold:
                self.fail_after_hold = False
                raise subprocess.CalledProcessError(255, arguments)
            return ''
        if arguments[:2] == ['scontrol', 'release']:
            self.jobs[arguments[-1]].update(Priority='9000', Reason='None')
            return ''
        if arguments[0] == 'scancel':
            job = self.jobs[arguments[-1]]
            assert job['JobState'] == 'PENDING' and job['Reason'] == 'JobHeldUser' and job['Priority'] == '0'
            assert arguments == ['scancel', '--state=PENDING', '--user=sizz1997',
                                 '--name=' + job['JobName'], job['JobId']]
            assert '--ctld' not in arguments
            return ''
        if arguments[0] == 'sacct':
            assert '--format=State,Start,ElapsedRaw' in arguments
            return self.accounting
        raise AssertionError(f'Unexpected remote action: {arguments}')

    def transfer(self, files, *, pull):
        assert pull
        self.transfers.append(list(files))

    def completed(self, task):
        value = self.outputs.get(task['run_id'])
        if isinstance(value, Exception):
            raise value
        return value

    def poll_task(self, task, queue):
        row = self.state['tasks'][task['run_id']]
        assert row['status'] == 'running'
        self.normal_polls.append(task['run_id'])
        row['status'] = 'pending'
        return self.completed(task)


# @lat: [[evaluation#Evaluation and Verification#Calibrated Three Sweep Queue Reassignment]]
class RebalanceTests(unittest.TestCase):
    def setUp(self):
        runtime = ROOT / 'artifacts/runtime'
        runtime.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='verify-reassignment-', dir=runtime)
        self.addCleanup(self.temporary.cleanup)
        self.counter = 0
        self.clock = patch.object(rebalance.time, 'time', return_value=1000.0)
        self.clock.start()
        self.addCleanup(self.clock.stop)

    def controller(self, **kwargs):
        self.counter += 1
        return FakeController(Path(self.temporary.name) / str(self.counter), **kwargs)

    def request(self, controller, slots=2):
        return rebalance.rebalance_pending(controller, controller.tasks, controller.queue, slots)

    def rows(self, controller):
        return list(controller.state['tasks'].values())

    def test_slurm_parsers_reject_incomplete_or_ambiguous_evidence(self):
        controller = self.controller()
        info = controller.remote(['scontrol', 'show', 'job', '-o', '1000'])
        self.assertEqual(rebalance.parse_slurm_job(info)['Account'], 'uos')
        for bad in (info.replace('Account=uos', ''), info + ' JobId=1001',
                    info.replace('Priority=9000', 'Priority=N/A')):
            with self.assertRaises(ValueError):
                rebalance.parse_slurm_job(bad)
        self.assertEqual(rebalance.parse_accounting('CANCELLED by 1275|Unknown|0|')[0],
                         {'state': 'CANCELLED', 'start': 'Unknown', 'elapsed_raw': 0})
        for bad in ('CANCELLED|Unknown|', 'CANCELLED|Unknown|NaN|', 'CANCELLED|', '|Unknown|0|'):
            with self.assertRaises(ValueError):
                rebalance.parse_accounting(bad)

    def test_pair_intent_is_saved_before_hold_and_cancel(self):
        controller = self.controller()
        self.assertEqual(self.request(controller), 2)
        self.assertTrue(all(row['status'] == 'cancelling_for_local' for row in self.rows(controller)))
        actions = [call[:2] for call in controller.actions]
        hold = actions.index(['scontrol', 'hold'])
        cancel = next(index for index, call in enumerate(actions) if call[0] == 'scancel')
        self.assertLess(hold, cancel)
        self.assertEqual(actions[hold + 1], ['scontrol', 'show'])
        self.assertEqual(controller.transfers, [])
        rebalance.poll_rebalance(controller, controller.tasks[1], controller.queue)
        self.assertEqual(sum(call[0] == 'scancel' for call in controller.actions), 1)
        self.assertTrue(all(row['attempt'] == 2 for row in self.rows(controller)))

    def test_capacity_wait_and_existing_requests_are_respected(self):
        controller = self.controller(count=4)
        self.assertEqual(self.request(controller, 1), 0)
        self.assertEqual(self.request(controller, 3), 2)
        self.assertEqual(self.request(controller, 3), 0)
        self.assertEqual(self.request(controller, 4), 2)
        self.assertEqual(self.request(controller, 8), 0)
        young = self.controller()
        for row in self.rows(young):
            row['started_at'] = 941
        self.assertEqual(self.request(young), 0)
        for row in self.rows(young):
            row['started_at'] = 940
        self.assertEqual(self.request(young), 2)

    def test_only_complete_current_stage_and_unfixed_groups_move(self):
        for change in ('fixed', 'partial', 'phase', 'queue_running', 'name', 'incomplete_pair'):
            with self.subTest(change=change):
                controller = self.controller()
                tasks = controller.tasks
                if change == 'fixed':
                    self.rows(controller)[0]['fixed_host'] = 'ubai'
                elif change == 'partial':
                    tasks = tasks[:1]
                elif change == 'phase':
                    tasks[1]['seed'] = 2
                    self.rows(controller)[1]['task_sha256'] = identity.json_sha256(tasks[1])
                    (controller.root / 'tasks' / (tasks[1]['run_id'] + '.json')).write_text(json.dumps(tasks[1]))
                elif change == 'queue_running':
                    controller.queue[0]['state'] = 'RUNNING'
                elif change == 'name':
                    controller.queue[0]['name'] = 'foreign'
                else:
                    self.rows(controller)[1]['job_id'] = '1001'
                self.assertEqual(rebalance.rebalance_pending(controller, tasks, controller.queue, 2), 0)
                self.assertFalse(any(call[0] in {'scancel'} or call[:2] == ['scontrol', 'hold']
                                     for call in controller.actions))

    def test_wrong_ownership_and_task_identity_are_rejected(self):
        for key, value in (('Account', 'other'), ('UserId', 'other(42)'),
                           ('JobId', '9999'), ('JobName', 'c3-test-other')):
            controller = self.controller()
            controller.jobs['1000'][key] = value
            with self.assertRaises(ValueError):
                self.request(controller)
            self.assertTrue(all(row['status'] == 'running' for row in self.rows(controller)))
        controller = self.controller()
        self.rows(controller)[0]['task_sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            self.request(controller)
        self.assertEqual(controller.actions, [])

    def test_preexisting_hold_and_already_started_job_are_not_changed(self):
        for state, priority, reason in (('PENDING', '0', 'JobHeldUser'),
                                        ('PENDING', '0', 'JobHeldAdmin'),
                                        ('RUNNING', '9000', 'None')):
            controller = self.controller()
            controller.jobs['1000'].update(JobState=state, Priority=priority, Reason=reason)
            self.assertEqual(self.request(controller), 0)
            self.assertEqual(len(controller.actions), 1)

    def test_running_race_releases_only_the_requested_hold(self):
        controller = self.controller()
        controller.run_on_hold = True
        self.assertEqual(self.request(controller), 2)
        self.assertIn(['scontrol', 'release', '1000'], controller.actions)
        self.assertFalse(any(call[0] == 'scancel' for call in controller.actions))
        self.assertTrue(all(row['status'] == 'running' and row['host'] == 'ubai'
                            and row['attempt'] == 2 and 'rebalance_target' not in row
                            for row in self.rows(controller)))

    def test_uncertain_hold_is_resumed_without_duplicate_local_work(self):
        controller = self.controller()
        controller.fail_after_hold = True
        with self.assertRaises(subprocess.CalledProcessError):
            self.request(controller)
        self.assertTrue(all(row['status'] == 'cancelling_for_local' for row in self.rows(controller)))
        self.assertEqual(self.request(controller, 2), 0)
        rebalance.poll_rebalance(controller, controller.tasks[0], controller.queue)
        self.assertEqual(sum(call[:2] == ['scontrol', 'hold'] for call in controller.actions), 1)
        self.assertEqual(sum(call[0] == 'scancel' for call in controller.actions), 1)

    def test_cancel_requires_confirmed_hold(self):
        controller = self.controller()
        controller.hold_effective = False
        self.request(controller)
        self.assertFalse(any(call[0] == 'scancel' for call in controller.actions))
        self.assertTrue(all(row['status'] == 'running' for row in self.rows(controller)))

    def test_external_hold_during_request_remains_untouched(self):
        for reason, own_requested in (('JobHeldAdmin', False), ('JobHeldUser', False),
                                       ('JobHeldAdmin', True)):
            controller = self.controller()
            with patch.object(rebalance, 'poll_rebalance'):
                self.request(controller)
            controller.jobs['1000'].update(Priority='0', Reason=reason)
            for row in self.rows(controller):
                row['rebalance_hold_requested'] = own_requested
            rebalance.poll_rebalance(controller, controller.tasks[0], controller.queue)
            self.assertFalse(any(call[0] == 'scancel' or call[:2] == ['scontrol', 'release']
                                 for call in controller.actions))
            self.assertEqual(controller.jobs['1000']['Priority'], '0')
            self.assertTrue(all(row['status'] == 'running' and row['host'] == 'ubai'
                                for row in self.rows(controller)))

    def test_queue_absence_alone_never_releases_work(self):
        for accounting in ('', 'PENDING|Unknown|0|', 'COMPLETING|2026-09-15T00:00:00|0|',
                            'CANCELLED|Unknown|0|\nRUNNING|Unknown|0|'):
            controller = self.controller()
            self.request(controller)
            controller.accounting = accounting
            self.assertIsNone(rebalance.poll_rebalance(controller, controller.tasks[0], []))
            self.assertTrue(all(row['status'] == 'cancelling_for_local' for row in self.rows(controller)))
            self.assertEqual(controller.transfers, [])
        controller = self.controller()
        self.request(controller)
        before = len(controller.actions)
        self.assertIsNone(rebalance.poll_rebalance(controller, controller.tasks[0], None))
        self.assertEqual(len(controller.actions), before)

    def test_terminal_cancel_moves_both_and_restores_queue_attempt(self):
        for start in ('Unknown', 'None', '(null)'):
            controller = self.controller()
            self.request(controller)
            controller.accounting = f'CANCELLED by 1275|{start}|0|\n'
            self.assertIsNone(rebalance.poll_rebalance(controller, controller.tasks[0], []))
            for row in self.rows(controller):
                self.assertEqual((row['status'], row['host'], row['preferred_host'], row['attempt']),
                                 ('pending', None, 'local', 1))
                self.assertEqual(row['rebalance_target'], 'local')
                self.assertNotIn('job_id', row)
                self.assertEqual(row['rebalance_history'][0]['job_id'], '1000')
            self.assertEqual(set(controller.transfers[0]),
                             {task[key] for task in controller.tasks for key in ('result_file', 'log_file')})
            rebalance.poll_rebalance(controller, controller.tasks[1], [])
            self.assertTrue(all(len(row['rebalance_history']) == 1 for row in self.rows(controller)))
            self.assertEqual(controller.normal_polls, [])

    def test_started_cancel_and_other_terminal_states_use_normal_retry(self):
        for accounting in ('CANCELLED|2026-09-15T00:00:00|1|', 'CANCELLED|Unknown|1|',
                            'CANCELLED||0|', 'FAILED|Unknown|0|',
                            'CANCELLED|Unknown|0|\nPREEMPTED|2026-09-15T00:00:00|5|'):
            controller = self.controller()
            self.request(controller)
            controller.accounting = accounting
            rebalance.poll_rebalance(controller, controller.tasks[0], [])
            self.assertEqual(controller.normal_polls, [task['run_id'] for task in controller.tasks])
            self.assertTrue(all(row['attempt'] == 2 and 'rebalance_target' not in row
                                for row in self.rows(controller)))

    def test_complete_peer_is_reused_and_invalid_result_blocks_both(self):
        controller = self.controller()
        self.request(controller)
        first = controller.tasks[0]
        accepted = {'elapsed_seconds': 10.0, 'correct': 100}
        controller.outputs[first['run_id']] = accepted
        returned = rebalance.poll_rebalance(controller, first, [])
        self.assertIs(returned, accepted)
        self.assertEqual(self.rows(controller)[0]['status'], 'complete')
        self.assertEqual(self.rows(controller)[1]['status'], 'pending')
        self.assertEqual(self.rows(controller)[1]['rebalance_target'], 'local')
        self.assertEqual(controller.normal_polls, [])
        invalid = self.controller()
        self.request(invalid)
        invalid.outputs[invalid.tasks[1]['run_id']] = ValueError('Invalid result identity')
        with self.assertRaises(ValueError):
            rebalance.poll_rebalance(invalid, invalid.tasks[0], [])
        self.assertTrue(all(row['status'] == 'cancelling_for_local' for row in self.rows(invalid)))

    def test_single_job_and_cancelled_started_peer_reuse(self):
        single = self.controller(count=1, paired=False)
        self.assertEqual(self.request(single, 1), 1)
        rebalance.poll_rebalance(single, single.tasks[0], [])
        self.assertEqual(self.rows(single)[0]['status'], 'pending')
        controller = self.controller()
        self.request(controller)
        controller.accounting = 'FAILED|2026-09-15T00:00:00|100|'
        controller.outputs[controller.tasks[0]['run_id']] = {'elapsed_seconds': 100}
        rebalance.poll_rebalance(controller, controller.tasks[0], [])
        self.assertEqual(self.rows(controller)[0]['status'], 'complete')
        self.assertEqual(controller.normal_polls, [controller.tasks[1]['run_id']])


if __name__ == '__main__':
    unittest.main()
