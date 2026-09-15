"""Move held, never-started Slurm jobs to local workers after terminal proof."""
from __future__ import annotations

import json
import math
import re
import time
from typing import Any

from scripts.experiments.calibrated_three_sweeps import task_sha256

MIN_WAIT_SECONDS = 60
RETRY_SECONDS = 10
USER = 'sizz1997'
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'PREEMPTED', 'NODE_FAIL', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}


def parse_slurm_job(text: str) -> dict[str, str]:
    required = {'JobId', 'JobName', 'UserId', 'Account', 'JobState', 'Reason', 'Priority'}
    pairs = re.findall(r'(?:^|\s)(JobId|JobName|UserId|Account|JobState|Reason|Priority)=([^\s]+)', text)
    fields = dict(pairs)
    if set(fields) != required or len(pairs) != len(required):
        raise ValueError('Incomplete or ambiguous Slurm job identity')
    if not fields['JobId'].isdecimal() or not fields['Priority'].isdecimal():
        raise ValueError('Invalid Slurm job identity')
    return fields


def parse_accounting(text: str) -> list[dict[str, Any]]:
    records = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.split('|')
        if len(fields) < 3 or not fields[0].strip() or not fields[2].strip().isdecimal():
            raise ValueError('Incomplete Slurm terminal accounting')
        records.append({'state': fields[0].split()[0].rstrip('+'),
                        'start': fields[1].strip(), 'elapsed_raw': int(fields[2].strip())})
    return records


def _group(controller: Any, job_id: str) -> list[tuple[dict, dict]]:
    members = []
    for run_id, row in controller.state['tasks'].items():
        if row.get('host') != 'ubai' or str(row.get('job_id')) != job_id:
            continue
        task = json.loads((controller.root / 'tasks' / (run_id + '.json')).read_text())
        if task['run_id'] != run_id or row.get('task_sha256') != task_sha256(task):
            raise ValueError('Assignment task identity changed during reassignment')
        members.append((task, row))
    return members


def _compatible(members: list[tuple[dict, dict]]) -> bool:
    if not members or len(members) > 2:
        return False
    phases = {'collect': 'calibration', 'smoke_clean': 'smoke', 'smoke_noise': 'smoke',
              'theta_train': 'theta', 'theta_validation': 'theta', 'dense': 'theta',
              'theta_replay': 'replay', 'noise': 'noise'}
    tasks = [task for task, _ in members]
    if any(task.get('host_label') is not None or task['kind'] not in phases for task in tasks):
        return False
    if len({phases[task['kind']] for task in tasks}) != 1:
        return False
    if tasks[0]['kind'] == 'noise' and len({task['seed'] for task in tasks}) != 1:
        return False
    pairs = {row.get('pair_id') for _, row in members}
    return len(pairs) == 1 and ((len(members) == 2) == (next(iter(pairs)) is not None))


def _identity(controller: Any, job_id: str, members: list[tuple[dict, dict]],
              *, before_request: bool = False) -> dict[str, str]:
    info = parse_slurm_job(controller.remote(['scontrol', 'show', 'job', '-o', job_id]))
    names = {row.get('slurm_name') for _, row in members}
    expected_account = getattr(controller, 'rebalance_account', None)
    if (len(names) != 1 or info['JobId'] != job_id or info['JobName'] not in names
            or not info['JobName'].startswith(controller.prefix)
            or info['UserId'].split('(')[0] != USER or info['Account'] != expected_account):
        raise ValueError('Slurm ownership does not match the assigned job')
    for _, row in members:
        old = row.get('rebalance_job_identity')
        if old and any(old[key] != info[key] for key in ('JobId', 'JobName', 'UserId', 'Account')):
            raise ValueError('Slurm identity changed after reassignment was requested')
    if before_request and (info['JobState'] != 'PENDING' or int(info['Priority']) == 0
                           or info['Reason'].startswith('JobHeld')):
        return {}
    return info


def _clear_request(row: dict) -> None:
    for key in list(row):
        if key.startswith('rebalance_') and key not in {'rebalance_history', 'rebalance_target'}:
            row.pop(key)


def _history(row: dict, outcome: str, accounting: list[dict] | None = None) -> None:
    entry = {key: row[key] for key in ('job_id', 'slurm_name', 'pair_id', 'attempt', 'started_at') if key in row}
    entry.update(outcome=outcome, recorded_at=time.time(), accounting=accounting)
    row.setdefault('rebalance_history', []).append(entry)


def _restore_remote(controller: Any, job_id: str, members: list[tuple[dict, dict]],
                    info: dict[str, str]) -> None:
    # A user hold applied to a running job must not affect a later Slurm requeue.
    release_own_hold = (any(row.get('rebalance_hold_requested') for _, row in members)
                        and info['Priority'] == '0' and info['Reason'] != 'JobHeldAdmin'
                        and (info['Reason'] == 'JobHeldUser'
                             or info['JobState'] in {'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED'}))
    if release_own_hold:
        controller.remote(['scontrol', 'release', job_id])
    for _, row in members:
        _history(row, 'kept_remote')
        _clear_request(row)
        row['status'] = 'running'
    controller.save()
    controller.event('queue_reassignment_kept_remote', job_id=job_id, state=info['JobState'])


def rebalance_pending(controller: Any, tasks: list[dict], queue: list[dict] | None,
                      local_slots: int) -> int:
    if queue is None or local_slots <= 0:
        return 0
    active = {task['run_id'] for task in tasks}
    reserved = sum(row['status'] == 'cancelling_for_local' for row in controller.state['tasks'].values())
    available = max(0, local_slots - reserved)
    requested = 0
    for job in queue:
        if job['state'] != 'PENDING' or not job['name'].startswith(controller.prefix):
            continue
        members = _group(controller, str(job['job_id']))
        if (not _compatible(members) or len(members) > available
                or any(task['run_id'] not in active or row['status'] != 'running'
                       or row.get('fixed_host') or row.get('slurm_name') != job['name']
                       for task, row in members)):
            continue
        starts = [row.get('started_at') for _, row in members]
        if any(not isinstance(value, (int, float)) or not math.isfinite(value)
               or time.time() - value < MIN_WAIT_SECONDS for value in starts):
            continue
        if int(job.get('gpus', 0)) != len(members):
            continue
        info = _identity(controller, str(job['job_id']), members, before_request=True)
        if not info:
            continue
        # Save both members before the first state-changing Slurm command.
        for _, row in members:
            row.update(status='cancelling_for_local', rebalance_requested_at=time.time(),
                       rebalance_previous_attempt=row['attempt'] - 1,
                       rebalance_hold_requested=False, rebalance_job_identity=info)
        controller.save()
        controller.event('queue_reassignment_requested', job_id=job['job_id'],
                         run_ids=[task['run_id'] for task, _ in members])
        requested += len(members)
        available -= len(members)
        poll_rebalance(controller, members[0][0], queue)
    return requested


def poll_rebalance(controller: Any, task: dict, queue: list[dict] | None) -> dict | None:
    row = controller.state['tasks'][task['run_id']]
    if row['status'] != 'cancelling_for_local':
        return controller.completed(task)
    if queue is None:
        return None
    job_id = str(row['job_id'])
    members = _group(controller, job_id)
    if not _compatible(members) or any(member['status'] != 'cancelling_for_local' for _, member in members):
        raise ValueError('Reassignment must retain every member of the original job')
    queued = [job for job in queue if str(job['job_id']) == job_id]
    if len(queued) > 1:
        raise ValueError('Ambiguous queued Slurm identity')
    if queued:
        if queued[0]['name'] != row['slurm_name']:
            raise ValueError('Queued Slurm name changed during reassignment')
        info = _identity(controller, job_id, members)
        if info['JobState'] in {'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED'}:
            _restore_remote(controller, job_id, members, info)
            return None
        if info['JobState'] != 'PENDING':
            return None
        if int(info['Priority']) > 0:
            last = max(member.get('rebalance_last_hold_at', 0) for _, member in members)
            if time.time() - last < RETRY_SECONDS:
                return None
            for _, member in members:
                member.update(rebalance_hold_requested=True, rebalance_last_hold_at=time.time())
            controller.save()
            controller.remote(['scontrol', 'hold', job_id])
            info = _identity(controller, job_id, members)
            if info['JobState'] != 'PENDING':
                _restore_remote(controller, job_id, members, info)
                return None
        if (info['Priority'] != '0' or info['Reason'] != 'JobHeldUser'
                or not all(member.get('rebalance_hold_requested') for _, member in members)):
            _restore_remote(controller, job_id, members, info)
            return None
        last = max(member.get('rebalance_last_cancel_at', 0) for _, member in members)
        if time.time() - last >= RETRY_SECONDS:
            for _, member in members:
                member['rebalance_last_cancel_at'] = time.time()
            controller.save()
            # Slurm 23.11 filters scancel on the client. Hold first to prevent scheduling.
            controller.remote(['scancel', '--state=PENDING', '--user=' + USER,
                               '--name=' + row['slurm_name'], job_id])
        return None
    accounting = parse_accounting(controller.remote(
        ['sacct', '-n', '-X', '-j', job_id, '--format=State,Start,ElapsedRaw', '--parsable2']))
    if not accounting or any(record['state'] not in TERMINAL for record in accounting):
        return None
    never_started = all(record['state'] == 'CANCELLED'
                        and record['start'].lower() in {'unknown', 'none', '(null)'}
                        and record['elapsed_raw'] == 0 for record in accounting)
    files = []
    for member_task, _ in members:
        files.extend([member_task['result_file'], member_task['log_file']])
        if member_task['kind'] == 'collect':
            files.append(member_task['calibration_file'])
    controller.transfer(list(dict.fromkeys(files)), pull=True)
    # Validate every available result before releasing either assignment.
    results = {member_task['run_id']: controller.completed(member_task) for member_task, _ in members}
    retry_tasks = []
    for member_task, member in members:
        result = results[member_task['run_id']]
        previous_attempt = member.get('rebalance_previous_attempt')
        _history(member, 'complete' if result is not None else 'local' if never_started else 'terminal_retry', accounting)
        _clear_request(member)
        if result is not None:
            member.update(status='complete', elapsed_seconds=result['elapsed_seconds'], finished_at=time.time())
        elif never_started:
            member.update(status='pending', host=None, preferred_host='local', rebalance_target='local',
                          attempt=previous_attempt)
            for key in ('job_id', 'slurm_name', 'pair_id', 'started_at', 'pid', 'gpu', 'cpu_ids'):
                member.pop(key, None)
        else:
            member['status'] = 'running'
            retry_tasks.append(member_task)
    controller.save()
    controller.event('queue_reassignment_terminal', job_id=job_id, never_started=never_started,
                     reused=[run_id for run_id, result in results.items() if result is not None])
    for member_task in retry_tasks:
        controller.poll_task(member_task, queue)
    return results[task['run_id']]
