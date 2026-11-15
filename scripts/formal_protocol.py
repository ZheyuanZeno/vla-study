"""Frozen protocol checks. This module never creates an approval or runs a model."""
import hashlib
import json
from pathlib import Path

CONDITIONS=['Baseline','T1','T2','T3','B1','B2','B3']
INFERENCE={'seed':42,'order_seed':20261115,'horizon':280,'n_action_steps':1,'num_flow_steps':10,
           'use_amp':False,'matmul_allow_tf32':True,'cudnn_benchmark':True,'control_mode':'relative'}
ANALYSIS={'bootstrap_replicates':5000,'task_weighting':'equal',
          'resampling_unit':'seven_condition_initial_state_block','interval':'pointwise_95_percentile'}


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execution_code_hash():
    paths=sorted(Path('src').rglob('*.py'))+[Path('requirements-lock.txt'),Path('configs/source_versions.json'),Path(__file__)]
    return hashlib.sha256(b''.join(str(p.relative_to(Path.cwd()) if p.is_absolute() else p).encode()+b'\0'+p.read_bytes() for p in paths)).hexdigest()


def fingerprint(protocol):
    return hashlib.sha256(json.dumps({k:v for k,v in protocol.items() if k!='protocol_sha256'},sort_keys=True).encode()).hexdigest()


def validate_approval(protocol,approval):
    if approval.get('approved') is not True:
        raise PermissionError('Formal collection has not been explicitly approved')
    if approval.get('protocol_sha256')!=protocol['protocol_sha256']:
        raise PermissionError('Approval belongs to a different frozen protocol')
    if approval.get('approved_rollouts')!=protocol['planned_rollouts']:
        raise PermissionError('Approval does not cover this exact sample size')
    if not approval.get('user_message'):
        raise PermissionError('Record the explicit user approval message')


def validate_protocol(path,require_approval=True):
    protocol=json.loads(Path(path).read_text())
    if fingerprint(protocol)!=protocol['protocol_sha256']: raise ValueError('Protocol was modified')
    if protocol['execution_code_hash']!=execution_code_hash(): raise ValueError('Execution code differs from frozen protocol')
    source=json.loads(Path('configs/source_versions.json').read_text())
    if protocol['checkpoint_revision']!=source['checkpoint']['revision']: raise ValueError('Wrong checkpoint revision')
    if protocol['vision_language_base_revision']!=source['vision_language_base']['revision']: raise ValueError('Wrong VLM revision')
    if protocol['analysis']!=ANALYSIS: raise ValueError('Analysis protocol differs')
    if protocol['inference']!=INFERENCE: raise ValueError('Inference settings differ')
    import importlib.metadata as metadata
    for package,version in protocol['dependency_versions'].items():
        if metadata.version(package)!=version: raise ValueError(f'Dependency changed: {package}')
    import subprocess
    commit=subprocess.check_output(['git','-C','vendor/lerobot','rev-parse','HEAD'],text=True).strip()
    if commit!=source['lerobot']['commit']: raise ValueError('LeRobot source commit changed')
    if protocol['conditions']!=CONDITIONS: raise ValueError('Need exact seven conditions')
    n=protocol['sample_size_per_task']
    if n!=50 or protocol['state_pool_type']!='native':
        raise ValueError('Only the validated native 50-state proposal is currently supported; expanded pools need validation')
    if protocol['planned_rollouts']!=4*7*n: raise ValueError('Incorrect rollout total')
    tasks=protocol['tasks']
    if len(tasks)!=4 or len({(t['suite'],t['task_id']) for t in tasks})!=4: raise ValueError('Need four unique tasks')
    for name in ['selection','state_inventory']:
        if file_sha(protocol[name+'_file'])!=protocol[name+'_sha256']: raise ValueError(f'{name} evidence changed')
    selection=json.loads(Path(protocol['selection_file']).read_text())
    if not selection['four_tasks_selected']: raise ValueError('Screening not complete')
    if any(t['n_valid']!=20 or t['successes']<10 for t in selection['selected']): raise ValueError('Baseline selection gate not met')
    if [(t['suite'],t['task_id']) for t in tasks]!=[(t['suite'],t['task_id']) for t in selection['selected']]:
        raise ValueError('Tasks are not the first four passing the confirmed order')
    inventory=json.loads(Path(protocol['state_inventory_file']).read_text())
    from lerobot.envs.libero import _get_suite,get_task_init_states
    import numpy as np
    for task in tasks:
        for name in ['regions','geometry_audit','screening_results','pilot_results','pilot_verification']:
            if file_sha(task[name+'_file'])!=task[name+'_sha256']: raise ValueError(f'{name} changed')
        expected=next(r for r in inventory if (r['suite'],r['task_id'])==(task['suite'],task['task_id']))
        if expected['pool_sha256']!=task['native_pool_sha256'] or expected['unique_exact_states']!=50:
            raise ValueError('State pool is not the audited native pool')
        actual=np.asarray(get_task_init_states(_get_suite(task['suite']),task['task_id']))
        if hashlib.sha256(actual.tobytes()).hexdigest()!=task['native_pool_sha256']:
            raise ValueError('Installed initial-state pool changed')
        if task['state_ids']!=list(range(50)): raise ValueError('Native proposal must use the explicit 50 IDs exactly once')
        pilot_rows=[json.loads(line) for line in Path(task['pilot_results_file']).read_text().splitlines() if line.strip()]
        valid_pilot=[r for r in pilot_rows if r['status']!='infrastructure_failure']
        if any(r['code_hash']!=protocol['execution_code_hash'] or r['checkpoint_revision']!=protocol['checkpoint_revision'] for r in valid_pilot):
            raise ValueError('Pilot code/checkpoint differs from frozen proposal')
        verification=json.loads(Path(task['pilot_verification_file']).read_text())
        if verification['valid_rollouts']!=7 or not all(b['seven_complete'] and b['saved_actual_inputs_verified'] for b in verification['blocks']):
            raise ValueError('Selected task pilot not verified')
        audit=json.loads(Path(task['geometry_audit_file']).read_text())
        if not audit['all_passed'] or len(audit['records'])!=50: raise ValueError('Geometry not validated across native pool')
    if require_approval:
        approval_path=Path(protocol['approval_file'])
        if not approval_path.exists(): raise PermissionError(f'Formal approval is absent: {approval_path}')
        validate_approval(protocol,json.loads(approval_path.read_text()))
    return protocol


def validate_run_config(config):
    if config.get('study_phase')!='formal': raise ValueError('Formal config must be labelled formal')
    if not config.get('protocol_file') or not Path(config['protocol_file']).is_file():
        raise PermissionError('A frozen formal protocol and explicit approval are required')
    protocol=validate_protocol(config['protocol_file'],require_approval=True)
    if config['protocol_sha256']!=protocol['protocol_sha256']: raise ValueError('Wrong protocol hash in run config')
    task=next((t for t in protocol['tasks'] if (t['suite'],t['task_id'])==(config['suite'],config['task_id'])),None)
    if task is None: raise ValueError('Task not approved')
    expected={'conditions':CONDITIONS,'state_ids':task['state_ids'],'checkpoint':'HuggingFaceVLA/smolvla_libero',
        'seed':protocol['inference']['seed'],'order_seed':protocol['inference']['order_seed'],
        'horizon':280,'regions_file':task['regions_file'],'output_dir':task['formal_output_dir'],
        'record_video':protocol['record_video']}
    if any(config.get(k)!=v for k,v in expected.items()): raise ValueError('Run config differs from frozen approved protocol')
    return protocol
