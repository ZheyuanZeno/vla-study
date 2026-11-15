"""Build a reviewable native-50 proposal after selection and held-out pilots.

No approval file is written and no formal rollout is started.
"""
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import yaml
from scripts.formal_protocol import ANALYSIS,INFERENCE,CONDITIONS,file_sha,execution_code_hash,fingerprint

selection_file=Path('artifacts/screening_v1/selection.json')
selection=json.loads(selection_file.read_text())
if not selection['four_tasks_selected']: raise RuntimeError('Ordered screening not complete')
inventory_file=Path('artifacts/state_inventory.json');inventory=json.loads(inventory_file.read_text())
source=json.loads(Path('configs/source_versions.json').read_text())
config_dir=Path('configs/formal50');config_dir.mkdir(parents=True,exist_ok=True)
tasks=[];durations=[];peak_allocated=0;technical_attempts=0
for selected in selection['selected']:
    suite,task_id=selected['suite'],selected['task_id'];slug=f'{suite}_{task_id}'
    pilot=Path('artifacts/pilot_selected_v1')/slug
    verification=pilot/'verification.json';report=json.loads(verification.read_text())
    if report['valid_rollouts']!=7 or not all(b['seven_complete'] for b in report['blocks']):
        raise RuntimeError('Held-out pilot is not complete')
    results=pilot/'results.jsonl';rows=[json.loads(s) for s in results.read_text().splitlines()]
    valid=[r for r in rows if r['status']!='infrastructure_failure']
    if len(valid)!=7 or any(r['state_id']!=20 or r['study_phase']!='pilot' or r['code_hash']!=execution_code_hash() for r in valid):
        raise ValueError('Expected the prespecified state-20 pilot')
    durations.extend(r['elapsed_time'] for r in valid)
    peak_allocated=max(peak_allocated,max(r['gpu_peak_allocated_bytes'] for r in valid))
    technical_attempts+=len(rows)-len(valid)
    eligibility=json.loads(Path(selected['eligibility_file']).read_text())
    pool=next(r for r in inventory if (r['suite'],r['task_id'])==(suite,task_id))
    task={'suite':suite,'task_id':task_id,'state_ids':list(range(50)),
        'regions_file':selected['regions_file'],'regions_sha256':file_sha(selected['regions_file']),
        'geometry_audit_file':eligibility['audit_file'],'geometry_audit_sha256':file_sha(eligibility['audit_file']),
        'native_pool_sha256':pool['pool_sha256'],
        'screening_results_file':selected['results_file'],'screening_results_sha256':file_sha(selected['results_file']),
        'pilot_results_file':str(results),'pilot_results_sha256':file_sha(results),
        'pilot_verification_file':str(verification),'pilot_verification_sha256':file_sha(verification),
        'formal_output_dir':f'artifacts/formal_v1/{slug}','formal_config_file':str(config_dir/f'{slug}.yaml')}
    tasks.append(task)
mean_seconds=sum(durations)/len(durations)
protocol_path=Path('configs/formal50_protocol.json')
previous=json.loads(protocol_path.read_text()) if protocol_path.exists() else None
execution_commit=(previous['execution_git_commit'] if previous and previous['execution_code_hash']==execution_code_hash()
                  else subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
protocol={'version':'formal50-proposal-v1','sample_size_per_task':50,'planned_rollouts':1400,
    'state_pool_type':'native','conditions':CONDITIONS,'tasks':tasks,
    'checkpoint_revision':source['checkpoint']['revision'],
    'vision_language_base_revision':source['vision_language_base']['revision'],
    'execution_code_hash':execution_code_hash(),
    'execution_git_commit':execution_commit,
    'dependency_versions':{p:metadata.version(p) for p in ['torch','torchvision','transformers','lerobot','hf-libero','mujoco','robosuite','numpy']},
    'selection_file':str(selection_file),'selection_sha256':file_sha(selection_file),
    'state_inventory_file':str(inventory_file),'state_inventory_sha256':file_sha(inventory_file),
    'inference':INFERENCE,'analysis':ANALYSIS,'record_video':True,
    'approval_file':'.runtime/formal_approval.json',
    'development_data_use':'screening and pilot outcomes excluded from formal statistics',
    'native_pool_overlap':'formal state IDs 0-49 overlap screening IDs 0-19 and pilot ID 20; each state counted once in formal data',
    'expanded_100_state_pool_validated':False,
    'pilot_performance':{'n_real_rollouts':len(durations),'mean_rollout_seconds':mean_seconds,
        'gpu_peak_allocated_bytes':peak_allocated,'infrastructure_attempts':technical_attempts,
        'estimated_formal_hours_with_15pct_overhead':1400*mean_seconds*1.15/3600,
        'estimate_limitation':'one held-out state per task; excludes model startup and may underestimate failures/timeouts'}}
protocol['protocol_sha256']=fingerprint(protocol)
protocol_path=Path('configs/formal50_protocol.json')
if protocol_path.exists() and json.loads(protocol_path.read_text())!=protocol:
    raise RuntimeError('Existing frozen proposal differs; create a new version rather than overwriting')
protocol_path.write_text(json.dumps(protocol,indent=2,ensure_ascii=False))
for task in tasks:
    config={'study_phase':'formal','checkpoint':'HuggingFaceVLA/smolvla_libero',
        'suite':task['suite'],'task_id':task['task_id'],'state_ids':task['state_ids'],
        'conditions':CONDITIONS,'seed':42,'order_seed':20261115,'horizon':280,'record_video':True,
        'regions_file':task['regions_file'],'output_dir':task['formal_output_dir'],
        'max_rollouts':350,'config_version':'formal50-proposal-v1',
        'protocol_file':str(protocol_path),'protocol_sha256':protocol['protocol_sha256']}
    Path(task['formal_config_file']).write_text(yaml.safe_dump(config,sort_keys=False))
print(json.dumps({'protocol_file':str(protocol_path),'protocol_sha256':protocol['protocol_sha256'],
    'formal_rollouts_started':False,'approval_file_created':False,'pilot_performance':protocol['pilot_performance']},indent=2))
