"""Freeze a fresh 100-state (or 50-state subset) proposal without approving/running it."""
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import yaml
from scripts.formal_protocol import ANALYSIS,INFERENCE,CONDITIONS,file_sha,execution_code_hash,fingerprint
from src.runner.state_pool import load_state_pool

parser=argparse.ArgumentParser();parser.add_argument('--sample-size',type=int,choices=[50,100],default=100)
args=parser.parse_args();n=args.sample_size
selection_file=Path('artifacts/screening_v1/selection.json');selection=json.loads(selection_file.read_text())
assert selection['four_tasks_selected']
inventory_file=Path('artifacts/state_inventory.json');inventory=json.loads(inventory_file.read_text())
source=json.loads(Path('configs/source_versions.json').read_text())
config_dir=Path(f'configs/formal_fresh{n}');config_dir.mkdir(parents=True,exist_ok=True)
protocol_path=Path(f'configs/formal_fresh{n}_protocol.json')
previous=json.loads(protocol_path.read_text()) if protocol_path.exists() else None
current_hash=execution_code_hash()
commit=previous['execution_git_commit'] if previous and previous['execution_code_hash']==current_hash else subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
tasks=[];durations=[];peak=0;technical=0;generation_exclusions=0
for task in selection['selected']:
    suite,task_id=task['suite'],task['task_id'];slug=f'{suite}_{task_id}'
    pool_root=Path('artifacts/generated_states_v1')/slug;pool=load_state_pool(pool_root/'states.npz')
    assert len(pool.states)==101 and pool.manifest['pilot_state_id']==100
    pilot=Path('artifacts/pilot_fresh_v1')/slug
    verification_file=pilot/'verification.json';verification=json.loads(verification_file.read_text())
    assert verification['valid_rollouts']==7 and all(b['seven_complete'] and b['full_state_bundle_verified'] and b['state_pool_replay_verified'] and b['policy_rng_paired_verified'] for b in verification['blocks'])
    results=pilot/'results.jsonl';rows=[json.loads(s) for s in results.read_text().splitlines()]
    valid=[r for r in rows if r['status']!='infrastructure_failure']
    assert len(valid)==7 and all(r['state_id']==100 and r['code_hash']==current_hash and r['initial_state_pool_sha256']==pool.sha256 for r in valid)
    durations.extend(r['elapsed_time'] for r in valid);peak=max(peak,max(r['gpu_peak_allocated_bytes'] for r in valid));technical+=len(rows)-len(valid)
    generation_exclusions+=len(pool.manifest['exclusions'])
    eligibility=json.loads(Path(task['eligibility_file']).read_text())
    native=next(r for r in inventory if (r['suite'],r['task_id'])==(suite,task_id))
    tasks.append({'suite':suite,'task_id':task_id,'state_ids':list(range(n)),
        'regions_file':task['regions_file'],'regions_sha256':file_sha(task['regions_file']),
        'geometry_audit_file':eligibility['audit_file'],'geometry_audit_sha256':file_sha(eligibility['audit_file']),
        'native_pool_sha256':native['pool_sha256'],
        'screening_results_file':task['results_file'],'screening_results_sha256':file_sha(task['results_file']),
        'state_pool_file':str(pool.path),'state_pool_sha256':pool.sha256,
        'state_pool_manifest_file':str(pool_root/'pool.json'),'state_pool_manifest_sha256':pool.manifest_sha256,
        'pilot_results_file':str(results),'pilot_results_sha256':file_sha(results),
        'pilot_verification_file':str(verification_file),'pilot_verification_sha256':file_sha(verification_file),
        'formal_output_dir':f'artifacts/formal_fresh{n}_v1/{slug}',
        'formal_config_file':str(config_dir/f'{slug}.yaml')})
mean=sum(durations)/len(durations);total=4*7*n
protocol={'version':f'formal-fresh{n}-proposal-v1-20261116','sample_size_per_task':n,'planned_rollouts':total,
    'state_pool_type':'generated_seeded_reset','environment_reset_seed_mode':'per_state_bundle','conditions':CONDITIONS,'tasks':tasks,
    'checkpoint_revision':source['checkpoint']['revision'],'vision_language_base_revision':source['vision_language_base']['revision'],
    'execution_code_hash':current_hash,'execution_git_commit':commit,
    'dependency_versions':{p:metadata.version(p) for p in ['torch','torchvision','transformers','lerobot','hf-libero','mujoco','robosuite','numpy']},
    'selection_file':str(selection_file),'selection_sha256':file_sha(selection_file),
    'state_inventory_file':str(inventory_file),'state_inventory_sha256':file_sha(inventory_file),
    'inference':INFERENCE,'analysis':ANALYSIS,'analysis_output_dir':f'artifacts/formal_fresh{n}_v1/analysis','record_video':True,'approval_file':'.runtime/formal_approval.json',
    'development_data_use':'no screening/Pilot outcomes in formal analysis; formal object layouts distinct from all 50 native development layouts',
    'native_pool_overlap':False,'pilot_pool_overlap':False,'pilot_state_id':100,
    'initialization_scope':'fresh seeded LIBERO resets with exact seed+dynamic+fixture replay; not the original fixed benchmark state list',
    'state_selection_rule':'prespecified seeds; reject only initial-goal completion, near-duplicate object position layouts or illegal mask geometry; no policy outcomes used',
    'generation_geometry_exclusions':generation_exclusions,
    'pilot_performance':{'n_real_rollouts':len(durations),'mean_rollout_seconds':mean,'gpu_peak_allocated_bytes':peak,
        'infrastructure_attempts':technical,'estimated_formal_hours_with_15pct_overhead':total*mean*1.15/3600,
        'estimate_limitation':'one reserved state per task; timing and success are not population estimates'}}
protocol['protocol_sha256']=fingerprint(protocol)
if previous and previous!=protocol:raise ValueError('Existing frozen proposal differs; create a new version')
protocol_path.write_text(json.dumps(protocol,indent=2,ensure_ascii=False))
for task in tasks:
    config={'study_phase':'formal','checkpoint':'HuggingFaceVLA/smolvla_libero','suite':task['suite'],'task_id':task['task_id'],
        'state_ids':task['state_ids'],'conditions':CONDITIONS,'seed':42,'order_seed':20261115,'horizon':280,'record_video':True,
        'regions_file':task['regions_file'],'initial_state_pool':task['state_pool_file'],'initial_state_pool_manifest':task['state_pool_manifest_file'],
        'output_dir':task['formal_output_dir'],'max_rollouts':n*7,'config_version':protocol['version'],
        'protocol_file':str(protocol_path),'protocol_sha256':protocol['protocol_sha256']}
    Path(task['formal_config_file']).write_text(yaml.safe_dump(config,sort_keys=False))
print(json.dumps({'protocol_file':str(protocol_path),'protocol_sha256':protocol['protocol_sha256'],'planned_rollouts':total,
    'formal_rollouts_started':False,'approval_created':False,'pilot_performance':protocol['pilot_performance']},indent=2))
