"""Use only reserved generated state 100 for the four-task policy Pilot."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import yaml
from src.runner.state_pool import load_state_pool

selected=json.loads(Path('configs/screening/selected_tasks.json').read_text())['selected_tasks']
folder=Path('configs/pilot_fresh');folder.mkdir(parents=True,exist_ok=True)
for task in selected:
    slug=f"{task['suite']}_{task['task_id']}";root=Path('artifacts/generated_states_v1')/slug
    pool=load_state_pool(root/'states.npz')
    assert len(pool.states)==101 and pool.manifest['formal_state_ids']==list(range(100)) and pool.manifest['pilot_state_id']==100
    config={'study_phase':'pilot','checkpoint':'HuggingFaceVLA/smolvla_libero','suite':task['suite'],'task_id':task['task_id'],
        'state_ids':[100],'conditions':['Baseline','T1','T2','T3','B1','B2','B3'],
        'seed':42,'order_seed':20261115,'horizon':280,'record_video':True,
        'regions_file':task['regions_file'],'initial_state_pool':str(pool.path),'initial_state_pool_manifest':str(root/'pool.json'),
        'output_dir':f'artifacts/pilot_fresh_v1/{slug}','config_version':'generated-pool-reserved100-pilot-v1'}
    path=folder/f'{slug}.yaml';text=yaml.safe_dump(config,sort_keys=False)
    if path.exists() and path.read_text()!=text:raise ValueError('Refuse to alter a frozen pilot config')
    path.write_text(text);print(path)
