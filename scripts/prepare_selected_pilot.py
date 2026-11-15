"""Prepare a prespecified held-out pilot after all four tasks have been selected."""
import json
from pathlib import Path
import yaml

selection=json.loads(Path('artifacts/screening_v1/selection.json').read_text())
if not selection['four_tasks_selected']: raise RuntimeError('Complete ordered screening first')
protocol=yaml.safe_load(Path('configs/screening/protocol.yaml').read_text())
state=protocol['post_selection_pilot_state_id']
assert state not in protocol['state_ids']
folder=Path('configs/pilot_selected');folder.mkdir(parents=True,exist_ok=True)
paths=[]
for task in selection['selected']:
    suite,task_id=task['suite'],task['task_id']
    config={'study_phase':'pilot','checkpoint':'HuggingFaceVLA/smolvla_libero',
        'suite':suite,'task_id':task_id,'state_ids':[state],
        'conditions':['Baseline','T1','T2','T3','B1','B2','B3'],
        'seed':protocol['seed'],'order_seed':protocol['order_seed'],'horizon':280,
        'record_video':True,'regions_file':task['regions_file'],
        'output_dir':f'artifacts/pilot_selected_v1/{suite}_{task_id}',
        'config_version':'held-out-pilot-state20-v1'}
    path=folder/f'{suite}_{task_id}.yaml'
    text=yaml.safe_dump(config,sort_keys=False)
    if path.exists() and path.read_text()!=text: raise RuntimeError('Refuse to change an existing pilot config')
    path.write_text(text);paths.append(str(path))
print('\n'.join(paths))
