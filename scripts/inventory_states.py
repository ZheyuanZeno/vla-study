"""Read official state pools; counts are not evidence of statistical independence."""
import hashlib
import json
from pathlib import Path
import numpy as np
from lerobot.envs.libero import _get_suite, get_task_init_states

rows=[]
for suite_name in ['libero_spatial','libero_object']:
    suite=_get_suite(suite_name)
    for task_id in range(len(suite.tasks)):
        task=suite.get_task(task_id)
        states=np.asarray(get_task_init_states(suite,task_id))
        hashes=[hashlib.sha256(s.tobytes()).hexdigest() for s in states]
        rows.append({'suite':suite_name,'task_id':task_id,'task':task.name,'instruction':task.language,
            'available_states':len(states),'unique_exact_states':len(set(hashes)),
            'shape':list(states.shape),'state_hashes':hashes,'pool_sha256':hashlib.sha256(states.tobytes()).hexdigest()})
output=Path('artifacts/state_inventory.json'); output.write_text(json.dumps(rows,indent=2))
print(json.dumps([{k:v for k,v in r.items() if k not in ['state_hashes','instruction','shape','pool_sha256']} for r in rows],indent=2))
