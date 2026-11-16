"""Summarize measured pool/replay evidence; no policy or simulator run."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.runner.state_pool import load_state_pool,nearest_position_distance
from src.runner.storage import atomic_json

selected=json.loads(Path('configs/screening/selected_tasks.json').read_text())['selected_tasks']
rows=[]
for task in selected:
    slug=f"{task['suite']}_{task['task_id']}";root=Path('artifacts/generated_states_v1')/slug
    pool=load_state_pool(root/'states.npz')
    refs=list(np.load(root/'native_object_positions.npy'));distances=[]
    for positions in pool.object_positions:
        distance=nearest_position_distance(positions,refs);assert distance>=pool.manifest['minimum_object_position_linf_m']
        distances.append(distance);refs.append(positions)
    records=pool.manifest['records']
    rows.append({'suite':task['suite'],'task_id':task['task_id'],'n_validated':len(pool.states),
        'n_formal_reserved':len(pool.manifest['formal_state_ids']),'pilot_state_id':pool.manifest['pilot_state_id'],
        'minimum_nearest_object_position_linf_m':min(distances),
        'max_dynamic_replay_error':max(r['state_replay_max_abs_delta'] for r in records),
        'max_static_layout_replay_error':max(r['static_layout_replay_max_abs_delta'] for r in records),
        'max_object_replay_error':max(r['object_replay_max_abs_delta'] for r in records),
        'excluded_candidates':len(pool.manifest['exclusions']),'technical_failures':len(pool.manifest['infrastructure_failures']),
        'pool_sha256':pool.sha256,'pool_manifest_sha256':pool.manifest_sha256})
report={'pool_type':'real seeded LIBERO resets','tasks':rows,'total_qualified_states':sum(r['n_validated'] for r in rows),
        'formal_states_held_out_from_policy_pilot':sum(r['n_formal_reserved'] for r in rows),
        'reserved_pilot_states':len(rows),'formal_rollouts_started':False}
atomic_json('artifacts/generated_states_v1/validation_summary.json',report)
print(json.dumps(report,indent=2))
