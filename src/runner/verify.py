"""Check complete real paired blocks and saved policy inputs without inference."""
import argparse
import json
from pathlib import Path
import numpy as np
from src.runner.storage import atomic_json
from src.runner.state_pool import load_state_pool

CONDITIONS={'Baseline','T1','T2','T3','B1','B2','B3'}


def verify(root,require_seven=True):
    rows=[json.loads(s) for s in (root/'results.jsonl').read_text().splitlines() if s.strip()]
    completed=[r for r in rows if r['status']!='infrastructure_failure']
    keys=[(r['suite'],r['task_id'],r['state_id'],r['condition']) for r in completed]
    if len(keys)!=len(set(keys)): raise ValueError('Duplicate valid results')
    groups={key[:3] for key in keys}; checks=[]; pools={}
    for group in sorted(groups):
        group_rows=[r for r in completed if (r['suite'],r['task_id'],r['state_id'])==group]
        by_condition={r['condition']:r for r in group_rows}
        if require_seven and set(by_condition)!=CONDITIONS: raise ValueError(f'Incomplete seven-condition block {group}')
        bundle_ids={r.get('initial_state_bundle_id',r['state_sha256']) for r in group_rows}
        if len(bundle_ids)!=1: raise ValueError('Conditions use different full state bundles')
        env_seeds={r.get('environment_reset_seed',r['seed']) for r in group_rows}
        if len(env_seeds)!=1: raise ValueError('Environment reset seeds differ')
        rng_hashes={r.get('policy_cuda_rng_sha256') for r in group_rows}
        if len(rng_hashes)!=1: raise ValueError('Policy RNG state differs across conditions')
        source_states=[np.load(Path(r['attempt_dir'])/'initial_state.npy') for r in group_rows]
        assert all(np.array_equal(s,source_states[0]) for s in source_states)
        settled_paths=[Path(r['attempt_dir'])/'settled_state.npy' for r in group_rows]
        settled_delta=None
        if all(p.exists() for p in settled_paths):
            states=[np.load(p) for p in settled_paths]
            settled_delta=max(float(np.abs(s-states[0]).max()) for s in states)
            if settled_delta>1e-8: raise ValueError(f'Reset states differ: {settled_delta}')
        layouts=[Path(r['attempt_dir'])/'static_layout.npy' for r in group_rows]
        static_delta=None
        if all(path.exists() for path in layouts):
            snapshots=[np.load(path) for path in layouts]
            static_delta=max(float(np.abs(a-snapshots[0]).max()) for a in snapshots)
            if static_delta>1e-12: raise ValueError('Fixture/model layout differs across conditions')
        generated=any(r.get('initial_state_origin')=='generated_seeded_reset' for r in group_rows)
        if generated and (static_delta is None or None in rng_hashes): raise ValueError('Generated block lacks complete state/RNG evidence')
        for row in group_rows:
            if generated:
                pool_path=row['initial_state_pool_file']
                if pool_path not in pools:pools[pool_path]=load_state_pool(pool_path,row['initial_state_pool_manifest_file'])
                pool=pools[pool_path];state_id=row['state_id']
                assert row['initial_state_pool_sha256']==pool.sha256
                assert row['initial_state_pool_manifest_sha256']==pool.manifest_sha256
                assert row['environment_reset_seed']==int(pool.reset_seeds[state_id])
                assert np.array_equal(np.load(Path(row['attempt_dir'])/'initial_state.npy'),pool.states[state_id])
                assert np.max(np.abs(np.load(Path(row['attempt_dir'])/'settled_state.npy')-pool.settled_states[state_id]))<=pool.manifest['state_replay_atol']
                assert np.max(np.abs(np.load(Path(row['attempt_dir'])/'static_layout.npy')-pool.static_layouts[state_id]))<=pool.manifest['static_layout_replay_atol']
            p=Path(row['attempt_dir']); original=np.load(p/'original_rgb_float.npy'); masked=np.load(p/'masked_rgb_float.npy')
            geometry=row['mask_geometry']
            assert row['env_steps']>0 and row['inference_calls']==row['env_steps']
            if geometry is None:
                assert row['condition']=='Baseline' and np.array_equal(original,masked)
                continue
            x0,y0,x1,y1=geometry['bbox']; side=geometry['side']
            assert x1-x0==y1-y0==side and side**2==geometry['pixel_area']
            outside=np.ones((512,512),dtype=bool); outside[y0:y1,x0:x1]=False
            assert np.array_equal(original[outside],masked[outside])
            assert np.all(masked[y0:y1,x0:x1]==np.float32(128/255))
            target=np.load(p/'target_pixels.npy'); actual=float(target[y0:y1,x0:x1].sum()/target.sum())
            assert actual==row['target_coverage']
            if row['condition'].startswith('T'): assert actual>=[.25,.5,.75][int(row['condition'][1])-1]
            else: assert actual==0
        for level in '123':
            if 'T'+level in by_condition and 'B'+level in by_condition:
                assert by_condition['T'+level]['mask_geometry']['side']==by_condition['B'+level]['mask_geometry']['side']
        checks.append({'suite':group[0],'task_id':group[1],'state_id':group[2],
            'conditions':sorted(by_condition),'seven_complete':set(by_condition)==CONDITIONS,
            'settled_state_max_abs_delta':settled_delta,'static_layout_max_abs_delta':static_delta,
            'full_state_bundle_verified':static_delta is not None and settled_delta is not None,
            'state_pool_replay_verified':generated,'policy_rng_paired_verified':None not in rng_hashes,
            'saved_actual_inputs_verified':True})
    report={'valid_rollouts':len(completed),'infrastructure_attempts':len(rows)-len(completed),'blocks':checks}
    if not completed: raise ValueError('No completed real rollouts')
    atomic_json(root/'verification.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--root',required=True)
    parser.add_argument('--allow-partial',action='store_true'); args=parser.parse_args()
    print(json.dumps(verify(Path(args.root),require_seven=not args.allow_partial),indent=2))


if __name__=='__main__': main()
