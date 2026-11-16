"""Generate real seeded LIBERO layouts, replay and geometry audit them; no VLA inference."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import yaml
from lerobot.envs.libero import LiberoEnv,_get_suite,get_task_init_states,get_libero_dummy_action
from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad
import torch
from src.runner.state_pool import static_layout,object_geometry,model_signature,nearest_position_distance,sha256_file,load_state_pool
from src.runner.regions import prepare_regions
from src.runner.storage import atomic_json


def save_npz(path,**arrays):
    temporary=path.with_suffix('.tmp.npz')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def settle(env,n):
    observation=None
    for _ in range(n):observation,_,_,_=env._env.step(get_libero_dummy_action())
    return env._format_raw_obs(observation)


def saved_rgb(observation,path):
    raw=np.flip(observation['pixels']['image'],axis=(0,1)).copy()
    x=torch.as_tensor(raw).permute(2,0,1)[None].float()/255
    rgb=resize_with_pad(x,512,512,pad_value=0)[0].permute(1,2,0).numpy()
    Image.fromarray(np.rint(np.clip(rgb,0,1)*255).astype(np.uint8)).save(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--task-id',type=int,required=True)
    parser.add_argument('--suite',default='libero_spatial');parser.add_argument('--plan',default='configs/state_generation/fresh101_v1.yaml')
    parser.add_argument('--stop-after',type=int);args=parser.parse_args()
    plan=yaml.safe_load(Path(args.plan).read_text())
    selected=json.loads(Path(plan['selected_tasks_file']).read_text())['selected_tasks']
    task=next(t for t in selected if (t['suite'],t['task_id'])==(args.suite,args.task_id))
    regions=json.loads(Path(task['regions_file']).read_text())
    root=Path(plan['output_root'])/f'{args.suite}_{args.task_id}';root.mkdir(parents=True,exist_ok=True)
    source_paths=[Path(__file__),Path('src/runner/state_pool.py'),Path('src/runner/regions.py'),
                  Path('src/masking/geometry.py'),Path('src/masking/segmentation.py')]
    manifest={'plan':plan,'suite':args.suite,'task_id':args.task_id,'regions_sha256':sha256_file(task['regions_file']),
        'generator_code_sha256':hashlib.sha256(b''.join(p.read_bytes() for p in source_paths)).hexdigest(),
        'dependencies':{p:metadata.version(p) for p in ['hf-libero','robosuite','mujoco','numpy']}}
    if (root/'generation_manifest.json').exists():
        if json.loads((root/'generation_manifest.json').read_text())!=manifest:raise ValueError('Generation plan/code changed; use a new version')
    else:atomic_json(root/'generation_manifest.json',manifest)
    if (root/'pool.json').exists():
        pool=load_state_pool(root/'states.npz');print(f'Already validated {len(pool.states)} states; no generation repeated');return
    suite=_get_suite(args.suite)
    env=LiberoEnv(suite,args.task_id,args.suite,obs_type='pixels_agent_pos',init_states=False)
    try:
        signature=model_signature(env)
        native=np.asarray(get_task_init_states(suite,args.task_id))
        native_path=root/'native_object_positions.npy'
        if native_path.exists():native_positions=np.load(native_path)
        else:
            collected=[]
            for state in native:
                env._env.seed(plan['policy_seed']);env._env.reset();env._env.set_init_state(state)
                settle(env,plan['num_steps_wait']);_,positions,_=object_geometry(env);collected.append(positions)
            native_positions=np.stack(collected);np.save(native_path,native_positions)
        accepted=[];excluded=[];resolved={};technical=[]
        for directory in sorted(root.glob('candidates/*/attempt*')):
            record_path=directory/'record.json'
            if not record_path.exists():
                prior=json.loads((directory/'attempt_manifest.json').read_text())
                prior.update(status='infrastructure_failure',error_type='InterruptedGenerationAttempt')
                atomic_json(record_path,prior)
            record=json.loads(record_path.read_text())
            if record['status']=='validated':accepted.append(record);resolved[record['candidate_index']]=record
            elif record['status']=='excluded_geometry_or_initial_state':excluded.append(record);resolved[record['candidate_index']]=record
            else:technical.append(record)
        accepted.sort(key=lambda r:r['state_id'])
        if [r['state_id'] for r in accepted]!=list(range(len(accepted))):raise ValueError('Noncontiguous accepted IDs')
        positions_list=[]
        for record in accepted:
            with np.load(Path(record['attempt_dir'])/'bundle.npz',allow_pickle=False) as bundle:positions_list.append(bundle['object_positions'].copy())
        new_count=0
        for candidate in range(plan['max_candidates_per_task']):
            if len(accepted)>=plan['accepted_states_per_task']:break
            if candidate in resolved:continue
            seed=plan['seed_base']+args.task_id*plan['seed_stride_per_task']+candidate
            parent=root/'candidates'/f'seed{seed}';directory=parent/f'attempt{len(list(parent.glob("attempt*")))+1:03d}'
            directory.mkdir(parents=True)
            record={'candidate_index':candidate,'environment_seed':seed,'suite':args.suite,'task_id':args.task_id,'attempt_dir':str(directory)}
            atomic_json(directory/'attempt_manifest.json',record)
            try:
                env._env.seed(seed);env._env.reset()
                state=env._env.get_sim_state().copy();layout=static_layout(env)
                observation=settle(env,plan['num_steps_wait'])
                settled=env._env.get_sim_state().copy();names,positions,matrices=object_geometry(env)
                if not np.isfinite(state).all() or not np.isfinite(settled).all():raise RuntimeError('Nonfinite physics state')
                save_npz(directory/'bundle.npz',state=state,settled_state=settled,static_layout=layout,
                    object_positions=positions,object_matrices=matrices,object_names=np.array(names))
                saved_rgb(observation,directory/'original.png')
                reasons=[]
                if env._env.check_success():reasons.append('goal_already_satisfied_after_settling')
                distance=nearest_position_distance(positions,list(native_positions)+positions_list)
                if distance is not None and distance<plan['minimum_object_position_linf_m']:reasons.append('near_duplicate_movable_object_positions')
                record['nearest_object_position_linf_m']=distance
                env._env.seed(seed);env._env.reset();env._env.set_init_state(state)
                restored_layout=static_layout(env)
                observation=settle(env,plan['num_steps_wait'])
                restored_state=env._env.get_sim_state().copy();_,restored_positions,restored_matrices=object_geometry(env)
                delta=float(np.max(np.abs(settled-restored_state)));layout_delta=float(np.max(np.abs(layout-restored_layout)))
                position_delta=float(np.max(np.abs(positions-restored_positions)))
                record.update(state_replay_max_abs_delta=delta,static_layout_replay_max_abs_delta=layout_delta,
                    object_replay_max_abs_delta=position_delta)
                if delta>plan['state_replay_atol'] or layout_delta>plan['static_layout_replay_atol'] or position_delta>plan['state_replay_atol']:
                    raise RuntimeError('Full state bundle does not reproduce exactly with its reset seed')
                if env._env.check_success():reasons.append('goal_satisfied_on_reload')
                try:prepare_regions(env,observation,regions,'Baseline',directory)
                except ValueError as exc:reasons.append('mask_geometry_ineligible: '+str(exc))
                if reasons:
                    record.update(status='excluded_geometry_or_initial_state',reasons=reasons);excluded.append(record)
                else:
                    record.update(status='validated',state_id=len(accepted),state_sha256=hashlib.sha256(state.tobytes()).hexdigest(),
                        static_layout_sha256=hashlib.sha256(layout.tobytes()).hexdigest(),mask_pairs=json.loads((directory/'mask_pairs.json').read_text()))
                    accepted.append(record);positions_list.append(positions);new_count+=1
                atomic_json(directory/'record.json',record)
                atomic_json(root/'progress.json',{'accepted':len(accepted),'target':plan['accepted_states_per_task'],
                    'excluded':len(excluded),'infrastructure_failures':len(technical),'last_candidate':candidate})
                print(json.dumps({k:v for k,v in record.items() if k!='mask_pairs'}),flush=True)
            except Exception as exc:
                record.update(status='infrastructure_failure',error_type=type(exc).__name__,traceback=traceback.format_exc())
                atomic_json(directory/'record.json',record);raise
            if args.stop_after is not None and new_count>=args.stop_after:break
        if len(accepted)!=plan['accepted_states_per_task']:
            if args.stop_after is not None:
                print('Stopped at requested small generation checkpoint; pool is NOT validated or usable yet');return
            raise RuntimeError('Insufficient eligible unique states in the prespecified candidate budget')
        bundles=[]
        for record in accepted:
            with np.load(Path(record['attempt_dir'])/'bundle.npz',allow_pickle=False) as data:bundles.append({k:data[k].copy() for k in data.files})
        pool_path=root/'states.npz'
        save_npz(pool_path,states=np.stack([b['state'] for b in bundles]),reset_seeds=np.array([r['environment_seed'] for r in accepted],dtype=np.int64),
            settled_states=np.stack([b['settled_state'] for b in bundles]),static_layouts=np.stack([b['static_layout'] for b in bundles]),
            object_positions=np.stack([b['object_positions'] for b in bundles]),object_matrices=np.stack([b['object_matrices'] for b in bundles]),object_names=bundles[0]['object_names'])
        final={'status':'validated','version':plan['version'],'suite':args.suite,'task_id':args.task_id,'n_states':len(accepted),
            'formal_state_ids':list(range(plan['formal_state_ids'][0],plan['formal_state_ids'][1]+1)),
            'pilot_state_id':plan['pilot_state_id'],'policy_seed':plan['policy_seed'],'num_steps_wait':plan['num_steps_wait'],
            'minimum_object_position_linf_m':plan['minimum_object_position_linf_m'],
            'state_replay_atol':plan['state_replay_atol'],'static_layout_replay_atol':plan['static_layout_replay_atol'],
            'model_signature':signature,'pool_sha256':sha256_file(pool_path),'generation_manifest':manifest,
            'native_pool_sha256':hashlib.sha256(native.tobytes()).hexdigest(),'native_positions_sha256':sha256_file(native_path),
            'records':accepted,'exclusions':excluded,'infrastructure_failures':technical,
            'policy_inference_performed':False,'formal_rollouts_started':False}
        atomic_json(root/'pool.json',final)
        load_state_pool(pool_path)
        print(f'VALIDATED: {len(accepted)} real distinct bundles; formal IDs 0-99; Pilot ID 100; no model rollout',flush=True)
    finally:env.close()


if __name__=='__main__':main()
