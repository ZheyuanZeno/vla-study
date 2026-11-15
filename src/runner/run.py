"""Run small paired real experiments using pinned LeRobot rollout and processors."""
import os
# Freeze locally cached VLM/tokenizer references before importing Hub clients.
os.environ['HF_HUB_OFFLINE']='1'
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import socket
import subprocess
import time
import traceback

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from PIL import Image
import torch
import yaml
from huggingface_hub import snapshot_download
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv as LiberoConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.scripts.lerobot_eval import rollout
from lerobot.utils.random_utils import set_seed
from src.masking.policy_hook import image_intervention
from src.runner.regions import prepare_regions
from src.runner.storage import atomic_json, rebuild_index, recover_interrupted

CONDITIONS=['Baseline','T1','T2','T3','B1','B2','B3']


def execute_episode(vec,policy,processors,config,regions,state_id,condition,output):
    output.mkdir(parents=True,exist_ok=True)
    env=vec.envs[0]
    original_reset,original_step=env.reset,env.step
    context={'ready':False,'square':None,'geometry':None}
    steps=[]; frames=[]
    state=np.asarray(env._init_states[state_id])
    def reset(*args,**kwargs):
        obs,info=original_reset(*args,**kwargs)
        if not context['ready']:
            if regions is not None:
                context['square'],context['geometry']=prepare_regions(env,obs,regions,condition,output)
            elif condition!='Baseline':
                raise ValueError('Occlusion requires reviewed regions')
            context['ready']=True
            np.save(output/'initial_state.npy',state)
            settled=np.asarray(env._env.get_sim_state())
            np.save(output/'settled_state.npy',settled)
            context['settled_state_sha256']=hashlib.sha256(settled.tobytes()).hexdigest()
            for key,rgb in obs['pixels'].items(): Image.fromarray(rgb).save(output/f'raw_{key}.png')
            if config['record_video']: frames.append(np.flip(obs['pixels']['image'],axis=(0,1)).copy())
        return obs,info
    def step(action):
        assert action.shape==(7,) and np.isfinite(action).all()
        obs,reward,terminated,truncated,info=original_step(action)
        steps.append({'env_step':len(steps)+1,'action':action.tolist(),
            'eef_pos':obs['robot_state']['eef']['pos'].tolist(),'success':bool(info['is_success']),
            'reward':float(reward),'terminated':bool(terminated),'truncated':bool(truncated)})
        with (output/'steps.jsonl').open('a') as stream: stream.write(json.dumps(steps[-1])+'\n')
        if config['record_video']: frames.append(np.flip(obs['pixels']['image'],axis=(0,1)).copy())
        return obs,reward,terminated,truncated,info
    env.reset,env.step=reset,step
    env.init_state_id=state_id
    for processor in processors: processor.reset()
    policy.reset()
    set_seed(config['seed'])
    torch.cuda.reset_peak_memory_stats()
    started=time.monotonic()
    result={'success':None,'error_type':None,'status':'infrastructure_failure','mask_geometry':None,
            'target_coverage':None,'background_anchor':None if regions is None else regions['background_anchor'],
            'state_sha256':hashlib.sha256(state.tobytes()).hexdigest()}
    try:
        with image_intervention(policy,lambda:context['square'],output) as calls:
            with torch.no_grad():
                data=rollout(vec,policy,*processors[:2],*processors[2:],seeds=[config['seed']])
            result['inference_calls']=calls()
        assert len(steps)>0 and result['inference_calls']==len(steps)
        success=bool(data['success'].any().item())
        assert success==any(s['success'] for s in steps), 'Success predicate mismatch'
        if success: status='success'
        elif any(s['terminated'] for s in steps): status='valid_task_failure'
        else:
            assert len(steps)==env._max_episode_steps
            status='task_timeout'
        result.update(success=success,status=status)
    except Exception as exc:
        result.update(error_type=type(exc).__name__,traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        env.reset,env.step=original_reset,original_step
        result.update(env_steps=len(steps),elapsed_time=time.monotonic()-started,
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        result['settled_state_sha256']=context.get('settled_state_sha256')
        result['mask_geometry']=context['geometry']
        result['target_coverage']=None if context['geometry'] is None else context['geometry']['target_coverage']
        (output/'steps.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in steps))
        if frames:
            try: imageio.mimsave(output/'rollout.mp4',frames,fps=30,macro_block_size=1)
            except Exception as exc:
                result.update(success=None,status='infrastructure_failure',error_type='VideoWriteError',traceback=str(exc))
    return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',required=True)
    parser.add_argument('--stop-after',type=int,default=None,help='Stop cleanly after this many new episodes; resume uses identical order')
    args=parser.parse_args(); config=yaml.safe_load(Path(args.config).read_text())
    if config['study_phase'] not in ['mask_smoke','pilot','screening']:
        raise ValueError('Formal collection is disabled until explicit protocol approval')
    if not config['conditions'] or len(set(config['conditions']))!=len(config['conditions']) or not set(config['conditions'])<=set(CONDITIONS):
        raise ValueError('Invalid or duplicate conditions')
    if len(set(config['state_ids']))!=len(config['state_ids']) or not config['state_ids']:
        raise ValueError('Need unique explicit state IDs')
    count=len(config['state_ids'])*len(config['conditions'])
    if count>config.get('max_rollouts',7) or count>=1400:
        raise ValueError('Configured rollout count exceeds explicit small-run limit')
    if count>7 and config['study_phase']=='pilot':
        from src.runner.verify import verify
        verify(Path(config['pilot_gate_dir']))
    if args.stop_after is not None and args.stop_after<1: raise ValueError('stop-after must be positive')
    phase1=json.loads(Path('artifacts/environment.json').read_text())
    assert phase1['status']=='passed' and phase1['simulator']['real_reset_and_step']
    regions=json.loads(Path(config['regions_file']).read_text()) if config.get('regions_file') else None
    if regions is not None: assert regions['suite']==config['suite'] and regions['task_id']==config['task_id']
    elif config['conditions']!=['Baseline']: raise ValueError('Need regions for occlusion')
    digest=hashlib.sha256(json.dumps({'config':config,'regions':regions},sort_keys=True).encode()).hexdigest()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    root=Path(config['output_dir']); root.mkdir(parents=True,exist_ok=True)
    sources=sorted(Path('src').rglob('*.py'))+[Path('requirements-lock.txt'),Path('configs/source_versions.json')]
    code_hash=hashlib.sha256(b''.join(str(p).encode()+b'\0'+p.read_bytes() for p in sources)).hexdigest()
    manifest={'config_hash':digest,'code_hash':code_hash,'git_commit':commit,'config':config,'regions':regions}
    if (root/'manifest.json').exists():
        previous=json.loads((root/'manifest.json').read_text())
        if previous['config_hash']!=digest or previous.get('code_hash')!=code_hash:
            raise ValueError('Resume config/code differs from the frozen run; use a new output directory')
    else: atomic_json(root/'manifest.json',manifest)
    recover_interrupted(root)
    rows=rebuild_index(root)
    done={(r['state_id'],r['condition']) for r in rows if r['status']!='infrastructure_failure'}
    order=[(s,c) for s in config['state_ids'] for c in config['conditions']]
    random.Random(config['order_seed']).shuffle(order)
    order_record={'order_seed':config['order_seed'],'full_order':order}
    if (root/'execution_order.json').exists():
        saved=json.loads((root/'execution_order.json').read_text())
        if saved!={'order_seed':config['order_seed'],'full_order':[list(pair) for pair in order]}:
            raise ValueError('Saved condition order does not match configuration')
    else: atomic_json(root/'execution_order.json',order_record)
    pending=[pair for pair in order if pair not in done]
    if not pending:
        print('All configured episodes already completed; no inference run.'); return
    checkpoint=json.loads(Path('configs/source_versions.json').read_text())
    assert config['checkpoint']==checkpoint['checkpoint']['repo_id']
    revision=checkpoint['checkpoint']['revision']
    torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=True
    snapshot=snapshot_download(config['checkpoint'],revision=revision,local_files_only=True)
    base=Path(os.environ['HF_HOME'])/'hub/models--HuggingFaceTB--SmolVLM2-500M-Instruct/refs/main'
    assert base.read_text().strip()==checkpoint['vision_language_base']['revision']
    cfg=PreTrainedConfig.from_pretrained(snapshot); cfg.device='cuda'; cfg.pretrained_path=Path(snapshot)
    env_cfg=LiberoConfig(task=config['suite'],task_ids=[config['task_id']],episode_length=config['horizon'])
    envs=make_env(env_cfg,n_envs=1,use_async_envs=False)
    vec=envs[config['suite']][config['task_id']]
    try:
        policy=SmolVLAPolicy.from_pretrained(snapshot,config=cfg,strict=True); policy.eval()
        assert next(policy.parameters()).device.type=='cuda'
        pre,post=make_pre_post_processors(cfg,pretrained_path=snapshot,preprocessor_overrides={
            'device_processor':{'device':'cuda'},'rename_observations_processor':{'rename_map':{}}})
        env_pre,env_post=make_env_pre_post_processors(env_cfg,cfg)
        processors=(env_pre,env_post,pre,post)
        runtime_flags={'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
            'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
            'seed_scope':'reset immediately before each rollout','use_amp':cfg.use_amp,
            'parameter_dtypes':sorted({str(p.dtype) for p in policy.parameters()})}
        atomic_json(root/'inference_settings.json',{'policy':asdict(cfg),'runtime':runtime_flags})
        new_count=0
        for state_id,condition in pending:
            if not 0<=state_id<len(vec.envs[0]._init_states): raise ValueError('State index would wrap; prohibited')
            episode=root/'episodes'/f'state{state_id:03d}_{condition}'
            attempt=episode/f'attempt{len(list(episode.glob("attempt*")))+1:03d}'
            metadata={'suite':config['suite'],'task_id':config['task_id'],'state_id':state_id,'condition':condition,
                'seed':config['seed'],'order_seed':config['order_seed'],'checkpoint_revision':revision,
                'git_commit':commit,'code_hash':code_hash,'config_hash':digest,'study_phase':config['study_phase'],
                'data_origin':'real_rollout','attempt_dir':str(attempt),'runtime_flags':runtime_flags,
                'machine_id':config.get('machine_id',socket.gethostname()),'config_version':config.get('config_version','v1')}
            atomic_json(attempt/'attempt_manifest.json',metadata)
            result=execute_episode(vec,policy,processors,config,regions,state_id,condition,attempt)
            result.update(metadata)
            result.update(task_id=config['task_id'],suite=config['suite'],state_id=state_id,condition=condition,
                seed=config['seed'],order_seed=config['order_seed'],machine_id=config.get('machine_id',socket.gethostname()),
                checkpoint_revision=revision,git_commit=commit,config_hash=digest,study_phase=config['study_phase'],
                task_instruction=vec.envs[0].task_description,attempt_dir=str(attempt),
                control_mode=vec.envs[0].control_mode,horizon=config['horizon'],action_execution_horizon=cfg.n_action_steps)
            atomic_json(attempt/'result.json',result)
            rows=rebuild_index(root)
            print(json.dumps(result,ensure_ascii=False),flush=True)
            if result['status']=='infrastructure_failure': raise RuntimeError(f'Technical failure; see {attempt}')
            new_count+=1
            if args.stop_after is not None and new_count>=args.stop_after: break
        complete={s: sorted(r['condition'] for r in rows if r['state_id']==s and r['status']!='infrastructure_failure') for s in config['state_ids']}
        atomic_json(root/'completeness.json',{'conditions':complete,'seven_condition_blocks':{s:set(cs)==set(CONDITIONS) for s,cs in complete.items()}})
    finally:
        vec.close()


if __name__=='__main__':
    try:
        main()
    except BaseException as exc:
        error_dir=Path('artifacts/runner_errors'); error_dir.mkdir(parents=True,exist_ok=True)
        atomic_json(error_dir/f'{time.time_ns()}.json',{'status':'infrastructure_failure','success':None,
            'error_type':type(exc).__name__,'traceback':traceback.format_exc(),'argv':__import__('sys').argv})
        raise
