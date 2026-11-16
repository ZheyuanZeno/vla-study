"""Bind validated pools and prove the entire reset, including static fixtures."""
import hashlib
import json
import numpy as np
from src.runner.state_pool import load_state_pool,validate_pool_for_env,sha256_file,static_layout,object_geometry


def check_pool_role(pool,config):
    manifest=pool.manifest
    if config['seed']!=manifest['policy_seed']:raise ValueError('Policy seed differs from pool protocol')
    ids=config['state_ids']
    if any(not 0<=i<len(pool.states) for i in ids):raise ValueError('State index outside pool')
    if config['study_phase'] in ['pilot','mask_smoke']:
        if ids!=[manifest['pilot_state_id']]:raise ValueError('Fresh formal states cannot be used for pilot performance checks')
    elif config['study_phase']=='formal':
        if not set(ids)<=set(manifest['formal_state_ids']):raise ValueError('Reserved Pilot state is not a formal unit')
    else:raise ValueError('Generated pools are not used to change completed task screening')


def bind_configured_pool(vec,config):
    path=config.get('initial_state_pool')
    if not path:return None
    pool=load_state_pool(path,config.get('initial_state_pool_manifest'))
    check_pool_role(pool,config)
    env=vec.envs[0];validate_pool_for_env(pool,env)
    if env.num_steps_wait!=pool.manifest['num_steps_wait']:raise ValueError('Settling duration differs from state validation')
    if sha256_file(config['regions_file'])!=pool.manifest['generation_manifest']['regions_sha256']:
        raise ValueError('Regions differ from the state-pool geometry audit')
    env._init_states=pool.states
    env.init_states=True
    return pool


def initial_state_evidence(env,pool,state_id,source_state,output):
    layout=static_layout(env);names,positions,matrices=object_geometry(env)
    np.save(output/'static_layout.npy',layout)
    np.save(output/'object_positions.npy',positions)
    np.save(output/'object_matrices.npy',matrices)
    (output/'object_names.json').write_text(json.dumps(names))
    result={'static_layout_sha256':hashlib.sha256(layout.tobytes()).hexdigest(),
            'initial_state_bundle_id':hashlib.sha256(np.asarray(source_state).tobytes()+layout.tobytes()).hexdigest()}
    if pool is not None:
        settled=np.asarray(env._env.get_sim_state())
        deltas={'state':float(np.abs(settled-pool.settled_states[state_id]).max()),
                'static_layout':float(np.abs(layout-pool.static_layouts[state_id]).max()),
                'object_positions':float(np.abs(positions-pool.object_positions[state_id]).max()),
                'object_matrices':float(np.abs(matrices-pool.object_matrices[state_id]).max())}
        tolerance=pool.manifest['state_replay_atol'];static_tol=pool.manifest['static_layout_replay_atol']
        if deltas['state']>tolerance or deltas['static_layout']>static_tol or deltas['object_positions']>tolerance or deltas['object_matrices']>tolerance:
            raise RuntimeError(f'Generated state bundle replay differs: {deltas}')
        if env._env.check_success():raise RuntimeError('Initial state already satisfies the goal')
        result['generated_state_replay_deltas']=deltas
    return result
