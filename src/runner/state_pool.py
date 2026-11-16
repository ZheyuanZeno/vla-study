"""Replayable initial-state bundles: dynamic state plus environment reset seed.

The seed reconstructs fixture transforms that are absent from MuJoCo's flattened
state vector. Duplicate detection uses movable-object positions, not time or robot noise.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def static_layout(env):
    model=env._env.sim.model
    return np.column_stack((model.body_pos,model.body_quat)).copy()


def object_geometry(env):
    sim=env._env.sim
    names=sorted(env._env.env.objects_dict)
    objects=env._env.env.objects_dict
    positions=np.stack([sim.data.get_body_xpos(objects[name].root_body) for name in names]).copy()
    matrices=np.stack([sim.data.get_body_xmat(objects[name].root_body) for name in names]).copy()
    return names,positions,matrices


def model_signature(env):
    model=env._env.sim.model
    return {'nq':model.nq,'nv':model.nv,'na':model.na,'body_names':list(model.body_names),
            'joint_names':list(model.joint_names)}


def nearest_position_distance(positions,references):
    if len(references)==0:return None
    references=np.asarray(references)
    if references.shape[1:]!=np.asarray(positions).shape:raise ValueError('Object topology differs')
    return float(np.max(np.abs(references-positions),axis=(1,2)).min())


@dataclass
class StatePool:
    path: Path
    manifest: dict
    states: np.ndarray
    reset_seeds: np.ndarray
    settled_states: np.ndarray
    static_layouts: np.ndarray
    object_positions: np.ndarray
    object_matrices: np.ndarray
    object_names: list[str]
    sha256: str
    manifest_sha256: str


def load_state_pool(path,manifest_path=None):
    path=Path(path);manifest_path=Path(manifest_path) if manifest_path else path.with_name('pool.json')
    manifest=json.loads(manifest_path.read_text())
    if manifest['status']!='validated':raise ValueError('State pool validation is incomplete')
    digest=sha256_file(path)
    if digest!=manifest['pool_sha256']:raise ValueError('State pool checksum mismatch')
    with np.load(path,allow_pickle=False) as data:
        arrays={k:data[k].copy() for k in ['states','reset_seeds','settled_states','static_layouts','object_positions','object_matrices']}
        names=data['object_names'].tolist()
    states=arrays['states'];n=len(states)
    if states.dtype!=np.float64 or states.ndim!=2 or n!=manifest['n_states']:raise ValueError('Invalid state matrix')
    signature=manifest['model_signature']
    if states.shape[1]!=1+signature['nq']+signature['nv']+signature['na']:raise ValueError('State width differs from model')
    if arrays['reset_seeds'].shape!=(n,) or arrays['reset_seeds'].dtype.kind not in 'iu':raise ValueError('Invalid reset seeds')
    if np.any(arrays['reset_seeds']<0) or np.any(arrays['reset_seeds']>=2**32):raise ValueError('Seed outside NumPy range')
    if arrays['settled_states'].shape!=states.shape:raise ValueError('Invalid settled states')
    if arrays['static_layouts'].shape!=(n,len(signature['body_names']),7):raise ValueError('Invalid static layouts')
    if arrays['object_positions'].shape!=(n,len(names),3):raise ValueError('Invalid object geometry')
    if arrays['object_matrices'].shape!=(n,len(names),3,3):raise ValueError('Invalid object orientation')
    if any(not np.isfinite(a).all() for a in arrays.values()):raise ValueError('Nonfinite state data')
    records=manifest['records']
    if len(records)!=n:raise ValueError('Missing state provenance')
    for i,record in enumerate(records):
        if record['state_id']!=i or record['environment_seed']!=int(arrays['reset_seeds'][i]):raise ValueError('State/seed mapping differs')
        if hashlib.sha256(states[i].tobytes()).hexdigest()!=record['state_sha256']:raise ValueError('State checksum mismatch')
    return StatePool(path,manifest,**arrays,object_names=names,sha256=digest,manifest_sha256=sha256_file(manifest_path))


def validate_pool_for_env(pool,env):
    if model_signature(env)!=pool.manifest['model_signature']:raise ValueError('Pool model topology differs from environment')
    if sorted(env._env.env.objects_dict)!=pool.object_names:raise ValueError('Pool object names differ')
