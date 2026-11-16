"""Synthetic structural fixtures; not simulator states and never experiment data."""
import hashlib,json
import numpy as np
import pytest
from src.runner.state_pool import load_state_pool,nearest_position_distance,sha256_file


def fixture_pool(tmp_path,seeds=None):
    states=np.arange(8,dtype=np.float64).reshape(2,4)
    seeds=np.array([42,43] if seeds is None else seeds,dtype=np.int64)
    path=tmp_path/'states.npz'
    np.savez_compressed(path,states=states,reset_seeds=seeds,settled_states=states,
        static_layouts=np.zeros((2,2,7)),object_positions=np.zeros((2,1,3)),
        object_matrices=np.tile(np.eye(3),(2,1,1,1)),object_names=np.array(['synthetic_object']))
    manifest={'data_origin':'synthetic_unit_test','status':'validated','pool_sha256':sha256_file(path),'n_states':2,
        'model_signature':{'nq':2,'nv':1,'na':0,'body_names':['synthetic_a','synthetic_b']},
        'records':[{'state_id':i,'environment_seed':int(seeds[i]),'state_sha256':hashlib.sha256(states[i].tobytes()).hexdigest()} for i in range(2)]}
    (tmp_path/'pool.json').write_text(json.dumps(manifest))
    return path


def test_pool_provenance_shapes_and_checksum(tmp_path):
    path=fixture_pool(tmp_path);pool=load_state_pool(path)
    assert pool.states.shape==(2,4) and pool.reset_seeds.tolist()==[42,43]
    path.write_bytes(path.read_bytes()+b'tampered')
    with pytest.raises(ValueError,match='checksum'):load_state_pool(path)


def test_incomplete_pool_or_invalid_seed_cannot_enter_runner(tmp_path):
    path=fixture_pool(tmp_path,[-1,43])
    with pytest.raises(ValueError,match='Seed'):load_state_pool(path)
    manifest=json.loads((tmp_path/'pool.json').read_text());manifest['status']='in_progress'
    (tmp_path/'pool.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='incomplete'):load_state_pool(path)


def test_duplicate_detection_uses_object_positions_not_robot_or_time():
    positions=np.array([[.1,.2,.3],[.2,.3,.4]])
    assert nearest_position_distance(positions,[positions.copy()])==0
    moved=positions.copy();moved[1,0]+=.01
    assert nearest_position_distance(moved,[positions])>.009
