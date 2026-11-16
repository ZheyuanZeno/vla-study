"""Real seeded-reset and fixed-seed reload probe; no policy inference."""
import json
from pathlib import Path
import numpy as np
from lerobot.envs.libero import LiberoEnv,_get_suite,get_libero_dummy_action

out=Path('artifacts/state_generation_probe_v2');out.mkdir(exist_ok=True)
env=LiberoEnv(_get_suite('libero_spatial'),0,'libero_spatial',obs_type='pixels_agent_pos',init_states=False)

def geometry():
    sim=env._env.sim
    return {name:np.concatenate([sim.data.get_body_xpos(obj.root_body),sim.data.get_body_xmat(obj.root_body).ravel()])
            for name,obj in env._env.env.objects_dict.items()}

try:
    records=[]
    for seed in [2026111600,2026111601,2026111602]:
        env._env.seed(seed);env._env.reset()
        sim=env._env.sim
        state=env._env.get_sim_state().copy()
        positions=sim.model.body_pos.copy();quaternions=sim.model.body_quat.copy()
        raw_geometry=geometry()
        for _ in range(10):env._env.step(get_libero_dummy_action())
        settled=env._env.get_sim_state().copy();settled_geometry=geometry()
        initial_success=bool(env._env.check_success())
        env._env.seed(42);env._env.reset();env._env.set_init_state(state)
        loaded=env._env.sim
        static_delta=float(np.abs(positions-loaded.model.body_pos).max())
        quat_delta=float(np.abs(quaternions-loaded.model.body_quat).max())
        for _ in range(10):env._env.step(get_libero_dummy_action())
        restored=env._env.get_sim_state().copy()
        record={'seed':seed,'initial_success':initial_success,'state_shape':list(state.shape),
            'roundtrip_max_abs_delta':float(np.abs(settled-restored).max()),
            'model_body_pos_delta':static_delta,'model_body_quat_delta':quat_delta,
            'raw_object_geometry':{k:v.tolist() for k,v in raw_geometry.items()},
            'settled_object_geometry':{k:v.tolist() for k,v in settled_geometry.items()}}
        records.append(record);print(json.dumps(record),flush=True)
        np.save(out/f'state_seed{seed}.npy',state)
    (out/'probe.json').write_text(json.dumps(records,indent=2))
finally:env.close()
