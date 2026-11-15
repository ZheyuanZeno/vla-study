"""真实 reset 状态的 RGB/geom segmentation 对齐检查，保持模拟场景不变。"""
import sys
sys.path.insert(0,str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from robosuite import macros
from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
from lerobot.envs.libero import LiberoEnv, _get_suite
from src.masking.segmentation import render_segmentation, policy_segmentation
from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad

out=Path('artifacts/segmentation/task0-state0'); out.mkdir(parents=True,exist_ok=True)
env=LiberoEnv(_get_suite('libero_spatial'),0,'libero_spatial',obs_type='pixels_agent_pos')
try:
    obs,_=env.reset(seed=42)
    sim=env._env.sim
    rgb=sim.render(width=256,height=256,camera_name='agentview')
    conv=IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
    rgb=rgb[::conv]
    Image.fromarray(rgb).save(out/'render_direct.png')
    Image.fromarray(obs['pixels']['image']).save(out/'observation.png')
    errors={name:float(np.abs(a.astype(float)-obs['pixels']['image'].astype(float)).mean()) for name,a in [('identity',rgb),('vertical',rgb[::-1]),('horizontal',rgb[:,::-1]),('both',rgb[::-1,::-1])]}
    print('alignment_errors',errors,flush=True)
    assert errors['identity'] < 1.0, 'RGB 渲染与真实 observation 不对齐'
    seg=render_segmentation(env)
    np.save(out/'segmentation_raw.npy',seg)
    # 与官方 LIBERO processor 的旋转、SmolVLA resize 顺序一致。
    seg=np.flip(seg,axis=(0,1)).copy()
    seg512=torch.nn.functional.interpolate(torch.tensor(seg).permute(2,0,1)[None].float(),size=(512,512),mode='nearest')[0].permute(1,2,0).numpy().astype(np.int32)
    np.save(out/'segmentation_policy.npy',seg512)
    image=resize_with_pad(torch.tensor(np.flip(rgb,axis=(0,1)).copy()).permute(2,0,1)[None].float()/255,512,512,pad_value=0)[0].permute(1,2,0).numpy()
    Image.fromarray(np.rint(image*255).astype(np.uint8)).save(out/'rgb_policy.png')
    names={str(i):sim.model.geom_id2name(i) for i in range(sim.model.ngeom)}
    counts=[]
    for geom in np.unique(seg512[...,1]):
        counts.append({'geom_id':int(geom),'name':names.get(str(geom)),'visible_pixels':int((seg512[...,1]==geom).sum())})
    (out/'geoms.json').write_text(json.dumps({'geom_names':names,'visible':counts,'rgb_mean_absolute_error':errors['identity'],'task':env.task,'instruction':env.task_description},indent=2))
    print(json.dumps(counts,indent=2))
finally:
    env.close()
