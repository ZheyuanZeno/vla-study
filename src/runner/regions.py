"""Construct episode-fixed target/background masks from the real reset state."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from src.masking.geometry import paired_squares, coverage
from src.masking.segmentation import render_segmentation, policy_segmentation


def prepare_regions(env, observation, regions, condition, output):
    import mujoco
    from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad
    import torch
    seg=policy_segmentation(render_segmentation(env))
    sim=env._env.sim
    names={i:sim.model.geom_id2name(i) for i in range(sim.model.ngeom)}
    target_ids=[i for i,n in names.items() if n in regions['target_geom_names']]
    allowed_ids=[i for i,n in names.items() if n in regions['background_allowed_geom_names']]
    if len(target_ids) != len(regions['target_geom_names']):
        raise ValueError('Frozen target geom names are missing')
    target=(seg[...,0]==int(mujoco.mjtObj.mjOBJ_GEOM))&np.isin(seg[...,1],target_ids)
    allowed=(seg[...,0]==int(mujoco.mjtObj.mjOBJ_GEOM))&np.isin(seg[...,1],allowed_ids)
    squares={}; geometry={}
    for level,fraction in [('1',.25),('2',.5),('3',.75)]:
        t,b=paired_squares(target,fraction,regions['background_anchor'],allowed)
        for prefix,square in [('T',t),('B',b)]:
            squares[prefix+level]=square
            geometry[prefix+level]={**square.to_dict(),'target_coverage':coverage(target,square)}
    rgb=observation['pixels']['image']
    tensor=torch.tensor(np.flip(rgb,axis=(0,1)).copy()).permute(2,0,1)[None].float()/255
    resized=resize_with_pad(tensor,512,512,pad_value=0)[0].permute(1,2,0).numpy()
    overlay=np.rint(resized*255).astype(np.uint8); overlay[target]=[255,0,0]
    Image.fromarray(overlay).save(output/'target_overlay.png')
    np.save(output/'target_pixels.npy',target)
    np.save(output/'background_allowed.npy',allowed)
    np.save(output/'segmentation.npy',seg)
    (output/'mask_pairs.json').write_text(json.dumps(geometry,indent=2))
    return (None if condition=='Baseline' else squares[condition]),geometry.get(condition)
