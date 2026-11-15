"""Read real geom IDs using robosuite's render context.

robosuite 1.4.0 read_pixels(segmentation=True) multiplies uint8 by 256,
which raises on NumPy 2. We use the same rendering and ID mapping, with
explicit int32 arithmetic locally, without patching upstream files.
"""
import numpy as np
import torch
from robosuite import macros
from robosuite.utils.binding_utils import _MjSim_render_lock
from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING


def decode_id_rgb(rgb):
    wide=np.asarray(rgb,dtype=np.int32)
    return wide[...,0]+wide[...,1]*256+wide[...,2]*65536


def render_segmentation(env, camera='agentview'):
    sim=env._env.sim
    width,height=env.observation_width,env.observation_height
    ctx=sim._render_context_offscreen
    with _MjSim_render_lock:
        ctx.render(width=width,height=height,camera_id=sim.model.camera_name2id(camera),segmentation=True)
        codes=decode_id_rgb(ctx.read_pixels(width,height,segmentation=False))
        mapping=np.full((ctx.scn.ngeom+1,2),-1,dtype=np.int32)
        for i in range(ctx.scn.ngeom):
            geom=ctx.scn.geoms[i]
            if geom.segid != -1:
                mapping[geom.segid+1]=(geom.objtype,geom.objid)
        codes[codes>=len(mapping)]=0
        segmentation=mapping[codes]
    convention=IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
    return segmentation[::convention].copy()


def policy_segmentation(raw, size=(512,512)):
    rotated=np.flip(raw,axis=(0,1)).copy()
    return torch.nn.functional.interpolate(
        torch.as_tensor(rotated).permute(2,0,1)[None].float(),size=size,mode='nearest'
    )[0].permute(1,2,0).numpy().astype(np.int32)
