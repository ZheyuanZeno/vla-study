"""Create real initial-frame geometry evidence; this script does not approve eligibility."""
import argparse
import json
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from PIL import Image
from libero.libero import get_libero_path
from lerobot.envs.libero import LiberoEnv,_get_suite
from src.masking.segmentation import render_segmentation,policy_segmentation
from src.masking.geometry import paired_squares,coverage,mask_rgb
from src.runner.regions import prepare_regions


def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',required=True);p.add_argument('--task-id',type=int,required=True)
    p.add_argument('--anchor',type=int,nargs=2,default=[320,460]);args=p.parse_args()
    suite=_get_suite(args.suite);task=suite.get_task(args.task_id)
    bddl=Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
    source=bddl.read_text();goal=source.split('(:goal',1)[1]
    matches=re.findall(r'\((On|In)\s+([\w_]+)\s+([\w_]+)\s*\)',goal)
    if len(matches)!=1: raise ValueError('Goal requires individual semantic review; no guessed target')
    predicate,manipuland,destination=matches[0]
    output=Path('artifacts/candidate_geometry')/f'{args.suite}_{args.task_id}';output.mkdir(parents=True,exist_ok=True)
    env=LiberoEnv(suite,args.task_id,args.suite,obs_type='pixels_agent_pos')
    try:
        obs,_=env.reset(seed=42);seg=policy_segmentation(render_segmentation(env))
        names={i:env._env.sim.model.geom_id2name(i) for i in range(env._env.sim.model.ngeom)}
        target_names=[name for name in names.values() if name and name.startswith(destination+'_')]
        if not target_names: raise ValueError('Destination has no named geoms')
        regions={'suite':args.suite,'task_id':args.task_id,'placement_destination':destination,
            'target_geom_names':target_names,'background_anchor':args.anchor,
            'background_allowed_geom_names':['table_visual'],'regions_version':'screening-v1',
            'source_image':str(output/'original.png'),'state_id_for_anchor_review':0,
            'bddl_file':str(bddl),'bddl_goal':goal,'manipulated_object':manipuland}
        prepare_regions(env,obs,regions,'Baseline',output)
        from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad
        import torch
        x=torch.tensor(np.flip(obs['pixels']['image'],axis=(0,1)).copy()).permute(2,0,1)[None].float()/255
        rgb=np.rint(resize_with_pad(x,512,512,pad_value=0)[0].permute(1,2,0).numpy()*255).astype(np.uint8)
        Image.fromarray(rgb).save(output/'original.png')
        pairs=json.loads((output/'mask_pairs.json').read_text())
        from src.masking.geometry import Square
        for condition in ['T1','T2','T3','B1','B2','B3']:
            box=pairs[condition];x0,y0,_,_=box['bbox']
            Image.fromarray(mask_rgb(rgb,Square(x0,y0,box['side']))).save(output/f'{condition}.png')
        (output/'regions_proposed.json').write_text(json.dumps(regions,indent=2))
        print(json.dumps(regions,indent=2))
    finally: env.close()


if __name__=='__main__':main()
