"""Audit the frozen task anchor on native reset states, without policy inference."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from lerobot.envs.libero import LiberoEnv,_get_suite
from src.runner.regions import prepare_regions
from src.runner.storage import atomic_json


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--regions',required=True)
    parser.add_argument('--output',required=True); args=parser.parse_args()
    regions=json.loads(Path(args.regions).read_text()); output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    env=LiberoEnv(_get_suite(regions['suite']),regions['task_id'],regions['suite'],obs_type='pixels_agent_pos')
    records=[]
    try:
        for state_id in range(len(env._init_states)):
            folder=output/f'state{state_id:03d}'; folder.mkdir(exist_ok=True)
            env.init_state_id=state_id
            record={'state_id':state_id,'suite':regions['suite'],'task_id':regions['task_id'],
                'source_state_sha256':hashlib.sha256(np.asarray(env._init_states[state_id]).tobytes()).hexdigest()}
            try:
                observation,_=env.reset(seed=42)
                prepare_regions(env,observation,regions,'Baseline',folder)
                record.update(status='passed',pairs=json.loads((folder/'mask_pairs.json').read_text()))
            except Exception as exc:
                record.update(status='region_or_infrastructure_failure',error_type=type(exc).__name__,error=str(exc))
            records.append(record)
            atomic_json(output/'audit.json',{'regions_file':args.regions,'records':records,
                'all_passed':len(records)==len(env._init_states) and all(r['status']=='passed' for r in records)})
            print(json.dumps(record),flush=True)
    finally:
        env.close()
    return 0 if all(r['status']=='passed' for r in records) else 1


if __name__=='__main__': sys.exit(main())
