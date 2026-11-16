"""Check or run the frozen proposal. Default is read-only; --run requires approval."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import yaml
from scripts.formal_protocol import validate_protocol,validate_run_config


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',default='configs/formal50_protocol.json')
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--run',action='store_true');group.add_argument('--analyze',action='store_true')
    args=parser.parse_args()
    protocol=validate_protocol(args.protocol,require_approval=args.run or args.analyze)
    if not args.run and not args.analyze:
        print(json.dumps({'protocol_sha256':protocol['protocol_sha256'],'planned_rollouts':protocol['planned_rollouts'],
            'tasks':[(t['suite'],t['task_id']) for t in protocol['tasks']],
            'checks_passed_except_approval':True,'approval_file_exists':Path(protocol['approval_file']).exists(),
            'formal_rollouts_started':False},indent=2));return
    if args.run:
        for task in protocol['tasks']:
            config=yaml.safe_load(Path(task['formal_config_file']).read_text())
            validate_run_config(config)
            subprocess.run([sys.executable,'-m','src.runner.run','--config',task['formal_config_file']],check=True)
        return
    inputs=[]
    expected_conditions=set(protocol['conditions'])
    for task in protocol['tasks']:
        root=Path(task['formal_output_dir']);results=root/'results.jsonl'
        rows=[json.loads(s) for s in results.read_text().splitlines() if s.strip()]
        valid=[r for r in rows if r['status']!='infrastructure_failure']
        expected={(state,condition) for state in task['state_ids'] for condition in expected_conditions}
        keys=[(r['state_id'],r['condition']) for r in valid]
        if len(keys)!=len(set(keys)) or set(keys)!=expected: raise ValueError('Formal sampling plan is not complete')
        if any(r['study_phase']!='formal' or r.get('protocol_sha256')!=protocol['protocol_sha256'] for r in valid):
            raise ValueError('Results do not match approved formal protocol')
        subprocess.run([sys.executable,'-m','src.runner.verify','--root',str(root)],check=True)
        inputs.append(str(results))
    subprocess.run([sys.executable,'-m','src.analysis.run','--input',*inputs,'--output',protocol.get('analysis_output_dir','artifacts/formal_v1/analysis')],check=True)


if __name__=='__main__':main()
