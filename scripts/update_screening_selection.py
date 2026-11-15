"""Apply the prespecified order/20-episode rule; never inspect masked outcomes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import yaml
from src.runner.storage import atomic_json

VALID={'success','valid_task_failure','task_timeout'}


def score_rows(rows,state_ids,checkpoint_revision):
    valid={}; faults=0
    for row in rows:
        if row['study_phase']!='screening' or row['condition']!='Baseline':
            raise ValueError('Only screening Baseline data may determine selection')
        if row['checkpoint_revision']!=checkpoint_revision: raise ValueError('Wrong checkpoint')
        if row['state_id'] not in state_ids: raise ValueError('Unexpected state')
        if row['status']=='infrastructure_failure': faults+=1; continue
        if row['status'] not in VALID: raise ValueError('Unknown status')
        if type(row['success']) is not bool or row['success']!=(row['status']=='success'):
            raise ValueError('Invalid Boolean outcome')
        if row['state_id'] in valid: raise ValueError('Duplicate completed state')
        valid[row['state_id']]=row
    if len({r['state_sha256'] for r in valid.values()})!=len(valid):
        raise ValueError('Repeated initial state cannot count as an independent screening sample')
    runtime={json.dumps(r.get('runtime_flags'),sort_keys=True) for r in valid.values()}
    if len(runtime)>1: raise ValueError('Mixed inference flags')
    return {'n_valid':len(valid),'successes':sum(r['success'] for r in valid.values()),
        'timeouts':sum(r['status']=='task_timeout' for r in valid.values()),'infrastructure_attempts':faults,
        'complete':set(valid)==set(state_ids)}


def evaluate(protocol,root,eligibility_root):
    selected=[]; assessments=[]
    for candidate in protocol['candidate_order']:
        suite,task=candidate['suite'],candidate['task_id']; name=f'{suite}_{task}'
        entry={**candidate}
        eligibility_path=eligibility_root/f'{name}.json'
        if not eligibility_path.exists():
            assessments.append({**entry,'status':'pending_geometry_review'}); break
        eligibility=json.loads(eligibility_path.read_text())
        if eligibility['decision']=='exclude_geometry':
            if not eligibility.get('reason') or not eligibility.get('visual_evidence'):
                raise ValueError('Geometry exclusions require recorded evidence')
            assessments.append({**entry,'status':'excluded_geometry','reason':eligibility['reason']}); continue
        if eligibility['decision']!='eligible': raise ValueError('Unresolved eligibility decision')
        audit=json.loads(Path(eligibility['audit_file']).read_text())
        if not audit['all_passed']: raise ValueError('Eligibility audit has not passed')
        results=root/name/'results.jsonl'
        if not results.exists():
            assessments.append({**entry,'status':'pending_baseline'}); break
        rows=[json.loads(line) for line in results.read_text().splitlines() if line.strip()]
        if any(r['suite']!=suite or r['task_id']!=task for r in rows): raise ValueError('Wrong task in data')
        score=score_rows(rows,protocol['state_ids'],protocol['checkpoint_revision'])
        entry.update(score)
        if not score['complete']:
            assessments.append({**entry,'status':'baseline_in_progress'}); break
        eligible=score['successes']>=protocol['minimum_successes']
        entry.update(status='selected' if eligible else 'excluded_baseline_below_threshold',
            regions_file=eligibility['regions_file'],eligibility_file=str(eligibility_path),results_file=str(results))
        assessments.append(entry)
        if eligible: selected.append(entry)
        if len(selected)==protocol['stop_after_selected_tasks']: break
    return {'selected':selected,'assessments':assessments,
        'four_tasks_selected':len(selected)==protocol['stop_after_selected_tasks'],
        'formal_collection_authorized':False}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--protocol',default='configs/screening/protocol.yaml')
    parser.add_argument('--root',default='artifacts/screening_v1'); args=parser.parse_args()
    protocol=yaml.safe_load(Path(args.protocol).read_text())
    report=evaluate(protocol,Path(args.root),Path('configs/screening/eligibility'))
    report['protocol']=protocol
    report['selection_code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    atomic_json(Path(args.root)/'selection.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
