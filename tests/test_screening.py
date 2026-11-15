"""Synthetic selection fixtures: no rollout outcomes are generated for experiments."""
import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('screening',Path('scripts/update_screening_selection.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def synthetic_rows(n_success=10):
    return [dict(study_phase='screening',condition='Baseline',checkpoint_revision='synthetic',
        state_id=i,state_sha256=f'synthetic-state-{i}',success=i<n_success,
        status='success' if i<n_success else 'task_timeout',runtime_flags={'synthetic':True}) for i in range(20)]


def test_twenty_unique_valid_states_required_and_ten_success_gate():
    score=module.score_rows(synthetic_rows(),list(range(20)),'synthetic')
    assert score['complete'] and score['successes']==10 and score['timeouts']==10
    assert not module.score_rows(synthetic_rows()[:19],list(range(20)),'synthetic')['complete']
    assert module.score_rows(synthetic_rows(9),list(range(20)),'synthetic')['successes']==9


def test_reject_duplicate_states_and_pilot_data():
    rows=synthetic_rows();rows[-1]['state_sha256']=rows[0]['state_sha256']
    with pytest.raises(ValueError,match='Repeated initial'):module.score_rows(rows,list(range(20)),'synthetic')
    rows=synthetic_rows();rows[0]['study_phase']='pilot'
    with pytest.raises(ValueError,match='Only screening'):module.score_rows(rows,list(range(20)),'synthetic')


def test_selection_keeps_first_four_passing_in_order_not_highest_rates(tmp_path):
    import json
    root=tmp_path/'results';eligibility=tmp_path/'eligibility';eligibility.mkdir()
    order=[{'suite':'synthetic','task_id':i} for i in range(6)]
    protocol={'candidate_order':order,'state_ids':list(range(20)),'checkpoint_revision':'synthetic',
              'minimum_successes':10,'stop_after_selected_tasks':4}
    for task,successes in enumerate([10,20,9,10,19,20]):
        audit=tmp_path/f'synthetic_audit_{task}.json';audit.write_text(json.dumps({'all_passed':True}))
        (eligibility/f'synthetic_{task}.json').write_text(json.dumps({'decision':'eligible',
            'audit_file':str(audit),'regions_file':'synthetic_regions.json'}))
        folder=root/f'synthetic_{task}';folder.mkdir(parents=True)
        rows=[dict(row,suite='synthetic',task_id=task) for row in synthetic_rows(successes)]
        (folder/'results.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    result=module.evaluate(protocol,root,eligibility)
    assert [r['task_id'] for r in result['selected']]==[0,1,3,4]
    assert result['four_tasks_selected'] and result['assessments'][-1]['task_id']==4
