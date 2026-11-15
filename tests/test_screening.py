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
