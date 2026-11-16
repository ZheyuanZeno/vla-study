"""All arrays and rows in this file are explicitly synthetic unit-test fixtures."""
import numpy as np
import pytest
from src.analysis.bootstrap import CONDITIONS,blocks_from_rows,summarize


def synthetic_rows():
    return [dict(suite='synthetic',task_id=task,state_id=state,condition=c,status='success',success=True,
        state_sha256=f'synthetic-state-{task}-{state}',checkpoint_revision='synthetic',
        study_phase='formal',data_origin='synthetic') for task in range(4) for state in range(2) for c in CONDITIONS]


def test_perfect_pairing_gives_zero_gap_in_every_resample():
    correlated=np.repeat(np.array([[0],[1],[0],[1]]),7,axis=1)
    result=summarize({'synthetic:0':correlated},5000,1)
    assert result['equal_task_average']['gap_ci95_pointwise']==[[0.,0.]]*3
    assert result['equal_task_average']['large_minus_small_ci95']==[0.,0.]


def test_tasks_are_equally_weighted_despite_unequal_block_counts():
    result=summarize({'synthetic:0':np.ones((1,7)),'synthetic:1':np.zeros((9,7))},5000,1)
    assert all(x==.5 for x in result['equal_task_average']['success_rates'].values())


def test_reject_incomplete_duplicate_mismatched_and_synthetic_real_reports():
    rows=synthetic_rows()
    with pytest.raises(ValueError,match='Synthetic'): blocks_from_rows(rows)
    with pytest.raises(ValueError,match='Incomplete'): blocks_from_rows(rows[:-1],allow_synthetic=True)
    with pytest.raises(ValueError,match='Duplicate'): blocks_from_rows(rows+[rows[0]],allow_synthetic=True)
    wrong=[dict(r) for r in rows]; wrong[0]['state_sha256']='different'
    with pytest.raises(ValueError,match='different initial'): blocks_from_rows(wrong,allow_synthetic=True)
    assert len(blocks_from_rows(rows,allow_synthetic=True)[0])==4


def test_pilot_cannot_enter_formal_analysis():
    rows=synthetic_rows()
    for row in rows: row['study_phase']='pilot'
    with pytest.raises(ValueError,match='No formal data'): blocks_from_rows(rows,allow_synthetic=True)


def test_infrastructure_failure_requires_repaired_complete_block():
    rows=synthetic_rows(); fault=dict(rows[-1],success=None,status='infrastructure_failure')
    with pytest.raises(ValueError,match='Incomplete'): blocks_from_rows(rows[:-1]+[fault],allow_synthetic=True)
    blocks,failures=blocks_from_rows(rows+[fault],allow_synthetic=True)
    assert len(failures)==1 and len(blocks)==4


def test_full_fixture_bundle_mismatch_and_aliased_units_are_rejected():
    rows=synthetic_rows()
    for row in rows:row['initial_state_bundle_id']=f"synthetic-bundle-{row['task_id']}-{row['state_id']}"
    rows[0]['initial_state_bundle_id']='wrong-fixture-layout'
    with pytest.raises(ValueError,match='bundles'):blocks_from_rows(rows,allow_synthetic=True)
    rows=synthetic_rows()
    for row in rows:row['initial_state_bundle_id']=f"synthetic-same-bundle-{row['task_id']}"
    with pytest.raises(ValueError,match='Duplicate physical'):blocks_from_rows(rows,allow_synthetic=True)
