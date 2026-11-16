"""Synthetic metadata-only role checks; no simulation or policy outputs."""
from types import SimpleNamespace
import numpy as np
import pytest
from src.runner.state_pool_runtime import check_pool_role


def test_reserved_pilot_is_separate_from_formal_candidates():
    pool=SimpleNamespace(states=np.zeros((101,4)),manifest={'policy_seed':42,'pilot_state_id':100,'formal_state_ids':list(range(100))})
    check_pool_role(pool,{'seed':42,'study_phase':'pilot','state_ids':[100]})
    check_pool_role(pool,{'seed':42,'study_phase':'formal','state_ids':list(range(100))})
    with pytest.raises(ValueError,match='formal states'):check_pool_role(pool,{'seed':42,'study_phase':'pilot','state_ids':[0]})
    with pytest.raises(ValueError,match='Reserved'):check_pool_role(pool,{'seed':42,'study_phase':'formal','state_ids':[100]})
    with pytest.raises(ValueError,match='screening'):check_pool_role(pool,{'seed':42,'study_phase':'screening','state_ids':[100]})
    with pytest.raises(ValueError,match='Policy seed'):check_pool_role(pool,{'seed':43,'study_phase':'pilot','state_ids':[100]})
