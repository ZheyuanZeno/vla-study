"""Synthetic permission records; these never approve any actual experiment."""
import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('formal_protocol',Path('scripts/formal_protocol.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_formal_approval_requires_exact_protocol_size_and_explicit_message():
    protocol={'protocol_sha256':'synthetic-sha','planned_rollouts':1400}
    with pytest.raises(PermissionError):module.validate_approval(protocol,{})
    with pytest.raises(PermissionError):module.validate_approval(protocol,{'approved':True,'protocol_sha256':'wrong'})
    with pytest.raises(PermissionError):module.validate_approval(protocol,{'approved':True,'protocol_sha256':'synthetic-sha','approved_rollouts':2800})
    with pytest.raises(PermissionError):module.validate_approval(protocol,{'approved':True,'protocol_sha256':'synthetic-sha','approved_rollouts':1400})
    module.validate_approval(protocol,{'approved':True,'protocol_sha256':'synthetic-sha','approved_rollouts':1400,'user_message':'synthetic fixture only'})


def test_protocol_fingerprint_detects_a_changed_sampling_plan():
    a={'sample_size':50}; a['protocol_sha256']=module.fingerprint(a)
    assert module.fingerprint(a)==a['protocol_sha256']
    b=dict(a,sample_size=100)
    assert module.fingerprint(b)!=a['protocol_sha256']


def test_formal_run_without_a_frozen_protocol_is_rejected():
    with pytest.raises(PermissionError):module.validate_run_config({'study_phase':'formal'})
