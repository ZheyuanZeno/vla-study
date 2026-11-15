"""Synthetic interrupted-attempt fixtures; never real experiment outcomes."""
import json
from src.runner.storage import atomic_json,recover_interrupted,rebuild_index


def test_interrupted_attempt_is_a_technical_fault_and_valid_failure_is_preserved(tmp_path):
    root=tmp_path
    abandoned=root/'episodes/state000_T1/attempt001'
    metadata={'data_origin':'synthetic','suite':'synthetic','task_id':0,'state_id':0,'condition':'T1'}
    atomic_json(abandoned/'attempt_manifest.json',metadata)
    (abandoned/'steps.jsonl').write_text('{"synthetic_step":1}\n{"synthetic_step":2}\n')
    completed=root/'episodes/state000_T1/attempt002'
    valid=dict(metadata,status='task_timeout',success=False,env_steps=280)
    atomic_json(completed/'result.json',valid)
    recover_interrupted(root); rows=rebuild_index(root)
    fault=next(r for r in rows if r.get('error_type')=='InterruptedAttempt')
    assert fault['success'] is None and fault['status']=='infrastructure_failure' and fault['env_steps']==2
    assert next(r for r in rows if r['status']=='task_timeout')['success'] is False
    recover_interrupted(root)
    assert rebuild_index(root)==rows
