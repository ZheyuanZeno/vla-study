"""Atomic per-attempt results and rebuildable JSONL index."""
import json
import os
from pathlib import Path


def atomic_json(path, data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as stream:
        json.dump(data,stream,ensure_ascii=False,indent=2,default=str)
        stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def rebuild_index(root):
    rows=[json.loads(p.read_text()) for p in sorted(root.glob('episodes/*/attempt*/result.json'))]
    temporary=root/'results.jsonl.tmp'
    with temporary.open('w') as stream:
        for row in rows: stream.write(json.dumps(row,ensure_ascii=False,default=str)+'\n')
        stream.flush(); os.fsync(stream.fileno())
    temporary.replace(root/'results.jsonl')
    return rows


def recover_interrupted(root):
    """Retain abandoned attempts as technical faults before retrying, never as failures."""
    for attempt in sorted(root.glob('episodes/*/attempt*')):
        if (attempt/'result.json').exists(): continue
        metadata=attempt/'attempt_manifest.json'
        if not metadata.exists():
            raise ValueError(f'Interrupted attempt lacks metadata; inspect manually: {attempt}')
        row=json.loads(metadata.read_text())
        step_file=attempt/'steps.jsonl'
        step_count=len(step_file.read_text().splitlines()) if step_file.exists() else 0
        row.update(status='infrastructure_failure',success=None,error_type='InterruptedAttempt',
            env_steps=step_count,elapsed_time=None,gpu_peak_allocated_bytes=None,
            gpu_peak_reserved_bytes=None,inference_calls=None,mask_geometry=None,target_coverage=None)
        atomic_json(attempt/'result.json',row)
