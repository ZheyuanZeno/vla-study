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
