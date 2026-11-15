"""下载不可变 revision 并记录来源；不更改 checkpoint 文件。"""
import json
from pathlib import Path
from huggingface_hub import snapshot_download

REPO = 'HuggingFaceVLA/smolvla_libero'
REVISION = '6721902bc4d61e50a3bfdb11dfb4cb626f05d102'
path = snapshot_download(REPO, revision=REVISION)
output = Path('artifacts/checkpoint.json')
output.parent.mkdir(exist_ok=True)
output.write_text(json.dumps({'repo_id': REPO, 'revision': REVISION, 'snapshot_path': path}, indent=2))
print(path)
