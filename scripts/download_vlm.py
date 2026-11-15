"""Pin the auxiliary VLM/processor cache used by the untouched checkpoint config."""
import json
from pathlib import Path
from huggingface_hub import snapshot_download

root=Path(__file__).resolve().parents[1]
source=json.loads((root/'configs/source_versions.json').read_text())['vision_language_base']
snapshot=Path(snapshot_download(source['repo_id'],revision=source['revision'],
    allow_patterns=['*.json','*.model','*.txt','*.safetensors','*.jinja']))
# The original checkpoint and tokenizer processor refer to the repository name.
# Freeze this project-local alias, then run evaluation offline.
reference=snapshot.parent.parent/'refs/main'
reference.parent.mkdir(parents=True,exist_ok=True)
reference.write_text(source['revision'])
(root/'artifacts/vision_language_base.json').write_text(json.dumps(
    {**source,'snapshot_path':str(snapshot),'project_local_main_alias':str(reference)},indent=2))
print(snapshot)
