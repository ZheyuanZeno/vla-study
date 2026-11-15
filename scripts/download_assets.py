"""锁定模拟资产 revision，接入 hf-libero 的本地资产查找接口。"""
import importlib.util
import json
from pathlib import Path
from huggingface_hub import snapshot_download

repo = 'lerobot/libero-assets'
revision = '0b3ea86be5fe169d0fd036ae63d1070ec09e90f6'
root = Path(__file__).resolve().parents[1]
snapshot = Path(snapshot_download(repo, repo_type='dataset', revision=revision))
required = ['articulated_objects', 'stable_scanned_objects', 'turbosquid_objects', 'stable_hope_objects', 'scenes']
for name in required:
    if not (snapshot / name).is_dir():
        raise RuntimeError(f'资产缺失: {name}')
spec = importlib.util.find_spec('libero')
link = Path(list(spec.submodule_search_locations)[0]) / 'libero/assets'
if link.is_symlink():
    if link.resolve() != snapshot.resolve():
        raise RuntimeError(f'已有其他资产链接，不自动替换: {link}')
elif link.exists():
    raise RuntimeError(f'已有本地资产目录，不自动替换: {link}')
else:
    link.symlink_to(snapshot, target_is_directory=True)
(root/'artifacts/libero-assets.json').write_text(json.dumps(
    {'repo_id': repo, 'revision': revision, 'snapshot_path': str(snapshot), 'installed_link': str(link)}, indent=2))
print(snapshot)
