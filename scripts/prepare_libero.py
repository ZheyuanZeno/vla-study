"""创建项目独立的 LIBERO 路径配置，不触碰 ~/.libero。"""
import importlib.util
import os
from pathlib import Path
import yaml

root = Path(__file__).resolve().parents[1]
spec = importlib.util.find_spec('libero')
if spec is None:
    raise RuntimeError('请先安装 hf-libero')
base = Path(list(spec.submodule_search_locations)[0]) / 'libero'
config_dir = Path(os.environ.get('LIBERO_CONFIG_PATH', root / '.runtime/libero'))
config_dir.mkdir(parents=True, exist_ok=True)
config = dict(benchmark_root=str(base), bddl_files=str(base/'bddl_files'),
              init_states=str(base/'init_files'), datasets=str(root/'.cache/datasets'),
              assets=str(base/'assets'))
path = config_dir/'config.yaml'
if path.exists():
    existing = yaml.safe_load(path.read_text())
    if existing != config:
        raise RuntimeError(f'已有配置不同，请检查 {path}')
else:
    path.write_text(yaml.safe_dump(config))
print(path)
