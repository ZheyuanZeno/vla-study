#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/activate.sh
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -c constraints.txt -e 'vendor/lerobot[smolvla,libero]' pytest
python scripts/prepare_libero.py
python scripts/download_assets.py
python -m pip check
python -m pip freeze > artifacts/pip-freeze.txt
conda env export > artifacts/conda-environment.yml
