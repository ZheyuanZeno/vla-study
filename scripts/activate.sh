# 从任意目录 source；CONDA_BASE 可指定非默认安装位置。
_vla_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if ! type conda >/dev/null 2>&1; then
    _vla_conda_base="${CONDA_BASE:-$HOME/miniconda3}"
    if [[ ! -f "$_vla_conda_base/etc/profile.d/conda.sh" ]]; then
        echo '找不到 Conda；请设置 CONDA_BASE 或先初始化 conda。' >&2
        return 1
    fi
    source "$_vla_conda_base/etc/profile.d/conda.sh"
elif [[ "$(type -t conda)" != function ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate smolvla-libero
cd "$_vla_root"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LIBERO_CONFIG_PATH="$PWD/.runtime/libero"
export HF_HOME="$PWD/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
# 避免 ~/.local/lib/python3.10 中已有 ROS/其他项目包污染实验。
export PYTHONNOUSERSITE=1
unset _vla_root _vla_conda_base
