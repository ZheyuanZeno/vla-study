# 在项目根目录执行 source scripts/activate.sh
source /home/zy/miniconda3/etc/profile.d/conda.sh
conda activate smolvla-libero
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LIBERO_CONFIG_PATH="$PWD/.runtime/libero"
export HF_HOME="$PWD/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
# 避免 ~/.local/lib/python3.10 中已有 ROS/其他项目包污染实验。
export PYTHONNOUSERSITE=1
