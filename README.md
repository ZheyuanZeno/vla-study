# SmolVLA + LIBERO 视觉遮挡鲁棒性实验

当前按阶段实施：先验证本地环境，再运行真实 checkpoint 的单个 Baseline episode。
原始需求见 docs/task_specification.txt。工程文档使用中文。

```bash
cd /home/zy/vla_study
source scripts/activate.sh
python scripts/check_environment.py --simulator
```

安装依据和完整命令见 [安装文档](docs/installation.md)。

## 数据与阶段约束

- 只执行固定 checkpoint 推理，不训练；首轮 batch=1、单模拟环境。
- 环境通过、模型推理通过、完整 rollout 和任务成功是独立状态；真实 task failure 也是有效结果。
- 未通过真实 Baseline 前，不开始 T1/B1 或七条件 Pilot。
- 任务、遮挡区域与配对状态须经实际验证。Pilot 数据不能混入正式统计。
- 不会自动启动 1400/2800 次正式实验；需用户明确批准。
- 日志位于 logs/，检查结果、图像和模型版本证据位于 artifacts/。

环境验证通过后运行官方 Baseline（固定模型 revision，1 个 episode）：

```bash
bash scripts/run_baseline.sh
```

T1/B1、Pilot 和分析命令将在相应阶段真实验收后加入，当前不提供占位成功输出。

## 已验证（2026-11-15）

Phase 1：真实 LIBERO reset/step、EGL 双摄像头和 CUDA 运算通过。
Phase 2：任务 0、初始状态 0，75 步，success predicate=True。
单个样本不代表整体成功率。官方 rollout 计时约 24 秒，含首次加载约 82 秒。
证据：artifacts/baseline/20261115T191502Z/evidence.json；录像位于同目录 videos/。
冻结配置：configs/baseline_frozen.json；完整依赖：requirements-lock.txt。

Phase 3 已验证：同状态 T1、B1 都完成真实 rollout（各 79 步，success=True）。

```bash
source scripts/activate.sh
python -m src.runner.run --config configs/mask_smoke.yaml
python -m src.runner.run --config configs/pilot.yaml
```

每份输出 manifest 固定配置和 Git commit；代码改变后须使用新目录。
状态 ID 不允许模运算绕回。同一 block 每个条件的 seed 在 rollout 前重置；条件顺序单独随机化。
官方 Phase 2 的 seed 在模型初始化前设置，因此其随机流起点与 Pilot 不同；该诊断样本和 mask smoke 都不混入 Pilot 或正式分析。
遮挡坐标、分割兼容处理和 hook 见 [遮挡说明](docs/masking.md)。
