# SmolVLA + LIBERO 视觉遮挡鲁棒性实验

本机 Ubuntu 22.04.5 / RTX 3060 6GB，使用隔离 Conda 环境 smolvla-libero。全部计算在本机，不使用 Docker、远程推理或训练。

仓库包含代码、环境定义、冻结配置和准备阶段元数据。目录说明、克隆恢复步骤及原始数据的备份范围见 [仓库与恢复说明](docs/repository.md)。模型缓存、Conda 环境二进制和原始视频保留在本机。

真实 Baseline 已成功：LIBERO-Spatial 任务 0、状态 0，75 步，success predicate=True。证据在 artifacts/baseline/20261115T191502Z/，包含实际 camera inputs、动作、设备信息、结果和录像。单个样本不代表整体成功率。

同状态 T1/B1 真实测试通过。首个完整七条件开发 Pilot 已完成：6 次成功，T2 timeout；未删除失败样本。最新 v2 Pilot 对齐官方 TF32/cuDNN 开关，七条件均真实成功，并保留完整运行设置和恢复验证证据。旧诊断与 v2 不合并，重复同一状态不算新增独立样本。

## 从新终端运行

```bash
cd /home/zy/vla_study
source scripts/activate.sh
# 真实 import / CUDA / EGL / reset / step 检查
python scripts/check_environment.py --simulator
# 官方 evaluator + 只读证据记录，1 个 episode
bash scripts/run_baseline.sh
# 同一状态的 T1/B1，首次运行写入 mask_smoke_v2
python -m src.runner.run --config configs/mask_smoke.yaml
# 1 task × 1 state × 7 conditions；已完成项自动跳过
python -m src.runner.run --config configs/pilot.yaml
# 不运行模型，仅验证成组完整性、配对状态和实际 mask 输入
python -m src.runner.verify --root artifacts/pilot_block002
```

首次安装见 [安装文档](docs/installation.md)。已有 vlaBaseline、系统 Python、NVIDIA 驱动和内核均未改动。激活脚本禁用 user site，避免 ~/.local 中旧 ROS 包干扰。

## 配置、恢复与证据

runner 使用官方 LeRobot rollout、环境与 checkpoint processors；固定 policy、语言、相对控制、280-step horizon 和原 checkpoint 的 n_action_steps=1。每条件开始 reset 模拟器、policy queue 和 processors；seed 在每次 rollout 前重置，条件顺序用独立 order_seed 随机化。官方 Phase 2 的 seed 在模型初始化前设置，随机流起点与 Pilot 不同，该诊断样本不混入 Pilot。

配置文件选定 suite/task、显式 state_ids、conditions 和输出目录。`--stop-after 1` 可在一条新 rollout 完成后停止；去掉该参数恢复，完整条件顺序不变。每个 attempt 单独保存；技术失败和未完成中断会保留，不能变成 success=False 的任务失败。success、valid_task_failure、task_timeout、infrastructure_failure 分开记录。相同配置/执行代码可恢复；执行代码或配置改变须用新输出目录，禁止覆盖已有结果。

每个输出目录包含 manifest.json、inference_settings.json、execution_order.json、results.jsonl、completeness.json。episode 目录含真实 initial/settled state、实际 resize 后 original/masked RGB float arrays 和 PNG、目标分割、bbox/面积/覆盖率、真实动作、结果与视频。JSONL 是可重建索引，原始 result.json 为逐 attempt 原子写入记录。

小规模执行默认最多 7 次。多状态 Pilot 须先通过真实七条件 gate，再显式设置 max_rollouts、pilot_gate_dir 和状态子集。筛选可使用单任务 Baseline 配置（20 状态，max_rollouts=20）；用户已确认 Spatial 0–9、再 Object 0–9；筛选已完成并选定 Spatial 0–3，详情见 docs/screening.md。正式执行器要求用户明确批准冻结方案，不会自动执行 1400/2800 次。

原生 Spatial/Object 各 10 个任务均有 50 个精确不同的初始状态；完整 inventory 在 artifacts/state_inventory.json。不通过取模或重复同一确定性执行凑 100 个样本。

## 分析与测试

分析与模拟器解耦，正式默认只读 formal 数据并要求四任务完整配对；5,000 次任务内整块 bootstrap 后四任务等权汇总。Pilot 与正式数据不混合，synthetic fixtures 只用于测试。分析命令和四类图见 [分析说明](docs/analysis.md)。目前没有正式实验数据。

```bash
source scripts/activate.sh
python -m pytest tests -q
```

区域依据、NumPy 2 分割兼容处理、精确 hook 和几何约定见 [遮挡说明](docs/masking.md)。正式任务、样本量、状态清单和批准状态见 [协议准备状态](docs/protocol_readiness.md) 与 configs/protocol_draft.yaml。原始需求完整保留在 docs/task_specification.txt。

四个任务的冻结背景 anchor 均在各自全部 50 个原生状态的真实 reset/segmentation 上通过三档几何审计；审计不等同于 policy rollout。

## 2026-11-16 验收归档

已按确认顺序选定 Spatial 0–3，80 次 Baseline 筛选成功数分别为 18/20、16/20、18/20、19/20。预定状态 20 的跨任务七条件 Pilot 共 28 次，21 次成功、7 次 timeout，技术故障为 0；全部完整性、配对与实际输入检查通过。

新池已生成并验证：每任务 101 个完整状态包，ID 0–99 留作正式候选，ID 100 专用 Pilot；与原生开发姿态去重。新池真实 Pilot 共 28 条，25 success、3 timeout、0 技术故障，完整状态/随机数/实际输入核验与全部视频解码通过。25 项测试通过。

首选 2800 条约 34.4 小时，备选 1400 条约 17.2 小时（小 Pilot 均时加 15% 预留，估计有限）。两份方案均通过只读检查，尚无批准文件或正式数据。见 [冻结方案审阅](docs/formal_proposal.md)、[100 状态扩展池](docs/generated_state_pools.md)。旧 configs/formal50_protocol.json 是旧代码下原生池复用的历史归档。

```bash
source scripts/activate.sh
# 已完成 Pilot 可核验/恢复；只使用保留 ID 100
python -m src.runner.run --config configs/pilot_fresh/libero_spatial_0.yaml
python -m src.runner.verify --root artifacts/pilot_fresh_v1/libero_spatial_0
# 当前冻结方案只读检查，不启动正式采集
python scripts/run_formal.py --protocol configs/formal_fresh100_protocol.json
python scripts/run_formal.py --protocol configs/formal_fresh50_protocol.json
```

正式 1400/2800 次采集需要用户明确批准具体方案，准备脚本不自动批准。
