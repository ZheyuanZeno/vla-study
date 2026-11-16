# 仓库结构与恢复说明

此仓库保存 SmolVLA + LIBERO 遮挡鲁棒性项目的代码、固定环境定义、实验配置、中文文档及准备阶段的关键元数据。2026-11-16 整理上传时没有运行实验或测试，也没有批准正式采集。

## 内容布局

- `src/masking/`：图像遮挡 hook、分割与几何。
- `src/runner/`：真实 rollout、状态包恢复、配对核验和中断恢复。
- `src/analysis/`：独立的配对 bootstrap 与绘图实现。
- `scripts/`：环境安装、固定资源下载、准备与运行入口。
- `configs/`：版本、区域、开发/Pilot 配置和未批准的正式方案。
- `tests/`：实现验证代码；上传整理期间没有执行。
- `docs/`：安装、方法、实际验证范围及正式方案说明。
- `reproducibility/`：准备阶段元数据和冻结初始状态池的压缩快照，附逐文件 SHA256。

`artifacts/`、`logs/`、`.runtime/`、`.cache/` 和 `vendor/` 是本机生成内容，默认不直接纳入 Git。原始数据没有删除或移动。仓库不包含 Conda 环境目录、NVIDIA 驱动、模型权重、上游源码副本或 GitHub 登录凭据。

## 克隆后恢复环境

在 Ubuntu 22.04、本机可用的 NVIDIA 驱动与 Conda 基础上，按照 [安装文档](installation.md) 创建 `smolvla-libero` 环境、检出固定版本的 LeRobot，再安装固定依赖和资源。`environment.yml` 定义基础 Conda 环境，`constraints.txt` 固定关键安装约束，`requirements-lock.txt` 保留已验证环境的完整 pip 快照。锁定快照中的 LeRobot editable URL 指向固定 commit；安装流程见文档，不能只运行基础 `conda env create` 就视为全部依赖已装好。

激活脚本自动定位项目根目录。默认 Conda 安装目录为 `$HOME/miniconda3`；其他位置可先设置 `CONDA_BASE`，或在已初始化 Conda 的 shell 中执行：

```bash
source scripts/activate.sh
```

环境和源数据的历史记录可能含原机器绝对路径，它们是不可变的审计记录，不代表新机器必须使用相同用户名。安装脚本会生成新机器专用的 `.runtime/libero` 配置；不要复制原机器的运行配置或凭据。

## 元数据恢复包

`reproducibility/preparation-metadata.tar.gz` 保留当前两份 fresh 正式方案直接引用的状态池、池清单、区域审计、筛选和 Pilot 结果索引/核验记录，以及关键环境和检查日志。`reproducibility/manifest.json` 记录完整路径、字节数和 SHA256。文件按原始字节归档，没有重新计算或替换实验结果。

如需恢复，在**新克隆且尚无 artifacts/logs 的目录**中解包：

```bash
tar -xzf reproducibility/preparation-metadata.tar.gz
```

这是元数据快照，不是原始实验数据的完整备份。原始 episode 的实际输入数组、动作日志、图像、视频和逐候选生成图仍只在原机器。索引中的 `attempt_dir` 等路径会指向未随仓库上传的文件；仅凭恢复包不能重新执行完整输入/视频核验。若需要审阅这些证据，须另行复制原始 `artifacts/` 和 `logs/`，并保持目录结构。

已冻结 proposal 的代码 hash、依赖版本和数据引用保持原值。换机器后应按固定依赖恢复；审计 metadata 中的原始设备和环境路径保持历史含义。正式方案的只读检查不会创建批准，正式采集仍需要用户明确批准具体协议与条数。

## 当前边界

准备阶段已验证的结果与时间估计见 [正式方案](formal_proposal.md)。新生成初始化分布区别于官方固定 benchmark 状态表，见 [状态池说明](generated_state_pools.md)。没有正式统计数据，Pilot 成败不构成研究结论。

仓库初始可见性设为私有。尚未为本项目另行指定开源许可证；上游 LeRobot、LIBERO、模型及资产分别遵守各自许可证和使用条件。
