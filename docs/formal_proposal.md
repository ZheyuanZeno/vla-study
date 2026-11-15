# 正式采集方案审阅（尚未批准）

当前可准备的方案是 4 tasks × 7 conditions × 50 native states = 1400 rollouts。没有启动正式采集；批准文件不会由准备脚本自动创建。

## 已固定的任务与方法

按用户确认的 Spatial 0–9，再 Object 0–9 顺序，首四个合格任务是 Spatial 0、1、2、3。每任务完整完成 20 次 Baseline，成功数分别为 18、16、18、19。筛选不读取遮挡结果。Object 未进入后续筛选，因此当前任务范围只覆盖这四个 Spatial 任务。

四个任务的 placement destination 都是 BDDL goal 中的 plate_1。目标由真实 geom segmentation 定义，目标方形为 reset 可见像素质心处达到 25%/50%/75% 覆盖的最小整数方形；同状态 T/B 同尺寸。灰色 RGB 128、硬边界、不透明，在外部相机 resize 后、SigLIP normalization 前持续应用。腕部相机、状态、语言和物理场景保持原流程。

背景 anchor：任务 0/2/3 为 [320,460]，任务 1 为 [190,470]。任务 1 最初 [320,460] 会盖住一只碗，已在任何该任务 Baseline 或遮挡 rollout 前更换为真实图像确认的空桌面位置，保留拒绝记录。全部四任务的全部 50 个原生 reset 状态通过三档几何审计。

固定 checkpoint revision、辅助 VLM revision、依赖、代码 hash、区域文件 hash、初始状态池 hash 和显式状态 ID 清单将写入 configs/formal50_protocol.json。模型不训练、不微调。batch=1、单环境、relative control、horizon=280、n_action_steps=1、flow steps=10、use_amp=false、TF32/cuDNN 开关与官方 evaluator 对齐。每 rollout 前重置模型/processor 缓存与 seed，条件顺序随机化并冻结。

## 原生池复用的边界

**原生 50 状态方案不是 50 个从未使用过的新状态。** 筛选使用 ID 0–19，额外 Pilot 使用 ID 20；正式候选池为原生 ID 0–49。正式统计只对每个状态计一次七条件 block，不把筛选或 Pilot 的结果合并进来，也不把同一确定性执行的重复次数当作新增独立状态。

这一方案需要明确披露开发/筛选与正式状态池重叠。若研究协议要求开发状态与正式状态完全分离，或者坚持首选 100 状态方案，应先构建并验证更多真实初始状态，再冻结新的方案，而不能把 50 个状态重复两遍。官方方法依据和待验证项见 docs/initial_state_generation.md。

## 数据与分析

保留所有 success、valid task failure 和 timeout。技术故障保留 attempt 记录并在相同实验单位重试；不完整 block 不进入最终分析。执行器去重、支持中断恢复。保存实际模型输入、分割、mask geometry、动作、真实 success predicate、时间、GPU 分配峰值、视频和版本标识。

正式分析独立运行。每任务计算条件 Success Rate 与 background-target Gap，再对四任务等权平均。5,000 次 bootstrap 在每任务内整块重采样 initial-state 七条件 block，输出 pointwise 95% percentile interval 及 Large Gap-Small Gap。筛选和 Pilot 结果均不进入正式统计。

## 检查与执行入口

四任务状态 20 的七条件 Pilot 全部完成并核验后，才能生成 proposal。准备脚本只写配置，不写批准：

```bash
source scripts/activate.sh
python scripts/build_formal_proposal.py
# 只读校验，不构建模拟环境、不运行模型
python scripts/run_formal.py --protocol configs/formal50_protocol.json
```

正式执行使用同一冻结协议和配置，且需要用户明确批准该 protocol hash 和恰好 1400 次采集。没有批准记录时，--run 会被拒绝；本轮不会运行 --run。完整正式数据齐备后，--analyze 才会检查计划完整性并生成真实统计和图像。

运行时间预算将来自四任务真实 Pilot 的耗时，标注单状态估计局限；不将其当作成功率或运行时间的总体保证。
