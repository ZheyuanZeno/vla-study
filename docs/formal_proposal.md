# 正式采集方案审阅（2026-11-16，尚未批准）

首选为 4 tasks × 100 fresh states × 7 conditions = 2800 rollouts；备选取同一新池前 50 状态，为 1400 条。两份冻结方案均已通过只读验证，没有创建批准记录，没有启动正式采集。

## 已固定的任务与方法

按用户确认的 Spatial 0–9、再 Object 0–9 顺序选出首四个合格任务 Spatial 0–3，每任务 20 次 Baseline 成功数分别为 18、16、18、19。没有按遮挡差异选择任务，Object 无需继续筛选。

四任务目标均为 BDDL 中的 plate_1。使用真实 geom segmentation 的可见目标像素质心，求达到 25%/50%/75% 覆盖的最小整数方形；同状态 T/B 同尺寸。外部相机 resize 后、SigLIP normalization 前应用 RGB 128 不透明硬边 mask。背景 anchor 为任务 0/2/3 的 [320,460]、任务 1 的 [190,470]，已冻结并保留原始几何审计与拒绝记录。

checkpoint、基础 VLM、依赖、代码、区域、数据和证据 hash 均已写入方案。batch=1，单环境，relative control，horizon=280，n_action_steps=1，flow steps=10，use_amp=false；TF32/cuDNN 与官方 evaluator 对齐。policy seed=42，条件顺序由 order_seed=20261115 冻结。环境 reset seed 随状态包保存并在七条件间相同。

## 新状态与真实验收

每任务 101 个状态包中 ID 0–99 为正式候选，ID 100 为 Pilot，二者分离。所有新布局与同任务原生 50 个开发布局及已接受新布局按可动物体位置去重。404 个状态通过真实重载；生成仅排除 1 个背景几何不合法候选，没有使用 policy 结果筛状态。它们是同一 BDDL 的 seeded reset 分布，**不是官方 benchmark 固定初始状态列表**；详见 [新池说明](generated_state_pools.md)。

预留 ID 100 的四任务真实七条件 Pilot 共 28 条，25 success、3 timeout、0 技术故障；逐任务成功数为 7、4、7、7。全部配对动态/设施最大差为 0，完整状态包、CUDA RNG 起点、实际输入及池引用均核验通过，28 个视频逐帧解码且帧数等于实际 env_steps+1。首次单条停止后成功恢复，无重复计数。Pilot 仅每任务一个状态，不能推断整体遮挡效果。

证据：artifacts/pilot_fresh_v1/validation_summary.json、各任务 verification.json、video_verification.json；日志 logs/pilot-fresh-spatial-*.log、logs/verify-pilot-fresh-*.log。25 项测试通过，见 logs/tests-fresh-pilot-frozen.log；synthetic fixtures 不进入实验报告。

## 时间与资源

真实 Pilot 均时 38.44 秒/条，PyTorch 分配峰值 1.195 GiB；该指标不包含桌面/EGL 等整卡占用。此前模型运行时整卡占用约 3.1GB，6GB GPU 已通过真实运行。

按均时加 15% 预留估算：2800 条约 34.4 小时，1400 条约 17.2 小时。每任务一个状态的耗时样本有限，失败率与时长均不是总体保证；实测 rollout 均时包含 reset/准备/推理/step，不包含初次加载和末尾录像编码。可中断后按相同协议恢复。

## 冻结方案与只读检查

- 首选：[formal_fresh100_protocol.json](../configs/formal_fresh100_protocol.json)，SHA256 `f9e74055a4a593194ccf1bdb47b89b5cafbd7abf82ebaab7dc711730e71e4f4e`。
- 备选：[formal_fresh50_protocol.json](../configs/formal_fresh50_protocol.json)，SHA256 `894b1ed240e8e9de35a322d0baea1a7d7084f2e863090ad7352076de83f61092`。
- 执行代码 SHA256：`700c87eae7a77096824697d9c40f4868c2ca2ce5001a94f420dda1256cd0b334`。

```bash
source scripts/activate.sh
# 只读校验，不创建模拟环境、不调用模型
python scripts/run_formal.py --protocol configs/formal_fresh100_protocol.json
python scripts/run_formal.py --protocol configs/formal_fresh50_protocol.json
# 解码已有真实 Pilot 录像
python scripts/verify_pilot_videos.py
```

两份检查均返回 checks_passed_except_approval=true、approval_file_exists=false、formal_rollouts_started=false。结果见 logs/check-formal-fresh100.log 和 logs/check-formal-fresh50.log。

正式执行的 --run 与 --analyze 入口要求 `.runtime/formal_approval.json` 记录用户明确批准、对应 protocol SHA 和精确条数；准备工具不写该文件。批准任一方案后只执行该方案，不叠加执行两个样本量。旧 configs/formal50_protocol.json 是旧代码版本的原生池复用归档，不是当前入口。

## 正式分析

保留 success、任务失败和 timeout；技术故障保留 attempt 后在同单位重试，不完整 block 禁止分析。按任务内七条件初始状态整块 bootstrap 5000 次、四任务等权汇总，输出点态 95% percentile interval、background−target Gap 与 Large Gap−Small Gap。筛选、Pilot 和 synthetic 数据均不进入正式统计。当前没有正式统计结果。

正式启动需要明确批准，是原始任务第十节“未经我的明确批准，不要自动启动 1400 或 2800 次正式实验”的要求。准备工作已完成，等待该批准。
