# 候选任务筛选

用户已确认顺序：LIBERO-Spatial 0–9，随后 LIBERO-Object 0–9。协议固定在 configs/screening/protocol.yaml。

依次检查：任务具有明确固定 placement destination、目标可见、三档同尺寸背景方形合法。目标区域来自真实 BDDL goal 和模拟器 geom segmentation；保存初始帧、目标 overlay 与三档遮挡预览，目视核对后冻结配置。对该任务全部 50 个原生 reset 状态检查几何合法性；这些是几何审计，不是 50 次模型 rollout。

几何合格后，固定使用状态 ID 0–19 运行 20 次未遮挡 Baseline，checkpoint、推理数值设置、控制模式和 horizon 保持不变。每任务完整收集 20 个有效结果后，以成功数至少 10 判断通过。技术故障单独保留并修复，不算任务失败，也不能通过跳过难样本得到 20 个结果。不得依据 T/B 效果选择任务。

按顺序保留最先四个同时满足几何与 Baseline 门槛的任务，然后停止筛选。Spatial 中若已找到四个，不再额外指定 Object 配额。所有筛选数据标记 screening，不进入正式结果；之前的 Pilot/Baseline 诊断数据也不计入本次 20 次。

```bash
source scripts/activate.sh
python -m src.runner.run --config configs/screening/libero_spatial_0.yaml
python scripts/update_screening_selection.py
```

单任务运行可用相同命令恢复；候选选择摘要在 artifacts/screening_v1/selection.json。几何证据在 configs/screening/eligibility/ 及其引用的真实图像和 audit.json。评估输入、动作、视频与每个 attempt 结果仍由原 runner 保存。

正式样本量与正式状态池还未批准，此筛选不授权正式采集。
