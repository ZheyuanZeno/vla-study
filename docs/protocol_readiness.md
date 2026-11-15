# 正式实验前的当前状态

本文件是准备状态，不能视为正式采集批准。正式 runner 入口仍禁用。

已完成：原生 Ubuntu/CUDA/EGL 与真实模型验证；任务 0 的真实 Baseline；同状态 T1/B1；任务 0 的七条件 Pilot；20 个候选任务的状态池读取；单元测试验证 paired bootstrap 和统计隔离。最新 Pilot 的运行设置记录在其 inference_settings.json，阶段证据分别保留，不合并成独立样本。

尚未完成：Research Plan 规定的候选任务 suite/ID 顺序确认；按该顺序做每任务 20 次 Baseline 筛选；四任务的 placement destination 与背景 anchor 验证；四任务的状态列表与排除准则冻结；用户明确批准正式样本量。不会根据哪个任务遮挡差异更大进行选择。

## 状态数量限制

实际读取 LIBERO-Spatial 和 LIBERO-Object 共 20 个任务的原生初始状态文件，每个都是 50 个，精确去重后仍为 50。详见 artifacts/state_inventory.json 和 logs/state-inventory.log。数组不同不等于已经证明统计独立，但重复同一状态/确定性执行不能增加独立样本数。

本次未生成新初始状态。现有 benchmark 官方路径是加载固定 init-state 集，再用 set_init_state 精确配对。不能通过取模或换 seed 冒充 100 个新状态。100 状态方案需要另行验证生成分布、合法 reset、去重和可复现性；50 状态备选也仍需协议批准，并明确 Pilot/筛选与正式状态池的关系。

## 运行规模与故障

现有短 Pilot 只覆盖一个任务/状态，不能将它的单次成功率、均时或技术故障率推广到正式四任务。正式时间预算需以候选任务的 screening 和跨状态 Pilot 测得耗时计算，并留出 timeout、重试、加载与录像开销。

PyTorch 显存分配峰值与 nvidia-smi 整机占用口径不同，后者还包含桌面和 EGL 等。正式预算同时保存这两种口径，不能用分配峰值代替整个 GPU 的使用量。

## 待冻结内容

草案见 configs/protocol_draft.yaml：checkpoint/基础 VLM/依赖版本、四任务及筛选排除记录、目标分割与 background anchors、初始状态列表和独立单位、原始推理数值设置、horizon、action execution horizon、样本量、故障修复与分析规则。

正式统计代码已实现，但尚无正式数据；单元测试所用 synthetic fixtures 不进入任何实验报告。只有在四任务与样本量等协议完整、并获用户明确批准后，才能启用和执行 1400/2800 次正式采集。

## 本次最终验收证据

- 最新 Pilot：artifacts/pilot_block002，七条件全部完成且 success=True；7 次不是 7 个独立状态。
- 恢复：先 --stop-after 1 完成 B3，随后恢复剩余六条；再次执行跳过全部已完成项，总 attempt 数仍为 7。
- 七次 settled state 最大绝对差为 0，实际模型遮挡输入核验通过，七个录像都可解码。
- v2 均时约 25.19 秒/rollout，PyTorch 分配峰值约 1.195GiB。该均时包含 reset/区域准备/推理/step，不含模型加载和末尾视频编码。
- 旧 v1 的 T2 timeout 完整保留；它未显式固定 TF32/cuDNN 数值开关，仅作开发诊断，不与 v2 合并。
- 任务 0 在全部 50 个原生状态上完成真实 reset/segmentation 区域审计，三档 T/B 均合法；这不是 50 次 policy rollout。证据为 artifacts/region_audit_task0/audit.json。
- 14 项单元/导出测试通过，日志 logs/tests-final.log；统计 fixtures 明确 synthetic，不是实验数据。

Phase 1–3 的真实运行证据已齐备；Phase 4 的单状态七条件 Pilot 已通过，但任务正式筛选尚未执行。Phase 5 的执行器和统计模块已实现并验证上述范围，尚无四任务正式统计。Phase 6 是未完成草案，等待任务顺序、筛选、状态池和样本量等协议确认。
