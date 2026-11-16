# 100 状态首选方案：真实扩展池验证

2026-11-16，使用已筛选的 Spatial 0–3，按 configs/state_generation/fresh101_v1.yaml 的固定种子计划生成。全部计算在本机，生成时没有调用 SmolVLA，也没有查看或筛选 policy 的成功率。

每任务得到 101 个合格状态：正式候选 ID 0–99 共 100 个，ID 100 专用于七条件 Pilot。四任务共 404 个状态包，正式候选 400 个，Pilot 保留 4 个。前 100 个状态不会用于本次 Pilot；原生开发池的 Baseline/Pilot 结果不进入新正式方案。

## 为什么状态向量之外还要保存 seed

实测 MuJoCo 的 92 维 flattened state 包含 time、qpos、qvel，却不包含所有 model.body_pos/body_quat。LIBERO 随机 reset 会改变固定设施位姿。只恢复向量时，动态状态可以恰好相同，设施位置仍可能差约 1 厘米。

状态包因而包含环境 reset seed、原始动态状态、10 步 settling 后的参考状态、固定设施/模型 body transforms、可动物体的世界坐标与旋转矩阵。重载时先用该状态的 seed 经官方 reset 重建模型，再 set_init_state，使用相同的 10 步 no-op settling。policy 的 Torch seed 独立固定为 42，环境 seed 逐状态保存并在七条件之间保持一致。

真实生成审计中，404 个状态的动态状态、固定设施和物体位置重载误差均为 0；rollout 接入时还逐次核对物体旋转矩阵。记录见 artifacts/generated_states_v1/validation_summary.json 及各任务 pool.json。初始目标已完成、几何不合格或物体位置近重复的候选按预先规则排除；数值异常、重载错误等技术故障会停止，不会伪装为合格状态。

## 区别于换 seed 重跑同一姿态

去重依据是五个可动物体 settling 后的世界位置，不以 time、机器人噪声或 seed 字符串作为新增样本依据。比较所有物体 x/y/z 的最大绝对坐标差，预先设置小于 0.1mm 为近重复。相对同任务全部 50 个原生开发状态和已接受的新状态，实际最小布局距离约 3.49mm。独立采样仍是统计假设，不能仅由 hash 不同来证明。

生成共出现 1 个几何排除：Spatial 2 的 seed 2026131615，大号背景框覆盖了非背景像素。原始图像、状态和原因完整保留；没有根据 policy 成败排除任何候选，没有移动已冻结的 anchor。其余候选均通过，技术故障为 0。

本池来自同一 LIBERO BDDL 任务的 seeded reset，并通过初始目标未完成、目标可见、三档背景合法、无近重复与准确重载检查。它不是原 benchmark 发布的固定状态列表；正式结果应明确说明这一初始化分布。

## 运行与审计

```bash
source scripts/activate.sh
# 已完成时校验现有池，不重新生成
python scripts/generate_state_pool.py --task-id 0
python scripts/report_state_pools.py
# 仅使用保留 ID 100，不允许使用正式 ID 0–99 做 Pilot 成败检查
python -m src.runner.run --config configs/pilot_fresh/libero_spatial_0.yaml
python -m src.runner.verify --root artifacts/pilot_fresh_v1/libero_spatial_0
```

执行器记录 raw state、完整状态包 ID、设施快照、环境 seed、policy seed、CUDA RNG 起点和重载误差；自动 cleanup reset 也回到当前单位，保持 seed/设施一致。分析程序拒绝七条件完整状态包不同或同一物理单位换 ID 重复计数。

生成审计不等于完整 model rollout。保留状态的真实七条件 Pilot 已完成 28 条（25 success、3 timeout、0 技术故障），完整状态包、随机数起点、实际输入与全部录像核验通过。新的 100 状态/2800 次及取前 50 状态/1400 次方案均已冻结并通过只读检查，见 [方案审阅](formal_proposal.md)。两者均需要用户明确批准后才执行正式采集。

原生 50 状态方案的归档对应当时的执行代码版本；新增池接入后已冻结新的 100 状态方案和新的 50 状态子集方案。生成清单中的 policy_inference_performed=false 指生成/筛选过程没有调用 policy；后续专用 Pilot ID 100 的实际使用由独立 rollout 日志记录，清单保持不可变。
