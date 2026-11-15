# 分析协议与命令

分析模块不导入模拟器、策略或 GPU 库，独立读取 JSONL。正式分析默认要求四个任务，拒绝 incomplete blocks、重复完成条件、不同 checkpoint revision、状态配对错误和 synthetic 数据。technical failure 保留于技术故障统计；尚未补齐的整个 block 会阻止分析，不把故障当任务失败。valid task failure 与 timeout 都计为 success=0。

每个任务先按 initial-state block 得到七条件 Success Rate 和三个 `SR_background - SR_target`。四任务等权平均，不按 episode 数量加权。

bootstrap 默认 5,000 次。每个任务内部有放回重采样初始状态，七条件作为同一整块移动。每个 replicate 再等权平均四任务。置信区间为 pointwise 95% percentile，不是 simultaneous interval。Large Gap - Small Gap 在同一 bootstrap replicate 内计算，保留相关性。

```bash
source scripts/activate.sh
python -m src.runner.verify --root artifacts/pilot_block001
# 正式四任务数据齐全后才能运行；当前没有正式数据。
python -m src.analysis.run \
  --input TASK1/results.jsonl TASK2/results.jsonl TASK3/results.jsonl TASK4/results.jsonl \
  --output artifacts/formal_analysis
```

输出 statistics.json 及四类 PNG：severity_success_rates、per_task_success_rates、paired_gaps、baseline_comparison。

Pilot 与 formal 分开：正式默认 phase=formal，绝不合并 pilot。需要纯开发检查 Pilot 分析接口时，显式使用 `--phase pilot --expected-tasks 1`；一个状态的 bootstrap 区间退化，不能用于研究结论。当前验收以真实 rollout/输入核验为主，不将单状态 Pilot 的点估计或区间当成最终结论。

测试 tests/test_analysis.py 的数据全部显式标记 synthetic，只用于检验配对、等权、拒绝不完整/重复数据和阶段隔离。它们不是实验结果。
