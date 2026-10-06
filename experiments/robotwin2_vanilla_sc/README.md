# RoboTwin 2.0 普通自条件扩散策略实验

本目录记录在 RoboTwin 2.0 官方图像版 DP（Diffusion Policy，扩散策略）上加入 Vanilla Self-Conditioning（普通自条件）的实现与评测。

## 对照评测

四个任务均完成 Vanilla Self-Conditioning 与 Original DP（原始扩散策略）评测。两者使用 `demo_clean`、50 条示范、Aloha-AgileX、seed 42、joint 控制，评测目标为 100 回合，指令类型为 seen（已见）。训练配置审查显示，除 `policy.use_self_condition`（是否启用自条件）外保持一致；Original DP 数据集别名指向同一任务数据。

| 任务 | 普通自条件 | 原始 DP | 差值（百分点） | 跳过种子（SC / DP） |
|---|---:|---:|---:|---:|
| `beat_block_hammer` | 19% | 25% | -6 | 33 / 33 |
| `handover_block` | 20% | 18% | +2 | 9 / 9 |
| `stack_bowls_three` | 50% | 57% | -7 | 73 / 73 |
| `pick_dual_bottles` | 21% | 24% | -3 | 2 / 2 |
| 四任务等权平均 | 27.5% | 31% | -3.5 | — |

评测摘要显示每对方法的跳过种子数量相同，但没有保存具体 seed ID（种子编号）；结果按原始 `_result.txt` 中的成功率记录，不推断跳过种子的身份。表格文件为 `results/robotwin_comparison.csv`，便于阅读的版本为 `results/comparison.md`，每项服务器原始摘要保存在 `results/<task>/<method>/_result.txt`。本目录没有评测 MP4 视频。

## 上游版本与实现

- RoboTwin：`RoboTwin-Platform/RoboTwin`，基线提交 `ea8b211`。
- XPolicyLab 子模块：基线提交 `fa431ecd893ee706883e64fe5fe1464ec8cd928d`。
- `source_snapshot/` 保存修改后的三个源文件；`patches/xpolicylab_dp_vanilla_sc.patch` 可在上述 XPolicyLab 版本根目录应用。
- 方法只加入普通自条件：训练时以 50% 概率使用同一扩散时间步的 detached feedback（停止梯度的反馈），另外 50% 使用零反馈；推理时将干净动作估计传到下一去噪步。官方 scheduler（调度器）和训练损失保持不变。

## 文件范围

目录包含代码快照、补丁、配置、日志和评测结果，不包含模型 checkpoint（模型检查点）、任务数据集或处理后的数据集。训练和评测日志快照在 `artifacts/` 与 `in_progress/`。
