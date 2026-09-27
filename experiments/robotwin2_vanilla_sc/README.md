# RoboTwin 2.0 普通自条件扩散策略实验

本目录记录在 RoboTwin 2.0 官方图像版 DP（Diffusion Policy，扩散策略）上加入 Vanilla Self-Conditioning（普通自条件）的实现与评测。

## 上游版本

- RoboTwin：`RoboTwin-Platform/RoboTwin`，基线提交 `ea8b211`。
- XPolicyLab 子模块：基线提交 `fa431ecd893ee706883e64fe5fe1464ec8cd928d`。
- `source_snapshot/` 保存本次修改后的三个源文件；`patches/xpolicylab_dp_vanilla_sc.patch` 可在上述 XPolicyLab 版本根目录用 `git apply` 应用。

## 已完成结果：beat_block_hammer

| 任务 | 数据 | 机器人 | 示范数 | 训练轮数 / 更新步 | 评测回合 | 成功数 | 成功率 | 跳过种子 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `beat_block_hammer` | `demo_clean` | Aloha-AgileX | 50 | 600 / 25,800 | 100 | 19 | 19% | 33 |

模型参数量为 96,822,734（U-Net 为 85,646,222）。评测结果和原始评测日志位于 `results/beat_block_hammer/` 与 `artifacts/eval.log`。评测没有生成 MP4 视频。

## 其他任务进度

本次记录时 `handover_block` 与 `stack_bowls_three` 的训练仍在运行，`in_progress/` 中附有配置和日志快照；这些不是最终结果。`pick_dual_bottles` 尚无训练结果。

## 代码与配置

仅加入普通自条件：训练时以 50% 概率使用同一扩散时间步的 detached feedback（已停止梯度的反馈），另 50% 使用零反馈；推理时把干净动作估计传到下一去噪步，保持官方 scheduler（调度器）和训练损失不变。启用后动作维度为 14，U-Net 输入/输出为 28/14。

原始配置、解析后的配置和 Hydra（配置管理框架）运行快照在 `config/`；完整训练及评测日志在 `artifacts/`；实验计划和过程记录在 `notes/`。

## 发布内容范围

包含代码快照、补丁、配置、日志和结果表；不包含模型 checkpoint（模型检查点）、任务数据集或处理后的数据集。`results/robotwin_results.csv` 可直接作为结果表格使用。
