# RoboTwin 2.0 对照评测结果

评测设置：`demo_clean`，50 条示范，Aloha-AgileX，seed 42，joint 控制，目标 100 episodes（回合），seen instruction（已见指令）。下表的成功率直接取自服务器 `_result.txt`。

| 任务 | Vanilla Self-Conditioning（普通自条件） | Original DP（原始扩散策略） | 差值（百分点） | 跳过种子（SC / DP） |
|---|---:|---:|---:|---:|
| `beat_block_hammer` | 19% | 25% | -6 | 33 / 33 |
| `handover_block` | 20% | 18% | +2 | 9 / 9 |
| `stack_bowls_three` | 50% | 57% | -7 | 73 / 73 |
| `pick_dual_bottles` | 21% | 24% | -3 | 2 / 2 |
| 四任务等权平均 | 27.5% | 31% | -3.5 | — |

两个方法的每个任务跳过种子数相同；评测摘要只保存跳过数量，未记录具体 seed ID（种子编号）。结果原文分别保存在各任务的 `vanilla_self_conditioning/` 和 `original_dp/` 子目录中。
