# DEBUG.md — 走到厨房 live 空参数 (campaign #4 batch 2 收官阻塞)

## OBSERVE
- Repro: GUI smoke (artifacts/smoke-20260612-155941) — "走到厨房" 两次尝试均
  "navigate_to requires a label OR numeric x and y" 0.0s; 树显示
  strategy=navigate_to_skill, verify=visited('kitchen'), params 不可见。
- R6 修复 (null 剥离+backfill _has) 与 R7 修复 (param pass) 已在 HEAD, 仍复现。
- 注意: navigate label 声明 required: False (label OR x/y 二选一) → R7 param
  pass 检不出 missing — 只有 backfill 能救, 而 backfill 需要 verify 在 parse
  时刻已是 visited(...)。
- 离线实验 (R9, 简化 vocab): deepseek 原始 JSON = strategy "navigate_skill",
  verify "nearest_room() == 'kitchen'" (invalid → 步被 DROP), params
  {"room": "kitchen"} — 简化 vocab 下根本不教 visited。LIVE vocab 教 visited
  (navigate_to_point.py:52 verify_hint), 故 live 原始 JSON 形状未知 — 必须抓。

## HYPOTHESIZE (R10 按序证伪)
| # | 假设 | 证据 |
|---|------|------|
| H-O | live 原始 JSON 的 params 用错键 (如 {"room": "kitchen"}), verify 同时是别的 (非 visited) → 步被 verify validator 改写/replan 后 verify 才变 visited, backfill 时刻匹配不上 | 离线实验显示 deepseek 爱用 room 键 + 自造 verify |
| H-P | live 原始 verify 是 visited 但 params 用 {"room": ...} → backfill 只查 label/x/y, 设了 label — 应该成功 (若证实 = backfill 没跑, 查调用序) | 代码读 |
| H-Q | _build_goal_tree → _validate_sub_goal 在某条件下重建 SubGoal 丢 params (e.g. foreach/answer 分支) | 未读全 |
| H-R | engine 层 (run_turn_unified/observation) 展示的树 ≠ 执行的树 (展示后处理) | 0.0s + 两层显示差异 |

## EXPERIMENT (R10 首项)
决定性: headless boot_habitat_agent (VECTOR_HABITAT_GUI=0) + 真 registry vocab
(engine._build_registry_vocab_kwargs) + tap backend 抓 raw → 对照 final tree
params。一次实验同时裁 H-O/H-P/H-Q。

## CONCLUDE
(待 R10)

## CONCLUDE (R10)
- 全部 R9 假设 (H-O/P/Q/R) 证伪: live 原始 JSON 完美 ({"label": "kitchen"},
  verify visited('kitchen')) — decompose/backfill/param pass 从未坏过。
- 真相链 (raw 日志 + --verbose 重放):
  1. 首次尝试真实错误 = "no object matching 'kitchen'; world model EMPTY —
     start sysnav": **apartment 预设 rooms 设计上为空** (catalog 注释明示,
     语义房间等 HM3D/DQ-1) → seed_room_landmarks 种了 0 个房间。行为诚实正确;
     campaign #3 的厨房 GUI 战果在带房间的场景上。
  2. **真内核 bug**: Layer-1 重试清空 strategy → selector Priority-3 alias
     match 用 `{}` 重路由 → label 丢失 → 后续尝试降级为 "requires a label"。
  3. **误导面**: 步视图只显示末次错误 → 真因被埋, 误导了 R5/R6/R9 三轮诊断。
- 修复: (a) alias match 继承 sub_goal.strategy_params (重路由绝不剥绑定);
  (b) 末错≠首错时 error 附加 "[attempt 1: ...]" + result_data.attempt_errors;
  (c) smoke 脚本 --scenario, 默认 house (有语义房间)。
- 回归测试: tests/unit/vcli/test_retry_fidelity.py (4)。
- file:line: strategy_selector.py:213 (alias match), vgg_harness.py 重试循环。
