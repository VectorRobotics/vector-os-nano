# Vector OS Nano 代码库设计审查与优化方案 (2026-06-12)

## 一、根因画像 — 为什么 bug 一直在同一个家族里

这三天反复出现的 "PASS 但什么都没发生"，根源只有一个:**整个验证体系只看"终态"，从不看"变化"** — 内核 (goal_executor.py:341) 在执行后才评估 verify 谓词,从不在执行前留基线,所以"本来就满足"和"动作真的生效"在数据结构 (StepRecord) 里是同一个 True;下面每一层 (habitat server、sysnav_bridge、各 skill) 又各自重复实现了"已到达就提前退出"的静默捷径,且结果里不带 moved/duration/already_there 任何运动证据。雪上加霜的是,本应兜底的护城河 (evidence gate) 在所有 playground 世界被 `is_robot=True` 一刀切关闭 (trace_store.py:243),而 verify 谓词本身又有两套手写来源 (`_VERIFY_MAP` vs `verify_hint`) 且大量缺省为 `"True"` 哨兵 — 等于运动类指令在快路径上**结构性地无验证**。第二个家族是**跨层契约漂移**:同一个参数/语义在 词表→选择器→原语/技能→世界 之间被改名不改义 (turn 的度数被当弧度)、改义不报错 (枚举默认值兜底、未知参数静默丢弃)、或承诺没实现 (harness "依赖失败会跳过" 的注释对应的代码不存在)。"走到厨房"种子 bug 是两个家族的交点:房间是区域却被压成点 + 物体 standoff 语义 (tol 1.5/verify 1.6 ≥ 初始距离 1.53m),初始状态即满足 verify → 0 运动 PASS。点修永远修不完,因为缺的是**机制级不变量**,不是某一处代码。

## 二、确认的设计缺陷清单 (按优先级)

26 条发现去重合并为 18 项。严重度取对抗复核后的 verdict 值。

### P0 — false-PASS 内核家族 (critical)

| # | 标题 | 文件 | 严重度 | 本质 | 修法方向 |
|---|---|---|---|---|---|
| 1 | verify 只在执行后跑一次,无前置基线 | vcli/cognitive/goal_executor.py:341/441 | critical | 任何初始状态已满足谓词的目标退化为 0 秒 no-op PASS,无"本来就在"的诚实信号 (种子 bug 的内核形态) | 执行前先评估谓词,StepRecord 加 `pre_satisfied` 字段 (规则 6 增量式);pre 已 True 则上报"已满足、未动作";动作步要求状态差 (位移/verify 由假变真) 才算证据 |
| 2 | habitat server navigate_to 先查到达再迈步,且 reached 按 tol×2.0 打分 | playground/habitat/server.py:551/581 | critical | 到达判定先于运动 + tol=1.5 时 3.0m 内即"reached",返回不含 moved/elapsed/already_there,上层无从分辨 | 返回三值契约 {reached, already_there, moved_m, elapsed_s};reached 严格按调用方 tol;中间路点用固定小阈值,tol 只用于终点 |
| 3 | 房间目标套用物体 standoff 语义 | skills/navigate_to_point.py:145 + vcli/habitat_runtime.py:202 | critical | seed_room_landmarks 把房间矩形压成中心点,navigate 对所有 label 一律 tol≥1.5/verify at_position(1.6) — 而正确的 `visited(room)` 区域内谓词早已存在并绑定却无人使用 (playground/world.py:156) | landmark 携带 rect 于 properties;type=room 分支:目标取矩形内可导航点、verify 用 `visited('<room>')`;1.5 standoff 仅留给 type=object。注意 test_room_landmarks.py:87-90 把错误契约固化进了测试,需同步改 |

### P1 — 护城河失效与 verify 双脑 (high)

| # | 标题 | 文件 | 严重度 | 本质 | 修法方向 |
|---|---|---|---|---|---|
| 4 | `is_robot=True` 整体绕过 evidence gate,playground 恰好全是 True | vcli/cognitive/trace_store.py:243/276 + playground/world.py:86 | high | 在 owner 实测的世界里 verified ≡ trace.success,护城河结构性关闭;goal_executor W1.1 文档声称的行为与实际接线矛盾 (漂移而非设计) | 改为按步豁免:世界注册"确无符号后置条件的策略集合" (类比 answer_only),playground 声明空集;被豁免步上报 "completed (unverified)" 而非 verified |
| 5 | verify 来源双脑:内核 `_VERIFY_MAP` 手写 + 默认 "True",忽略 skill.verify_hint;`to_schemas` 把未声明 hint 全部缺省成 "True" | vcli/engine.py:1218 + core/skill.py:380 | high | 违反规则 3 (单一来源):快路径与 LLM 路径产出不同 verify;全部 go2 运动技能 (walk/turn/explore/navigate/stop…) 两条路径都拿到哨兵 "True" — 运动指令整体无验证 | 删 _VERIFY_MAP,快路径读同一 verify_hint;注册时 hint 必填或显式打 `unverified` 标签;给运动技能补真实谓词 (walk: 位移 ≥ 请求×0.5;turn: 朝向差;stop: 速度==0) |
| 6 | WalkSkill 成功不蕴含行走 | skills/go2/walk.py:117/140 | high | 0 距离/0 速度/撞墙原地 (try_step 零位移) 全部 success=True,result_data 回显**指令**距离冒充测量,无起点位姿 | 命令前后采位姿,result_data 带 start/moved_m;moved≪requested 报失败或打标;distance≤0 走 bad_params |
| 7 | 横移方向在 habitat 底盘是保证性 no-op PASS | skills/go2/walk.py:106 + playground/habitat/base.py:88 | high | base 丢弃 vy 仅留日志返回 True,skill 从不查 `supports_holonomic` — "往左走一米"=3.3 秒静止+success | execute 时检查 base 能力,横移不支持则 fail loud (diagnosis=lateral_unsupported,建议 turn+walk);发布给 planner 的枚举按能力过滤 |
| 8 | sysnav navigate_to 用欧氏距离判到达 | playground/habitat/sysnav_bridge.py:345 | high | 穿墙 false PASS (隔墙 1.5m 内即 reached) + 首次轮询即可零运动通过;verify 的 at_position 同为欧氏,救不回来 | 进度判定保持 oracle-free,终点到达用一次 geodesic_distance (verify 侧 oracle 已被许可,base.py:219);并返回 already_there/moved |
| 9 | navigate_to_point 的 success ≡ 传输层 reached 标志,result_data 无运动证据 | skills/navigate_to_point.py:169 | high | "drove there" 与 "never moved" 返回完全相同的形状;verify/replan/用户报告三者都分辨不了 | 与 #2/#6 同一契约:所有运动技能的 SkillResult 强制携带前后状态 (start/moved_m/duration_s/already_there) |
| 10 | foreach source_path 解析失败 = 0 次迭代 + 整树 PASS | vcli/cognitive/goal_executor.py:585 | high | 空展开使 owner 的 verify 成为死代码,且不产生 FailureRecord,replan 永远无法纠正 — 闭环在此结构性断开 | 区分"解析为空列表" (诚实上报 0 项后通过) 与"路径未解析" (failure_class=exec_error,附 producer 实际可用键名喂给 replan) |
| 11 | harness "依赖失败会跳过"是幻影代码;harness 与 executor 失败语义相反 | vcli/cognitive/vgg_harness.py:579 | high | 注释承诺的 skip 机制不存在 → navigate 失败后照样从错误位置真实执行 pick (硬件上即"错误状态下动作"危险);executor break、harness continue,同一棵树两套语义;max_redecompose 是死配置 | 实现承诺:传递失败步集合,传递依赖失败者发 skipped StepRecord (新 failure_class);统一两条路径语义;删除 max_redecompose 或实现之 |
| 12 | turn 的 angle 以度教学、改名 angle_rad 不换算、原语按弧度积分 | vcli/cognitive/strategy_selector.py:396 + primitives/locomotion.py:134 | high | -90 度变 90 弧度 (~2.4 圈后超时);小度数则 57 倍错误幅度下 PASS;派生词表教原语却零参数文档;_execute_primitive 静默丢弃未识别参数;现有测试把 bug 固化 | 原语参数 schema (名/类型/**单位**) 与函数同源声明,词表与归一化都从它派生,接缝处 math.radians;丢参数改为 fail loud |
| 13 | 枚举值静默兜底:未知 direction 走 forward / 转 right;wrapper 把 enum/default 从 schema 剥掉 | vcli/tools/skill_wrapper.py:88 + walk.py:103 + turn.py:82 | high | 错方向运动 + success=True;LLM 根本看不到合法值集合 (mcp/tools.py:167 倒是传了 — 同一契约两个 wrapper 已漂移) | _build_schema 透传 enum/default (各一行);技能内按枚举校验,非法值返回 bad_params 附合法集 (规则 8) |
| 14 | MOTOR_KEYWORDS 子串嗅探双向误判:walk/turn 非 motor (真机免确认+跳过 post-state 证据),STOP 是 motor (急停被确认框拦截) | vcli/tools/skill_wrapper.py:70 | high | 关键词嗅探代替显式声明;急停进确认流程违反"E-stop 独立"安全规则 (P0 旁路目前救了精确停止词,但 LLM 路径仍中招) | Skill 协议加显式 `is_motor`/`actuates` 属性,注册时必填;stop 类硬豁免确认;凡触底盘/臂的技能一律附 robot_state_after |
| 15 | 主通道 60s socket 超时 vs 无上限节拍运动 → 行协议永久错位 | playground/habitat/bridge.py:135 | high | "走 20 米"=67s>60s,readline 抛裸 socket.timeout 绕过错误契约,之后每个响应错位一格 (pano 当 navigate 结果解析),无恢复路径 | 协议加单调递增 id 并回显,错位即丢弃/重连;读超时按 op 估算 (duration/路径长+余量);服务端 cap duration;timeout 包装为 HabitatBridgeError 并标记 bridge 死亡强制重连 |

### P2 — 次级 (medium/low)

| # | 标题 | 文件 | 严重度 | 本质 | 修法方向 |
|---|---|---|---|---|---|
| 16 | 节拍运动独占 op 线程,行驶期间 pano/render 全饿死 | playground/habitat/server.py:545 | medium | sim-oracle 模式下导航 30s 内语义建图/用户可见话题全冻结,恢复时位姿大跳 | 最小修:waypoint 间协作让出 op 线程服务排队 render;或 motion worker + ticket 轮询 |
| 17 | 后置式 step timeout:慢但完成的物理动作记失败,整树重放已成功的运动 | vcli/cognitive/goal_executor.py:391 + vgg_harness.py:209 | medium | 超时后不跑 verify、不 capture,replan 上下文只带失败不带"已完成" → 双倍运动/时间/LLM 成本 | 超时后仍跑 verify (verified 则 pass+timing 警告);re-decompose 上下文注入已完成步 + Blackboard 种子,只规划余下部分 |
| 18 | replan 参数继承按策略名 "latest wins",目标盲 | vcli/cognitive/vgg_harness.py:311 | medium | 同策略多步全部继承同一份最新绑定 (两个 navigate 都拿到 sofa);Case-7 点修把 fail-loud 变成了静默猜测 | 按 (策略, sub_goal 名) 匹配;仅当前树该策略只有一步时才允许策略级回退;歧义时留空走 bad_params 喂 replan |
| 19 | navigate_to 先读位姿后拿 lease (与 walk 顺序相反) | playground/habitat/server.py:531 | low | 并发 cmd_vel 流下用过期起点规划 + 首次 _write_pose 回拉位姿 (窗口毫秒级,影响 cm 级) | _acquire_lease 提到位姿读取/规划之前,no_path 早退时释放;加回归测试 |
| 20 | 首行畸形 JSON 经未绑定 `req` 引发 NameError 杀死整个 server | playground/habitat/server.py:837 | low | 清理路径被跳过,stderr 被 DEVNULL 吞掉,只见"连接关闭" | try 前 `req={}` 或 op 局部化 (stream handler 已是正确写法,照抄) |

## 三、优化方案 — 分批次

原则:第一批是**内核级机制**,一刀切掉 false-PASS 整个家族;后面批次才做语义和契约修复。每批 TDD:先写 red 测试钉住缺陷,再实现。每批结束跑 `.venv/bin/python -m pytest tests/vcli tests/unit/vcli -q` + 更新 `docs/agent-kernel-STATUS.md`、`docs/ARCHITECTURE.md` (规则 9)、`docs/tricky-bugs.md` 追加 Case 9+。

### 批次一:杜绝 false-PASS 家族 (内核机制,非点修) — 覆盖 #1/#4/#5/#10

**目标**:建立三条内核不变量,让"初始状态即满足/哨兵 verify/零证据成功"在机制上不可能静默通过:

- **不变量 I (前置基线)**:GoalExecutor 在派发策略**前**评估 verify 谓词,StepRecord 增量式加 `pre_satisfied: bool = False` (规则 6)。pre_satisfied 的动作步:(a) 对用户/trace 诚实上报"目标已满足,未执行动作";(b) 要计为 evidence-backed 必须有状态差 (verify 假→真,或 result_data 位移/时长超下限)。一个咽喉点取代 N 处各自为政的 arrival check。
- **不变量 II (证据门按步豁免)**:删除 trace_store.py 的 `if is_robot: return True`,改为世界在 4 件套注册时声明"无符号后置条件的策略豁免集" (机制照抄已有的 answer_only);PlaygroundWorld 声明空集 (它有完整 oracle)。run_turn_unified 区分上报 "verified" vs "completed (unverified)"。
- **不变量 III (verify 单一来源)**:删 engine.py `_VERIFY_MAP`,快路径从 registry 的 `skill.verify_hint` 派生 (参数代入);`to_schemas` 不再静默缺省 "True",改为显式 `unverified: true` 标签;为 go2 运动技能声明真实 hint (walk 位移、turn 朝向、stop 速度==0、房间导航 visited)。加一个一致性测试:任何注册技能在快路径与 LLM 路径必须得到同一 verify。
- **顺手**:foreach 未解析 source_path → failure_class=exec_error 附 producer 真实键名 (#10,十几行 + 测试)。

**改动面**:goal_executor.py、trace_store.py、types.py (增量字段)、engine.py、core/skill.py、vocab_from_registry.py、playground/world.py、go2 各 skill 的 verify_hint 声明。
**风险**:中。pre_satisfied 与豁免集会让一批现有"宽松通过"的测试转红 — 这正是目的,逐个改判;frozen dataclass 严格增量,不破构造。注意 `feedback_no_parallel_agents`:测试串行跑,不要三个 agent 并行 pytest。
**预计规模**:~600-900 行 (含测试),2-3 个 engineer-day,单 agent 或 Alpha+Beta 分内核/技能两线。

### 批次二:房间=区域 + 导航传输层诚实契约 — 覆盖 #2/#3/#8/#9 (种子 bug 根治)

**目标**:'走到厨房' 这类房间目标按区域语义判定;导航三层 (server、sysnav_bridge、skill) 全部返回运动证据。

- seed_room_landmarks 在 properties 保留 rect;navigate_to_point 按 `properties.type=='room'` 分支:目标取矩形内可导航点,verify_hint 改 `visited('<room>')`;1.5 standoff 仅限 object。修正 test_room_landmarks.py 固化的错误断言。
- server.navigate_to:lease 前先算 geodesic,≤tol 返回 `{reached, already_there: true, moved: 0.0}`;中间路点固定小阈值,tol 仅终点;`reached = gd <= tol` (删 ×2.0);返回必含 moved_m/elapsed_s。
- sysnav_bridge:终点到达加一次 geodesic 校验;同样返回 already_there/moved。
- navigate_to_point/walk/turn 的 result_data 契约化:start/moved_m/duration_s/already_there 必填,与批次一的状态差证据对接;walk 拒绝 distance≤0,moved≪requested 报失败;turn 记录前后朝向。

**改动面**:habitat_runtime.py、navigate_to_point.py、server.py、sysnav_bridge.py、walk.py、turn.py + 测试 (含"已在房内""穿墙""撞墙零位移"三个回归用例)。
**风险**:低-中,全在 playground/skills 层,内核不动。
**预计规模**:~500 行,2 engineer-day。

### 批次三:跨层契约单一来源 — 覆盖 #7 (横移)/#12/#13/#14

- 原语参数 schema (名/类型/单位) 与函数同源声明;词表 params-help 与 selector 归一化从它派生;deg→rad 在接缝换算;_execute_primitive 丢参数改 fail loud。修正 test_level44 固化的弧度断言。
- skill_wrapper._build_schema 透传 enum/default (对齐 mcp/tools.py);walk/turn 枚举校验返回 bad_params;WalkSkill 检查 supports_holonomic,横移不支持 fail loud。
- 删 MOTOR_KEYWORDS 嗅探:Skill 协议加显式 `is_motor` (注册时必填),stop 类硬豁免确认,motor 技能一律附 robot_state_after。

**改动面**:strategy_selector.py、goal_executor.py (原语派发)、primitives/、skill_wrapper.py、walk/turn/stop、core/skill.py。
**风险**:中 (Skill 协议加字段触及全部技能,但是增量式 + 注册期校验,fail loud 即测试可见)。
**预计规模**:~500-700 行,2 engineer-day。

### 批次四:harness/传输层健壮性 — 覆盖 #11/#15/#16/#17/#18/#19/#20

- harness 依赖失败跳过 (skipped StepRecord + 新 failure_class);统一 harness/executor 失败语义;删死配置 max_redecompose (不实现 Layer-2,见第五节)。
- bridge 协议加请求 id + 按 op 估算读超时 + timeout→HabitatBridgeError+强制重连;server cap duration。
- 后置超时改"仍跑 verify";re-decompose 上下文注入已完成步。
- replan 继承按 (策略, sub_goal) 匹配。
- 三个小修:waypoint 间协作让出 op 线程、navigate_to lease 顺序、`req={}` 初始化。

**改动面**:vgg_harness.py、goal_executor.py、bridge.py、server.py。
**风险**:低-中,协议加 id 需 bridge/server 同步改 (一次 PR 内完成,无版本兼容问题——同进程拉起)。
**预计规模**:~600 行,2-3 engineer-day。

## 四、需要 CEO 拍板的决策点

1. **房间目标的成功定义**
   A. "进入房间矩形" (`visited(room)`),目标点取矩形内可导航点 — 语义正确,种子 bug 按构造消失;
   B. 保留中心点 + 动态 tol (中心到矩形边界距离) — 改动小但仍是点语义,墙角房间仍可能站在房外通过。
   **推荐 A**。正确谓词已实现已绑定,只差接线。

2. **evidence gate 豁免粒度**
   A. 按步豁免 + 世界声明豁免策略集 (playground 空集),被豁免步上报 "unverified" — 护城河在能开的地方全开;
   B. 维持按世界开关,仅把 playground 改成 False — 改动最小,但真机上线时护城河又是全关,问题原样复发。
   **推荐 A**。这是"verify 是护城河" (规则 5) 的唯一兑现方式;B 只是把雷往后挪。

3. **verify_hint 缺省策略**
   A. 注册时必填,无 hint 拒绝注册 (fail loud) — 最严,但要求一次性补齐所有技能;
   B. 允许缺省但 schema 打 `unverified: true` 标签,planner/证据门拒绝把此类步当承重步,用户侧显示 "unverified" — 渐进。
   **推荐 B 先行 (批次一),真机接入前升级为 A**。

4. **节拍运动与渲染的并发架构**
   A. 最小修:waypoint 间协作让出 op 线程 (批次四,~50 行);
   B. 大修:motion worker 线程 + ticket/轮询完成,或运动全部改 stream 驱动。
   **推荐 A**。B 是真机/长程导航阶段的事,现在做是过度工程。

5. **种子 bug 是否需要今天先打点修** (navigate verify tol 1.6→收紧 + 房间 verify 临时换 visited)
   A. 等批次一/二机制修 (1 周内);B. 今天先点修保 demo 可演示。
   **推荐 B + A 并行**:点修两行 (verify_hint tol、room 分支),不与机制修冲突;若本周无 demo 需求则直接 A。

## 五、不建议做的事 (过度重构警告)

- **不要实现 Layer-2 中途重分解** (max_redecompose 承诺的功能)。Layer-1 重试 + Layer-3 整树 replan 配合批次四的"已完成步注入"已够用;中途改树是高复杂度低收益。删掉死配置和文档承诺即可。
- **不要把 bridge 行协议重写成完整 RPC 框架** (gRPC/zmq)。加请求 id + 超时分级 + 强制重连三件事解决全部已知问题;协议重写会拖一周且引入新风险面。
- **不要重写 habitat server 为全异步**。决策点 4 的协作让出足够;sim-oracle 是过渡形态,nav stack 激活后导航本就走 stream 通道。
- **不要一次性重构全部 18 个技能的 SkillResult**。只对**运动类**技能强制前后状态契约 (批次二);describe/detect 等观测类技能现状无害。
- **不要动 frozen dataclass 的既有字段**、不要改 nav stack planner 参数 (既有反馈红线)、不要为修测试而放宽 verify — 规则 5:沙箱只能更严。
- **不要并行跑多个 agent 各自起 pytest/MuJoCo** — 64GB 内存红线,批次内测试串行执行。

**执行建议**:批次一立即以 `/sdd init` 立项 (spec 即上文三条不变量,验收 = 种子场景 '走到厨房' 在 spawn 距中心 <1.6m 时上报 "already satisfied / unverified" 而非 verified PASS);批次二紧随,三/四可与二并行 (文件不重叠,git worktree 隔离)。总量约 2200-2700 行含测试,4 个批次约 1.5-2 周。

---
*来源: 6 维并行代码审查 workflow (35 agents, 2.37M tokens), 26 条发现经对抗核实
(3 critical / 17 high / 4 medium / 2 low), 去重合并为 18 项。种子缺陷
('走到厨房' 零运动 PASS) 已实测钉死: success=True / wall=0.00s / moved=0.00m /
最终位置不在厨房矩形内 (~/sandbox/live_repro_kitchen_noop.py)。
状态: 待 CEO 决策 (第四节 5 个决策点); 批准后批次一以 /sdd init 立项。*
