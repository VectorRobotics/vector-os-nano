# VGG: 可验证目标图 — 设计规范

**Verified Goal Graph (VGG) — Design Specification**

版本: 0.3 (Phase 1+2+3 implemented, v2.0 abort + reliability upgrades)
日期: 2026-04-14
作者: Vector Robotics Architect
状态: 生产就绪。全局 abort 信号、prompt caching、3-layer retry 均已实装。

---

## 一句话

机器人的行为永远是一棵**透明的、可验证的目标树** — 不是不透明的代码，不是黑盒技能调用。LLM 是认知引擎，不是代码生成器。

---

## 当前实装状态 (Current Implementation Status) — 2026-04-08

### Phase 1 ✓ 完成
- GoalDecomposer: LLM → GoalTree JSON (with template matching)
- GoalVerifier: safe verification sandbox
- StrategySelector: rule-based strategy selection
- GoalExecutor: topological execution with fallback
- Primitives API: 25 functions across 4 categories (locomotion/navigation/perception/world)
- CLI integration: is_complex() keyword detection
- Test coverage: 187 tests, all green

### Phase 2 ✓ 完成
- CodeExecutor: RestrictedPython sandbox (velocity clamped, AST validated, 30s timeout)
- StrategyStats: persistent success rate tracking (~/.vector_os_nano/strategy_stats.json)
- ExperienceCompiler: traces → parameterized GoalTree templates
- TemplateLibrary: template matching + instantiation
- StrategySelector upgrade: stats-driven selection (confidence >= 3 samples, success > 50%)
- GoalDecomposer upgrade: TemplateLibrary lookup before LLM call
- GoalExecutor upgrade: per-step stats recording, auto-save
- CLI enhancement: GoalTree plan display, step-by-step feedback
- Test coverage: 87 additional tests, all green

### 总计 (Total)
- 274+ new tests in vgg_test_harness.py
- vcli/cognitive/ + vcli/primitives/: 8 + 5 modules
- Zero production code in src/ (adheres to Scribe constraints)
- Ready for live testing on real Go2

### 下一步 (Next)
- Phase 3 (planned): RL Executor integration
- V-Graph ceiling filter completion (in progress)
- Door-chain nav stack refinement (in progress)
- Experience compiler training on real user sessions

---

## 1. 为什么需要 VGG

### 1.1 现有范式的根本缺陷

| 范式 | 代表 | 做到了 | 牺牲了 |
|------|------|--------|--------|
| 预编译技能 | 当前 22 skills | 安全、可预测、可审计 | 无法组合、不能泛化新任务 |
| LLM 选工具 | 当前 VectorEngine | 灵活的意图理解 | LLM 只是调度器，行为被技能限死 |
| Code-as-Policy | CaP-X (NVIDIA/Berkeley) | 灵活组合、自适应 | 黑盒代码、无法验证中间状态、安全风险 |
| 端到端 VLA | RT-2, pi0 | 从数据学习 | 完全不可解释、海量数据、无法调试 |
| 行为树 | BehaviorTree.CPP | 可视化、可调试 | 手动编写、不能泛化 |

**核心矛盾**：每种方法都在灵活性、透明性、安全性三者之间做了取舍。没有一个同时拥有三者。

### 1.2 CaP-X 的启发与不足

CaP-X (https://capgym.github.io/) 的关键突破：
- LLM 生成可执行代码组合感知+控制原语（Code-as-Policy）
- 多轮视觉差分反馈（VDM）
- 自动技能库合成
- RL 后训练（GRPO）：7B 模型从 20% 到 72% 成功率
- 仿真到实机迁移差距极小（80% sim → 84% real）

CaP-X 的局限：
- 生成的代码是一整块，中间状态不可见
- 无法验证每一步是否成功
- 策略失败后无结构化的回退机制
- 面向桌面操作臂，未考虑四足导航场景
- 接触密集任务（倒水、插入）仍然脆弱

### 1.3 Vector OS Nano 的独特优势

我们已经有的基础：
- SceneGraph — 空间世界模型（房间、门、物体）
- 22 个预编译技能 + SkillFlow 路由
- VectorEngine — 多轮 tool_use agent 循环
- CategorizedToolRegistry + IntentRouter
- Go2 全套导航栈（FAR + TARE + localPlanner）
- VLM 感知（Ollama gemma4）
- 仿真到实机管道

缺的是：目标分解、验证机制、策略选择、自我进化。VGG 补上这些。

---

## 2. 核心理念

### 2.1 Agent 的六个维度

一个真正 agentic 的机器人需要覆盖全部六个维度：

```
目标 (Goals)    — 它想达成什么
信念 (Beliefs)  — 它认为世界是什么样的
规划 (Plans)    — 它打算怎么做
行动 (Actions)  — 它能做什么
感知 (Percept)  — 它怎么感知世界
学习 (Learning) — 它怎么从经验中进步
```

现有系统的覆盖：
- 技能系统：只有 Actions
- Code-as-Policy：Plans + Actions，没有 Beliefs 和 Learning
- VLA：Percept → Actions，其他全没有
- **VGG 目标**：覆盖全部六个维度

### 2.2 VGG 的核心主张

1. **LLM 输出结构，不输出代码。** LLM 生成结构化的目标分解（可审计、可验证），不是一整块不透明的代码。
2. **每一步都可验证。** 每个子目标有明确的成功条件，由系统自动检查，不依赖 LLM 判断。
3. **多策略执行。** 同一个子目标可以用预编译技能、生成代码、RL 策略、或人工介入来执行。系统自动选最优。
4. **Code-as-Policy 是一种策略，不是全部。** 它和预编译技能并列存在于策略市场中。
5. **经验自动积累。** 成功的目标树变成模板，失败的策略降低优先级，系统随使用自我进化。

---

## 3. 架构设计

### 3.1 三层架构

```
┌──────────────────────────────────────────────────┐
│              认知层 (Cognitive Layer)              │
│                                                   │
│  IntentParser → GoalDecomposer → StrategySelector │
│       ↕              ↕               ↕            │
│   WorldModel    ExperienceDB    GoalVerifier      │
│                                                   │
├──────────────────────────────────────────────────┤
│              能力层 (Capability Layer)             │
│                                                   │
│  策略执行器:                                       │
│    SkillExecutor  — 22 个预编译技能 (快、安全)     │
│    CodeExecutor   — LLM 生成代码 (灵活、较慢)     │
│    RLExecutor     — RL 训练策略 (快、特化) [未来]  │
│    HumanExecutor  — 人工遥操作 (兜底)              │
│                                                   │
│  原语 API (~20 个函数):                            │
│    locomotion | navigation | perception | world    │
│                                                   │
├──────────────────────────────────────────────────┤
│              基础层 (Foundation Layer)             │
│                                                   │
│  SceneGraph | VLM (Ollama) | TerrainAccumulator   │
│  ROS2 Nav Stack (FAR + TARE + localPlanner)       │
│  Go2 Hardware (MuJoCo sim / Unitree real)         │
└──────────────────────────────────────────────────┘
```

### 3.2 数据流

```
用户: "去厨房看看桌上有没有杯子"
  │
  ▼
IntentParser (LLM)
  输入: 自然语言 + 当前世界模型摘要
  输出: Goal { target: "verify(cup, kitchen_table)" }
  │
  ▼
GoalDecomposer (LLM)
  输入: Goal + 世界模型 + 经验模板库
  输出: GoalTree (结构化 JSON)
  │
  ▼
GoalTree:
  ├─ reach(kitchen)         verify: nearest_room()=="kitchen"
  ├─ observe(kitchen_table) verify: "table" in scene_desc
  └─ detect(cup)            verify: detect_objects("cup").count > 0
  │
  ▼
StrategySelector
  对每个子目标:
    查询策略市场 → 选成功率最高的策略
    reach(kitchen) → navigate_skill (成功率 92%)
  │
  ▼
Executor
  执行选中的策略
  每步执行后 → GoalVerifier 检查成功条件
  成功 → 下一个子目标
  失败 → StrategySelector 选备选策略
  │
  ▼
ExperienceCompiler
  整棵目标树执行完 → 记录结果
  成功 → 提取为模板
  失败 → 更新策略成功率
```

### 3.3 目标树数据结构

```python
@dataclass(frozen=True)
class SubGoal:
    """一个可验证的子目标。"""
    name: str                          # "reach_kitchen"
    description: str                   # "导航到厨房"
    verify: str                        # Python 表达式: "nearest_room() == 'kitchen'"
    timeout_sec: float                 # 60.0
    depends_on: tuple[str, ...] = ()   # ("previous_subgoal",)
    fail_action: str = ""              # "door_chain_fallback"

@dataclass(frozen=True)
class GoalTree:
    """LLM 生成的结构化目标分解。"""
    goal: str                          # "verify(cup, kitchen_table)"
    sub_goals: tuple[SubGoal, ...]     # 有序子目标列表
    context: str                       # 世界模型快照

@dataclass(frozen=True)
class ExecutionTrace:
    """一次目标树执行的完整追踪。"""
    goal_tree: GoalTree
    steps: tuple[StepRecord, ...]      # 每步的策略、结果、耗时
    success: bool
    total_duration_sec: float

@dataclass(frozen=True)
class StepRecord:
    """单步执行记录。"""
    sub_goal: str                      # "reach_kitchen"
    strategy: str                      # "navigate_skill"
    strategy_confidence: float         # 0.92
    result: str                        # "success" | "failed" | "timeout"
    verify_result: bool                # True
    duration_sec: float                # 26.3
    fallback_used: bool                # False
```

---

## 4. 五个核心创新

### 4.1 活的世界模型 (Active World Model)

将 SceneGraph 从被动数据库升级为推理引擎：

| 能力 | 现在 (SceneGraph) | 升级后 (WorldModel) |
|------|-------------------|---------------------|
| 空间查询 | nearest_room() → "hallway" | 保留 |
| 预测 | 无 | predict("穿过门A") → "到达厨房" |
| 不确定性 | 无 | certainty("杯子在桌上") → 0.7, 上次观测: 5min前 |
| 几何推理 | 无 | can_pass("门框") → True, 间距0.8m > 机身0.6m |
| 时序记忆 | 无 | last_seen("杯子") → 位置(10.2, 3.1), 时间戳 |

实现方式：在现有 SceneGraph 基础上增加：
- `ObjectMemory`：每个物体的最后观测位置 + 时间戳 + 置信度
- `predict(action)` → 基于房间拓扑和门连接的状态预测
- `certainty(fact)` → 基于时间衰减的置信度评估

### 4.2 LLM 生成目标树，不是代码

LLM 的输出格式是结构化 JSON，不是 Python 代码：

```json
{
  "goal": "verify(cup, kitchen_table)",
  "sub_goals": [
    {
      "name": "reach_kitchen",
      "description": "导航到厨房",
      "verify": "nearest_room() == 'kitchen'",
      "strategy_preference": ["navigate_skill", "door_chain_fallback"],
      "timeout": 60
    },
    {
      "name": "observe_table",
      "description": "观察厨房桌面",
      "verify": "'table' in describe_scene()",
      "depends_on": ["reach_kitchen"],
      "strategy_preference": ["turn_and_look", "code_as_policy"],
      "timeout": 15
    },
    {
      "name": "detect_cup",
      "description": "检测桌上是否有杯子",
      "verify": "detect_objects('cup').count > 0",
      "depends_on": ["observe_table"],
      "fail_action": "scan_room_360_then_report_not_found",
      "timeout": 10
    }
  ]
}
```

对比 Code-as-Policy：
- **CaP-X**：LLM 生成一大块 Python → 执行 → 要么成功要么整个失败
- **VGG**：LLM 生成目标树 → 每个节点独立执行和验证 → 失败的节点可以换策略重试

### 4.3 策略市场 (Strategy Marketplace)

每个子目标有多种执行策略，系统维护每种策略的历史表现：

```
策略注册表:
  reach(room):
    navigate_skill:      成功率 92%  平均耗时 25s  安全性 高
    code_as_policy:      成功率 78%  平均耗时 35s  安全性 中
    door_chain_fallback: 成功率 65%  平均耗时 45s  安全性 高
    human_teleop:        成功率 99%  平均耗时 ?    安全性 高

  detect(object):
    vlm_describe:        成功率 85%  平均耗时 3s   安全性 高
    scan_and_detect:     成功率 72%  平均耗时 15s  安全性 高
    code_as_policy:      成功率 68%  平均耗时 20s  安全性 中
```

策略选择公式：`score = success_rate * safety_weight / normalized_latency`

核心意义：
- Code-as-Policy 不是唯一选择，是策略之一
- 简单任务用预编译技能（快、安全）
- 复杂/新任务用 code-as-policy（灵活）
- 人工兜底永远可用
- 新策略（RL 训练的、经验合成的）可以随时加入

### 4.4 透明执行追踪

每个决策完整记录，任何人可读：

```
[10:23:45] 目标: 找厨房桌上的杯子
[10:23:45] 分解: reach(kitchen) → observe(table) → detect(cup)
[10:23:46] 策略: reach via navigate_skill (成功率 92%)
[10:23:46] 执行: navigate(room="kitchen")
[10:24:12] 验证: nearest_room()="kitchen" ✓ (26s)
[10:24:12] 策略: observe via turn_and_look (成功率 85%)
[10:24:13] 执行: turn(toward=table_position)
[10:24:15] 执行: describe_scene()
[10:24:18] 验证: "table" in scene ✓ (6s)
[10:24:18] 策略: detect via vlm_detect (成功率 80%)
[10:24:19] 执行: detect_objects("cup")
[10:24:22] 验证: cup_detected=True ✓ (4s)
[10:24:22] 目标完成: 杯子在厨房桌上 (总耗时 37s)
```

对比：
- 技能调度：看到 navigate() 被调用了，不知道为什么
- Code-as-Policy：看到一大块 Python 代码，不知道执行到哪了
- VGG：每步都有 what + why + result + time

### 4.5 经验编译器 (Experience Compiler)

成功的执行追踪自动提炼为可复用模板：

```yaml
template: find_object_in_room
  parameters: [object, room]
  sub_goals:
    - reach($room)    [verify: in_room($room)]
    - scan($room)     [verify: scene_described, max_directions: 4]
    - detect($object) [verify: found OR fully_scanned]
  stats:
    success_rate: 87%
    executions: 23
    common_failures:
      - room_not_explored: 3 times → auto-prepend explore()
      - object_moved: 2 times → expand scan area
```

模板不是代码 — 是结构化的经验。比代码更可解释，比固定技能更灵活。

编译流程：
1. 收集成功的 ExecutionTrace
2. LLM 识别相似模式 → 提取参数化模板
3. 模板注册到经验库
4. 未来的 GoalDecomposer 优先匹配模板
5. 模板的策略成功率持续更新

---

## 5. 原语 API 设计

为 CodeExecutor 和 GoalVerifier 提供的底层函数。包装现有接口，不新增 ROS2 节点。

### 5.1 运动 (locomotion)

| 函数 | 签名 | 来源 |
|------|------|------|
| `set_velocity` | `(vx, vy, vyaw) → None` | Go2 base.set_velocity() |
| `walk_forward` | `(distance_m) → bool` | 封装 set_velocity + 里程计 |
| `turn` | `(angle_rad) → bool` | 封装 set_velocity + 航向 |
| `stop` | `() → None` | base.set_velocity(0,0,0) |
| `stand` | `() → bool` | StandSkill |
| `sit` | `() → bool` | SitSkill |

### 5.2 导航 (navigation)

| 函数 | 签名 | 来源 |
|------|------|------|
| `get_position` | `() → (x, y, z)` | base.get_position() |
| `get_heading` | `() → float (rad)` | base.get_heading() |
| `nearest_room` | `() → str` | scene_graph.nearest_room() |
| `publish_goal` | `(x, y) → None` | /goal_point topic |
| `wait_until_near` | `(x, y, tol, timeout) → bool` | 轮询 get_position |
| `get_door_chain` | `(from, to) → list[(x,y)]` | scene_graph.get_door_chain() |

### 5.3 感知 (perception)

| 函数 | 签名 | 来源 |
|------|------|------|
| `capture_image` | `() → Image` | camera.capture() |
| `describe_scene` | `() → str` | VLM (Ollama gemma4) |
| `detect_objects` | `(query) → list[Object]` | VLM 检测 |
| `measure_distance` | `(angle_rad) → float` | lidar 单束测距 |
| `scan_360` | `() → list[(angle, dist)]` | LaserScan 数据 |

### 5.4 世界模型 (world)

| 函数 | 签名 | 来源 |
|------|------|------|
| `query_rooms` | `() → list[Room]` | SceneGraph |
| `query_doors` | `() → list[Door]` | SceneGraph |
| `query_objects` | `(room) → list[Object]` | WorldModel |
| `path_between` | `(room_a, room_b) → list[(x,y)]` | SceneGraph BFS |
| `certainty` | `(fact) → float` | WorldModel 时间衰减 |

---

## 6. 与现有架构的关系

### 6.1 不替代，叠加

```
现有架构 (保留):
  VectorEngine → tool_use → 22 skills     ← 简单任务仍走这条路

VGG 层 (新增):
  VectorEngine → VGG cognitive layer       ← 复杂任务走目标分解
                   ↓
              GoalTree → StrategySelector → Executor (可能调用 skills)
```

IntentRouter 增加一层判断：
- 简单指令（"站起来"、"去厨房"）→ 直接技能调用（零开销）
- 复杂任务（"检查所有房间的灯是否关了"）→ VGG 目标分解

### 6.2 文件结构（预计新增）

```
vector_os_nano/
  vcli/
    cognitive/                    # 新增：认知层
      goal_decomposer.py         # LLM → GoalTree
      goal_verifier.py           # 验证条件执行器
      strategy_selector.py       # 策略选择
      experience_compiler.py     # 经验 → 模板
      world_model.py             # 活世界模型（SceneGraph 增强）
    tools/
      code_executor.py            # 新增：代码执行沙盒
    primitives/                   # 新增：原语 API
      locomotion.py
      navigation.py
      perception.py
      world.py
```

---

## 7. 与其他方法的对比

| 维度 | 技能调度 | Code-as-Policy | VLA | **VGG** |
|------|---------|---------------|-----|---------|
| 灵活性 | 低 | 高 | 高 | 高 |
| 透明性 | 中 | 低 | 无 | **高** |
| 安全性 | 高 | 低 | 低 | **高** |
| 可验证 | 无 | 无 | 无 | **每步验证** |
| 自我进化 | 无 | 有限 | 需重训 | **经验编译** |
| 容错 | 无 | 代码级 | 无 | **策略级切换** |
| 可审计 | 看日志 | 看代码 | 不可能 | **目标树+追踪** |
| 延迟 | 最低 | 中 | 低 | 自适应* |

*VGG 对简单任务直接走技能（低延迟），复杂任务走目标分解（较高延迟但值得）。

---

## 8. 实施路线图

### Phase 1：原语 API + 目标分解引擎

- 暴露 ~20 个原语（包装现有接口）
- `GoalDecomposer`：LLM → 结构化目标树（JSON）
- `GoalVerifier`：每个子目标的成功条件自动检查
- 保留现有 22 skills 作为默认执行策略
- 测试：手动构造 5 个目标树，验证执行 + 验证流程

### Phase 2：策略市场 + CodeExecutor

- `StrategySelector`：根据成功率/安全性/延迟选策略
- `CodeExecutor`：RestrictedPython 沙盒（作为策略之一）
- 安全包络：速度钳位、超时、急停检查
- 执行追踪可视化（CLI 输出）
- IntentRouter 判断：简单 → 技能，复杂 → VGG

### Phase 3：活世界模型 + 视觉验证

- SceneGraph 增强：ObjectMemory、时间戳、置信度衰减
- VLM 验证回路：执行后拍照 → 文本差分 → 验证条件
- `predict(action)` 预测接口

### Phase 4：经验编译器

- 收集成功 ExecutionTrace → 提取参数化模板
- 模板注册到经验库
- 策略成功率持续更新
- GoalDecomposer 优先匹配已有模板

---

## 9. 风险和未决问题

### 风险

| 风险 | 影响 | 缓解 |
|------|------|------|
| LLM 生成的验证条件不正确 | 误判成功/失败 | 验证条件由预定义模板 + LLM 生成混合 |
| 目标分解延迟高 | 简单任务变慢 | IntentRouter 分流：简单 → 直接技能 |
| 安全性：CodeExecutor 执行不安全代码 | 机器人损坏 | RestrictedPython + 速度钳位 + 急停 |
| 策略成功率冷启动 | 初期选择不准 | 用预编译技能的已知表现作为初始值 |

### 未决问题

1. **验证条件的表达力**：纯 Python 表达式够用吗？是否需要更丰富的验证语言？
2. **目标树的深度**：允许多少层嵌套？嵌套太深会增加延迟和失败率。
3. **CodeExecutor 的沙盒边界**：允许调用哪些原语？是否允许网络访问？
4. **经验模板的泛化**：在 A 公寓学到的模板能迁移到 B 公寓吗？
5. **多机器人**：目标树能否跨多个机器人分配子目标？

---

## 10. 参考

- CaP-X: https://capgym.github.io/ — Code-as-Policy 系统评估框架
- CaP-X paper: https://arxiv.org/abs/2603.22435 — benchmark + CaP-Agent0 + CaP-RL
- Voyager: https://arxiv.org/abs/2305.16291 — LLM agent 自动课程 + 技能库
- SayCan (Google): LLM + affordance grounding
- Inner Monologue (Google): LLM + 环境反馈循环
