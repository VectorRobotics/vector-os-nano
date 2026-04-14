# v2.0 Spec: VectorEngine 统一架构 + VGG 全面升级

**Status**: DRAFT (待 CEO 审批)
**替代**: v1.9.0 abort signal spec (scope 扩大)

## 背景

v1.8.0 删除了 robo/cli/web 旧入口后，系统仍有两条执行路径：
- **vcli** → VectorEngine + VGG (新引擎)
- **MCP** → core/Agent.execute() (旧6阶段流水线)

两条路径共享 skills/hardware 但执行逻辑完全独立。abort 信号、bug 修复、新功能都要实现两遍。

审计发现 MCP 对旧 Agent 的依赖仅为：硬件引用 + SkillRegistry + WorldModel。实际调用 Agent.execute() 的只有 mcp/tools.py 的 4 个入口。迁移到 VectorEngine 改动量可控。

同时，VGG 框架审计发现 15+ 个弱点需要修复。

## 目标

### Phase 1: 统一架构 (MUST)

将 MCP 切换到 VectorEngine，删除旧执行管线。

**改动：**

| 文件 | 操作 | 说明 |
|------|------|------|
| `mcp/server.py` | 重写 | 接收 `(engine, hw_context)` 替代 `Agent` |
| `mcp/tools.py` | 重写 | `handle_tool_call` 用 `engine.run_turn()` 替代 `agent.execute()` |
| `mcp/resources.py` | 改写 | 从 `hw_context` 读硬件状态，不再依赖 Agent |
| `core/agent.py` | 瘦身 | 只保留 `__init__` + 硬件字段 + `_build_context()`，删除 execute/handle/plan/run_goal 约 700 行 |

**删除：**

| 文件/模块 | 行数 | 原因 |
|-----------|------|------|
| `core/agent_loop.py` | 243 | VGG GoalExecutor 替代 |
| `core/mobile_agent_loop.py` | 474 | VGG Harness 替代 |
| `core/tool_agent.py` | 355 | VectorEngine backends 替代 |
| `core/memory.py` | 263 | vcli Session 替代 |
| `core/plan_validator.py` | 375 | VGG GoalDecomposer 内部验证替代 |
| `llm/` 整个模块 | ~700 | vcli/backends/ 替代 |
| 21 个旧测试 + 18 个空 stub | ~1500 | 测试旧 Agent 逻辑 |

**保留 (共享基础设施)：**
- `core/types.py` — 全局数据类型
- `core/skill.py` — SkillRegistry + SkillContext + @skill
- `core/world_model.py` — WorldModel
- `core/scene_graph.py` — SceneGraph (VGG primitives 依赖)
- `core/spatial_memory.py` — SpatialMemory (mobile skills 依赖)
- `core/config.py` — 配置加载
- `core/executor.py` — TaskExecutor (保留，可选复用)
- `core/nav_client.py` — NavStackClient (导航技能依赖)

**验收标准：**
- [ ] `vector-cli` 功能不变，3651 个测试全绿
- [ ] `vector-os-mcp --sim` 启动成功，natural_language/run_goal/diagnostics/direct skill 全部可用
- [ ] MCP resources (world://*, camera://*) 正常返回
- [ ] `import vector_os_nano` 不报错
- [ ] `llm/` 目录不存在
- [ ] `from vector_os_nano.core.agent import Agent` 仍可用（瘦身后）

### Phase 2: 全局 Abort 信号 (MUST — 安全)

统一架构后，abort 只需实现一次。

**新文件：**

`vcli/cognitive/abort.py`:
```python
_abort = threading.Event()

def request_abort() -> None       # StopSkill / Ctrl+C / 新任务覆盖旧
def clear_abort() -> None         # 每个新任务开始时
def is_abort_requested() -> bool  # 所有阻塞循环检查
def wait_or_abort(seconds) -> bool  # 替代 time.sleep()，可中断
```

**改动：**

| 文件 | 改动 |
|------|------|
| `vcli/engine.py` | P0 stop 绕过（"stop/停/halt" 硬编码匹配，不走 LLM/VGG，<100ms） |
| `vcli/engine.py` | 每个 run_turn 开头 clear_abort()；新任务时 request_abort() 取消旧任务 |
| `vcli/cognitive/vgg_harness.py` | 每步执行前 + 每次重试前检查 abort |
| `vcli/cognitive/goal_executor.py` | 每个 SubGoal 执行前检查 abort；async skill 等待循环用 wait_or_abort |
| `skills/go2/stop.py` | execute() 内调 request_abort() |
| `skills/go2/explore.py` | _explore_cancel 挂到全局 abort |
| `skills/navigate.py` | dead_reckoning 每个 waypoint 间检查 abort |
| `hardware/sim/go2_ros2_proxy.py` | navigate_to() 0.5s poll 循环检查 abort |

**验收标准：**
- [ ] VGG 执行中说 "stop" → 机器人 <100ms 停止，VGG 线程退出
- [ ] "探索然后去厨房" → explore 完成后才 navigate
- [ ] explore 过程中 "stop" → 探索取消，不执行后续步骤
- [ ] navigate 过程中 "stop" → 导航立即取消
- [ ] 新命令覆盖旧任务 → 旧任务 abort，新任务启动

### Phase 3: VGG 框架升级 (SHOULD)

审计发现的 15 个弱点，按优先级修复。

#### 3.1 VGG 初始化健壮性

**问题**：`engine.init_vgg()` 捕获所有异常后静默禁用 VGG，用户不知道为什么 VGG 没工作。

**修复**：
- 失败时日志输出具体原因（哪个组件初始化失败）
- 通过 `on_text` 回调通知用户 "VGG unavailable: {reason}"
- 区分 "缺依赖"（正常降级）vs "初始化 bug"（需要修复）

| 文件 | 改动 |
|------|------|
| `vcli/engine.py` init_vgg() | 分离每个组件初始化，单独 try/except，记录失败原因 |

#### 3.2 GoalDecomposer 质量提升

**问题**：
- 不检查 skill 是否可用就生成 GoalTree（可能引用不存在的 skill）
- verify 表达式质量依赖 LLM，常生成无效表达式
- 没有 few-shot 示例库

**修复**：
- decompose() 接收 available_skills 参数，注入到 prompt
- 验证生成的 strategy 对应的 skill 确实存在
- 维护 5-10 个高质量 few-shot 示例（覆盖导航、探索、感知、组合任务）
- verify 表达式生成后做 AST 预检（在执行前）

| 文件 | 改动 |
|------|------|
| `vcli/cognitive/goal_decomposer.py` | 增加 skill 可用性检查 + few-shot 示例 + AST 预检 |

#### 3.3 GoalExecutor 错误传播

**问题**：
- 步骤失败后 error 信息丢失，不传到 VGGHarness 的 re-plan 上下文
- 没有区分 "skill 失败" vs "verify 失败" vs "超时"
- 缺少执行时间统计

**修复**：
- StepRecord 增加 `error_category: str` 字段（"skill_error" | "verify_failed" | "timeout" | "abort"）
- 失败信息注入到 re-decompose 的 world_context
- 每步记录 wall-clock 时间

| 文件 | 改动 |
|------|------|
| `vcli/cognitive/goal_executor.py` | 错误分类 + 上下文注入 |
| `vcli/cognitive/types.py` | StepRecord 增加 error_category |

#### 3.4 StrategySelector 数据驱动升级

**问题**：
- 关键词匹配规则硬编码（reach→navigate, observe→look...）
- 只在 stats 有 3+ 次记录且 >50% 成功率时才用数据驱动选择
- 不支持同义词/语义相似度

**修复**：
- 降低数据驱动阈值：2 次记录 + 40% 成功率即可
- 添加模糊匹配（description 关键词交集评分）
- 失败策略冷却：连续失败 2 次后降低权重

| 文件 | 改动 |
|------|------|
| `vcli/cognitive/strategy_selector.py` | 阈值调整 + 模糊匹配 + 冷却机制 |

#### 3.5 VisualVerifier 触发精度

**问题**：关键词触发过于宽泛（"look" 匹配所有含 look 的步骤，包括 "look at map"）

**修复**：
- 收紧触发条件：只在 verify 表达式含感知函数（detect_objects, describe_scene, nearest_room）时触发
- 非感知步骤即使 verify 失败也不触发 VLM

| 文件 | 改动 |
|------|------|
| `vcli/cognitive/visual_verifier.py` | 基于 verify 表达式内容触发，而非步骤名关键词 |

#### 3.6 ObjectMemory 双向同步

**问题**：SceneGraph → ObjectMemory 单向同步。ObjectMemory 发现的新物体不会回写 SceneGraph。

**修复**：
- 增加 ObjectMemory.sync_to_scene_graph() 方法
- GoalExecutor 每步执行后调用同步

| 文件 | 改动 |
|------|------|
| `vcli/cognitive/object_memory.py` | 增加 sync_to_scene_graph() |
| `vcli/cognitive/goal_executor.py` | 步骤完成后触发同步 |

### Phase 4: 引擎质量 (COULD)

#### 4.1 IntentRouter 混合语言支持

**问题**：中英文混合输入（"go to 厨房"）匹配不准。

**修复**：
- 中文分词后再匹配（jieba 或简单正则分割）
- 英文关键词和中文关键词统一 normalize

| 文件 | 改动 |
|------|------|
| `vcli/intent_router.py` | 中文分词 + 统一匹配 |

#### 4.2 Engine Context 上下文追踪

**问题**：用户说"这里有什么"、"回去"时无法解析上下文。

**修复**：
- Engine 维护 `_current_room` / `_previous_room`
- 每次导航成功后更新
- Prompt 注入当前房间信息

| 文件 | 改动 |
|------|------|
| `vcli/engine.py` | 增加 room context 追踪 |
| `vcli/prompt.py` | 动态注入 current_room |

#### 4.3 VGG CLI 进度流

**问题**：多步 VGG 任务执行时用户看不到进度。

**修复**：
- GoalExecutor 的 on_step 回调输出 `[2/5] navigating to kitchen...`
- VGGHarness 的 re-plan 输出 `[re-planning: step X failed]`

| 文件 | 改动 |
|------|------|
| `vcli/engine.py` | on_step → on_text 管道 |

## 执行顺序

```
Phase 1 (统一架构)  ← 先做，消除双路径
    ↓
Phase 2 (Abort 信号)  ← 安全关键，统一后只实现一次
    ↓
Phase 3 (VGG 升级)  ← 质量提升
    ↓
Phase 4 (引擎质量)  ← 锦上添花
```

Phase 1+2 为 blocking（必须完成才能发布）。
Phase 3+4 为 non-blocking（可以分批交付）。

## 预估影响

| 指标 | 变化 |
|------|------|
| 删除代码 | ~4000 行 (agent_loop + mobile_agent_loop + tool_agent + memory + plan_validator + llm/) |
| 新增代码 | ~300 行 (abort.py + MCP 重写) |
| 改动代码 | ~500 行 (engine + VGG 组件) |
| 删除测试 | ~40 个 (旧 Agent 测试 + stubs) |
| 新增测试 | ~80 个 (abort + MCP 新接口 + VGG 升级) |
| 净效果 | 代码量减少 ~3200 行，测试增加 ~40 个 |

## 风险

| 风险 | 概率 | 缓解 |
|------|------|------|
| MCP 切换后行为不一致 | 中 | Phase 1 新增 MCP E2E 测试覆盖所有工具 |
| 删除旧代码遗漏依赖 | 低 | grep 验证零引用后再删 |
| VGG 升级引入回归 | 中 | Phase 3 每个改动独立 PR + 跑全量测试 |
| abort 信号竞争条件 | 低 | threading.Event 本身线程安全，review 所有 check 点 |
