# VectorEngine — 统一执行引擎

## 概述

VectorEngine 是 Vector OS Nano 的唯一执行引擎。v2.0 起，CLI 和 MCP 共用同一引擎：

```
vector-cli  ─┐
             ├→ VectorEngine → VGG / tool_use → skill.execute()
vector-os-mcp┘
```

用户说自然语言，AI agent 通过工具系统同时控制机器人、编辑代码、诊断问题 —— 一个 session 里完成所有事。

```
用户: "探索的时候狗在转角撞墙"
  ↓
AI Agent (VectorEngine)
  ├── file_read("go2_vnav_bridge.py")     → 读路径跟随代码
  ├── file_edit(old="0.6", new="0.4")     → 改转弯速度
  ├── skill_reload("walk")                → 热加载，不用重启
  ├── explore()                           → 重新跑探索
  └── 回复: "改了转弯速度，重新探索中"
```

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│  vector-cli (vcli/cli.py)                                       │
│                                                                 │
│  用户输入 ──→ IntentRouter (意图分类)                             │
│                  │                                              │
│                  ↓                                              │
│              VectorEngine.run_turn()                             │
│                  │                                              │
│                  ├── DynamicSystemPrompt                         │
│                  │     ├── 角色设定 (缓存)                        │
│                  │     ├── 工具使用说明 (缓存)                     │
│                  │     ├── 硬件/技能/世界模型 (静态)               │
│                  │     └── [机器人状态] (每次刷新)                  │
│                  │           位置、房间、SceneGraph、              │
│                  │           导航状态、探索进度                     │
│                  │                                              │
│                  ├── CategorizedToolRegistry                     │
│                  │     ├── code:   文件读写编辑、bash、搜索         │
│                  │     ├── robot:  22个技能 + 场景图查询            │
│                  │     ├── diag:   ROS2话题/节点/日志、导航/地形    │
│                  │     └── system: 状态、仿真、热加载              │
│                  │                                              │
│                  ├── ToolHookRegistry                            │
│                  │     ├── pre_hook: 执行前回调                   │
│                  │     └── post_hook: 执行后回调(验证/统计)         │
│                  │                                              │
│                  ├── LLM 后端 (Anthropic / OpenRouter / 本地)     │
│                  ├── 权限系统 (7层检查)                            │
│                  └── Session (JSONL 持久化)                       │
└─────────────────────────────────────────────────────────────────┘
```

## Tool Call 完整流程

```
1. 用户输入自然语言
2. IntentRouter 关键词分类 → 选择相关工具类别
   "去厨房" → robot+diag (省 68% token)
   "改代码" → code+system (省 72% token)
   "你好"   → 全部工具 (无法判断意图)
3. VectorEngine 序列化:
   - system_prompt (DynamicSystemPrompt 刷新机器人状态)
   - messages (对话历史)
   - tools (只发选中类别的工具 schema)
4. LLM 返回 tool_use 调用
5. 引擎分区执行:
   - 只读 + 并发安全 → 并行 (ThreadPoolExecutor, 10 workers)
   - 写入 / 电机控制 → 串行
6. 每个工具执行:
   a. pre_hook 触发 (日志/预检)
   b. 权限检查 → allow / deny / ask(用户确认)
   c. tool.execute(params, context) → ToolResult
   d. post_hook 触发 (验证/统计)
   e. 电机技能 → 自动附加执行后状态(位置/房间)
7. 结果追加到 session
8. 循环回步骤 3，直到 LLM 返回 end_turn
9. 最终文本渲染到 CLI 面板
```

## CategorizedToolRegistry — 分类工具注册表

### 设计思路

继承自 `ToolRegistry`（完全向后兼容）。核心能力：
- 工具按类别分组管理
- 运行时动态启用/禁用整个类别
- 配合 IntentRouter 按意图只发送相关工具

```python
class CategorizedToolRegistry(ToolRegistry):
    _categories: dict[str, list[str]]   # 类别 → [工具名列表]
    _disabled: set[str]                 # 已禁用的类别

    def register(self, tool, category="default") -> None
    def enable_category(self, category: str) -> None
    def disable_category(self, category: str) -> None
    def to_anthropic_schemas(categories=None) -> list[dict]  # 可按类别过滤
    def list_categories(self) -> dict[str, list[str]]
```

### 工具类别

| 类别 | 工具 | 用途 |
|------|------|------|
| `code` | file_read, file_write, file_edit, bash, glob, grep | 代码读写编辑 |
| `robot` | 22个包装技能 + scene_graph_query + world_query | 机器人控制 + 空间查询 |
| `diag` | ros2_topics, ros2_nodes, ros2_log, nav_state, terrain_status | ROS2 诊断 |
| `system` | robot_status, start_simulation, web_fetch, skill_reload | 系统管理 |

### 扩展策略

| 阶段 | 策略 | 效果 |
|------|------|------|
| v1（当前） | 全部启用，IntentRouter 按意图路由 | 平均省 52% token |
| v1.1 | 延迟 schema — 先发名字，LLM 需要时再请求完整定义 | 再省 60% |
| v2 | 外部插件 — pyproject.toml entry_points 注册第三方工具 | 无限扩展 |

### 添加新工具（开发者工作流）

```python
# 1. 新建文件: vcli/tools/my_tool.py
@tool(name="my_tool", description="...", read_only=True, permission="allow")
class MyTool:
    input_schema = { "type": "object", "properties": { ... } }
    def execute(self, params, context) -> ToolResult: ...

# 2. 在 vcli/tools/__init__.py 中:
#    - discover_all_tools() 里加 import + 实例化
#    - _TOOL_CATEGORIES["my_category"] 里加工具名

# 完成。不需要改引擎、后端、权限、或任何其他文件。
```

## IntentRouter — 意图路由器

零成本关键词匹配，在 LLM 调用前选择相关工具类别：

```python
class IntentRouter:
    def route(self, user_message: str) -> list[str] | None:
        # 返回类别列表，或 None（发全部工具）

# 规则示例:
# "改"/"edit"/"code"/"bug"  → ["code", "system"]
# "去"/"走"/"explore"       → ["robot", "diag"]
# "topic"/"log"/"为什么"    → ["diag", "system"]
# "你好" (无关键词匹配)     → None → 全部工具
```

Token 节省效果：

| 场景 | 改前 (39 工具) | 改后 (路由) | 节省 |
|------|---------------|------------|------|
| "我在哪" | ~2500 tokens | ~800 tokens | 68% |
| "改速度" | ~2500 tokens | ~700 tokens | 72% |
| "你好" | ~2500 tokens | ~2500 tokens | 0% |
| 平均 | ~2500 tokens | ~1200 tokens | ~52% |

## Tool Protocol — 工具协议

每个工具实现这个接口（Protocol 类型，不需要继承）：

```python
class Tool(Protocol):
    name: str                           # 工具名
    description: str                    # LLM 看到的描述
    input_schema: dict[str, Any]        # JSON Schema 参数定义

    def execute(params, context) -> ToolResult          # 执行
    def check_permissions(params, context) -> PermissionResult  # 权限检查
    def is_read_only(params) -> bool                    # 只读？
    def is_concurrency_safe(params) -> bool             # 可并发？
```

`@tool` 装饰器自动注入 permissions、read_only、concurrency 的默认实现。

## SkillWrapperTool — 技能包装器

Robot skill（`@skill` 装饰器）自动包装为 LLM tool：

```
@skill(aliases=["stand", "站"]) class StandSkill  →  SkillWrapperTool("stand")
@skill(aliases=["navigate"])    class NavigateSkill →  SkillWrapperTool("navigate")
```

包装器增加的能力：
- **电机检测**: effects 中包含 "move"/"navigate"/"arm" → 需要用户授权
- **执行后状态**: 电机技能执行后，自动附加当前位置/房间到结果
- **恢复提示**: 失败时根据 diagnosis_code 给出下一步建议

```
成功: "Skill 'navigate' succeeded. Data: {room: kitchen}
       State: pos=(16.8, 2.3) room=kitchen"

失败: "Skill 'navigate' failed. (room_not_explored)
       Suggested: Room not explored yet. Run the explore skill first.
       Current state: {position: [10.0, 5.0], room: hallway}"
```

已知的恢复提示映射：

| diagnosis_code | 提示 |
|---------------|------|
| no_base | 没有连接机器人，用 start_simulation 启动仿真 |
| unknown_room | 房间不存在，用 scene_graph_query 查看可用房间 |
| room_not_explored | 房间未探索，先运行 explore |
| navigation_failed | 导航失败，用 nav_state 检查导航栈状态 |
| no_vlm | VLM 不可用，检查 Ollama 是否运行 |
| camera_failed | 摄像头未连接，用 robot_status 检查硬件 |

## ToolHookRegistry — 工具执行钩子

在每个工具执行前后触发回调，用于：
- 自动验证（电机技能后检查位置变化）
- 统计遥测（记录工具调用频率/耗时）
- 链式反应（文件编辑后自动格式化）

```python
class ToolHookRegistry:
    def add_pre_hook(self, hook: Callable) -> None    # 执行前
    def add_post_hook(self, hook: Callable) -> None   # 执行后
    def fire_pre(self, ctx: ToolHookContext) -> None
    def fire_post(self, ctx: ToolHookContext) -> None

@dataclass(frozen=True)
class ToolHookContext:
    tool_name: str
    params: dict
    result: ToolResult | None   # pre-hook 时为 None
    duration: float             # pre-hook 时为 0.0
```

钩子异常被吞掉，不会中断工具执行。

## DynamicSystemPrompt — 动态系统提示

**问题**: System prompt 启动时构建一次，之后机器人状态就过期了。

**解决**: `DynamicSystemPrompt` 是 list 的子类，重写 `__iter__()`。VectorEngine 每次 API 调用都会遍历 system prompt，所以机器人状态每轮都是最新的。

LLM 每次对话都看到：
```
[Robot State]
Position: (10.2, 5.3, 0.28) — hallway
Heading: 23 deg (NNE)
SceneGraph: 8 rooms (6 visited), 7 doors, 12 objects
Exploring: no
Nav stack: running
```

## RobotContextProvider — 机器人状态采集

从多个来源实时采集状态：

| 字段 | 数据源 | 更新频率 |
|------|--------|---------|
| 位置 (x, y, z) | `base.get_position()` | 每轮对话 |
| 朝向 (度数 + 方位) | `base.get_heading()` | 每轮对话 |
| 当前房间 | `scene_graph.nearest_room()` | 每轮对话 |
| SceneGraph 摘要 | `scene_graph.stats()` + `get_room_summary()` | 每轮对话 |
| 是否在探索 | `explore.is_exploring()` | 每轮对话 |
| 导航栈运行中？ | `explore.is_nav_stack_running()` | 每轮对话 |

优雅降级：没有 base → "No hardware connected"。没有 SceneGraph → 省略房间数据。

## 权限系统

7 层检查（优先级从高到低）：

1. `no_permission` 标志 → 全部放行
2. `deny_tools` 黑名单 → 拒绝
3. `tool.check_permissions()` 返回 deny → 拒绝
4. `session_allow`（用户说了 "always"）→ 放行
5. `is_read_only(params)` → 放行
6. `tool.check_permissions()` 返回 ask → 提示用户确认
7. 默认 → 提示用户确认

电机技能（navigate、walk、pick）→ 始终 ask。
只读工具（file_read、grep、ros2_topics）→ 始终 allow。

## 完整工具清单 (17 内置 + 22 技能)

### 内置工具

| 工具 | 类别 | 只读 | 权限 | 说明 |
|------|------|------|------|------|
| file_read | code | 是 | allow | 读取文件（带行号） |
| file_write | code | 否 | ask | 创建/覆盖文件 |
| file_edit | code | 否 | ask | 搜索替换 |
| bash | code | 否 | ask | 执行 shell 命令 |
| glob | code | 是 | allow | 按模式查找文件 |
| grep | code | 是 | allow | 搜索文件内容 |
| world_query | robot | 是 | allow | 查询世界模型对象 |
| scene_graph_query | robot | 是 | allow | 查询房间/门/物体/路径 |
| robot_status | system | 是 | allow | 硬件连接状态 |
| start_simulation | system | 否 | ask | 启动 MuJoCo 仿真 |
| web_fetch | system | 是 | allow | 抓取 URL |
| skill_reload | system | 否 | ask | 热加载技能模块 |
| ros2_topics | diag | 是 | allow | 列出/hz/echo ROS2 话题 |
| ros2_nodes | diag | 是 | allow | 列出/info ROS2 节点 |
| ros2_log | diag | 是 | allow | 读取机器人日志 |
| nav_state | diag | 是 | allow | 导航/探索状态 |
| terrain_status | diag | 是 | allow | 地形地图文件信息 |

### 包装的机器人技能 (22 个)

Walk, Turn, Stand, Sit, Lie Down, Stop, Explore, Navigate, Patrol,
Look, Describe Scene, Where Am I, Home, Scan, Wave, Pick, Place,
Handover, Detect, Describe, Gripper Open, Gripper Close

## Session 持久化

JSONL 格式，原子写入 + fsync：
```
{"type":"user","content":"去厨房","ts":"..."}
{"type":"assistant","text":"","tool_use":[{"name":"navigate","input":{"room":"kitchen"}}],"ts":"..."}
{"type":"tool_result","results":[{"content":"Skill 'navigate' succeeded..."}],"ts":"..."}
{"type":"assistant","text":"到了厨房，你要我看看有什么吗？","ts":"..."}
```

50 条记录自动压缩，防止上下文溢出。

## 探索事件流

探索期间，房间发现事件实时显示在 CLI：

```
vector> explore
  start_simulation(sim_type="go2") ... ok 2.1s
  explore() ... ok
  Entered hallway (1/8)
  Entered kitchen (2/8)
  Entered dining_room (3/8)
  ...
  Exploration finished — 8 rooms
```

由 `explore.py` 的 `set_event_callback()` 驱动，在 `vcli/cli.py` 启动时接入。

## 文件目录

```
vcli/
├── cli.py                  # 入口、REPL 循环、斜杠命令
├── engine.py               # VectorEngine — 多轮 tool_use agent 循环
├── intent_router.py        # IntentRouter — 意图路由（关键词 → 类别）
├── hooks.py                # ToolHookRegistry — 工具执行钩子
├── prompt.py               # 系统提示构建器（静态 + 动态块）
├── robot_context.py        # RobotContextProvider（实时机器人状态）
├── dynamic_prompt.py       # DynamicSystemPrompt（每轮自动刷新）
├── session.py              # JSONL session 持久化
├── config.py               # ~/.vector/config.yaml 加载器
├── permissions.py          # 7层权限检查器
├── backends/
│   ├── __init__.py         # LLMBackend Protocol + create_backend 工厂
│   ├── anthropic.py        # Anthropic Messages API（流式）
│   └── openai_compat.py    # OpenRouter / Ollama / vLLM
└── tools/
    ├── base.py             # Tool Protocol, @tool 装饰器,
    │                       # ToolRegistry, CategorizedToolRegistry
    ├── __init__.py         # discover_all_tools(), discover_categorized_tools()
    ├── file_tools.py       # file_read, file_write, file_edit
    ├── bash_tool.py        # bash
    ├── search_tools.py     # glob, grep
    ├── robot.py            # world_query, robot_status
    ├── sim_tool.py         # start_simulation
    ├── web_tool.py         # web_fetch
    ├── skill_wrapper.py    # SkillWrapperTool + wrap_skills() + 恢复提示
    ├── scene_graph_tool.py # scene_graph_query（7种查询）
    ├── ros2_tools.py       # ros2_topics, ros2_nodes, ros2_log
    ├── nav_tools.py        # nav_state, terrain_status
    └── reload_tool.py      # skill_reload（热加载）
```
