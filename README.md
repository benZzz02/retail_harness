# RefundPilot Harness — External τ-bench Retail

一个“不训练模型”的电商客服 Agent Harness 最小实现。主流程直接加载
Sierra Research 的外部 **τ-bench Retail** 环境；用户、订单、商品、工具执行、
状态变化和最终 reward 都来自上游仓库。多轮模式额外连接隐藏指令用户模拟器，
用于测试确认、改口和终止协议。

本项目只负责模型外围的运行层：

- 把外部工具 schema 转成统一的 `AgentContext`；
- 执行有步数上限的 agent loop；
- 将 `ToolAction` 路由到上游 `MockRetailDomainEnv.step(...)`；
- 将 Agent 的自然语言回合经上游 `Env.step(respond)` 发给用户模拟器；
- 保存 append-only JSONL trajectory；
- 调用上游 `Env.calculate_reward()` 验证终态；
- 审计失败工具及“Agent 声称完成后用户仍继续”的对话违规；
- 保留可替换的 `AgentProvider` 边界，后续可接任意模型。

## 一次跑通

需要 Python 3.10+（本机已用 Python 3.11 验证）：

```bash
cd /Users/ben/LLM-Agent/refundpilot-harness
./scripts/setup_tau_bench.sh
```

加载并查看一个外部 Retail task：

```bash
.venv/bin/python -m refundpilot tau-inspect --split test --task 0
```

跑 3 个可见样本的端到端接线测试：

```bash
.venv/bin/python -m refundpilot tau-smoke --split test --tasks 0 5 18
```

这个命令会实际执行官方工具并调用官方 reward，但它使用 benchmark 的 gold
actions，因此输出明确标记为 **ORACLE WIRING CHECK**。它证明“环境接通了”，
不代表任何模型的任务成功率。

使用本机已登录的 Codex LLM 做一次不暴露 gold actions 的逐步规划：

```bash
.venv/bin/python -m refundpilot tau-llm \
  --split test --task 0 --model gpt-5.6-luna --reasoning low
```

该模式每轮只向模型提供 Retail policy、工具 schema、任务和已有 observation，
模型返回一个结构化 `ToolAction` 或 `FinalAction`。由于 MVP 没有连接 live user
simulator，命令把初始明确请求视为最终授权，因此这是 one-turn LLM smoke test，
仍不是官方 τ-bench protocol。运行会把 τ-bench 中仿真的姓名、地址、订单号和
支付标识发送给所选模型服务。

连续跑前 10 个 one-turn 样本：

```bash
./scripts/run_tau_llm_first10.sh
```

使用 DeepSeek API 做相同的 blind one-turn 运行：

```bash
export DEEPSEEK_API_KEY="<your-key>"
.venv/bin/python -m refundpilot tau-llm \
  --provider deepseek --model deepseek-chat \
  --split test --task 0
```

DeepSeek 适配器使用官方 OpenAI-compatible Chat Completions endpoint 和 JSON
Output；也可以通过 `DEEPSEEK_BASE_URL` 指向兼容的代理服务。API key 只从环境变量
读取，不会写入轨迹或仓库。接口说明见
[DeepSeek Chat Completions API](https://api-docs.deepseek.com/api/create-chat-completion/)
和 [JSON Output](https://api-docs.deepseek.com/guides/json_mode/)。

一次真实 pilot 得到 5/10、70 次工具调用、0 次无效工具调用。后五题主要因缺少
确认后的用户改口而终态不匹配；详细协议和逐题诊断见
[`reports/llm_test_0_9_20260913.md`](reports/llm_test_0_9_20260913.md)。这个数字
不能作为官方 benchmark 成绩。

连接隐藏指令用户模拟器，跑一条真实多轮任务：

```bash
.venv/bin/python -m refundpilot tau-dialogue \
  --split test --task 5 \
  --agent-model gpt-5.6-luna \
  --user-model gpt-5.6-luna \
  --reasoning low --max-steps 30
```

此时 Agent 看不到 task instruction，只能看到模拟用户逐轮说出的内容。确认后的
改口会形成新 scope，任何状态变更都必须对最新 scope 重新确认。认证回合带一个
grounded 层：只使用隐藏指令里明确给出的 email，或从 benchmark account ID
中直接拆出的姓名与明确 ZIP，防止用户模拟器编造身份信息。

连续跑前 10 个多轮样本：

```bash
./scripts/run_tau_dialogue_first10.sh
```

2026-09-14 的最终独立 pilot 在 test/0–9 上得到 10/10 官方 exact-state reward、
71 次工具调用、0 次失败工具；更严格的 dialogue audit 为 9/10，因为 test/9 在
正确终态后出现一次模拟器 scope 回退。增加 scope 持久化约束后，test/9 定向
回归通过。完整逐题结果、失败证据和口径说明见
[`reports/dialogue_test_0_9_20260914.md`](reports/dialogue_test_0_9_20260914.md)。
这仍是 legacy 数据上的小样本 pilot，不是正式榜单成绩。

多轮模式可以独立选择 Agent 和用户模拟器模型：

```bash
export DEEPSEEK_API_KEY="<your-key>"
.venv/bin/python -m refundpilot tau-dialogue \
  --agent-provider deepseek --agent-model deepseek-chat \
  --user-provider codex --user-model gpt-5.6-luna \
  --split test --task 0 --reasoning low
```

如果需要两边都使用 DeepSeek，将 `--user-provider deepseek --user-model
deepseek-chat` 即可。原有 Codex 命令和默认行为保持不变。

命令结束会打印 session id。轨迹可回放：

```bash
.venv/bin/python -m refundpilot replay SESSION_ID
```

## 实际加载了什么

当前适配器在启动时实例化：

```python
from tau_bench.envs.retail.env import MockRetailDomainEnv

env = MockRetailDomainEnv(
    user_strategy="human",
    user_model="unused",
    task_split="test",
    task_index=0,
)
```

当前外部 Retail 数据规模为：

- 500 个用户；
- 1000 个订单；
- 50 个商品；
- 16 个官方工具，包括订单查询、取消、修改、退货、换货和转人工。

`tau-llm` 的 one-turn 模式不调用 `reset()`，用于快速验证工具和 reward 链路；
`tau-dialogue` 会把 Codex 用户模拟器注入上游环境并调用 `reset()`，后续每个
Agent 自然语言回合都通过 `Env.step(Action(name="respond"))` 推进。测试中的
scripted simulator 不访问网络，CI 不会卡在标准输入。

## Harness 边界

```text
hidden task instruction ──► User Simulator
                                ▲     │ customer turn / STOP
                                │     ▼
AgentProvider ◄──────────── TauDialogueRuntime ───► events.jsonl / audit
    │ ToolAction                    │ FinalAction
    ▼                               ▼
TauRetailEnvironment ─────► external tau_bench.Env.step(...)
    ├── official Retail data and tools
    ├── respond routing
    └── official terminal-state reward
```

`TauRetailEnvironment.describe()` 只暴露 instruction、数据规模、工具 schema 和
来源信息，不暴露 gold actions。只有显式的 `OracleReplayProvider` 会读取 gold
actions，且运行记录带 `evaluation_label=wiring_only`。

## 上游版本说明

为了让这个 MVP 在 Python 3.11 上最少改动地运行，当前锁定的是 Sierra 官方
legacy `tau-bench` 仓库的 commit：

```text
59a200c6d575d595120f1cb70fea53cef0632f6b
```

锁定信息见 `external/tau-bench.lock.json`，源码被 clone 到忽略版本控制的
`.external/tau-bench/`。上游已经提示该仓库的任务不是最新版本；最新修订任务
位于 [τ³-bench（仓库名 tau2-bench）](https://github.com/sierra-research/tau2-bench)。
所以这个版本适合展示“外部环境适配 + Harness 设计”，不应包装成最新榜单结果。

## 测试

外部环境与本地无依赖 fixture 一起测试：

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

测试覆盖：外部数据确实加载、工具 schema 数量、未知工具失败、LLM 结构化动作
解析、用户模拟器身份 grounding、多轮 respond 路由、完成后继续的对话违规、
官方 gold actions 得到 reward 1.0，以及 trajectory 可回放。系统 Python 未安装
`tau_bench` 时，外部集成测试会跳过，不会假装通过。

2026-09-13 的单任务真实 LLM smoke：`gpt-5.6-luna` 在未看到 gold actions 的
情况下依次完成用户认证、订单查询、两个商品检索和合并换货；`test/0` 的官方
状态 reward 为 1.0。该结果只有一个样本且修改了用户确认协议，不能写成测试集
成功率。

## 本地 fixture 的定位

原有 `python3 -m refundpilot demo` 使用 5 个小型 SQLite 退款案例。它现在仅用于
快速测试 Harness 自身的 provider/runtime/tool contract，不是主电商环境，也不
作为简历中的 benchmark 结果。主入口是 `tau-inspect` 和 `tau-smoke`。

## 下一步接模型

真实模型适配器只需实现同一个接口：

```python
class MyLLMProvider:
    name = "my-llm"

    def next_action(self, context):
        # 输入：context.user_request、context.tool_schemas、历史 observations
        # 输出：ToolAction(...) 或 FinalAction(...)
        ...
```

当前仓库已提供 `DeepSeekProvider` 和 `DeepSeekUserSimulator`，无需修改
Retail 环境、工具协议或评测器即可切换模型。

接模型后应单独报告 blind task success / pass^k，不能混用本项目的 oracle wiring
结果。
