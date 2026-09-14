# τ-bench Retail 多轮用户模拟器：test/0–9 运行报告

日期：2026-09-14（本地时区 Asia/Shanghai）

## 结论

Agent 和用户模拟器均使用 `gpt-5.6-luna`、`reasoning=low`，在 legacy
τ-bench Retail 的 test/0–9 上完成了一套独立多轮运行：

- 官方 exact-state reward：10/10；
- 工具调用：71 次，失败 0 次；
- 用户回合：50 次；
- Agent steps：111，平均 11.1/题；
- 事后加入“Agent 声称 completed 后用户仍继续”的 dialogue audit 后：9/10；
- 唯一被 dialogue audit 降级的是 test/9：业务终态正确，但模拟器错误复活了
  已在确认时取消的水杯诉求，导致一次多余转人工；
- 增加“变更后的 scope 不得复活”约束后，test/9 定向回归通过，官方 reward=1、
  失败工具=0、completion continuation=0。

因此，这里最诚实的说法不是“官方 benchmark 100%”，而是：

> 在 10 条 legacy τ-bench Retail 多轮 pilot 上取得 10/10 exact-state、9/10
> dialogue-audited pass，并修复了唯一被审计发现的用户模拟器 scope 回退问题。

样本很少、使用的是 legacy 数据，并且经历过针对模拟器协议的校准，不能当作正式
榜单成绩。

## 运行协议

- 外部环境：Sierra Research `tau-bench` Retail；
- 固定 commit：`59a200c6d575d595120f1cb70fea53cef0632f6b`；
- split：`test`；task index：0–9；
- Agent：`gpt-5.6-luna`；
- User simulator：`gpt-5.6-luna` + grounded authentication；
- 每题最大 30 个 Agent steps；
- Agent 只能看到当前用户话语、可见对话、Retail policy、工具 schema 和工具结果；
- 隐藏 task instruction 只发给用户模拟器，gold actions 不发给 Agent；
- `FinalAction` 是一轮用户可见回复，不自动结束会话；
- 用户回复 `###STOP###` 或上游环境进入 terminal tool 后才结束；
- 最终调用上游 reward 验证业务终态。

grounded authentication 只做受限事实解析：使用隐藏指令明确给出的 email，或从
`first_last_digits` 形式的 benchmark account ID 中拆出姓名，并读取明确的五位
ZIP。它不会读取 gold actions，也不会向 Agent 暴露隐藏指令。

## 最终 10 题批次

| Task | Steps | Tool calls | User turns | Failed tools | Reward | Completion continuation | Audited result |
|---:|---:|---:|---:|---:|---:|---:|:---|
| 0 | 9 | 5 | 5 | 0 | 1.0 | 0 | PASS |
| 1 | 9 | 6 | 4 | 0 | 1.0 | 0 | PASS |
| 2 | 14 | 10 | 5 | 0 | 1.0 | 0 | PASS |
| 3 | 12 | 9 | 4 | 0 | 1.0 | 0 | PASS |
| 4 | 16 | 11 | 6 | 0 | 1.0 | 0 | PASS |
| 5 | 11 | 6 | 6 | 0 | 1.0 | 0 | PASS |
| 6 | 10 | 6 | 5 | 0 | 1.0 | 0 | PASS |
| 7 | 10 | 6 | 5 | 0 | 1.0 | 0 | PASS |
| 8 | 10 | 6 | 5 | 0 | 1.0 | 0 | PASS |
| 9 | 10 | 6 | 5 | 0 | 1.0 | 1 | FAIL (dialogue audit) |

原始轨迹位于：

```text
.refundpilot_runs/dialogue-test-0-9-grounded/
```

批次运行时 CLI 打印的是 10/10，因为当时成功条件为“reward=1 + 所有工具成功 +
环境正常结束”。人工检查 test/9 后，Harness 新增
`continued_after_agent_completed` 审计项，并据原始 JSONL 重新计算为 9/10。
原日志未被覆盖。

## 校准批次与失败证据

在最终批次之前保留了一套未完成 grounded/auth/dialogue 校准的探索运行：

- 旧严格口径：7/10；
- 官方 exact-state reward：8/10；
- test/5：模拟器拒绝从 `mei_kovacs_8020` 提供 Mei Kovacs，认证重复后转人工；
- test/7：Agent 将可售 AC adapter 低亮度灯误判为不可售，选错 battery variant；
- test/9：模拟器编造 `mei.kovacs@example.com`，产生一次失败认证工具调用；之后虽
  达到正确终态，仍被 strict tool-success 指标判失败。

探索轨迹位于：

```text
.refundpilot_runs/dialogue-test-0-9/
```

这三种失败分别对应 Harness 应该分开的三层问题：用户模拟器真实性、Agent
业务推理、工具执行可靠性。test/7 在最终独立批次中选择了正确 AC variant，说明
该错误具有模型采样不稳定性；探索失败仍保留，不用后一次成功覆盖。

## test/9 定向回归

增加“确认触发的范围变更永久替换旧 scope，完成后不得返回 opening request”后：

```text
session: tau-test-009-20260913T164212-8ead407e
reward: 1.0
tool calls: 6
failed tools: 0
user turns: 5
completion continuations: 0
result: PASS
```

回归轨迹位于：

```text
.refundpilot_runs/dialogue-regression/
```

## 复现

```bash
cd /Users/ben/LLM-Agent/refundpilot-harness
./scripts/setup_tau_bench.sh
./scripts/run_tau_dialogue_first10.sh \
  .refundpilot_runs/dialogue-test-0-9-grounded
```

单题运行：

```bash
.venv/bin/python -m refundpilot \
  --runtime-dir .refundpilot_runs/dialogue-regression \
  tau-dialogue --split test --task 9 \
  --agent-model gpt-5.6-luna --user-model gpt-5.6-luna \
  --reasoning low --max-steps 30
```

## 局限

- 上游是已弃用的 legacy τ-bench；最新任务在 τ³-bench / `tau2-bench`；
- 仅 10 个连续样本，不能代表 115 题 test split；
- 同一模型同时扮演 Agent 与用户，存在 correlated behavior；
- 单次采样不足以估计稳定性，test/7 已显示同题跨运行可能改变结果；
- exact-state reward 不覆盖所有对话质量问题，因此另外增加了 completion
  continuation 审计；
- 运行使用登录态 Codex 配额，不提供逐调用美元成本。
