# Auto-Skill

Auto-Skill 是一个用于创建、测试、评估和迭代优化 Skill 的元 Skill。它不直接替你完成某一类任务，而是帮助你把“一个能解决任务的 Skill”打磨得更可靠。

它的核心思想很简单：用真实案例跑结果，让裁判基于输出给出判断，再由用户拍板，最后才交给编辑者修改 Skill。

## 什么时候使用

当你想做下面这些事时，可以使用 Auto-Skill：

- 从零创建一个新的 Skill
- 优化、打磨、校准已有 Skill
- 测试一个 Skill 是否真的有效
- 对 Skill 做 benchmark、真实案例测试或回归测试
- 根据测试结果反推 Skill 的设计缺陷
- 比较 v0、v1、v2 等不同版本的实际输出效果

## 核心流程

Auto-Skill 把一次优化拆成几个角色，避免同一个模型既当运动员又当裁判。

1. **建立评估契约**：先明确哪些指标可以机器判断，哪些只能由 LLM 辅助判断，哪些必须由人判断。
2. **执行者 Runner 跑测试**：用真实案例跑旧版、新版或无 Skill 版本，保存完整输出。
3. **裁判 Judge 做判断**：只看输出，不直接改 Skill，给出证据、置信度、风险和修改方向。
4. **用户拍板**：用户判断裁判方向是否正确；用户不认可时，不进入编辑。
5. **编辑者 Editor 修改**：只根据“裁判建议 + 用户拍板意见”的合成指令生成候选版。
6. **再次测试和复核**：旧版和候选版跑同一组题，裁判复核变化和副作用。
7. **用户决定是否继续**：接受并继续、接受并停止、拒绝、补测试、回退或停止。

一句话：**AI 负责跑量和整理证据，人负责把方向。**

## 关键原则

- 不允许只凭静态阅读就修改 Skill。
- 不允许没有真实案例就给出优化结论。
- 不允许把 LLM 裁判建议当成用户裁决。
- 不允许在用户接受前覆盖原 Skill。
- 不允许只给总结，必须展示左右版本的完整输出对比。
- 裁判必须按评价表给证据、置信度和风险。
- 编辑者只能在用户拍板后动手。
- 用户叫停、暂停或选择停止时，循环立即截止。
- Skill 内容应兼容标准 chat-completions / responses 风格消息，不依赖厂商或平台专有能力。

## 仓库结构

```text
.
├── README.md
├── SKILL.md
├── core/
│   ├── __init__.py
│   └── isolation.py
└── phases/
    ├── phase0_interview.py
    ├── phase1_baseline.py
    └── phase2_optimize.py
```

## 文件说明

- `SKILL.md`：Auto-Skill 的主体说明，包含触发条件、硬规则、角色分工和完整优化流程。
- `core/isolation.py`：通用 LLM 调用、测试运行、Judge/Edit/Gate 提示词构建。
- `phases/phase0_interview.py`：从自然语言需求访谈生成 Skill 初版。
- `phases/phase1_baseline.py`：跑 with-skill / without-skill 基线测试。
- `phases/phase2_optimize.py`：运行迭代优化循环。

## 使用方式

把本仓库作为一个 Skill 目录使用即可。最核心的是加载 `SKILL.md`。

如果需要使用脚本，准备 Python 3.10+ 环境，并设置兼容 OpenAI API 的环境变量：

```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

基线测试示例：

```bash
python phases/phase1_baseline.py \
  --skill-path ./SKILL.md \
  --evals ./evals.json \
  --output-dir ./outputs/baseline \
  --model gpt-4o
```

优化循环示例：

```bash
python phases/phase2_optimize.py \
  --skill-path ./SKILL.md \
  --train-evals ./train.json \
  --val-evals ./val.json \
  --output-dir ./outputs/iterations \
  --model gpt-4o
```

真实优化时不要使用自动接受模式。用户必须在裁判报告后拍板。

## 适合什么场景

Auto-Skill 尤其适合那些“能通过真实输出看出好坏，但单靠 prompt 很难一次写准”的任务，例如：

- 写作类 Skill
- 招聘评估类 Skill
- 文档生成类 Skill
- 数据处理类 Skill
- 代码辅助类 Skill
- 分析研究类 Skill

不同任务的评价方式不同。代码类可以跑测试，结构化输出可以跑解析器，写作风格和业务语气则必须由用户最终判断。

## 当前版本重点

当前版本重点强化了三件事：

1. **重要规则前置**：触发条件、硬规则、角色分工放在 `SKILL.md` 最前面。
2. **裁判更严谨**：裁判必须给评价维度、证据、置信度、风险和副作用。
3. **用户拍板优先**：裁判之后必须停下来让用户判断，编辑者只能按用户认可后的方向修改。