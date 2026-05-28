#!/usr/bin/env python3
"""
Auto-Skill Phase 0：半自动 Interview → 生成 Skill 初版

用法：
  python phase0_interview.py --skill-name "my-skill" --output-dir ./outputs

流程：
  1. 加载 interview 模板
  2. 逐个维度向用户追问（交互式 Q&A）
  3. 根据用户回答生成 SKILL.md 初版
  4. 用户确认后保存
"""

import json
import os
import sys
from pathlib import Path


INTERVIEW_DIMENSIONS = [
    {
        "id": "scope",
        "title": "范围与分类",
        "questions": [
            "这个 Skill 处理的任务有哪些类型？（如写作Skill：公文、汇报、邮件、随笔...）",
            "每种类型有什么独特的输入要求或输出格式？",
            "用户是否需要先选择类型，还是由 Skill 自动判断？",
        ],
    },
    {
        "id": "structure",
        "title": "输出结构与流程",
        "questions": [
            "任务的完成需要分几步？（如：大纲→内容→风格校准）",
            "每一步的输入输出是什么？",
            "是否有必须遵循的固定模板？",
        ],
    },
    {
        "id": "quality",
        "title": "质量与标准",
        "questions": [
            "输出需要符合什么标准？（如公文 GB/T 9704-2012）",
            "\"好\"的输出和\"差\"的输出区别是什么？",
            "有没有需要绝对避免的问题？（如 AI 味过重、格式错误、安全漏洞）",
        ],
    },
    {
        "id": "format",
        "title": "格式与交付",
        "questions": [
            "支持哪些输出格式？（md / docx / pdf / json / 代码文件）",
            "是否有特定的排版或结构要求？",
            "是否需要引用、附件、元数据？",
        ],
    },
    {
        "id": "examples",
        "title": "示例与反例",
        "questions": [
            "能否给一个\"理想输出\"的示例？（可直接粘贴）",
            "能否给一个\"绝对不能出现\"的反例？",
        ],
    },
    {
        "id": "compatibility",
        "title": "兼容性要求",
        "questions": [
            "这个 Skill 需要在哪些运行环境上使用？（标准 API / 本地工具 / 其他）",
            "是否有任何平台特有的限制？",
        ],
    },
]


def collect_answers_interactive() -> dict:
    """交互式收集用户回答。"""
    answers = {}
    print("=" * 60)
    print("Auto-Skill Phase 0：Interview")
    print("=" * 60)
    print()
    print("我将从 6 个维度向你提问，帮助你明确 Skill 的需求。")
    print("输入 'skip' 跳过当前问题，输入 'done' 提前结束。")
    print()

    for dim in INTERVIEW_DIMENSIONS:
        print(f"\n--- {dim['title']} ---")
        dim_answers = []
        for i, q in enumerate(dim["questions"]):
            print(f"\n[{dim['id']}.{i+1}] {q}")
            answer = input("> ").strip()
            if answer.lower() == "done":
                answers[dim["id"]] = dim_answers
                return answers
            if answer.lower() == "skip":
                dim_answers.append({"question": q, "answer": None})
            else:
                dim_answers.append({"question": q, "answer": answer})
        answers[dim["id"]] = dim_answers

    return answers


def load_answers_from_file(filepath: str) -> dict:
    """从 JSON 文件加载预先准备好的回答。"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_skill_prompt(answers: dict) -> str:
    """
    根据 Interview 回答生成 SKILL.md 的 system prompt。
    这个 prompt 会发给 LLM 来生成实际的 SKILL.md。
    """
    scope_answers = answers.get("scope", [])
    structure_answers = answers.get("structure", [])
    quality_answers = answers.get("quality", [])
    format_answers = answers.get("format", [])
    examples_answers = answers.get("examples", [])
    compat_answers = answers.get("compatibility", [])

    def fmt(qa_list):
        lines = []
        for qa in qa_list:
            if qa.get("answer"):
                lines.append(f"Q: {qa['question']}\nA: {qa['answer']}")
        return "\n".join(lines) or "(未提供)"

    return f"""你是一个 Skill 创建助手。根据以下 Interview 回答，生成一个完整的 SKILL.md 文件。

## 用户需求

### 范围与分类
{fmt(scope_answers)}

### 输出结构与流程
{fmt(structure_answers)}

### 质量与标准
{fmt(quality_answers)}

### 格式与交付
{fmt(format_answers)}

### 示例与反例
{fmt(examples_answers)}

### 兼容性要求
{fmt(compat_answers)}

## SKILL.md 格式要求

生成的 SKILL.md 必须包含以下 YAML frontmatter：

```yaml
---
name: {skill_name}
description: |
  {{简短描述 Skill 的功能和触发条件}}
compatibility: Standard OpenAI-compatible API, (可补充其他依赖)
---
```

## 重要约束

1. **兼容性**：所有指令必须兼容标准 OpenAI-compatible API 格式。禁止依赖厂商或平台专有功能（专用 thinking 字段、专用 reasoning 字段、固定输出限制、专用 IDE/CLI 钩子等）。
2. **结构**：使用清晰的标题层次（# ## ###），核心流程放在前面，详细规则在后面。
3. **示例**：包含 2-3 个具体的使用示例。
4. **长度**：SKILL.md 控制在 500 行以内。精简但完整。
5. **语言**：根据用户偏好选择中文或英文。默认使用与用户提问一致的语言。

请直接输出完整的 SKILL.md 内容，不要有任何前言或后记。"""


def save_skill(skill_content: str, output_dir: str, version: str = "v0"):
    """保存 SKILL.md 到指定目录。"""
    skill_dir = Path(output_dir) / version
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_path = skill_dir / "SKILL.md"
    with open(skill_path, "w", encoding="utf-8") as f:
        f.write(skill_content)
    print(f"\nSkill 已保存到: {skill_path}")
    return str(skill_path)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Auto-Skill Phase 0: Interview")
    parser.add_argument("--skill-name", required=True, help="Skill 名称")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--answers-file", help="预先准备的 Interview 回答 JSON 文件")
    parser.add_argument("--interactive", action="store_true", default=True,
                        help="交互式模式（默认）")
    args = parser.parse_args()

    # 收集回答
    if args.answers_file:
        answers = load_answers_from_file(args.answers_file)
        print(f"从 {args.answers_file} 加载了 Interview 回答")
    else:
        answers = collect_answers_interactive()

    # 保存 Interview 回答
    answers_path = Path(args.output_dir) / args.skill_name / "interview_answers.json"
    answers_path.parent.mkdir(parents=True, exist_ok=True)
    with open(answers_path, "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)
    print(f"Interview 回答已保存到: {answers_path}")

    # 生成 SKILL.md prompt
    prompt = generate_skill_prompt(answers)
    prompt_path = Path(args.output_dir) / args.skill_name / "generation_prompt.txt"
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt)
    print(f"生成提示词已保存到: {prompt_path}")
    print()
    print("=" * 60)
    print("下一步：将 generation_prompt.txt 发送给 LLM 生成 SKILL.md")
    print("生成后，使用 phase1_baseline.py 运行基线测试。")
    print("=" * 60)


if __name__ == "__main__":
    main()
