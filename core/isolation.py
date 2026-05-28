"""
Auto-Skill 核心模块：LLM 角色隔离 + API 调用封装

兼容标准 OpenAI-compatible API 格式，不依赖厂商或平台专有功能。
所有角色（Runner, Judge, Editor）通过此模块调用，
确保隔离——同一模型不会既当裁判又当运动员。
"""

import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

# --- Data structures ---

@dataclass
class RunResult:
    """单次 LLM 运行的结果"""
    eval_id: str
    config: str          # "with_skill" | "without_skill" | "with_v{N}"
    output: str
    total_tokens: int
    duration_ms: int
    model: str = ""

@dataclass
class JudgeReport:
    """Judge 阶段的诊断报告"""
    version: int
    skill_type: str      # "writing" | "engineering" | "analysis" | "other"
    user_feedback: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    val_results: list = field(default_factory=list)
    scene_coverage: dict = field(default_factory=dict)
    side_effects: list = field(default_factory=list)
    root_causes: list = field(default_factory=list)
    suggestions: list = field(default_factory=list)

@dataclass
class EditProposal:
    """Edit 阶段的编辑提案"""
    version: int
    edits: list = field(default_factory=list)  # [{type, target, old, new, reason}]
    total_change_chars: int = 0
    budget_used_pct: float = 0.0


# --- LLM call (OpenAI-compatible) ---

def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> dict:
    """
    标准 OpenAI-compatible API 调用。不依赖任何厂商或平台专有功能。
    返回 {"content": str, "total_tokens": int, "duration_ms": int}
    """
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")

    import requests

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    t0 = time.time()
    resp = requests.post(
        f"{base_url}/chat/completions",
        headers=headers,
        json=payload,
        timeout=300,
    )
    elapsed_ms = int((time.time() - t0) * 1000)

    if resp.status_code != 200:
        raise RuntimeError(f"API error {resp.status_code}: {resp.text}")

    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    total_tokens = data.get("usage", {}).get("total_tokens", 0)

    return {
        "content": content,
        "total_tokens": total_tokens,
        "duration_ms": elapsed_ms,
    }


# --- Runner ---

def run_skill_test(
    eval_id: str,
    eval_name: str,
    prompt: str,
    skill_content: str,
    config: str,
    output_dir: str,
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> RunResult:
    """
    运行单个测试用例。

    当 config="without_skill" 时，skill_content 为空字符串（纯模型能力）。
    当 config="with_skill" 时，skill_content 作为 system prompt 注入。
    """
    system_prompt = skill_content if skill_content else "You are a helpful assistant."

    result = call_llm(
        system_prompt=system_prompt,
        user_prompt=prompt,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )

    # Save output
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "output.md"), "w", encoding="utf-8") as f:
        f.write(result["content"])

    # Save timing
    with open(os.path.join(output_dir, "timing.json"), "w", encoding="utf-8") as f:
        json.dump({
            "total_tokens": result["total_tokens"],
            "duration_ms": result["duration_ms"],
        }, f, indent=2)

    return RunResult(
        eval_id=eval_id,
        config=config,
        output=result["content"],
        total_tokens=result["total_tokens"],
        duration_ms=result["duration_ms"],
        model=model,
    )


# --- Judge templates ---

JUDGE_SYSTEM_PROMPT = """You are a Skill Optimization Judge. Your job is to analyze Skill outputs, provide a rigorous evaluation table, and identify candidate improvement directions for the human to approve.

## Your Role
- You can ONLY read outputs. You cannot modify the Skill.
- You must diagnose based on the skill type (writing/engineering/analysis/other).
- User feedback has the HIGHEST priority. If user says X is wrong, X IS wrong.
- Your recommendation is advisory. The human decides whether an Editor may act.

## Diagnostic Dimensions by Type

### All types (required):
1. Functional correctness - does output follow the Skill's defined rules?
2. Multi-scene adaptability - does it work across different scenarios?
3. Side effects - did new rules break old capabilities?
4. User satisfaction - would the user accept this output?

### Writing/Creative skills (extra):
5. AI-voice elimination - mechanical structures, cliche phrases
6. Style consistency - does tone match expectations?
7. Format compliance - does output follow required templates?

### Engineering/Code skills (extra):
5. Executability - can the code run?
6. Correctness - does output follow syntax/protocol?
7. Performance - token usage, response time
8. Security - any vulnerabilities introduced?

### Analysis/Decision skills (extra):
5. Logical rigor - any reasoning gaps?
6. Data accuracy - are cited facts correct?
7. Conclusion reliability - any over-extrapolation?

## Output Format
Output a JSON object with these fields:
{
  "skill_type": "writing|engineering|analysis|other",
  "overall_assessment": "brief summary",
  "user_feedback_summary": ["point 1", "point 2"],
  "metrics": {"pass_rate": 0.0, "avg_tokens": 0, "avg_duration_ms": 0},
  "val_performance": [{"eval_id": "", "scene": "", "score": "", "issues": []}],
  "scene_coverage": {"covered": [], "missing": []},
  "side_effects": [{"rule": "", "effect": "", "severity": "high|medium|low"}],
  "root_causes": [{"cause": "", "frequency": 0, "affected_evals": []}],
  "suggestions": [{"target": "SKILL.md line/section", "action": "", "reason": ""}],
  "verdict": "accept|reject|needs_more_testing"
}
"""


def build_judge_prompt(
    skill_content: str,
    skill_version: int,
    run_results: list[RunResult],
    previous_report: Optional[str] = None,
    user_feedback: Optional[str] = None,
    previous_skill_content: Optional[str] = None,
) -> str:
    """Build the Judge prompt with all diagnostic data."""
    results_text = "\n\n".join([
        f"### Eval: {r.eval_id} ({r.config})\n```\n{r.output[:3000]}\n```\n"
        f"Tokens: {r.total_tokens}, Duration: {r.duration_ms}ms"
        for r in run_results
    ])

    prompt = f"""## Current Skill (v{skill_version})
```
{skill_content}
```
"""
    if previous_skill_content:
        prompt += f"""
## Previous Skill (v{skill_version - 1})
```
{previous_skill_content}
```
"""
    prompt += f"""
## Run Results
{results_text}
"""
    if user_feedback:
        prompt += f"""
## User Feedback (HIGHEST PRIORITY)
{user_feedback}
"""
    if previous_report:
        prompt += f"""
## Previous Judge Report
{previous_report}
"""

    prompt += """

## Instructions
Analyze the outputs above. Focus on:
1. What SPECIFICALLY went wrong (quote exact text)
2. Which Skill rules caused the problem (cite line/section)
3. What should change (be specific, not "improve tone" but "add rule: use casual address for WeChat articles")
4. Are there side effects from previous edits?

Output your analysis as a JSON object per the system prompt format."""
    return prompt


# --- Edit templates ---

EDIT_SYSTEM_PROMPT = """You are a Skill Optimization Editor. Your job is to propose precise, minimal edits to a SKILL.md file based on a diagnostic report.

## Rules
1. Total change per round ≤ 30% of SKILL.md character count (text learning rate budget)
2. Edits must be specific: target exact section/line, show old text and new text
3. Prioritize high-impact, low-risk edits
4. Check rejected_edits.json - don't repeat previously rejected edits
5. You can ONLY propose edits. You cannot run tests.

## Edit Types
- ADD: Insert new rule/constraint after existing content
- MODIFY: Change existing rule/example
- DELETE: Remove rule causing negative effects
- RESTRUCTURE: Reorder/merge sections

## Output Format
Output a JSON object:
{
  "edits": [
    {
      "type": "ADD|MODIFY|DELETE|RESTRUCTURE",
      "target": "section name or line range",
      "old_text": "exact text to replace (empty for ADD)",
      "new_text": "replacement text (empty for DELETE)",
      "reason": "why this edit, linked to Reflect finding",
      "confidence": "high|medium|low"
    }
  ],
  "total_change_chars": 0,
  "budget_used_pct": 0.0,
  "explanation": "brief summary of edit strategy"
}
"""


def build_edit_prompt(
    skill_content: str,
    judge_report: str,
    rejected_edits: Optional[list] = None,
) -> str:
    """Build the Edit prompt."""
    budget = len(skill_content) * 0.3

    prompt = f"""## Current SKILL.md
```
{skill_content}
```

## Judge Report And Human-Approved Direction
{judge_report}

## Text Learning Rate Budget
Max change: {int(budget)} characters (30% of {len(skill_content)} chars)
"""
    if rejected_edits:
        prompt += f"""
## Previously Rejected Edits (DO NOT REPEAT)
```json
{json.dumps(rejected_edits, indent=2, ensure_ascii=False)}
```
"""
    prompt += """
## Instructions
Propose edits only for the issues and direction approved by the human after the Judge report.
Each edit must target a specific location in SKILL.md.
Stay within the text learning rate budget.
Output as JSON per the system prompt format."""
    return prompt


# --- Gate templates ---

GATE_SYSTEM_PROMPT = """You are a Skill Optimization Judge for candidate review. Your job is to present held-out validation results for HUMAN judgment.

## Rules
1. You can run held-out validation tests, but CANNOT modify the Skill
2. You present complete raw outputs side-by-side (v{N} vs v{N-1}) - never summarize
3. Your recommendation is advisory only. The human makes the final decision.
4. Held-out validation set must NOT have been seen by the Editor.

## Output Format
Output a JSON object:
{
  "recommendation": "accept|reject|partial",
  "reason": "brief justification",
  "comparisons": [
    {
      "eval_id": "",
      "scene": "",
      "v_old_key_features": "",
      "v_new_key_features": "",
      "direction": "improved|degraded|unchanged",
      "side_effects": []
    }
  ],
  "new_rules_effects": [
    {"rule": "", "expected": "", "actual": "positive|negative|neutral"}
  ],
  "uncovered_scenarios": []
}
"""


def build_gate_prompt(
    skill_new: str,
    skill_old: str,
    val_results_new: list[RunResult],
    val_results_old: list[RunResult],
    edit_proposal: EditProposal,
) -> str:
    """Build the Gate prompt with full comparison data."""
    comparisons = []
    for r_new, r_old in zip(val_results_new, val_results_old):
        comparisons.append(f"""
### Eval: {r_new.eval_id}
#### v{NEW}
```
{r_new.output}
```
#### v{OLD}
```
{r_old.output}
```
""")

    prompt = f"""## Edit Summary
{len(edit_proposal.edits)} edits proposed, {edit_proposal.total_change_chars} chars changed ({edit_proposal.budget_used_pct:.1f}% budget)

## Full Comparison
{''.join(comparisons)}

## Instructions
Compare the outputs. Look for:
1. Did the edit fix the intended issue?
2. Did it break anything else?
3. Is the overall direction correct?

Output as JSON per the system prompt format.
IMPORTANT: Your recommendation is advisory. The human decides."""
    return prompt
