#!/usr/bin/env python3
"""
Auto-Skill Phase 1：基线测试

用法：
  python phase1_baseline.py \\
    --skill-path ./path/to/SKILL.md \\
    --evals ./evals.json \\
    --output-dir ./outputs/baseline \\
    --model gpt-4o \\
    --api-key $OPENAI_API_KEY

功能：
  1. 加载 Skill 和测试用例
  2. 对每个用例并行运行 with_skill 和 without_skill（双配置）
  3. 收集输出、时间、token 消耗
  4. 生成 baseline 报告
"""

import json
import os
import sys
from pathlib import Path

# Add parent to path for core imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.isolation import call_llm, run_skill_test, RunResult


def load_skill(skill_path: str) -> str:
    """Load SKILL.md content."""
    with open(skill_path, "r", encoding="utf-8") as f:
        return f.read()


def load_evals(evals_path: str) -> list:
    """Load test cases from JSON."""
    with open(evals_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_baseline(
    skill_content: str,
    evals: list,
    output_dir: str,
    model: str = "gpt-4o",
    api_key: str = None,
    base_url: str = None,
) -> dict:
    """
    运行基线测试：对每个用例跑 with_skill 和 without_skill。
    返回 {results: [...], timing: {...}, summary: {...}}
    """
    results = []
    output_path = Path(output_dir)

    for ev in evals:
        eval_id = ev["id"]
        eval_name = ev.get("name", eval_id)
        prompt = ev["prompt"]

        print(f"\n[{eval_id}] {eval_name}")

        # Run without_skill
        print(f"  without_skill...", end=" ", flush=True)
        r_without = run_skill_test(
            eval_id=eval_id,
            eval_name=eval_name,
            prompt=prompt,
            skill_content="",
            config="without_skill",
            output_dir=str(output_path / "results" / eval_id / "without_skill"),
            model=model,
            api_key=api_key,
            base_url=base_url,
        )
        print(f"tokens={r_without.total_tokens}, time={r_without.duration_ms}ms")

        # Run with_skill
        print(f"  with_skill...", end=" ", flush=True)
        r_with = run_skill_test(
            eval_id=eval_id,
            eval_name=eval_name,
            prompt=prompt,
            skill_content=skill_content,
            config="with_skill",
            output_dir=str(output_path / "results" / eval_id / "with_skill"),
            model=model,
            api_key=api_key,
            base_url=base_url,
        )
        print(f"tokens={r_with.total_tokens}, time={r_with.duration_ms}ms")

        results.append({
            "eval_id": eval_id,
            "eval_name": eval_name,
            "without_skill": {
                "tokens": r_without.total_tokens,
                "duration_ms": r_without.duration_ms,
            },
            "with_skill": {
                "tokens": r_with.total_tokens,
                "duration_ms": r_with.duration_ms,
            },
        })

    # Compute summary
    without_tokens = [r["without_skill"]["tokens"] for r in results]
    with_tokens = [r["with_skill"]["tokens"] for r in results]
    without_time = [r["without_skill"]["duration_ms"] for r in results]
    with_time = [r["with_skill"]["duration_ms"] for r in results]

    def mean(vals):
        return sum(vals) / len(vals) if vals else 0

    summary = {
        "num_evals": len(evals),
        "without_skill": {
            "avg_tokens": mean(without_tokens),
            "avg_duration_ms": mean(without_time),
        },
        "with_skill": {
            "avg_tokens": mean(with_tokens),
            "avg_duration_ms": mean(with_time),
        },
        "delta": {
            "tokens": mean(with_tokens) - mean(without_tokens),
            "duration_ms": mean(with_time) - mean(without_time),
        },
    }

    # Save baseline data
    baseline = {
        "skill_path": "",
        "model": model,
        "results": results,
        "summary": summary,
    }

    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "baseline.json", "w", encoding="utf-8") as f:
        json.dump(baseline, f, ensure_ascii=False, indent=2)

    # Generate markdown report
    report = generate_baseline_report(baseline, evals)
    with open(output_path / "baseline_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n基线报告已保存到: {output_path}/baseline_report.md")
    return baseline


def generate_baseline_report(baseline: dict, evals: list) -> str:
    """Generate human-readable baseline report."""
    s = baseline["summary"]
    lines = [
        "# 基线测试报告",
        "",
        f"**模型**: {baseline['model']}",
        f"**测试用例数**: {baseline['num_evals']}",
        "",
        "## 汇总对比",
        "",
        "| 指标 | Without Skill | With Skill | Δ |",
        "|------|-------------|-----------|----|",
        f"| 平均 Token | {s['without_skill']['avg_tokens']:.0f} | {s['with_skill']['avg_tokens']:.0f} | {s['delta']['tokens']:+.0f} |",
        f"| 平均时间(ms) | {s['without_skill']['avg_duration_ms']:.0f} | {s['with_skill']['avg_duration_ms']:.0f} | {s['delta']['duration_ms']:+.0f} |",
        "",
        "## 逐用例详情",
        "",
    ]

    for r, ev in zip(baseline["results"], evals):
        lines.append(f"### {r['eval_id']}: {r['eval_name']}")
        lines.append(f"**Prompt**: {ev['prompt'][:100]}...")
        lines.append(f"**Expected**: {ev.get('expected_output_summary', '-')}")
        lines.append("")
        lines.append(f"| 配置 | Token | 时间(ms) |")
        lines.append(f"|------|-------|---------|")
        lines.append(f"| Without Skill | {r['without_skill']['tokens']} | {r['without_skill']['duration_ms']} |")
        lines.append(f"| With Skill | {r['with_skill']['tokens']} | {r['with_skill']['duration_ms']} |")
        lines.append("")

    return "\n".join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Auto-Skill Phase 1: Baseline Testing")
    parser.add_argument("--skill-path", required=True, help="SKILL.md 路径")
    parser.add_argument("--evals", required=True, help="测试用例 JSON 文件路径")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--model", default="gpt-4o", help="LLM 模型 (default: gpt-4o)")
    parser.add_argument("--api-key", help="OpenAI API Key (或环境变量 OPENAI_API_KEY)")
    parser.add_argument("--base-url", help="API Base URL (或环境变量 OPENAI_BASE_URL)")
    args = parser.parse_args()

    skill_content = load_skill(args.skill_path)
    evals = load_evals(args.evals)

    print(f"Skill: {args.skill_path} ({len(skill_content)} chars)")
    print(f"Evals: {args.evals} ({len(evals)} cases)")
    print(f"Model: {args.model}")
    print()

    run_baseline(
        skill_content=skill_content,
        evals=evals,
        output_dir=args.output_dir,
        model=args.model,
        api_key=args.api_key or os.environ.get("OPENAI_API_KEY"),
        base_url=args.base_url or os.environ.get("OPENAI_BASE_URL"),
    )


if __name__ == "__main__":
    main()
