#!/usr/bin/env python3
"""
Auto-Skill Phase 2：优化循环（Ratchet Loop）

用法：
  python phase2_optimize.py \\
    --skill-path ./path/to/SKILL.md \\
    --train-evals ./train.json \\
    --val-evals ./val.json \\
    --output-dir ./outputs/iterations \\
    --model gpt-4o \\
    --max-rounds 10 \\
    --api-key $OPENAI_API_KEY

流程（每轮）：
  1. Runner：采样训练集，运行当前版本并保存完整输出
  2. Judge：独立调用分析输出，生成评价表和修改方向
  3. 用户拍板：认可、修正、补测试或停止；用户认可后才进入编辑
  4. Editor：基于“裁判建议 + 用户拍板意见”的合成输入生成候选版
  5. Runner/Judge：运行 held-out 验证集，展示 v{N} vs v{N-1} 完整对比
  6. 用户拍板 → accept_continue / accept_stop / reject / needs_more / stop
"""

import json
import os
import sys
from pathlib import Path
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.isolation import (
    call_llm, run_skill_test, RunResult, EditProposal,
    JUDGE_SYSTEM_PROMPT, build_judge_prompt,
    EDIT_SYSTEM_PROMPT, build_edit_prompt,
    GATE_SYSTEM_PROMPT, build_gate_prompt,
)


def rollout(
    skill_content: str,
    train_evals: list,
    sample_size: int,
    output_dir: str,
    iteration: int,
    model: str,
    api_key: str,
    base_url: str,
) -> list[RunResult]:
    """从训练集采样并运行 with_skill。"""
    import random
    sample = random.sample(train_evals, min(sample_size, len(train_evals)))

    results = []
    for ev in sample:
        eval_id = ev["id"]
        r = run_skill_test(
            eval_id=eval_id,
            eval_name=ev.get("name", eval_id),
            prompt=ev["prompt"],
            skill_content=skill_content,
            config=f"with_v{iteration}",
            output_dir=str(Path(output_dir) / f"iteration-{iteration}" / "rollout" / eval_id),
            model=model,
            api_key=api_key,
            base_url=base_url,
        )
        results.append(r)
        print(f"  Rollout [{eval_id}]: tokens={r.total_tokens}, time={r.duration_ms}ms")

    return results


def reflect(
    skill_content: str,
    skill_version: int,
    rollout_results: list[RunResult],
    output_dir: str,
    iteration: int,
    model: str,
    api_key: str,
    base_url: str,
    previous_report: str = None,
    user_feedback: str = None,
    previous_skill: str = None,
) -> str:
    """调用 Judge（独立 LLM 调用）生成诊断报告。"""
    prompt = build_judge_prompt(
        skill_content=skill_content,
        skill_version=skill_version,
        run_results=rollout_results,
        previous_report=previous_report,
        user_feedback=user_feedback,
        previous_skill_content=previous_skill,
    )

    result = call_llm(
        system_prompt=JUDGE_SYSTEM_PROMPT,
        user_prompt=prompt,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.3,
    )

    # Save report
    report_path = Path(output_dir) / f"iteration-{iteration}" / "reflect_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(result["content"])

    print(f"  Reflect 报告已保存: {report_path}")
    return result["content"]


def edit(
    skill_content: str,
    reflect_report: str,
    output_dir: str,
    iteration: int,
    model: str,
    api_key: str,
    base_url: str,
    rejected_edits: list = None,
) -> EditProposal:
    """调用 Editor（独立 LLM 调用）生成编辑提案。"""
    prompt = build_edit_prompt(
        skill_content=skill_content,
        reflect_report=reflect_report,
        rejected_edits=rejected_edits,
    )

    result = call_llm(
        system_prompt=EDIT_SYSTEM_PROMPT,
        user_prompt=prompt,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.3,
    )

    # Parse JSON from response
    content = result["content"]
    # Try to extract JSON block
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        print("  WARNING: Could not parse Editor JSON output, saving raw")
        data = {"edits": [], "total_change_chars": 0, "budget_used_pct": 0}

    proposal = EditProposal(
        version=iteration,
        edits=data.get("edits", []),
        total_change_chars=data.get("total_change_chars", 0),
        budget_used_pct=data.get("budget_used_pct", 0),
    )

    # Save proposal
    proposal_path = Path(output_dir) / f"iteration-{iteration}" / "edit_proposal.json"
    proposal_path.parent.mkdir(parents=True, exist_ok=True)
    with open(proposal_path, "w", encoding="utf-8") as f:
        json.dump({
            "version": iteration,
            "edits": proposal.edits,
            "total_change_chars": proposal.total_change_chars,
            "budget_used_pct": proposal.budget_used_pct,
        }, f, ensure_ascii=False, indent=2)

    print(f"  Edit 提案: {len(proposal.edits)} edits, {proposal.total_change_chars} chars "
          f"({proposal.budget_used_pct:.1f}% budget)")
    return proposal


def apply_edits(skill_content: str, edits: list) -> str:
    """
    Apply edits to SKILL.md content.
    Each edit has: type, target, old_text, new_text.
    Returns modified content.
    """
    modified = skill_content
    for ed in edits:
        old = ed.get("old_text", "")
        new = ed.get("new_text", "")
        ed_type = ed.get("type", "MODIFY")

        if ed_type == "DELETE":
            if old in modified:
                modified = modified.replace(old, "")
        elif ed_type == "ADD":
            # ADD appends to the end of the referenced section
            # Simple implementation: append after the last occurrence of target
            target = ed.get("target", "")
            if target and target in modified:
                idx = modified.rfind(target) + len(target)
                modified = modified[:idx] + "\n\n" + new + modified[idx:]
            else:
                modified += "\n\n" + new
        elif ed_type == "MODIFY":
            if old in modified:
                modified = modified.replace(old, new, 1)
        elif ed_type == "RESTRUCTURE":
            # RESTRUCTURE is a complex edit — apply as MODIFY for now
            if old in modified:
                modified = modified.replace(old, new, 1)

    return modified


def gate(
    skill_new: str,
    skill_old: str,
    val_evals: list,
    output_dir: str,
    iteration: int,
    model: str,
    api_key: str,
    base_url: str,
    edit_proposal: EditProposal,
) -> tuple[list[RunResult], list[RunResult], str]:
    """
    运行 held-out 验证集，生成对比报告。
    返回 (v_new_results, v_old_results, gate_report)
    """
    val_results_new = []
    val_results_old = []

    val_dir = Path(output_dir) / f"iteration-{iteration}" / "gate"
    val_dir.mkdir(parents=True, exist_ok=True)

    for ev in val_evals:
        eval_id = ev["id"]
        eval_name = ev.get("name", eval_id)

        # Run with new skill
        r_new = run_skill_test(
            eval_id=eval_id,
            eval_name=eval_name,
            prompt=ev["prompt"],
            skill_content=skill_new,
            config=f"with_v{iteration}",
            output_dir=str(val_dir / eval_id / "v_new"),
            model=model,
            api_key=api_key,
            base_url=base_url,
        )
        val_results_new.append(r_new)

        # Run with old skill
        r_old = run_skill_test(
            eval_id=eval_id,
            eval_name=eval_name,
            prompt=ev["prompt"],
            skill_content=skill_old,
            config=f"with_v{iteration - 1}",
            output_dir=str(val_dir / eval_id / "v_old"),
            model=model,
            api_key=api_key,
            base_url=base_url,
        )
        val_results_old.append(r_old)

        print(f"  Gate [{eval_id}]: v{iteration} tokens={r_new.total_tokens} | "
              f"v{iteration - 1} tokens={r_old.total_tokens}")

    # Generate gate report
    prompt = build_gate_prompt(
        skill_new=skill_new,
        skill_old=skill_old,
        val_results_new=val_results_new,
        val_results_old=val_results_old,
        edit_proposal=edit_proposal,
    )

    result = call_llm(
        system_prompt=GATE_SYSTEM_PROMPT,
        user_prompt=prompt,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.3,
    )

    # Save gate report
    gate_path = val_dir / "gate_report.md"
    with open(gate_path, "w", encoding="utf-8") as f:
        f.write(f"# Gate 验证报告 Iteration {iteration}\n\n")
        f.write(result["content"])
        f.write(
            "\n\n## 用户拍板区\n\n"
            "- 用户判断：accept_continue / accept_stop / reject / needs_more / stop\n"
            "- 用户修正意见：\n"
        )

    print(f"  Gate 报告已保存: {gate_path}")
    return val_results_new, val_results_old, result["content"]


def run_optimization_loop(
    skill_path: str,
    train_evals_path: str,
    val_evals_path: str,
    output_dir: str,
    model: str = "gpt-4o",
    api_key: str = None,
    base_url: str = None,
    max_rounds: int = 10,
    sample_size: int = 3,
    auto_accept: bool = False,
) -> str:
    """
    运行完整的 Phase 2 优化循环。
    返回用户已接受版本的 Skill 路径。
    """
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    base_url = base_url or os.environ.get("OPENAI_BASE_URL")

    # Load data
    with open(skill_path, "r", encoding="utf-8") as f:
        skill_content = f.read()
    with open(train_evals_path, "r", encoding="utf-8") as f:
        train_evals = json.load(f)
    with open(val_evals_path, "r", encoding="utf-8") as f:
        val_evals = json.load(f)

    output_path = Path(output_dir)
    rejected_edits = []
    previous_reflect = None
    skill_history = [skill_content]  # skill_history[0] = v0
    current_version = 0

    print(f"Auto-Skill Phase 2: 优化循环")
    print(f"  Skill: {skill_path} ({len(skill_content)} chars)")
    print(f"  Train: {train_evals_path} ({len(train_evals)} cases)")
    print(f"  Val: {val_evals_path} ({len(val_evals)} cases)")
    print(f"  Model: {model}")
    print(f"  Max rounds: {max_rounds}")
    print()

    for round_num in range(1, max_rounds + 1):
        print(f"{'='*60}")
        print(f"Round {round_num}/{max_rounds}")
        print(f"{'='*60}")

        # 1. Rollout
        print(f"[Rollout] Sampling {sample_size} from train set...")
        rollout_results = rollout(
            skill_content=skill_content,
            train_evals=train_evals,
            sample_size=sample_size,
            output_dir=output_dir,
            iteration=round_num,
            model=model,
            api_key=api_key,
            base_url=base_url,
        )

        # 2. Reflect
        print(f"[Reflect] Analyzing outputs...")
        previous_skill = skill_history[-2] if len(skill_history) >= 2 else None
        reflect_report = reflect(
            skill_content=skill_content,
            skill_version=current_version,
            rollout_results=rollout_results,
            output_dir=output_dir,
            iteration=round_num,
            model=model,
            api_key=api_key,
            base_url=base_url,
            previous_report=previous_reflect,
            previous_skill=previous_skill,
        )
        previous_reflect = reflect_report

        # 3. User direction decision. Editor cannot run before this point.
        if auto_accept:
            print("\n[AUTO] Direction approved (auto_accept=True, testing only)")
            direction_decision = "proceed"
            user_direction = "自动测试模式：认可 Judge 方向。"
        else:
            print("\n" + "=" * 60)
            print(f"请先查看 Judge 报告: {output_path}/iteration-{round_num}/reflect_report.md")
            print("用户拍板: proceed / revise / needs_more / stop")
            direction_decision = input("> ").strip().lower()
            if direction_decision == "revise":
                print("请输入用户修正后的方向，Editor 只能按这个方向生成候选版：")
                user_direction = input("> ").strip()
            else:
                user_direction = f"用户拍板：{direction_decision}"

        if direction_decision == "stop":
            print("  用户叫停，循环截止。")
            break
        if direction_decision == "needs_more":
            print("  → 需要补充测试用例或重跑裁判后再编辑。")
            break
        if direction_decision not in {"proceed", "revise"}:
            print(f"  Unknown direction decision '{direction_decision}', stopping before edit.")
            break

        editor_input = (
            reflect_report
            + "\n\n## 用户拍板后的编辑者输入\n\n"
            + user_direction
            + "\n\nEditor 只能根据以上用户认可或修正后的方向生成候选版；不得扩大修改范围。"
        )

        # 4. Edit
        print(f"[Edit] Generating edit proposals...")
        edit_proposal = edit(
            skill_content=skill_content,
            reflect_report=editor_input,
            output_dir=output_dir,
            iteration=round_num,
            model=model,
            api_key=api_key,
            base_url=base_url,
            rejected_edits=rejected_edits,
        )

        if not edit_proposal.edits:
            print("  No edits proposed. Stopping this loop.")
            break

        # Apply edits to create new version
        skill_new = apply_edits(skill_content, edit_proposal.edits)
        current_version = round_num

        # Save new skill version
        skill_version_dir = output_path / f"iteration-{round_num}"
        skill_version_dir.mkdir(parents=True, exist_ok=True)
        with open(skill_version_dir / "SKILL.candidate.md", "w", encoding="utf-8") as f:
            f.write(skill_new)

        # 5. Gate
        print(f"[Gate] Running held-out validation...")
        val_new, val_old, gate_report = gate(
            skill_new=skill_new,
            skill_old=skill_content,
            val_evals=val_evals,
            output_dir=output_dir,
            iteration=round_num,
            model=model,
            api_key=api_key,
            base_url=base_url,
            edit_proposal=edit_proposal,
        )

        # 6. User judgment
        if auto_accept:
            print("\n[AUTO] Accepting edit and continuing (auto_accept=True, testing only)")
            decision = "accept_continue"
        else:
            print("\n" + "=" * 60)
            print(f"请查看 Gate 报告: {output_path}/iteration-{round_num}/gate/gate_report.md")
            print("用户拍板: accept_continue / accept_stop / reject / needs_more / stop")
            decision = input("> ").strip().lower()

        if decision in {"accept", "accept_continue", "accept_stop"}:
            skill_content = skill_new
            skill_history.append(skill_new)
            print(f"  ✓ v{current_version} accepted.")
            if decision == "accept_stop":
                print("  用户选择接受并停止，循环截止。")
                break
        elif decision == "reject":
            rejected_edits.append({
                "round": round_num,
                "edits": edit_proposal.edits,
                "reason": "User rejected",
            })
            # Save rejected edits
            rej_path = output_path / "rejected_edits.json"
            with open(rej_path, "w", encoding="utf-8") as f:
                json.dump(rejected_edits, f, ensure_ascii=False, indent=2)
            print(f"  ✗ Edit rejected. Recorded in rejected_edits.json")
            # Keep old skill_content
        elif decision == "needs_more":
            print("  → 需要补充测试用例。请在 val.json 中添加新用例后继续。")
            break
        elif decision == "stop":
            print("  用户叫停，循环截止。")
            break
        else:
            print(f"  Unknown decision '{decision}', treating as reject")
            rejected_edits.append({
                "round": round_num,
                "edits": edit_proposal.edits,
                "reason": f"Unknown: {decision}",
            })

        # Check convergence
        consecutive_rejects = sum(
            1 for re in rejected_edits[-3:]
            if re.get("round", 0) >= round_num - 2
        )
        if consecutive_rejects >= 3:
            print("\n连续 3 轮无 accepted edit，收敛。")
            break

    # Save the latest user-accepted version. This is not automatically "final";
    # users may continue another optimization round from it.
    accepted_dir = output_path / "accepted"
    accepted_dir.mkdir(parents=True, exist_ok=True)
    with open(accepted_dir / "SKILL.md", "w", encoding="utf-8") as f:
        f.write(skill_content)

    print(f"\n用户已接受版本已保存到: {accepted_dir}/SKILL.md")
    return str(accepted_dir / "SKILL.md")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Auto-Skill Phase 2: Optimization Loop")
    parser.add_argument("--skill-path", required=True, help="SKILL.md 路径（当前版本）")
    parser.add_argument("--train-evals", required=True, help="训练集 JSON 路径")
    parser.add_argument("--val-evals", required=True, help="验证集 JSON 路径")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--model", default="gpt-4o", help="LLM 模型")
    parser.add_argument("--api-key", help="OpenAI API Key")
    parser.add_argument("--base-url", help="API Base URL")
    parser.add_argument("--max-rounds", type=int, default=10, help="最大优化轮次")
    parser.add_argument("--sample-size", type=int, default=3, help="每轮训练集采样数")
    parser.add_argument("--auto-accept", action="store_true",
                        help="自动接受所有编辑（仅用于自动化测试；真实优化不得跳过用户裁决）")
    args = parser.parse_args()

    run_optimization_loop(
        skill_path=args.skill_path,
        train_evals_path=args.train_evals,
        val_evals_path=args.val_evals,
        output_dir=args.output_dir,
        model=args.model,
        api_key=args.api_key or os.environ.get("OPENAI_API_KEY"),
        base_url=args.base_url or os.environ.get("OPENAI_BASE_URL"),
        max_rounds=args.max_rounds,
        sample_size=args.sample_size,
        auto_accept=args.auto_accept,
    )


if __name__ == "__main__":
    main()
