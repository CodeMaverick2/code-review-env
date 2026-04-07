"""
Inference Script — Code Review Environment
===========================================
MANDATORY environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

Defaults are set only for API_BASE_URL and MODEL_NAME:
    API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
    MODEL_NAME   = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

Usage:
    export HF_TOKEN=hf_...
    python inference.py
"""
from __future__ import annotations

import os
import sys
import json
import time
from typing import List, Optional

import httpx
from openai import OpenAI

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
ENV_URL: str = os.getenv("ENV_URL", "http://localhost:7860").rstrip("/")
BENCHMARK = "code-review-env"


# ---------------------------------------------------------------------------
# Structured stdout logging — MANDATORY format for OpenEnv submission
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# Curriculum ordering: easy → medium → medium-hard → hard
# Research (CAMRL, Curriculum RL): start with simpler tasks to build
# foundational skills, progress to harder multi-file and multi-language tasks.
TASK_IDS = [
    "bug-detection",        # easy: pure logic bugs, single file
    "security-audit",       # medium: OWASP Top-10, single file
    "async-review",         # medium-hard: async concurrency, subtle bugs
    "data-pipeline",        # hard: SQL injection + crypto + performance
    "comprehensive-review", # hard: multi-file Django, mixed issue types
    "api-security",         # hard: FastAPI auth/authz/injection
    "js-security",          # hard: JavaScript (cross-language generalization)
]

SYSTEM_PROMPT = """\
You are an expert software engineer performing a thorough, methodical code review.

Your mission: identify ALL real bugs, security vulnerabilities, and performance issues.

## REVIEW CHECKLIST — work through EVERY category for EVERY function:

### Security (check EVERY function for these)
- Hardcoded secrets / API keys / passwords / tokens
- SQL injection: f-strings/template literals/string concat in queries
- Command injection: shell=True, os.system(), execSync() with user input
- XSS: unsanitized user input in HTML templates / res.send()
- Path traversal: path.join/os.path.join with user-supplied paths
- IDOR: missing authorization — authenticated vs authorized
- Insecure deserialization: pickle.loads(), new Function(), eval() on user input
- Broken crypto: MD5/SHA1 for passwords; missing salt; weak PRNG
- JWT issues: missing expiry ('exp'), algorithm confusion, hardcoded secret
- Missing authentication on sensitive endpoints

### Bugs & Logic Errors (check EVERY function for these)
- Off-by-one errors in ranges, slices, loop bounds, retry conditions
- Wrong initial values (counters starting at 0 instead of 1)
- Race conditions (shared mutable state without locks/atomicity)
- Missing transaction atomicity (partial writes to DB)
- Wrong type arguments (int where object required, e.g. aiohttp timeout)
- State that accumulates across calls (class fields not reset)

### Performance (check EVERY loop and DB call)
- N+1 database queries (DB call inside a loop)
- Sequential async where gather() should be used
- One transaction per row in bulk operations
- Uncapped pagination (no max limit on per_page)

### Resource Management
- Unclosed sessions/connections/file handles
- Missing context managers (async with, with)

## RESPONSE FORMAT

For each issue you find, respond with ONE raw JSON object:
{"action_type": "flag_issue", "line_number": <int>, "filename": "<file>",
 "issue_type": "bug|security|performance|logic",
 "severity": "low|medium|high|critical",
 "description": "<specific explanation>",
 "fix_suggestion": "<concrete fix>",
 "confidence": <0.0-1.0>}

When finished, respond with:
{"action_type": "submit_review"}

## RULES
- Raw JSON only — no markdown fences, no extra text
- One action per response
- Line numbers are shown as "N|" prefix — use those EXACT numbers, do NOT count yourself
- Only flag REAL issues — no style preferences, no hypothetical issues
- Be precise: "SQL injection at line 19 via f-string in SELECT query" not just "SQL injection"
- Flag the EXACT line where the problem code is (the f-string line, not the function def)
- issue_type MUST be: "security" for injection/XSS/hardcoded secrets/crypto/auth, "bug" for logic/off-by-one/wrong values, "performance" for N+1/missing gather/uncapped pagination
"""


def chat_completion(messages: list) -> str:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[DEBUG] LLM call failed: {e}", flush=True)
        raise


def parse_action(text: str) -> dict:
    text = text.strip()

    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{") or part.startswith("["):
                text = part
                break

    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in ("{", "["):
            try:
                obj, _ = decoder.raw_decode(text, i)
                if isinstance(obj, dict):
                    return obj
                if isinstance(obj, list):
                    for item in obj:
                        if isinstance(item, dict):
                            return item
            except json.JSONDecodeError:
                continue

    return {"action_type": "submit_review"}


def run_keyword_fallback(base_url: str, task_id: str) -> dict:
    """Fallback: use the built-in /baseline endpoint (no LLM needed)."""
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.post(f"{base_url}/baseline")
            resp.raise_for_status()
            results = resp.json()
            score = results["baseline_scores"].get(task_id, {}).get("score", 0.0)
            return {"task_id": task_id, "score": score, "steps": 0, "method": "keyword_heuristic"}
    except Exception as e:
        print(f"[DEBUG] Keyword fallback failed: {e}", flush=True)
        return {"task_id": task_id, "score": 0.0, "steps": 0, "method": "error"}


def _build_progress_feedback(obs: dict) -> str:
    """Build a rich feedback string from observation progress data."""
    progress = obs.get("progress") or {}
    flagged_summary = obs.get("flagged_summary") or {}

    parts = []
    if progress:
        f1 = progress.get("f1", 0)
        precision = progress.get("precision", 0)
        recall = progress.get("recall", 0)
        tp = int(progress.get("true_positives", 0))
        total_gt = int(progress.get("total_ground_truth", 0))
        steps_rem = int(progress.get("steps_remaining", 0))
        unfound = progress.get("unfound_issue_types", [])

        parts.append(
            f"Score progress: {tp}/{total_gt} issues confirmed | "
            f"F1={f1:.2f} Precision={precision:.2f} Recall={recall:.2f} | "
            f"{steps_rem} steps remaining"
        )
        if unfound:
            parts.append(
                f"IMPORTANT — still need to find: {unfound}. "
                f"Search specifically for those issue types."
            )

    if flagged_summary:
        incorrect = flagged_summary.get("incorrect", 0)
        near = flagged_summary.get("near_misses", 0)
        if incorrect > 0:
            parts.append(
                f"WARNING: {incorrect} false positive(s) hurting precision. "
                f"Consider using clear_flag to remove uncertain flags."
            )
        if near > 0:
            parts.append(
                f"NOTE: {near} near-miss(es) — you're close on line numbers, "
                f"but slightly off. Re-check exact line and try reflagging."
            )

    return "\n".join(parts) if parts else ""


def _should_submit(obs: dict, step_count: int, max_steps: int) -> bool:
    """
    Smart submission: submit when recall is high or steps are nearly exhausted.
    Avoids wasting steps after all real issues are found.
    """
    progress = obs.get("progress", {})
    recall = progress.get("recall", 0.0)
    tp = int(progress.get("true_positives", 0))
    total_gt = int(progress.get("total_ground_truth", 0))
    steps_rem = int(progress.get("steps_remaining", 0))
    unfound = progress.get("unfound_issue_types", [])
    fp = int(progress.get("false_positives", 0))

    # All issues found
    if total_gt > 0 and tp >= total_gt:
        return True

    # No unfound categories and high recall
    if not unfound and recall >= 0.85:
        return True

    # High recall overall (≥80%) and precision is decent (not too many FPs)
    if recall >= 0.80 and (fp <= 2 or tp / max(tp + fp, 1) >= 0.6):
        return True

    # Very few steps left and we've done a reasonable scan
    if steps_rem <= 2 and step_count >= 5:
        return True

    return False


_cleared_lines: set = set()  # track lines we've already cleared to prevent loops


def _should_clear_flag(obs: dict, last_reward: float, last_action: dict) -> Optional[dict]:
    """
    Recovery strategy: if the last flag was a false positive with high penalty,
    suggest clearing it. Only clears each line ONCE to prevent flag/clear loops.
    """
    if last_reward is None or last_reward >= 0:
        return None
    if last_action.get("action_type") != "flag_issue":
        return None

    # Prevent loop: never clear the same line twice
    line_key = (last_action.get("filename"), last_action.get("line_number"))
    if line_key in _cleared_lines:
        return None

    progress = obs.get("progress", {})
    fp = int(progress.get("false_positives", 0))
    tp = int(progress.get("true_positives", 0))

    if fp > tp and last_reward <= -0.05:
        _cleared_lines.add(line_key)
        return {
            "action_type": "clear_flag",
            "line_number": last_action.get("line_number"),
            "filename": last_action.get("filename"),
        }

    return None


def run_task(task_id: str, http_client: httpx.Client) -> dict:
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    _cleared_lines.clear()  # reset per-task
    all_rewards: List[float] = []
    step_count = 0
    final_score = 0.0

    try:
        resp = http_client.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
        resp.raise_for_status()
        obs = resp.json()

        # Show code WITH line numbers — critical for LLM line-counting accuracy
        code_parts = []
        for fname, code in obs.get("code_files", {}).items():
            numbered_lines = "\n".join(
                f"{i+1:3d}| {line}" for i, line in enumerate(code.splitlines())
            )
            code_parts.append(f"=== {fname} ===\n{numbered_lines}")
        code_display = "\n\n".join(code_parts)

        code_metadata = obs.get("code_metadata") or {}
        function_ranges = code_metadata.get("function_ranges", [])
        fn_map_hint = ""
        if function_ranges:
            fn_lines = [f"  {fr['name']}() in {fr['file']} (lines {fr['start']}-{fr['end']})"
                        for fr in function_ranges]
            fn_map_hint = "\n\nFunction map:\n" + "\n".join(fn_lines)

        task_desc = obs.get("task_description", "")
        max_steps = obs.get("max_steps", 20)
        issue_categories = code_metadata.get("issue_categories", [])
        category_hint = ""
        if issue_categories:
            category_hint = f"\nIssue categories to look for: {sorted(set(issue_categories))}"

        state_features = code_metadata.get("state_features", [])
        complexity_label = "medium"
        if state_features and len(state_features) >= 4:
            complexity_score = state_features[3]
            complexity_label = "high" if complexity_score >= 1.0 else "medium" if complexity_score >= 0.5 else "low"

        reward_conditioning = (
            f"[TARGET: high-quality review, score ≥ 0.85. "
            f"Code complexity: {complexity_label}. "
            f"Be thorough — missing issues costs more than a single FP.]"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"{reward_conditioning}\n\n"
                    f"Task: {task_desc}\n\n"
                    f"{code_display}"
                    f"{fn_map_hint}"
                    f"{category_hint}\n\n"
                    f"You have {max_steps} steps total. "
                    f"Work through the checklist systematically, function by function. "
                    f"Flag each issue one at a time as a raw JSON object."
                ),
            },
        ]

        done = False
        last_action: dict = {}
        last_reward: Optional[float] = None

        while not done and step_count < max_steps:
            recovery_action = _should_clear_flag(obs, last_reward, last_action)
            if recovery_action and step_count < max_steps - 1:
                action = recovery_action
                action_text = json.dumps(action)
            else:
                try:
                    action_text = chat_completion(messages)
                except Exception as e:
                    print(f"[DEBUG] LLM unavailable ({e})", flush=True)
                    try:
                        http_client.post(f"{ENV_URL}/step", json={"action_type": "submit_review"}, timeout=30)
                    except Exception:
                        pass
                    break

                action = parse_action(action_text)

                if action.get("action_type") != "submit_review" and _should_submit(obs, step_count, max_steps):
                    action = {"action_type": "submit_review"}
                    action_text = json.dumps(action)

            try:
                step_resp = http_client.post(f"{ENV_URL}/step", json=action, timeout=30)
                step_resp.raise_for_status()
                obs = step_resp.json()
            except Exception as e:
                step_count += 1
                log_step(step=step_count, action="error", reward=0.0, done=True, error=str(e))
                break

            done = obs.get("done", False)
            step_count += 1
            last_reward = obs.get("reward")
            if done:
                final_score = last_reward or obs.get("current_score", 0.0)
            else:
                final_score = obs.get("current_score", 0.0)
            last_action = action

            # Build feedback for next LLM turn
            progress_feedback = _build_progress_feedback(obs)
            env_feedback = obs.get("feedback", "")
            combined_feedback = env_feedback
            if progress_feedback:
                combined_feedback += f"\n{progress_feedback}"

            messages.append({"role": "assistant", "content": action_text})
            if combined_feedback:
                messages.append({"role": "user", "content": combined_feedback})

            max_history = 2 + 24
            if len(messages) > max_history:
                messages = messages[:2] + messages[-(max_history - 2):]

            atype = action.get("action_type", "")
            reward_val = float(last_reward) if last_reward is not None else 0.0
            all_rewards.append(reward_val)
            action_str = f"{atype}({action.get('filename', '')}:{action.get('line_number', '')})" if atype == "flag_issue" else atype
            log_step(step=step_count, action=action_str, reward=reward_val, done=done, error=None)

            if atype == "submit_review":
                final_score = obs.get("reward", obs.get("current_score", 0.0)) or 0.0
                break

            time.sleep(0.3)

    except Exception as e:
        print(f"[DEBUG] Task {task_id} error: {e}", flush=True)
    finally:
        log_end(
            success=final_score >= 0.5,
            steps=step_count,
            score=final_score,
            rewards=all_rewards,
        )

    return {
        "task_id": task_id,
        "score": float(final_score),
        "steps": step_count,
        "method": "llm",
    }


def main():
    use_llm = bool(API_KEY and API_BASE_URL)

    try:
        with httpx.Client(timeout=10) as probe:
            health = probe.get(f"{ENV_URL}/health")
            health.raise_for_status()
    except Exception as e:
        print(f"[DEBUG] Cannot reach environment at {ENV_URL}: {e}", flush=True)
        sys.exit(1)

    results = {}

    if use_llm:
        with httpx.Client(timeout=60) as client:
            for task_id in TASK_IDS:
                result = run_task(task_id, client)
                results[task_id] = result
    else:
        for task_id in TASK_IDS:
            log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
            result = run_keyword_fallback(ENV_URL, task_id)
            results[task_id] = result
            log_end(
                success=result["score"] >= 0.5,
                steps=0,
                score=result["score"],
                rewards=[],
            )

    overall = sum(r["score"] for r in results.values()) / len(results)
    for task_id, r in results.items():
        print(f"[DEBUG] {task_id:30s}  score={r['score']:.4f}", flush=True)
    print(f"[DEBUG] Overall average: {overall:.4f}", flush=True)

    return results


if __name__ == "__main__":
    main()
