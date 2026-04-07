"""
Inference script for the Code Review Environment.

Environment variables:
    API_BASE_URL  — LLM API endpoint (e.g. https://openrouter.ai/api/v1)
    MODEL_NAME    — Model identifier (e.g. openai/gpt-4o-mini)
    HF_TOKEN      — API key for the LLM provider (also accepts OPENAI_API_KEY)
    ENV_URL       — Environment base URL (default: localhost:7860)

Usage:
    export API_BASE_URL=https://openrouter.ai/api/v1
    export MODEL_NAME=openai/gpt-4o-mini
    export HF_TOKEN=sk-...
    python inference.py
"""
from __future__ import annotations

import os
import sys
import json
import time
from typing import Optional

import httpx

API_BASE_URL: str = os.environ.get("API_BASE_URL", "").rstrip("/")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN: str = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY", "")
ENV_URL: str = os.environ.get("ENV_URL", "http://localhost:7860").rstrip("/")

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
- Count lines carefully from line 1 (including blank lines and comments)
- Only flag REAL issues — no style preferences, no hypothetical issues
- Be precise: "SQL injection at line 19 via f-string in SELECT query" not just "SQL injection"
- Flag the EXACT line where the problem code is (the f-string line, not the function def)
"""


def chat_completion(messages: list) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai")

    kwargs = {"api_key": HF_TOKEN or "no-key"}
    if API_BASE_URL:
        kwargs["base_url"] = API_BASE_URL

    client = OpenAI(**kwargs)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"    LLM call failed: {e}")
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
        print(f"    Keyword fallback failed: {e}")
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


def _should_clear_flag(obs: dict, last_reward: float, last_action: dict) -> Optional[dict]:
    """
    Recovery strategy: if the last flag was a false positive with high penalty,
    suggest clearing it to recover partial reward and improve precision.

    Returns a clear_flag action dict if we should recover, else None.
    """
    if last_reward is None or last_reward >= 0:
        return None
    if last_action.get("action_type") != "flag_issue":
        return None

    # Only clear if it was a clear FP (no near-miss indicator in feedback)
    # and we've got too many false positives
    progress = obs.get("progress", {})
    fp = int(progress.get("false_positives", 0))
    tp = int(progress.get("true_positives", 0))

    # If FP > TP and last reward was notably negative, clear the bad flag
    if fp > tp and last_reward <= -0.05:
        return {
            "action_type": "clear_flag",
            "line_number": last_action.get("line_number"),
            "filename": last_action.get("filename"),
        }

    return None


def run_task(task_id: str, http_client: httpx.Client) -> dict:
    try:
        resp = http_client.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
        resp.raise_for_status()
        obs = resp.json()
    except Exception as e:
        print(f"    Reset failed: {e} — falling back to keyword heuristic")
        return run_keyword_fallback(ENV_URL, task_id)

    code_display = "\n\n".join(
        f"=== {fname} (starting at line 1) ===\n{code}"
        for fname, code in obs.get("code_files", {}).items()
    )

    # Include function map hint if available
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
    n_gt = len(obs.get("code_files", {}))  # rough complexity hint
    category_hint = ""
    if issue_categories:
        category_hint = f"\nIssue categories to look for: {sorted(set(issue_categories))}"

    # RC-GRPO style reward conditioning (2025): tell the agent what quality level
    # it should aim for, so it calibrates confidence appropriately.
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
    step_count = 0
    final_score = 0.0
    last_action: dict = {}
    last_reward: Optional[float] = None
    consecutive_fp = 0

    while not done and step_count < max_steps:
        # --- Auto clear_flag recovery: undo recent FP if hurting precision ---
        recovery_action = _should_clear_flag(obs, last_reward, last_action)
        if recovery_action and step_count < max_steps - 1:
            action = recovery_action
            action_text = json.dumps(action)
            print(f"    Auto-recovery: clearing FP at {action.get('filename')}:{action.get('line_number')}")
        else:
            # --- Normal LLM action ---
            try:
                action_text = chat_completion(messages)
            except Exception as e:
                print(f"    LLM unavailable ({e}) — submitting and falling back to keyword heuristic")
                try:
                    http_client.post(f"{ENV_URL}/step", json={"action_type": "submit_review"}, timeout=30)
                except Exception:
                    pass
                return run_keyword_fallback(ENV_URL, task_id)

            action = parse_action(action_text)

            # Smart submission: inject submit if progress shows we're done
            if action.get("action_type") != "submit_review" and _should_submit(obs, step_count, max_steps):
                print(f"    Smart submit at step {step_count + 1} (recall target met)")
                action = {"action_type": "submit_review"}
                action_text = json.dumps(action)

        try:
            step_resp = http_client.post(f"{ENV_URL}/step", json=action, timeout=30)
            step_resp.raise_for_status()
            obs = step_resp.json()
        except Exception as e:
            print(f"    Step error: {e}")
            break

        done = obs.get("done", False)
        step_count += 1
        last_reward = obs.get("reward")
        # Use terminal reward (final grade) when done, else intermediate score
        if done:
            final_score = last_reward or obs.get("current_score", 0.0)
        else:
            final_score = obs.get("current_score", 0.0)
        last_action = action

        # Track consecutive FPs for logging
        if last_reward is not None and last_reward < 0 and action.get("action_type") == "flag_issue":
            consecutive_fp += 1
        else:
            consecutive_fp = 0

        # Build rich feedback for next LLM turn
        progress_feedback = _build_progress_feedback(obs)
        env_feedback = obs.get("feedback", "")
        combined_feedback = env_feedback
        if progress_feedback:
            combined_feedback += f"\n{progress_feedback}"

        messages.append({"role": "assistant", "content": action_text})
        if combined_feedback:
            messages.append({"role": "user", "content": combined_feedback})

        # Context window management: keep system + initial prompt + last 12 exchanges
        # This prevents token limit errors on long episodes (25+ steps)
        max_history = 2 + 24  # system + initial user + 12 assistant/user pairs
        if len(messages) > max_history:
            messages = messages[:2] + messages[-(max_history - 2):]

        atype = action.get("action_type", "")
        print(f"    Step {step_count:2d}: {atype:20s} | reward={str(last_reward):8s} | score={obs.get('current_score', 0.0):.3f}")

        if atype == "submit_review":
            final_score = obs.get("reward", obs.get("current_score", 0.0)) or 0.0
            break

        time.sleep(0.3)

    return {
        "task_id": task_id,
        "score": float(final_score),
        "steps": step_count,
        "method": "llm",
    }


def main():
    use_llm = bool(HF_TOKEN and API_BASE_URL)

    print("Code Review Environment — Inference")
    print(f"  Model   : {MODEL_NAME}")
    print(f"  API URL : {API_BASE_URL or '(not set — using keyword heuristic)'}")
    print(f"  Env URL : {ENV_URL}")
    print(f"  Tasks   : {TASK_IDS}\n")

    try:
        with httpx.Client(timeout=10) as probe:
            health = probe.get(f"{ENV_URL}/health")
            health.raise_for_status()
            print(f"  Health: {health.json()}\n")
    except Exception as e:
        print(f"ERROR: Cannot reach environment at {ENV_URL}: {e}")
        sys.exit(1)

    results = {}

    if use_llm:
        with httpx.Client(timeout=60) as client:
            for task_id in TASK_IDS:
                print(f"Running task: {task_id}")
                result = run_task(task_id, client)
                results[task_id] = result
                print(f"  → score: {result['score']:.4f}  ({result['steps']} steps, method={result['method']})\n")
    else:
        print("HF_TOKEN / API_BASE_URL not set — using built-in keyword heuristic baseline.\n")
        for task_id in TASK_IDS:
            print(f"Running task: {task_id}")
            result = run_keyword_fallback(ENV_URL, task_id)
            results[task_id] = result
            print(f"  → score: {result['score']:.4f}\n")

    print("=" * 50)
    print("INFERENCE RESULTS")
    print("=" * 50)
    for task_id, r in results.items():
        print(f"  {task_id:30s}  score={r['score']:.4f}")

    overall = sum(r["score"] for r in results.values()) / len(results)
    print(f"\n  Overall average: {overall:.4f}")
    print("=" * 50)

    return results


if __name__ == "__main__":
    main()
