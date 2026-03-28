"""
Inference script for the Code Review Environment.

Environment variables:
    API_BASE_URL  — LLM API endpoint (e.g. https://openrouter.ai/api/v1)
    MODEL_NAME    — Model identifier (e.g. openai/gpt-4o-mini)
    HF_TOKEN      — API key for the LLM provider
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

import httpx

API_BASE_URL: str = os.environ.get("API_BASE_URL", "").rstrip("/")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN: str = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY", "")
ENV_URL: str = os.environ.get("ENV_URL", "http://localhost:7860").rstrip("/")

TASK_IDS = ["bug-detection", "security-audit", "comprehensive-review"]

SYSTEM_PROMPT = """\
You are an expert software engineer performing a thorough code review.

Your job is to identify bugs, security vulnerabilities, and performance issues in code.

For each issue you find, respond with a single JSON object:
  {"action_type": "flag_issue", "line_number": <int>, "filename": "<file>", "issue_type": "bug|security|performance|logic", "severity": "low|medium|high|critical", "description": "<explanation>", "fix_suggestion": "<fix>"}

When done, respond with:
  {"action_type": "submit_review"}

Rules:
- Respond with raw JSON only — no markdown fences, no extra text
- One action per response
- Be precise with line numbers (count from line 1)
- Only flag real issues, not style preferences
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
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.0,
        max_tokens=400,
    )
    return response.choices[0].message.content.strip()


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
    with httpx.Client(timeout=30) as client:
        resp = client.post(f"{base_url}/baseline")
        resp.raise_for_status()
        results = resp.json()
        score = results["baseline_scores"].get(task_id, {}).get("score", 0.0)
        return {"task_id": task_id, "score": score, "steps": 0, "method": "keyword_heuristic"}


def run_task(task_id: str, http_client: httpx.Client) -> dict:
    resp = http_client.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
    resp.raise_for_status()
    obs = resp.json()

    code_display = "\n\n".join(
        f"=== {fname} ===\n{code}"
        for fname, code in obs.get("code_files", {}).items()
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Task: {obs.get('task_description', '')}\n\n"
                f"{code_display}\n\n"
                f"Review this code carefully. Flag every issue you find. "
                f"You have {obs.get('max_steps', 20)} steps total."
            ),
        },
    ]

    done = False
    step_count = 0
    max_steps = obs.get("max_steps", 20)
    final_score = 0.0

    while not done and step_count < max_steps:
        action_text = chat_completion(messages)
        action = parse_action(action_text)

        try:
            step_resp = http_client.post(f"{ENV_URL}/step", json=action, timeout=30)
            step_resp.raise_for_status()
            obs = step_resp.json()
        except Exception as e:
            print(f"    Step error: {e}")
            break

        done = obs.get("done", False)
        step_count += 1
        final_score = obs.get("current_score", 0.0)
        reward = obs.get("reward")

        messages.append({"role": "assistant", "content": action_text})
        messages.append({
            "role": "user",
            "content": (
                f"Feedback: {obs.get('feedback', '')} "
                f"(step {step_count}/{max_steps}, score: {obs.get('current_score', 0.0):.3f})"
            ),
        })

        atype = action.get("action_type", "")
        print(f"    Step {step_count:2d}: {atype:20s} | reward={str(reward):8s} | score={obs.get('current_score', 0.0):.3f}")

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
                print(f"  → score: {result['score']:.4f}  ({result['steps']} steps)\n")
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
