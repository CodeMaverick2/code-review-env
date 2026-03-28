"""
Demo script for the Code Review Environment.

Runs a complete episode against the live environment using the
keyword-heuristic agent (no API key required).

Usage:
    python demo.py
    python demo.py --url https://tejasghatule-code-review-env.hf.space
    python demo.py --task security-audit
"""
from __future__ import annotations

import argparse
import json
import sys
import os

import httpx

DEFAULT_URL = "https://tejasghatule-code-review-env.hf.space"
TASKS = ["bug-detection", "security-audit", "comprehensive-review"]


def run_keyword_agent(base_url: str, task_id: str) -> dict:
    """Run the built-in keyword-heuristic agent via the /baseline endpoint."""
    with httpx.Client(timeout=30) as client:
        # Health check
        health = client.get(f"{base_url}/health")
        health.raise_for_status()
        print(f"  Health : {health.json()}")

        # Reset
        resp = client.post(f"{base_url}/reset", json={"task_id": task_id})
        resp.raise_for_status()
        obs = resp.json()

        print(f"  Task   : {obs['task_id']} ({obs.get('difficulty', '')})")
        print(f"  Files  : {list(obs['code_files'].keys())}")
        print(f"  Steps  : 0 / {obs['max_steps']}")
        print()

        # Use /baseline endpoint (deterministic, no LLM)
        baseline = client.post(f"{base_url}/baseline")
        baseline.raise_for_status()
        results = baseline.json()

        return results


def run_manual_episode(base_url: str, task_id: str) -> None:
    """Walk through a full episode step-by-step to demonstrate the API."""
    with httpx.Client(timeout=30) as client:
        print(f"=== Episode Demo: {task_id} ===\n")

        # 1. Reset
        resp = client.post(f"{base_url}/reset", json={"task_id": task_id})
        resp.raise_for_status()
        obs = resp.json()

        print(f"Task      : {obs['task_description'][:120]}...")
        print(f"Files     : {list(obs['code_files'].keys())}")
        print(f"Max steps : {obs['max_steps']}")
        print(f"Score     : {obs['current_score']}")
        print()

        # 2. Flag a known issue (task-specific)
        actions = {
            "bug-detection": {
                "action_type": "flag_issue",
                "line_number": 6,
                "filename": "utils.py",
                "issue_type": "bug",
                "severity": "high",
                "description": "Off-by-one: range(len(numbers) + 1) causes IndexError",
                "fix_suggestion": "Change to range(len(numbers))",
            },
            "security-audit": {
                "action_type": "flag_issue",
                "line_number": 8,
                "filename": "app.py",
                "issue_type": "security",
                "severity": "high",
                "description": "Hardcoded SECRET_KEY in source code",
                "fix_suggestion": "Use os.environ.get('SECRET_KEY')",
            },
            "comprehensive-review": {
                "action_type": "flag_issue",
                "line_number": 8,
                "filename": "models.py",
                "issue_type": "security",
                "severity": "critical",
                "description": "Plaintext password storage in database",
                "fix_suggestion": "Use Django's make_password / check_password",
            },
        }

        action = actions.get(task_id, actions["bug-detection"])
        print(f"Step 1 — flag_issue at {action['filename']}:{action['line_number']}")
        resp = client.post(f"{base_url}/step", json=action)
        resp.raise_for_status()
        obs = resp.json()
        print(f"  Feedback : {obs['feedback']}")
        print(f"  Reward   : {obs['reward']}")
        print(f"  Score    : {obs['current_score']}")
        print()

        # 3. Request a hint
        print("Step 2 — request_hint")
        resp = client.post(f"{base_url}/step", json={"action_type": "request_hint"})
        resp.raise_for_status()
        obs = resp.json()
        print(f"  Feedback : {obs['feedback']}")
        print()

        # 4. Submit
        print("Step 3 — submit_review")
        resp = client.post(f"{base_url}/step", json={"action_type": "submit_review"})
        resp.raise_for_status()
        obs = resp.json()
        print(f"  Feedback : {obs['feedback']}")
        print(f"  Final score : {obs['reward']:.4f}")
        print(f"  Done        : {obs['done']}")
        print()

        # 5. Check state
        state = client.get(f"{base_url}/state")
        state.raise_for_status()
        s = state.json()
        print(f"State — episode_id: {s['episode_id']}, steps: {s['step_count']}, submitted: {s['submitted']}")


def main():
    parser = argparse.ArgumentParser(description="Code Review Environment demo")
    parser.add_argument("--url", default=DEFAULT_URL, help="Environment base URL")
    parser.add_argument("--task", default="bug-detection", choices=TASKS)
    parser.add_argument("--baseline", action="store_true", help="Run full baseline on all tasks")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    print(f"Code Review Environment — Demo")
    print(f"  URL  : {base_url}")
    print(f"  Task : {args.task}\n")

    if args.baseline:
        print("Running keyword-heuristic baseline on all tasks...\n")
        results = run_keyword_agent(base_url, args.task)
        print(json.dumps(results, indent=2))
    else:
        run_manual_episode(base_url, args.task)


if __name__ == "__main__":
    main()
