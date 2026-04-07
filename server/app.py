"""
FastAPI application for the Code Review Environment.

Endpoints:
  POST /reset    — start new episode
  POST /step     — take an action
  GET  /state    — get episode state
  GET  /health   — health check
  GET  /tasks    — list all tasks + action schema
  POST /grader   — grade a set of findings (stateless)
  POST /baseline — run keyword-heuristic baseline on all tasks
  WS   /ws       — persistent WebSocket session
  GET  /docs     — Swagger UI (auto-generated)
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import asyncio
import dataclasses
import random
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import ReviewAction, Issue
from server.environment import CodeReviewEnvironment
from server.graders import (
    grade_episode, grade_episode_detailed, run_keyword_baseline,
    compute_code_state_features, RewardNormalizer,
)
from tasks.data import ALL_TASKS, TASK_IDS



def _serialize(obj) -> dict:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        d = dataclasses.asdict(obj)
        # asdict handles nested dataclasses and lists recursively
        return d
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Cannot serialize {type(obj)}")


_env_instance = CodeReviewEnvironment()
_reward_normalizer = RewardNormalizer(window_size=100)


def _make_app() -> FastAPI:
    try:
        from openenv.core.env_server import create_fastapi_app
        base = create_fastapi_app(CodeReviewEnvironment)
        return base
    except Exception:
        pass

    _app = FastAPI(
        title="Code Review Environment",
        description=(
            "An OpenEnv environment for training AI agents to perform "
            "code review and security audits."
        ),
        version="1.0.0",
    )

    _app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @_app.get("/health")
    async def health():
        return {"status": "healthy"}

    @_app.post("/reset")
    async def reset(body: dict = None):
        body = body or {}
        task_id = body.get("task_id")
        seed = body.get("seed")
        episode_id = body.get("episode_id")
        obs = _env_instance.reset(task_id=task_id, seed=seed, episode_id=episode_id)
        return _serialize(obs)

    @_app.post("/step")
    async def step(body: dict):
        action = ReviewAction.from_dict(body)
        obs = _env_instance.step(action)
        return _serialize(obs)

    @_app.get("/state")
    async def state():
        return _serialize(_env_instance.state)

    @_app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        ws_env = CodeReviewEnvironment()
        try:
            while True:
                raw = await websocket.receive_text()
                msg = json.loads(raw)
                msg_type = msg.get("type", "")

                if msg_type == "reset":
                    data = msg.get("data", {})
                    obs = ws_env.reset(
                        task_id=data.get("task_id"),
                        seed=data.get("seed"),
                        episode_id=data.get("episode_id"),
                    )
                    await websocket.send_text(json.dumps({
                        "type": "observation",
                        "data": _serialize(obs),
                    }))

                elif msg_type == "step":
                    action = ReviewAction.from_dict(msg.get("data", {}))
                    obs = ws_env.step(action)
                    await websocket.send_text(json.dumps({
                        "type": "observation",
                        "data": _serialize(obs),
                    }))

                elif msg_type == "state":
                    await websocket.send_text(json.dumps({
                        "type": "state",
                        "data": _serialize(ws_env.state),
                    }))

                elif msg_type == "close":
                    break

                else:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "data": f"Unknown message type: {msg_type}",
                    }))

        except WebSocketDisconnect:
            pass
        except Exception as e:
            try:
                await websocket.send_text(json.dumps({"type": "error", "data": str(e)}))
            except Exception:
                pass

    return _app


app = _make_app()


@app.get("/tasks")
async def list_tasks():
    tasks_list = []
    for task in ALL_TASKS.values():
        tasks_list.append({
            "task_id": task["task_id"],
            "difficulty": task["difficulty"],
            "description": task["description"],
            "language": task.get("language", "python"),
            "max_steps": task["max_steps"],
            "num_issues": len(task["ground_truth_issues"]),
            "files": list(task["code_files"].keys()),
        })

    action_schema = {
        "type": "object",
        "description": "ReviewAction — one action per /step call",
        "required": ["action_type"],
        "properties": {
            "action_type": {
                "type": "string",
                "enum": ["flag_issue", "clear_flag", "request_hint", "submit_review"],
                "description": (
                    "flag_issue: mark a line as problematic. "
                    "clear_flag: remove a previous flag. "
                    "request_hint: get a hint (-0.01 reward). "
                    "submit_review: end episode and receive final grade."
                ),
            },
            "line_number": {
                "type": "integer",
                "description": "Line number of the issue (required for flag_issue / clear_flag)",
            },
            "filename": {
                "type": "string",
                "description": "File where the issue is (required for flag_issue / clear_flag)",
            },
            "issue_type": {
                "type": "string",
                "enum": ["bug", "security", "performance", "logic"],
                "description": "Category of issue (required for flag_issue)",
            },
            "severity": {
                "type": "string",
                "enum": ["low", "medium", "high", "critical"],
                "description": "Severity level (required for flag_issue)",
            },
            "description": {
                "type": "string",
                "description": "Human-readable description of the issue",
            },
            "fix_suggestion": {
                "type": "string",
                "description": "Optional suggested fix",
            },
        },
        "examples": [
            {
                "action_type": "flag_issue",
                "line_number": 6,
                "filename": "utils.py",
                "issue_type": "bug",
                "severity": "high",
                "description": "Off-by-one error in range()",
                "fix_suggestion": "Change range(len(numbers) + 1) to range(len(numbers))",
            },
            {"action_type": "submit_review"},
        ],
    }

    return {
        "tasks": tasks_list,
        "action_schema": action_schema,
        "total_tasks": len(tasks_list),
    }


class GraderRequest(BaseModel):
    task_id: str
    flagged_issues: List[Dict[str, Any]]

@app.post("/grader")
async def run_grader(request: GraderRequest):
    task = ALL_TASKS.get(request.task_id)
    if not task:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown task_id '{request.task_id}'. Valid: {TASK_IDS}",
        )

    flagged = [Issue.from_dict(i) for i in request.flagged_issues]
    ground_truth = [Issue.from_dict(gt) for gt in task["ground_truth_issues"]]
    detailed = grade_episode_detailed(flagged, ground_truth)

    return {
        "task_id": request.task_id,
        "difficulty": task["difficulty"],
        "score": detailed["score"],
        "max_score": 1.0,
        "f1": detailed["f1"],
        "precision": detailed["precision"],
        "recall": detailed["recall"],
        "severity_accuracy": detailed["severity_accuracy"],
        "details": {
            "total_flagged": len(flagged),
            "true_positives": detailed["true_positives"],
            "false_positives": detailed["false_positives"],
            "false_negatives": detailed["false_negatives"],
            "near_misses": detailed["near_misses"],
            "total_ground_truth": len(ground_truth),
            "per_file": detailed["per_file"],
        },
    }


@app.post("/baseline")
async def run_baseline():
    results = {}
    for task_id, task in ALL_TASKS.items():
        findings = run_keyword_baseline(task)
        ground_truth = [Issue.from_dict(gt) for gt in task["ground_truth_issues"]]
        score = grade_episode(findings, ground_truth)
        results[task_id] = {
            "difficulty": task["difficulty"],
            "score": score,
            "findings_count": len(findings),
            "ground_truth_count": len(ground_truth),
        }

    overall = sum(r["score"] for r in results.values()) / len(results)
    return {
        "baseline_scores": results,
        "overall_average": round(overall, 4),
        "method": "keyword_heuristic",
        "note": (
            "Run 'python baseline.py' with OPENAI_API_KEY for the LLM-based baseline. "
            "This endpoint uses a deterministic regex heuristic."
        ),
    }


class CurriculumRequest(BaseModel):
    agent_performance: Optional[Dict[str, Any]] = None
    easy_threshold: float = 0.30
    hard_threshold: float = 0.70


@app.post("/curriculum")
async def curriculum_task_selector(request: CurriculumRequest):
    """
    CAMRL-style curriculum task selector (Curriculum-based Asymmetric Multi-Task RL, TPAMI 2023).

    Given agent performance metrics per task, returns the recommended next task_id
    based on curriculum phase:
      - easy phase  (avg_success < 0.30): focus on task with fewest issues
      - medium phase (0.30-0.70):         mix easy/hard (70% easy, 30% hard)
      - hard phase  (avg_success > 0.70): focus on least-solved hard tasks

    Body:
      agent_performance: {task_id: {success_rate: 0.5, episodes: 10, avg_score: 0.4}}
      easy_threshold: float (default 0.3)
      hard_threshold: float (default 0.7)
    """
    perf = request.agent_performance or {}
    easy_thresh = request.easy_threshold
    hard_thresh = request.hard_threshold

    # Build difficulty estimate per task: (1 - success_rate) × complexity
    task_difficulty: Dict[str, float] = {}
    for task_id, task in ALL_TASKS.items():
        n_issues = len(task["ground_truth_issues"])
        complexity = min(1.0, n_issues / 10.0)
        task_perf = perf.get(task_id, {})
        success_rate = float(task_perf.get("success_rate", task_perf.get("avg_score", 0.0)))
        task_difficulty[task_id] = round((1.0 - success_rate) * complexity, 4)

    # Determine curriculum phase
    if perf:
        all_success = [float(p.get("success_rate", p.get("avg_score", 0.0))) for p in perf.values()]
        avg_success = sum(all_success) / len(all_success)
    else:
        avg_success = 0.0

    if avg_success < easy_thresh:
        phase = "easy"
        # Focus on task with lowest ground truth issue count (most approachable)
        recommended = min(ALL_TASKS.keys(), key=lambda t: len(ALL_TASKS[t]["ground_truth_issues"]))
    elif avg_success > hard_thresh:
        phase = "hard"
        # Focus on hardest unsolved task (highest difficulty score)
        recommended = max(task_difficulty, key=task_difficulty.get)
    else:
        phase = "medium"
        # Mix: pick a task proportional to difficulty (harder = more likely)
        import random
        weights = list(task_difficulty.values())
        total_w = sum(weights) or 1.0
        probs = [w / total_w for w in weights]
        recommended = random.choices(list(task_difficulty.keys()), weights=probs, k=1)[0]

    return {
        "recommended_task_id": recommended,
        "curriculum_phase": phase,
        "avg_success_rate": round(avg_success, 4),
        "task_difficulty_scores": task_difficulty,
        "thresholds": {"easy": easy_thresh, "hard": hard_thresh},
        "method": "CAMRL",
    }


@app.get("/reward_normalizer")
async def get_reward_normalizer_stats():
    """
    Return current RewardNormalizer statistics for the running environment.
    Useful for monitoring VL Norm across training runs.
    """
    return _reward_normalizer.to_dict()


@app.post("/record_episode")
async def record_episode(body: Dict[str, Any]):
    """
    Record a completed episode's return and length for VL Norm statistics.
    Body: {"episode_return": 0.72, "episode_length": 12}
    """
    episode_return = float(body.get("episode_return", 0.0))
    episode_length = int(body.get("episode_length", 1))
    _reward_normalizer.update(episode_return, episode_length)
    normalized = _reward_normalizer.normalize(episode_return, episode_length)
    return {
        "normalized_return": normalized,
        "stats": _reward_normalizer.to_dict(),
    }


class TRLRolloutRequest(BaseModel):
    task_id: Optional[str] = None
    seed: Optional[int] = None
    actions: List[Dict[str, Any]]  # Pre-generated action sequence from LLM


@app.post("/trl_rollout")
async def trl_rollout(request: TRLRolloutRequest):
    """
    Run a full episode from a pre-generated action sequence.

    Designed for TRL GRPOTrainer custom rollout_fn integration:
    - Takes a sequence of LLM-generated actions
    - Runs them through the environment
    - Returns trajectory dict with per-step rewards and final score

    This enables offline rollout: LLM generates all actions first,
    then this endpoint evaluates them, matching TRL's batch-rollout pattern.

    Body:
      task_id: str (optional, random if not set)
      seed: int (optional)
      actions: [{action_type, line_number, filename, ...}, ...]

    Returns:
      trajectory: [{step, action, reward, feedback, done}]
      episode_return: float (sum of step rewards)
      final_score: float (terminal grade)
      normalized_return: float (episode_return / num_steps)
      state_features: [float] (12-dim feature vector at end of episode)
    """
    rollout_env = CodeReviewEnvironment()
    obs = rollout_env.reset(task_id=request.task_id, seed=request.seed)

    trajectory = []
    episode_return = 0.0
    final_score = 0.0

    for step_idx, action_dict in enumerate(request.actions):
        action = ReviewAction.from_dict(action_dict)
        obs_step = rollout_env.step(action)
        step_data = _serialize(obs_step)

        reward = step_data.get("reward") or 0.0
        episode_return += reward

        trajectory.append({
            "step": step_idx + 1,
            "action": action_dict,
            "reward": reward,
            "reward_breakdown": step_data.get("reward_breakdown", {}),
            "feedback": step_data.get("feedback", ""),
            "current_score": step_data.get("current_score", 0.0),
            "done": step_data.get("done", False),
        })

        if step_data.get("done", False):
            final_score = step_data.get("reward", step_data.get("current_score", 0.0)) or 0.0
            break

    n_steps = max(len(trajectory), 1)
    # Record in global normalizer for VL Norm statistics
    _reward_normalizer.update(episode_return, n_steps)
    normalized = _reward_normalizer.normalize(episode_return, n_steps)

    # Get final state features
    final_progress = rollout_env._compute_progress(rollout_env._task["max_steps"] if rollout_env._task else 20)

    return {
        "task_id": request.task_id,
        "trajectory": trajectory,
        "episode_return": round(episode_return, 4),
        "final_score": round(final_score, 4),
        "normalized_return": normalized,
        "num_steps": n_steps,
        "state_features": final_progress.get("state_features", []),
        "final_progress": {k: v for k, v in final_progress.items() if k != "state_features"},
    }


def main():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
