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
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import ReviewAction, Issue
from server.environment import CodeReviewEnvironment
from server.graders import grade_episode, run_keyword_baseline
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
    score = grade_episode(flagged, ground_truth)

    tp = sum(
        1 for f in flagged
        if any(
            True for gt in ground_truth
            if abs(f.line_number - gt.line_number) <= 2
            and f.filename == gt.filename
        )
    )

    return {
        "task_id": request.task_id,
        "difficulty": task["difficulty"],
        "score": score,
        "max_score": 1.0,
        "details": {
            "total_flagged": len(flagged),
            "true_positives": tp,
            "false_positives": len(flagged) - tp,
            "total_ground_truth": len(ground_truth),
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


def main():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
