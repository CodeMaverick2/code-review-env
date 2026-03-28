"""
HTTP client for the Code Review Environment.

Usage:
    from client import CodeReviewEnv, ReviewAction

    with CodeReviewEnv(base_url="http://localhost:7860").sync() as env:
        result = env.reset(task_id="bug-detection")
        obs = result.observation
        print(obs.task_description)
        print(obs.code_files)

        # Flag an issue
        result = env.step(ReviewAction(
            action_type="flag_issue",
            line_number=6,
            filename="utils.py",
            issue_type="bug",
            severity="high",
            description="Off-by-one in range()"
        ))
        print(result.observation.feedback)

        # Submit
        result = env.step(ReviewAction(action_type="submit_review"))
        print(f"Score: {result.reward:.3f}")
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Optional, Generic, TypeVar
from models import ReviewAction, ReviewObservation, ReviewState, Issue

ObsT = TypeVar("ObsT")


class StepResult(Generic[ObsT]):
    def __init__(
        self,
        observation: ObsT,
        reward: Optional[float] = None,
        done: bool = False,
    ):
        self.observation = observation
        self.reward = reward
        self.done = done

    def __repr__(self) -> str:
        return (
            f"StepResult(done={self.done}, reward={self.reward}, "
            f"score={getattr(self.observation, 'current_score', None)})"
        )


try:
    import httpx
    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False

try:
    from openenv.core.http_env_client import HTTPEnvClient as _OfficialClient
    _HAS_OPENENV_CLIENT = True
except ImportError:
    _HAS_OPENENV_CLIENT = False


class SyncCodeReviewEnv:

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")
        if not _HAS_HTTPX:
            raise ImportError("httpx is required: pip install httpx")
        import httpx
        self._client = httpx.Client(timeout=30.0)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self._client.close()

    def reset(
        self,
        task_id: Optional[str] = None,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
    ) -> StepResult[ReviewObservation]:
        body = {}
        if task_id:
            body["task_id"] = task_id
        if seed is not None:
            body["seed"] = seed
        if episode_id:
            body["episode_id"] = episode_id

        resp = self._client.post(f"{self.base_url}/reset", json=body)
        resp.raise_for_status()
        obs = ReviewObservation.from_dict(resp.json())
        return StepResult(observation=obs, reward=obs.reward, done=obs.done)

    def step(self, action: ReviewAction) -> StepResult[ReviewObservation]:
        body = action.to_dict()
        resp = self._client.post(f"{self.base_url}/step", json=body)
        resp.raise_for_status()
        obs = ReviewObservation.from_dict(resp.json())
        return StepResult(observation=obs, reward=obs.reward, done=obs.done)

    def state(self) -> ReviewState:
        resp = self._client.get(f"{self.base_url}/state")
        resp.raise_for_status()
        data = resp.json()
        return ReviewState(
            task_id=data.get("task_id", ""),
            difficulty=data.get("difficulty", ""),
            episode_id=data.get("episode_id"),
            step_count=data.get("step_count", 0),
            flagged_issues=[Issue.from_dict(i) for i in data.get("flagged_issues", [])],
            current_score=data.get("current_score", 0.0),
            submitted=data.get("submitted", False),
        )

    def health(self) -> dict:
        resp = self._client.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    def list_tasks(self) -> dict:
        resp = self._client.get(f"{self.base_url}/tasks")
        resp.raise_for_status()
        return resp.json()


class CodeReviewEnv:

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url

    def sync(self) -> SyncCodeReviewEnv:
        return SyncCodeReviewEnv(self.base_url)

    def __enter__(self):
        self._sync = self.sync()
        return self._sync

    def __exit__(self, *args):
        if hasattr(self, "_sync"):
            self._sync.close()
