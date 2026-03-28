from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class Issue:
    line_number: int
    filename: str
    issue_type: str   # bug | security | performance | logic
    severity: str     # low | medium | high | critical
    description: str = ""
    fix_suggestion: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "line_number": self.line_number,
            "filename": self.filename,
            "issue_type": self.issue_type,
            "severity": self.severity,
            "description": self.description,
            "fix_suggestion": self.fix_suggestion,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Issue":
        return cls(
            line_number=int(d.get("line_number", 0)),
            filename=str(d.get("filename", "")),
            issue_type=str(d.get("issue_type", "bug")),
            severity=str(d.get("severity", "medium")),
            description=str(d.get("description", "")),
            fix_suggestion=d.get("fix_suggestion"),
        )


try:
    from openenv.core.env_server import (
        Action as _BaseAction,
        Observation as _BaseObservation,
        State as _BaseState,
    )
except ImportError:
    @dataclass
    class _BaseAction:
        metadata: Dict[str, Any] = field(default_factory=dict)

    @dataclass
    class _BaseObservation:
        done: bool = False
        reward: Optional[float] = None
        metadata: Dict[str, Any] = field(default_factory=dict)

    @dataclass
    class _BaseState:
        episode_id: Optional[str] = None
        step_count: int = 0


@dataclass
class ReviewAction(_BaseAction):
    """
    Agent action during a code review episode.

    action_type:
      flag_issue    — mark a line as containing an issue
      clear_flag    — remove a previously flagged issue
      request_hint  — get a hint (-0.01 reward)
      submit_review — end the episode and receive final grade
    """
    action_type: str = "flag_issue"
    line_number: Optional[int] = None
    filename: Optional[str] = None
    issue_type: Optional[str] = None
    severity: Optional[str] = None
    description: str = ""
    fix_suggestion: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "action_type": self.action_type,
            "line_number": self.line_number,
            "filename": self.filename,
            "issue_type": self.issue_type,
            "severity": self.severity,
            "description": self.description,
            "fix_suggestion": self.fix_suggestion,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ReviewAction":
        return cls(
            action_type=str(d.get("action_type", "flag_issue")),
            line_number=d.get("line_number"),
            filename=d.get("filename"),
            issue_type=d.get("issue_type"),
            severity=d.get("severity"),
            description=str(d.get("description", "")),
            fix_suggestion=d.get("fix_suggestion"),
        )


@dataclass
class ReviewObservation(_BaseObservation):
    """
    Observation returned after each reset/step call.
    code_files is only populated on reset; subsequent steps omit it.
    """
    task_id: str = ""
    task_description: str = ""
    code_files: Dict[str, str] = field(default_factory=dict)
    language: str = "python"
    flagged_issues: List[Issue] = field(default_factory=list)
    step_count: int = 0
    max_steps: int = 20
    hints_remaining: int = 3
    feedback: str = ""
    current_score: float = 0.0
    done: bool = False
    reward: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_description": self.task_description,
            "code_files": self.code_files,
            "language": self.language,
            "flagged_issues": [i.to_dict() for i in self.flagged_issues],
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "hints_remaining": self.hints_remaining,
            "feedback": self.feedback,
            "current_score": self.current_score,
            "done": self.done,
            "reward": self.reward,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ReviewObservation":
        return cls(
            task_id=d.get("task_id", ""),
            task_description=d.get("task_description", ""),
            code_files=d.get("code_files", {}),
            language=d.get("language", "python"),
            flagged_issues=[Issue.from_dict(i) for i in d.get("flagged_issues", [])],
            step_count=d.get("step_count", 0),
            max_steps=d.get("max_steps", 20),
            hints_remaining=d.get("hints_remaining", 3),
            feedback=d.get("feedback", ""),
            current_score=d.get("current_score", 0.0),
            done=d.get("done", False),
            reward=d.get("reward"),
        )


@dataclass
class ReviewState(_BaseState):
    task_id: str = ""
    difficulty: str = ""
    episode_id: Optional[str] = None
    step_count: int = 0
    flagged_issues: List[Issue] = field(default_factory=list)
    current_score: float = 0.0
    submitted: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "difficulty": self.difficulty,
            "episode_id": self.episode_id,
            "step_count": self.step_count,
            "flagged_issues": [i.to_dict() for i in self.flagged_issues],
            "current_score": self.current_score,
            "submitted": self.submitted,
        }
