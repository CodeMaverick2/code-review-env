"""
Core environment logic for the Code Review Environment.
"""
from __future__ import annotations

import random
import uuid
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional, List

from models import Issue, ReviewAction, ReviewObservation, ReviewState
from tasks.data import ALL_TASKS, TASK_IDS
from server.graders import grade_episode, compute_live_score, match_issue

try:
    from openenv.core.env_server import Environment as _BaseEnv
    _HAS_OPENENV = True
except ImportError:
    _HAS_OPENENV = False

    class _BaseEnv:  # type: ignore[no-redef]
        pass


class CodeReviewEnvironment(_BaseEnv):
    """
    A code review and security audit environment.

    The agent receives code files and must identify bugs, security
    vulnerabilities, and performance issues by flagging them with
    exact line numbers, types, and severity ratings.

    Episode flow:
      1. reset(task_id) — agent sees code files and task description
      2. step(flag_issue) — flag a problem; get per-step reward
      3. step(clear_flag) — remove an incorrectly flagged issue
      4. step(request_hint) — get a hint (costs -0.01 reward)
      5. step(submit_review) — episode ends, final grade is returned
         (or auto-ends when max_steps is reached)
    """

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self) -> None:
        self._state = ReviewState()
        self._task: Optional[dict] = None
        self._ground_truth: List[Issue] = []
        self._hint_index: int = 0

    def reset(
        self,
        task_id: Optional[str] = None,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> ReviewObservation:
        """Start a new review episode."""
        if seed is not None:
            random.seed(seed)

        if task_id is None or task_id not in ALL_TASKS:
            task_id = random.choice(TASK_IDS)

        self._task = ALL_TASKS[task_id]
        self._ground_truth = [
            Issue.from_dict(gt)
            for gt in self._task["ground_truth_issues"]
        ]
        self._hint_index = 0

        self._state = ReviewState(
            task_id=task_id,
            difficulty=self._task["difficulty"],
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            flagged_issues=[],
            current_score=0.0,
            submitted=False,
        )

        return ReviewObservation(
            task_id=task_id,
            task_description=self._task["description"],
            code_files=self._task["code_files"],
            language=self._task.get("language", "python"),
            flagged_issues=[],
            step_count=0,
            max_steps=self._task["max_steps"],
            hints_remaining=len(self._task.get("hints", [])),
            feedback=(
                f"New episode started. Task: {self._task['difficulty'].upper()}. "
                f"Review the code carefully and flag all issues you find. "
                f"Use 'submit_review' when done."
            ),
            current_score=0.0,
            done=False,
            reward=None,
        )

    def step(
        self,
        action: ReviewAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> ReviewObservation:
        """Process one agent action and return the new observation."""
        if self._task is None:
            return ReviewObservation(
                done=True,
                reward=0.0,
                feedback="Episode not initialized. Call reset() first.",
            )

        if self._state.submitted:
            return ReviewObservation(
                task_id=self._state.task_id,
                task_description="",
                code_files={},
                flagged_issues=list(self._state.flagged_issues),
                step_count=self._state.step_count,
                max_steps=self._task["max_steps"],
                hints_remaining=0,
                feedback="Episode already submitted. Call reset() to start a new episode.",
                current_score=self._state.current_score,
                done=True,
                reward=0.0,
            )

        if isinstance(action, dict):
            action = ReviewAction.from_dict(action)

        self._state.step_count += 1
        reward, feedback = self._process_action(action)

        max_steps = self._task["max_steps"]
        auto_end = self._state.step_count >= max_steps and not self._state.submitted
        done = self._state.submitted or auto_end

        if auto_end and not self._state.submitted:
            # Grade what was submitted so far
            final = grade_episode(self._state.flagged_issues, self._ground_truth)
            self._state.current_score = final
            reward = final * 0.5  # partial credit for auto-end
            feedback += (
                f" Max steps reached. Auto-graded: {final:.3f}. "
                f"Submit earlier for best score."
            )
            self._state.submitted = True

        live = compute_live_score(self._state.flagged_issues, self._ground_truth)
        self._state.current_score = live

        return ReviewObservation(
            task_id=self._state.task_id,
            task_description="",
            code_files={},
            language=self._task.get("language", "python"),
            flagged_issues=list(self._state.flagged_issues),
            step_count=self._state.step_count,
            max_steps=max_steps,
            hints_remaining=max(0, len(self._task.get("hints", [])) - self._hint_index),
            feedback=feedback,
            current_score=live,
            done=done,
            reward=reward,
        )

    @property
    def state(self) -> ReviewState:
        return self._state

    def _process_action(self, action: ReviewAction):
        atype = (action.action_type or "").strip().lower()

        if atype == "flag_issue":
            return self._handle_flag(action)
        elif atype == "clear_flag":
            return self._handle_clear(action)
        elif atype == "request_hint":
            return self._handle_hint()
        elif atype == "submit_review":
            return self._handle_submit()
        else:
            return 0.0, (
                f"Unknown action_type '{action.action_type}'. "
                "Use: flag_issue | clear_flag | request_hint | submit_review"
            )

    def _handle_flag(self, action: ReviewAction):
        if action.line_number is None:
            return -0.02, "flag_issue requires 'line_number'."
        if not action.filename:
            return -0.02, "flag_issue requires 'filename'."
        if action.issue_type not in ("bug", "security", "performance", "logic", None):
            action.issue_type = "bug"
        if action.severity not in ("low", "medium", "high", "critical", None):
            action.severity = "medium"

        for existing in self._state.flagged_issues:
            if (existing.line_number == action.line_number
                    and existing.filename == action.filename):
                return 0.0, (
                    f"Line {action.line_number} in {action.filename} already flagged. "
                    "Use clear_flag first if you want to change the finding."
                )

        new_issue = Issue(
            line_number=action.line_number,
            filename=action.filename or "",
            issue_type=action.issue_type or "bug",
            severity=action.severity or "medium",
            description=action.description or "",
            fix_suggestion=action.fix_suggestion,
        )

        is_tp = any(
            match_issue(new_issue, gt)
            for gt in self._ground_truth
        )

        self._state.flagged_issues.append(new_issue)

        if is_tp:
            reward = 0.10
            feedback = (
                f"Good catch! Issue flagged at {action.filename}:{action.line_number}. "
                f"[+0.10 reward — correct finding]"
            )
        else:
            reward = -0.05
            feedback = (
                f"Issue flagged at {action.filename}:{action.line_number}. "
                f"[-0.05 reward — no matching ground-truth issue nearby]"
            )

        return reward, feedback

    def _handle_clear(self, action: ReviewAction):
        if action.line_number is None or not action.filename:
            return -0.02, "clear_flag requires 'line_number' and 'filename'."

        before = len(self._state.flagged_issues)
        removed = None
        self._state.flagged_issues = [
            f for f in self._state.flagged_issues
            if not (f.line_number == action.line_number
                    and f.filename == action.filename)
        ]

        if len(self._state.flagged_issues) == before:
            return 0.0, (
                f"No flagged issue found at {action.filename}:{action.line_number}."
            )

        removed_issue = Issue(
            line_number=action.line_number,
            filename=action.filename,
            issue_type="bug",
            severity="medium",
        )
        was_tp = any(match_issue(removed_issue, gt) for gt in self._ground_truth)

        if was_tp:
            reward = -0.03
            feedback = (
                f"Removed a correct finding at {action.filename}:{action.line_number}. "
                f"[-0.03 reward]"
            )
        else:
            reward = 0.03
            feedback = (
                f"Removed a false positive at {action.filename}:{action.line_number}. "
                f"[+0.03 reward — good correction]"
            )

        return reward, feedback

    def _handle_hint(self):
        hints = self._task.get("hints", [])
        if self._hint_index >= len(hints):
            return -0.01, "No more hints available for this task."

        hint = hints[self._hint_index]
        self._hint_index += 1
        remaining = len(hints) - self._hint_index
        return -0.01, f"Hint {self._hint_index}/{len(hints)}: {hint} ({remaining} hints left)"

    def _handle_submit(self):
        self._state.submitted = True
        final_score = grade_episode(self._state.flagged_issues, self._ground_truth)
        self._state.current_score = final_score

        tp_count = sum(
            1 for f in self._state.flagged_issues
            if any(match_issue(f, gt) for gt in self._ground_truth)
        )
        total_gt = len(self._ground_truth)
        total_flagged = len(self._state.flagged_issues)

        feedback = (
            f"Review submitted! Final score: {final_score:.3f}. "
            f"Found {tp_count}/{total_gt} real issues. "
            f"Total flags: {total_flagged} "
            f"({'perfect' if total_flagged == tp_count else f'{total_flagged - tp_count} false positives'})."
        )

        return final_score, feedback
