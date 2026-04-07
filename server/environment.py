"""
Core environment logic for the Code Review Environment.
"""
from __future__ import annotations

import random
import uuid
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional, List, Dict, Any, Set

from models import Issue, ReviewAction, ReviewObservation, ReviewState
from tasks.data import ALL_TASKS, TASK_IDS
from server.graders import (
    grade_episode, compute_live_score, match_issue, match_quality,
    compute_code_metadata, grade_episode_detailed,
    graduated_near_reward, compute_potential, compute_code_state_features,
)

try:
    from openenv.core.env_server import Environment as _BaseEnv
    _HAS_OPENENV = True
except ImportError:
    _HAS_OPENENV = False

    class _BaseEnv:  # type: ignore[no-redef]
        pass


# Reward constants
_BASE_TP_REWARD = 0.10
_NEAR_MISS_REWARD = 0.03
_BASE_FP_PENALTY = -0.05
_SEVERITY_EXACT_BONUS = 0.02        # when severity exactly matches GT
_TEMPORAL_BONUS = 0.02              # early correct flag (first 40% of steps)
_CONFIDENCE_TP_BONUS = 0.01         # high-confidence TP
_CONFIDENCE_FP_EXTRA = -0.03        # high-confidence FP (penalty multiplier)
_HINT_COST = -0.01
_REMOVE_TP_PENALTY = -0.03
_REMOVE_FP_REWARD = 0.03
_VALIDATION_PENALTY = -0.02
# Flood protection: escalating FP penalty
_FP_FLOOD_THRESHOLD = 3             # FPs before escalation kicks in
_FP_FLOOD_MULTIPLIER = 1.5          # each extra FP beyond threshold costs 1.5x more

_SEV_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}


class CodeReviewEnvironment(_BaseEnv):
    """
    A code review and security audit RL environment.

    The agent receives code files and must identify bugs, security
    vulnerabilities, and performance issues by flagging them with
    exact line numbers, types, and severity ratings.

    Reward design:
    - True positive flag: +0.10 base, +0.02 severity exact match,
      +0.02 early (first 40% steps), +0.01 high-confidence TP
    - Near-miss (±3-5 lines): +0.03 partial credit
    - False positive: -0.05 base, escalating penalty after 3rd FP,
      extra -0.03 for high-confidence FP
    - Clear false positive: +0.03
    - Clear true positive: -0.03
    - Hint: -0.01
    - Submit: final F1+severity score (0.0–1.0)
    - Auto-end (max_steps): full grade score (no penalty)
    """

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self) -> None:
        self._state = ReviewState()
        self._task: Optional[dict] = None
        self._ground_truth: List[Issue] = []
        self._hint_index: int = 0
        self._code_metadata: Dict[str, Any] = {}
        self._fp_count: int = 0           # total false positives this episode
        self._matched_gt_indices: Set[int] = set()  # GT indices already matched
        self._episode_rewards: List[float] = []  # for VL return normalization

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
        self._fp_count = 0
        self._matched_gt_indices = set()
        self._episode_rewards = []

        self._state = ReviewState(
            task_id=task_id,
            difficulty=self._task["difficulty"],
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            flagged_issues=[],
            current_score=0.0,
            submitted=False,
        )

        issue_categories = list({gt.issue_type for gt in self._ground_truth})
        self._code_metadata = compute_code_metadata(
            self._task["code_files"],
            issue_categories=issue_categories,
        )
        # Pre-compute initial state features (progress=empty at reset)
        self._code_metadata["state_features"] = compute_code_state_features(
            self._code_metadata, progress={}
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
                f"Use 'submit_review' when done. "
                f"Issue categories present: {sorted(set(issue_categories))}."
            ),
            current_score=0.0,
            done=False,
            reward=None,
            reward_breakdown={},
            progress={},
            flagged_summary={},
            code_metadata=self._code_metadata,
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
        reward, feedback, reward_breakdown = self._process_action(action)

        # Track episode rewards for VL return normalization
        if reward is not None:
            self._episode_rewards.append(float(reward))

        max_steps = self._task["max_steps"]
        auto_end = self._state.step_count >= max_steps and not self._state.submitted
        done = self._state.submitted or auto_end

        if auto_end and not self._state.submitted:
            # Auto-end: grade in full (no penalty for hitting step limit)
            final = grade_episode(self._state.flagged_issues, self._ground_truth)
            self._state.current_score = final
            reward = final  # full score, no 0.5x penalty
            reward_breakdown = {"auto_end_grade": final, "total": final}
            feedback += (
                f" Step budget exhausted — auto-graded: {final:.3f}. "
                f"Submit earlier next time for slightly cleaner feedback."
            )
            self._state.submitted = True

        live = compute_live_score(self._state.flagged_issues, self._ground_truth)
        self._state.current_score = live

        progress = self._compute_progress(max_steps)
        flagged_summary = self._compute_flagged_summary()

        # PRM-style dense signal: expected reward-to-go
        # Based on Process Reward Models research: give agent an estimate of
        # how much reward is still available, so it can plan remaining steps.
        tp_found = len(self._matched_gt_indices)
        total_gt = len(self._ground_truth)
        issues_remaining = total_gt - tp_found
        # Expected: each remaining TP gives ~0.12 (base + avg severity bonus)
        expected_reward_to_go = round(issues_remaining * 0.12, 3)

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
            reward_breakdown=reward_breakdown,
            progress=progress,
            flagged_summary=flagged_summary,
            code_metadata={},  # Only populated on reset
            metadata={
                "issues_remaining": issues_remaining,
                "expected_reward_to_go": expected_reward_to_go,
            },
        )

    @property
    def state(self) -> ReviewState:
        return self._state

    # ------------------------------------------------------------------
    # Progress and summary helpers
    # ------------------------------------------------------------------

    def _compute_progress(self, max_steps: int) -> Dict[str, Any]:
        """Compute live precision/recall/f1, step stats, and unfound issue types."""
        flagged = self._state.flagged_issues
        gt = self._ground_truth

        tp = 0
        fp = 0
        matched: Set[int] = set()
        found_types: Set[str] = set()

        for flag in flagged:
            hit = False
            for i, g in enumerate(gt):
                if i not in matched and match_issue(flag, g):
                    tp += 1
                    matched.add(i)
                    found_types.add(g.issue_type)
                    hit = True
                    break
            if not hit:
                fp += 1

        fn = len(gt) - len(matched)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        all_types = {g.issue_type for g in gt}
        unfound_types = sorted(all_types - found_types)

        steps_used = self._state.step_count
        steps_remaining = max(0, max_steps - steps_used)

        # Variable-Length Return Normalization (VL Norm 2025):
        # normalized_return = cumulative_reward / max(steps_used, 1)
        # This makes return comparable across episodes of different length,
        # which is key for multi-task RL where tasks have different max_steps.
        cumulative_reward = sum(self._episode_rewards)
        normalized_return = round(cumulative_reward / max(steps_used, 1), 4)

        progress = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "true_positives": float(tp),
            "false_positives": float(fp),
            "total_ground_truth": float(len(gt)),
            "steps_used": float(steps_used),
            "steps_remaining": float(steps_remaining),
            "unfound_issue_types": unfound_types,
            "normalized_return": normalized_return,
            "cumulative_reward": round(cumulative_reward, 4),
        }

        # 12-dim state feature vector for RL policy/value networks (code2vec/PBRS literature)
        progress["state_features"] = compute_code_state_features(
            self._code_metadata, progress=progress
        )

        return progress

    def _compute_flagged_summary(self) -> Dict[str, Any]:
        """Compute correct/incorrect/near_miss counts."""
        flagged = self._state.flagged_issues
        gt = self._ground_truth

        correct = 0
        near_misses = 0
        incorrect = 0
        matched_gt: Set[int] = set()

        for flag in flagged:
            matched = False
            for i, g in enumerate(gt):
                if i in matched_gt:
                    continue
                if match_issue(flag, g):
                    correct += 1
                    matched_gt.add(i)
                    matched = True
                    break

            if not matched:
                is_near = False
                for i, g in enumerate(gt):
                    if i in matched_gt:
                        continue
                    if match_quality(flag, g) == "near":
                        is_near = True
                        break
                if is_near:
                    near_misses += 1
                else:
                    incorrect += 1

        return {
            "total_flagged": len(flagged),
            "correct": correct,
            "incorrect": incorrect,
            "near_misses": near_misses,
        }

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

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
            ), {}

    def _handle_flag(self, action: ReviewAction):
        if action.line_number is None:
            return _VALIDATION_PENALTY, "flag_issue requires 'line_number'.", {"validation_penalty": _VALIDATION_PENALTY}
        if not action.filename:
            return _VALIDATION_PENALTY, "flag_issue requires 'filename'.", {"validation_penalty": _VALIDATION_PENALTY}
        if action.issue_type not in ("bug", "security", "performance", "logic", None):
            action.issue_type = "bug"
        if action.severity not in ("low", "medium", "high", "critical", None):
            action.severity = "medium"

        # Duplicate check
        for existing in self._state.flagged_issues:
            if (existing.line_number == action.line_number
                    and existing.filename == action.filename):
                return 0.0, (
                    f"Line {action.line_number} in {action.filename} already flagged. "
                    "Use clear_flag first to change it."
                ), {"duplicate": 0.0}

        new_issue = Issue(
            line_number=action.line_number,
            filename=action.filename or "",
            issue_type=action.issue_type or "bug",
            severity=action.severity or "medium",
            description=action.description or "",
            fix_suggestion=action.fix_suggestion,
        )

        # Classify: TP, near-miss (with line distance), or FP
        is_tp = False
        is_near = False
        near_line_diff = 0
        matched_gt_issue: Optional[Issue] = None
        matched_gt_idx: Optional[int] = None

        for i, gt in enumerate(self._ground_truth):
            q = match_quality(new_issue, gt)
            if q == "exact" and i not in self._matched_gt_indices:
                is_tp = True
                matched_gt_issue = gt
                matched_gt_idx = i
                break
            elif q == "near" and not is_near:
                is_near = True
                near_line_diff = abs(new_issue.line_number - gt.line_number)

        self._state.flagged_issues.append(new_issue)

        # PBRS: compute potential before and after this flag
        tp_before = len(self._matched_gt_indices)
        total_gt = len(self._ground_truth)

        reward_breakdown: Dict[str, float] = {}

        if is_tp and matched_gt_issue is not None and matched_gt_idx is not None:
            self._matched_gt_indices.add(matched_gt_idx)
            tp_after = len(self._matched_gt_indices)

            base_reward = _BASE_TP_REWARD
            reward_breakdown["base_tp"] = base_reward

            # Severity exact match bonus
            severity_bonus = 0.0
            if new_issue.severity == matched_gt_issue.severity:
                severity_bonus = _SEVERITY_EXACT_BONUS
                reward_breakdown["severity_exact"] = severity_bonus

            # Temporal bonus: TP caught in first 40% of max_steps
            max_steps = self._task["max_steps"]
            early_threshold = max(1, int(max_steps * 0.4))
            temporal_bonus = 0.0
            if self._state.step_count <= early_threshold:
                temporal_bonus = _TEMPORAL_BONUS
                reward_breakdown["temporal_bonus"] = temporal_bonus

            # Confidence calibration: high confidence TP → small bonus
            confidence_bonus = 0.0
            if action.confidence is not None and action.confidence >= 0.7:
                confidence_bonus = _CONFIDENCE_TP_BONUS
                reward_breakdown["confidence_bonus"] = confidence_bonus

            # PBRS: Φ(s') - Φ(s)  (potential-based shaping, policy-invariant)
            phi_before = compute_potential(tp_before, total_gt)
            phi_after = compute_potential(tp_after, total_gt)
            pbrs_bonus = round(phi_after - phi_before, 4)
            reward_breakdown["pbrs_shaping"] = pbrs_bonus

            reward = base_reward + severity_bonus + temporal_bonus + confidence_bonus + pbrs_bonus
            reward_breakdown["total"] = round(reward, 4)

            sev_note = f", severity +{severity_bonus:.2f}" if severity_bonus else ""
            temp_note = f", early +{temporal_bonus:.2f}" if temporal_bonus else ""
            conf_note = f", conf +{confidence_bonus:.2f}" if confidence_bonus else ""
            pbrs_note = f", progress +{pbrs_bonus:.2f}" if pbrs_bonus > 0 else ""
            feedback = (
                f"Correct! Issue at {action.filename}:{action.line_number} confirmed. "
                f"[+{reward:.2f}{sev_note}{temp_note}{conf_note}{pbrs_note}]"
            )

        elif is_near:
            # Graduated near-miss: smooth exponential decay by line distance
            near_reward = graduated_near_reward(near_line_diff)
            reward_breakdown["near_miss"] = near_reward
            reward_breakdown["line_diff"] = float(near_line_diff)
            reward_breakdown["total"] = near_reward
            feedback = (
                f"Close! Near a real issue at {action.filename}:{action.line_number}. "
                f"[+{near_reward:.3f} — {near_line_diff} lines off, adjust line number]"
            )
            reward = near_reward

        else:
            # False positive — with flood protection
            self._fp_count += 1

            base_penalty = _BASE_FP_PENALTY
            reward_breakdown["base_fp"] = base_penalty

            # Escalating penalty after FP_FLOOD_THRESHOLD FPs
            flood_penalty = 0.0
            if self._fp_count > _FP_FLOOD_THRESHOLD:
                extra = self._fp_count - _FP_FLOOD_THRESHOLD
                flood_penalty = round(-0.02 * extra * _FP_FLOOD_MULTIPLIER, 3)
                reward_breakdown["flood_penalty"] = flood_penalty

            # High-confidence FP: extra penalty
            confidence_penalty = 0.0
            if action.confidence is not None and action.confidence >= 0.7:
                confidence_penalty = _CONFIDENCE_FP_EXTRA
                reward_breakdown["confidence_penalty"] = confidence_penalty

            reward = base_penalty + flood_penalty + confidence_penalty
            reward_breakdown["total"] = round(reward, 4)

            flood_note = f", over-flagging -{abs(flood_penalty):.2f}" if flood_penalty else ""
            conf_note = f", high-confidence penalty {confidence_penalty:.2f}" if confidence_penalty else ""
            feedback = (
                f"No match at {action.filename}:{action.line_number}. "
                f"[{reward:.2f} — false positive{flood_note}{conf_note}]"
            )

        return reward, feedback, reward_breakdown

    def _handle_clear(self, action: ReviewAction):
        if action.line_number is None or not action.filename:
            return _VALIDATION_PENALTY, "clear_flag requires 'line_number' and 'filename'.", {"validation_penalty": _VALIDATION_PENALTY}

        removed_issue = None
        new_list = []
        for f in self._state.flagged_issues:
            if f.line_number == action.line_number and f.filename == action.filename:
                removed_issue = f
            else:
                new_list.append(f)

        if removed_issue is None:
            return 0.0, (
                f"No flagged issue found at {action.filename}:{action.line_number}."
            ), {"no_op": 0.0}

        self._state.flagged_issues = new_list

        # Check if removed issue was TP
        was_tp = any(match_issue(removed_issue, gt) for gt in self._ground_truth)

        if was_tp:
            # Un-track it from matched set
            for i, gt in enumerate(self._ground_truth):
                if match_issue(removed_issue, gt):
                    self._matched_gt_indices.discard(i)
                    break
            reward = _REMOVE_TP_PENALTY
            reward_breakdown = {"removed_tp": reward, "total": reward}
            feedback = (
                f"Removed a correct finding at {action.filename}:{action.line_number}. "
                f"[{reward:.2f}]"
            )
        else:
            # Removing a FP — decrement counter
            self._fp_count = max(0, self._fp_count - 1)
            reward = _REMOVE_FP_REWARD
            reward_breakdown = {"removed_fp": reward, "total": reward}
            feedback = (
                f"Removed a false positive at {action.filename}:{action.line_number}. "
                f"[+{reward:.2f} — good correction]"
            )

        return reward, feedback, reward_breakdown

    def _handle_hint(self):
        hints = self._task.get("hints", [])

        adaptive_hint = self._get_adaptive_hint()
        if adaptive_hint:
            return _HINT_COST, f"Hint: {adaptive_hint} ({_HINT_COST} reward)", {"hint_cost": _HINT_COST}

        if self._hint_index >= len(hints):
            return _HINT_COST, "No more hints available for this task.", {"hint_cost": _HINT_COST}

        hint = hints[self._hint_index]
        self._hint_index += 1
        remaining = len(hints) - self._hint_index
        return _HINT_COST, f"Hint {self._hint_index}/{len(hints)}: {hint} ({remaining} hints left)", {"hint_cost": _HINT_COST}

    def _get_adaptive_hint(self) -> Optional[str]:
        """Generate a context-aware hint based on current episode state."""
        flagged = self._state.flagged_issues
        gt = self._ground_truth

        if not gt:
            return None

        tp_count = len(self._matched_gt_indices)
        fp_count = len(flagged) - tp_count - sum(
            1 for f in flagged
            if any(match_quality(f, g) == "near" for g in gt)
        )

        issue_categories = self._code_metadata.get("issue_categories", [])

        # Many false positives: over-flagging
        if fp_count > tp_count and fp_count >= 2:
            return (
                "You are over-flagging. Focus only on confident, concrete findings. "
                "Consider using clear_flag to remove uncertain flags."
            )

        # No correct flags at all yet
        if len(flagged) > 0 and tp_count == 0:
            if issue_categories:
                cats = ", ".join(sorted(set(issue_categories)))
                return (
                    f"Focus on [{cats}] issues. "
                    "None of your current flags match real issues. Re-examine carefully."
                )

        # Found some but missed whole categories
        if tp_count > 0 and issue_categories:
            found_types: Set[str] = set()
            for i in self._matched_gt_indices:
                found_types.add(gt[i].issue_type)
            missed = sorted(set(issue_categories) - found_types)
            if missed:
                missed_str = ", ".join(missed)
                return (
                    f"Good progress! You've found some issues but haven't flagged any "
                    f"[{missed_str}] issues yet — look again for those specifically."
                )

        return None  # Fall through to static hints

    def _handle_submit(self):
        self._state.submitted = True
        final_score = grade_episode(self._state.flagged_issues, self._ground_truth)
        self._state.current_score = final_score

        tp_count = len(self._matched_gt_indices)
        total_gt = len(self._ground_truth)
        total_flagged = len(self._state.flagged_issues)
        fp_count = total_flagged - tp_count

        # Breakdown for detailed feedback
        detailed = grade_episode_detailed(self._state.flagged_issues, self._ground_truth)

        feedback = (
            f"Review submitted! Final score: {final_score:.3f}. "
            f"Found {tp_count}/{total_gt} issues. "
            f"Precision: {detailed['precision']:.2f}, Recall: {detailed['recall']:.2f}, "
            f"F1: {detailed['f1']:.2f}. "
        )
        if fp_count > 0:
            feedback += f"{fp_count} false positive(s). "
        if detailed["false_negatives"] > 0:
            fn = detailed["false_negatives"]
            feedback += f"{fn} issue(s) missed."

        reward_breakdown = {
            "final_f1": detailed["f1"],
            "severity_accuracy": detailed["severity_accuracy"],
            "final_score": final_score,
            "total": final_score,
        }
        return final_score, feedback, reward_breakdown
