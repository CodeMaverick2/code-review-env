"""
Tests for CodeReviewEnvironment.

Run with:  pytest tests/ -v
Or:        python -m pytest tests/ -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from models import ReviewAction, ReviewObservation, ReviewState
from server.environment import CodeReviewEnvironment
from tasks.data import ALL_TASKS, TASK_IDS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    return CodeReviewEnvironment()


@pytest.fixture
def env_bug(env):
    env.reset(task_id="bug-detection")
    return env


@pytest.fixture
def env_sec(env):
    env.reset(task_id="security-audit")
    return env


@pytest.fixture
def env_hard(env):
    env.reset(task_id="comprehensive-review")
    return env


# ---------------------------------------------------------------------------
# reset() tests
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_returns_observation(self, env):
        obs = env.reset()
        assert isinstance(obs, ReviewObservation)

    def test_reset_done_is_false(self, env):
        obs = env.reset()
        assert obs.done is False

    def test_reset_reward_is_none(self, env):
        obs = env.reset()
        assert obs.reward is None

    def test_reset_has_code_files(self, env):
        obs = env.reset()
        assert isinstance(obs.code_files, dict)
        assert len(obs.code_files) > 0

    def test_reset_step_count_zero(self, env):
        obs = env.reset()
        assert obs.step_count == 0

    def test_reset_no_flagged_issues(self, env):
        obs = env.reset()
        assert obs.flagged_issues == []

    def test_reset_specific_task(self, env):
        for task_id in TASK_IDS:
            obs = env.reset(task_id=task_id)
            assert obs.task_id == task_id

    def test_reset_bug_detection(self, env):
        obs = env.reset(task_id="bug-detection")
        assert "utils.py" in obs.code_files

    def test_reset_security_audit(self, env):
        obs = env.reset(task_id="security-audit")
        assert "app.py" in obs.code_files

    def test_reset_comprehensive(self, env):
        obs = env.reset(task_id="comprehensive-review")
        assert "views.py" in obs.code_files
        assert "models.py" in obs.code_files

    def test_reset_with_seed_is_reproducible(self, env):
        obs1 = env.reset(seed=42)
        task1 = obs1.task_id
        obs2 = env.reset(seed=42)
        task2 = obs2.task_id
        assert task1 == task2

    def test_reset_clears_previous_state(self, env):
        env.reset(task_id="bug-detection")
        env.step(ReviewAction(
            action_type="flag_issue", line_number=6, filename="utils.py",
            issue_type="bug", severity="high", description="test"
        ))
        obs = env.reset(task_id="bug-detection")
        assert obs.flagged_issues == []
        assert obs.step_count == 0


# ---------------------------------------------------------------------------
# step() — flag_issue tests
# ---------------------------------------------------------------------------

class TestFlagIssue:
    def test_flag_increments_step_count(self, env_bug):
        obs = env_bug.step(ReviewAction(
            action_type="flag_issue", line_number=6, filename="utils.py",
            issue_type="bug", severity="high", description="test"
        ))
        assert obs.step_count == 1

    def test_flag_adds_to_flagged_issues(self, env_bug):
        obs = env_bug.step(ReviewAction(
            action_type="flag_issue", line_number=6, filename="utils.py",
            issue_type="bug", severity="high", description="test"
        ))
        assert len(obs.flagged_issues) == 1

    def test_flag_true_positive_gives_positive_reward(self, env_bug):
        obs = env_bug.step(ReviewAction(
            action_type="flag_issue", line_number=6, filename="utils.py",
            issue_type="bug", severity="high", description="off-by-one"
        ))
        assert obs.reward is not None and obs.reward > 0

    def test_flag_false_positive_gives_negative_reward(self, env_bug):
        obs = env_bug.step(ReviewAction(
            action_type="flag_issue", line_number=100, filename="utils.py",
            issue_type="bug", severity="low", description="nonexistent issue"
        ))
        assert obs.reward is not None and obs.reward < 0

    def test_flag_missing_line_number_gives_penalty(self, env_bug):
        obs = env_bug.step(ReviewAction(
            action_type="flag_issue", filename="utils.py",
            issue_type="bug", severity="high", description="test"
        ))
        assert obs.reward is not None and obs.reward <= 0

    def test_flag_duplicate_line_no_change(self, env_bug):
        env_bug.step(ReviewAction(
            action_type="flag_issue", line_number=6, filename="utils.py",
            issue_type="bug", severity="high", description="test"
        ))
        obs = env_bug.step(ReviewAction(
            action_type="flag_issue", line_number=6, filename="utils.py",
            issue_type="bug", severity="high", description="same line again"
        ))
        assert len(obs.flagged_issues) == 1  # not doubled

    def test_flag_multiple_issues(self, env_bug):
        for line in [6, 13, 33]:
            env_bug.step(ReviewAction(
                action_type="flag_issue", line_number=line, filename="utils.py",
                issue_type="bug", severity="medium", description=f"bug at {line}"
            ))
        obs = env_bug.state
        assert len(obs.flagged_issues) == 3


# ---------------------------------------------------------------------------
# step() — clear_flag tests
# ---------------------------------------------------------------------------

class TestClearFlag:
    def test_clear_removes_flag(self, env_bug):
        env_bug.step(ReviewAction(
            action_type="flag_issue", line_number=6, filename="utils.py",
            issue_type="bug", severity="high", description="test"
        ))
        obs = env_bug.step(ReviewAction(
            action_type="clear_flag", line_number=6, filename="utils.py",
            description=""
        ))
        assert len(obs.flagged_issues) == 0

    def test_clear_nonexistent_flag_no_reward(self, env_bug):
        obs = env_bug.step(ReviewAction(
            action_type="clear_flag", line_number=999, filename="utils.py",
            description=""
        ))
        assert obs.reward == 0.0

    def test_clear_false_positive_gives_positive_reward(self, env_bug):
        # First flag a FP
        env_bug.step(ReviewAction(
            action_type="flag_issue", line_number=100, filename="utils.py",
            issue_type="bug", severity="low", description="wrong"
        ))
        obs = env_bug.step(ReviewAction(
            action_type="clear_flag", line_number=100, filename="utils.py",
            description=""
        ))
        assert obs.reward is not None and obs.reward > 0


# ---------------------------------------------------------------------------
# step() — request_hint tests
# ---------------------------------------------------------------------------

class TestRequestHint:
    def test_hint_gives_small_negative_reward(self, env_bug):
        obs = env_bug.step(ReviewAction(action_type="request_hint"))
        assert obs.reward is not None and obs.reward < 0

    def test_hint_decrements_hints_remaining(self, env_bug):
        before = env_bug.state.step_count  # proxy check
        obs1 = env_bug.step(ReviewAction(action_type="request_hint"))
        obs2 = env_bug.step(ReviewAction(action_type="request_hint"))
        assert obs2.hints_remaining < obs1.hints_remaining

    def test_hint_content_in_feedback(self, env_bug):
        obs = env_bug.step(ReviewAction(action_type="request_hint"))
        assert "hint" in obs.feedback.lower() or "loop" in obs.feedback.lower()


# ---------------------------------------------------------------------------
# step() — submit_review tests
# ---------------------------------------------------------------------------

class TestSubmitReview:
    def test_submit_ends_episode(self, env_bug):
        obs = env_bug.step(ReviewAction(action_type="submit_review"))
        assert obs.done is True

    def test_submit_reward_is_float_in_range(self, env_bug):
        obs = env_bug.step(ReviewAction(action_type="submit_review"))
        assert obs.reward is not None
        assert 0.0 <= obs.reward <= 1.0

    def test_submit_all_bugs_gives_high_score(self, env_bug):
        # Flag all 3 correct bugs
        for line, sev in [(6, "high"), (13, "medium"), (33, "low")]:
            env_bug.step(ReviewAction(
                action_type="flag_issue", line_number=line, filename="utils.py",
                issue_type="bug", severity=sev, description=f"bug at line {line}"
            ))
        obs = env_bug.step(ReviewAction(action_type="submit_review"))
        assert obs.reward is not None and obs.reward >= 0.7

    def test_submit_no_flags_gives_zero(self, env_bug):
        obs = env_bug.step(ReviewAction(action_type="submit_review"))
        assert obs.reward == 0.0

    def test_submit_after_done_is_noop(self, env_bug):
        env_bug.step(ReviewAction(action_type="submit_review"))
        obs2 = env_bug.step(ReviewAction(action_type="submit_review"))
        assert obs2.done is True  # still done


# ---------------------------------------------------------------------------
# state property tests
# ---------------------------------------------------------------------------

class TestState:
    def test_state_returns_review_state(self, env):
        env.reset(task_id="bug-detection")
        st = env.state
        assert isinstance(st, ReviewState)

    def test_state_has_episode_id(self, env):
        env.reset(task_id="bug-detection")
        assert env.state.episode_id is not None

    def test_state_tracks_step_count(self, env_bug):
        env_bug.step(ReviewAction(action_type="request_hint"))
        assert env_bug.state.step_count == 1

    def test_state_tracks_flagged_issues(self, env_bug):
        env_bug.step(ReviewAction(
            action_type="flag_issue", line_number=6, filename="utils.py",
            issue_type="bug", severity="high", description="test"
        ))
        assert len(env_bug.state.flagged_issues) == 1


# ---------------------------------------------------------------------------
# Unknown action type
# ---------------------------------------------------------------------------

class TestUnknownAction:
    def test_unknown_action_type_no_crash(self, env_bug):
        obs = env_bug.step(ReviewAction(action_type="invalid_action"))
        assert obs is not None
        assert obs.done is False or obs.done is True


# ---------------------------------------------------------------------------
# Max steps auto-end
# ---------------------------------------------------------------------------

class TestMaxSteps:
    def test_episode_auto_ends_at_max_steps(self):
        """Verify episode ends when step budget is exhausted."""
        env = CodeReviewEnvironment()
        obs = env.reset(task_id="bug-detection")
        max_steps = obs.max_steps

        for _ in range(max_steps):
            obs = env.step(ReviewAction(action_type="request_hint"))
            if obs.done:
                break

        assert obs.done is True
