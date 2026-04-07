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


@pytest.fixture
def env_async(env):
    env.reset(task_id="async-review")
    return env


@pytest.fixture
def env_pipeline(env):
    env.reset(task_id="data-pipeline")
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

    def test_reset_has_code_metadata(self, env):
        """Reset observation should include code_metadata."""
        obs = env.reset(task_id="bug-detection")
        assert isinstance(obs.code_metadata, dict)
        assert "total_lines" in obs.code_metadata
        assert "num_functions" in obs.code_metadata
        assert "complexity_estimate" in obs.code_metadata

    def test_reset_code_metadata_has_issue_categories(self, env):
        """code_metadata should list the issue categories present in ground truth."""
        obs = env.reset(task_id="bug-detection")
        assert "issue_categories" in obs.code_metadata
        # bug-detection has only bug type issues
        assert "bug" in obs.code_metadata["issue_categories"]

    def test_reset_has_empty_progress(self, env):
        """Reset observation progress may be empty or absent (populated on step)."""
        obs = env.reset(task_id="bug-detection")
        assert isinstance(obs.progress, dict)

    def test_reset_has_empty_reward_breakdown(self, env):
        obs = env.reset(task_id="bug-detection")
        assert isinstance(obs.reward_breakdown, dict)

    def test_reset_async_task(self, env):
        obs = env.reset(task_id="async-review")
        assert obs.task_id == "async-review"
        assert "async.py" in obs.code_files

    def test_reset_pipeline_task(self, env):
        obs = env.reset(task_id="data-pipeline")
        assert obs.task_id == "data-pipeline"
        assert "pipeline.py" in obs.code_files


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

    def test_flag_has_reward_breakdown(self, env_bug):
        """Every step should have a reward_breakdown dict."""
        obs = env_bug.step(ReviewAction(
            action_type="flag_issue", line_number=6, filename="utils.py",
            issue_type="bug", severity="high", description="test"
        ))
        assert isinstance(obs.reward_breakdown, dict)
        assert len(obs.reward_breakdown) > 0

    def test_flag_has_progress(self, env_bug):
        """Every step should have a progress dict with required keys."""
        obs = env_bug.step(ReviewAction(
            action_type="flag_issue", line_number=6, filename="utils.py",
            issue_type="bug", severity="high", description="test"
        ))
        assert isinstance(obs.progress, dict)
        for key in ("precision", "recall", "f1", "true_positives", "steps_remaining"):
            assert key in obs.progress, f"Missing key: {key}"

    def test_flag_has_flagged_summary(self, env_bug):
        """Every step should have a flagged_summary dict."""
        obs = env_bug.step(ReviewAction(
            action_type="flag_issue", line_number=6, filename="utils.py",
            issue_type="bug", severity="high", description="test"
        ))
        assert isinstance(obs.flagged_summary, dict)
        assert "total_flagged" in obs.flagged_summary
        assert "correct" in obs.flagged_summary
        assert "incorrect" in obs.flagged_summary
        assert "near_misses" in obs.flagged_summary


# ---------------------------------------------------------------------------
# Near-miss tests
# ---------------------------------------------------------------------------

class TestNearMiss:
    def test_near_miss_gives_partial_credit(self, env_bug):
        """A flag within 3-5 lines of a GT issue should give +0.03 not -0.05."""
        # GT issue is at line 6 (off-by-one), so line 10 is 4 away = near miss
        obs = env_bug.step(ReviewAction(
            action_type="flag_issue", line_number=10, filename="utils.py",
            issue_type="bug", severity="high", description="near miss test"
        ))
        # Near miss gives +0.03
        assert obs.reward is not None and obs.reward > 0, (
            f"Expected near-miss +0.03 but got {obs.reward}"
        )
        assert obs.reward == pytest.approx(0.03, abs=0.01)

    def test_near_miss_counted_in_summary(self, env_bug):
        """Near-miss flags should appear in flagged_summary.near_misses."""
        # Line 10 is 4 lines from GT at line 6 → near miss
        obs = env_bug.step(ReviewAction(
            action_type="flag_issue", line_number=10, filename="utils.py",
            issue_type="bug", severity="high", description="near miss"
        ))
        assert obs.flagged_summary.get("near_misses", 0) >= 1

    def test_true_positive_not_counted_as_near_miss(self, env_bug):
        """An exact TP should not be counted as a near miss."""
        obs = env_bug.step(ReviewAction(
            action_type="flag_issue", line_number=6, filename="utils.py",
            issue_type="bug", severity="high", description="exact match"
        ))
        assert obs.flagged_summary.get("correct", 0) >= 1
        assert obs.flagged_summary.get("near_misses", 0) == 0


# ---------------------------------------------------------------------------
# Confidence field tests
# ---------------------------------------------------------------------------

class TestConfidenceField:
    def test_action_with_confidence(self, env_bug):
        """ReviewAction should accept a confidence field."""
        action = ReviewAction(
            action_type="flag_issue", line_number=6, filename="utils.py",
            issue_type="bug", severity="high", description="test",
            confidence=0.9
        )
        assert action.confidence == 0.9

    def test_high_confidence_tp_gets_bonus(self, env_bug):
        """High confidence + TP should give more than base 0.10."""
        obs = env_bug.step(ReviewAction(
            action_type="flag_issue", line_number=6, filename="utils.py",
            issue_type="bug", severity="high", description="test",
            confidence=0.9
        ))
        assert obs.reward is not None and obs.reward > 0.10

    def test_high_confidence_fp_gets_extra_penalty(self, env_bug):
        """High confidence + FP should give more penalty than -0.05."""
        obs = env_bug.step(ReviewAction(
            action_type="flag_issue", line_number=100, filename="utils.py",
            issue_type="bug", severity="low", description="wrong",
            confidence=0.9
        ))
        assert obs.reward is not None and obs.reward < -0.05

    def test_low_confidence_tp_base_reward_only(self, env_bug):
        """Low confidence + TP should give exactly base 0.10 (no bonus)."""
        obs = env_bug.step(ReviewAction(
            action_type="flag_issue", line_number=6, filename="utils.py",
            issue_type="bug", severity="high", description="test",
            confidence=0.5
        ))
        assert obs.reward is not None
        # Should be 0.10 base + possible temporal bonus but no confidence bonus
        assert obs.reward >= 0.10

    def test_no_confidence_field_is_none(self):
        """ReviewAction without confidence defaults to None."""
        action = ReviewAction(
            action_type="flag_issue", line_number=6, filename="utils.py",
        )
        assert action.confidence is None

    def test_confidence_in_action_to_dict(self):
        """confidence should round-trip through to_dict/from_dict."""
        action = ReviewAction(
            action_type="flag_issue", line_number=6, filename="utils.py",
            confidence=0.75
        )
        d = action.to_dict()
        assert d["confidence"] == 0.75
        action2 = ReviewAction.from_dict(d)
        assert action2.confidence == 0.75

    def test_related_lines_field(self):
        """ReviewAction should accept a related_lines field."""
        action = ReviewAction(
            action_type="flag_issue", line_number=6, filename="utils.py",
            related_lines=[6, 7, 8]
        )
        assert action.related_lines == [6, 7, 8]
        d = action.to_dict()
        assert d["related_lines"] == [6, 7, 8]
        action2 = ReviewAction.from_dict(d)
        assert action2.related_lines == [6, 7, 8]


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


# ---------------------------------------------------------------------------
# New task tests
# ---------------------------------------------------------------------------

class TestNewTasks:
    def test_async_review_task_exists(self, env):
        obs = env.reset(task_id="async-review")
        assert obs.task_id == "async-review"
        assert obs.done is False

    def test_async_review_has_correct_issue_count(self):
        from tasks.data import ALL_TASKS
        task = ALL_TASKS["async-review"]
        assert len(task["ground_truth_issues"]) == 6

    def test_async_review_has_async_py(self, env):
        obs = env.reset(task_id="async-review")
        assert "async.py" in obs.code_files
        code = obs.code_files["async.py"]
        assert "asyncio" in code
        assert "aiohttp" in code

    def test_async_review_max_steps(self):
        from tasks.data import ALL_TASKS
        task = ALL_TASKS["async-review"]
        assert task["max_steps"] == 20

    def test_data_pipeline_task_exists(self, env):
        obs = env.reset(task_id="data-pipeline")
        assert obs.task_id == "data-pipeline"
        assert obs.done is False

    def test_data_pipeline_has_correct_issue_count(self):
        from tasks.data import ALL_TASKS
        task = ALL_TASKS["data-pipeline"]
        assert len(task["ground_truth_issues"]) == 7

    def test_data_pipeline_has_pipeline_py(self, env):
        obs = env.reset(task_id="data-pipeline")
        assert "pipeline.py" in obs.code_files
        code = obs.code_files["pipeline.py"]
        assert "sqlite3" in code
        assert "hashlib" in code

    def test_data_pipeline_max_steps(self):
        from tasks.data import ALL_TASKS
        task = ALL_TASKS["data-pipeline"]
        assert task["max_steps"] == 25

    def test_task_count(self):
        from tasks.data import TASK_IDS
        assert len(TASK_IDS) >= 6

    def test_async_review_correct_tp_reward(self, env_async):
        """Flagging a known issue in async-review should give positive reward."""
        obs = env_async.step(ReviewAction(
            action_type="flag_issue", line_number=22, filename="async.py",
            issue_type="bug", severity="high",
            description="ClientSession not closed"
        ))
        assert obs.reward is not None and obs.reward > 0

    def test_data_pipeline_correct_tp_reward(self, env_pipeline):
        """Flagging a known SQL injection in pipeline.py should give positive reward."""
        obs = env_pipeline.step(ReviewAction(
            action_type="flag_issue", line_number=27, filename="pipeline.py",
            issue_type="security", severity="critical",
            description="SQL injection"
        ))
        assert obs.reward is not None and obs.reward > 0

    def test_all_tasks_have_hints(self):
        from tasks.data import ALL_TASKS
        for task_id, task in ALL_TASKS.items():
            assert "hints" in task, f"Task {task_id} missing hints"
            assert len(task["hints"]) >= 3, f"Task {task_id} has fewer than 3 hints"


# ---------------------------------------------------------------------------
# Observation serialization
# ---------------------------------------------------------------------------

class TestObservationSerialization:
    def test_reset_obs_to_dict_has_new_fields(self, env):
        """to_dict() should include all new fields."""
        obs = env.reset(task_id="bug-detection")
        d = obs.to_dict()
        assert "reward_breakdown" in d
        assert "progress" in d
        assert "flagged_summary" in d
        assert "code_metadata" in d

    def test_obs_from_dict_handles_missing_new_fields(self):
        """from_dict() should handle missing new fields gracefully."""
        d = {
            "task_id": "bug-detection",
            "task_description": "test",
            "code_files": {},
            "language": "python",
            "flagged_issues": [],
            "step_count": 0,
            "max_steps": 15,
            "hints_remaining": 3,
            "feedback": "",
            "current_score": 0.0,
            "done": False,
            "reward": None,
            # No reward_breakdown, progress, flagged_summary, code_metadata
        }
        obs = ReviewObservation.from_dict(d)
        assert obs.reward_breakdown == {}
        assert obs.progress == {}
        assert obs.flagged_summary == {}
        assert obs.code_metadata == {}

    def test_step_obs_to_dict_round_trip(self, env_bug):
        obs = env_bug.step(ReviewAction(
            action_type="flag_issue", line_number=6, filename="utils.py",
            issue_type="bug", severity="high", description="test"
        ))
        d = obs.to_dict()
        obs2 = ReviewObservation.from_dict(d)
        assert obs2.task_id == obs.task_id
        assert obs2.step_count == obs.step_count
        assert isinstance(obs2.reward_breakdown, dict)
        assert isinstance(obs2.progress, dict)
        assert isinstance(obs2.flagged_summary, dict)


# ---------------------------------------------------------------------------
# Severity exact match bonus
# ---------------------------------------------------------------------------

class TestSeverityBonus:
    def test_severity_match_gives_extra_reward(self, env_bug):
        """Exact severity match should give more than a severity mismatch."""
        # GT at line 6 is "high"
        obs_match = env_bug.step(ReviewAction(
            action_type="flag_issue", line_number=6, filename="utils.py",
            issue_type="bug", severity="high", description="exact severity"
        ))
        env_bug.reset(task_id="bug-detection")
        obs_wrong = env_bug.step(ReviewAction(
            action_type="flag_issue", line_number=6, filename="utils.py",
            issue_type="bug", severity="low", description="wrong severity"
        ))
        assert obs_match.reward > obs_wrong.reward

    def test_severity_bonus_in_reward_breakdown(self, env_bug):
        """reward_breakdown should include 'severity_exact' key on correct severity."""
        obs = env_bug.step(ReviewAction(
            action_type="flag_issue", line_number=6, filename="utils.py",
            issue_type="bug", severity="high", description="correct severity"
        ))
        assert "severity_exact" in obs.reward_breakdown

    def test_severity_mismatch_no_severity_bonus(self, env_bug):
        """Wrong severity should not include 'severity_exact' key."""
        obs = env_bug.step(ReviewAction(
            action_type="flag_issue", line_number=6, filename="utils.py",
            issue_type="bug", severity="low", description="wrong severity"
        ))
        assert "severity_exact" not in obs.reward_breakdown


# ---------------------------------------------------------------------------
# Flood protection (escalating FP penalty)
# ---------------------------------------------------------------------------

class TestFloodProtection:
    def test_many_fps_escalate_penalty(self, env_bug):
        """After 3 false positives, each subsequent FP should have larger penalty."""
        rewards = []
        for line in [101, 102, 103, 104, 105]:
            obs = env_bug.step(ReviewAction(
                action_type="flag_issue", line_number=line, filename="utils.py",
                issue_type="bug", severity="low", description="fp"
            ))
            if obs.reward is not None and obs.reward < 0:
                rewards.append(obs.reward)

        # The 4th and 5th FPs should have larger absolute penalty
        if len(rewards) >= 4:
            assert abs(rewards[-1]) >= abs(rewards[0]), (
                f"Expected escalating penalty but got {rewards}"
            )

    def test_fp_below_threshold_normal_penalty(self, env_bug):
        """First FP should get standard -0.05 penalty."""
        obs = env_bug.step(ReviewAction(
            action_type="flag_issue", line_number=200, filename="utils.py",
            issue_type="bug", severity="low", description="first fp"
        ))
        assert obs.reward is not None
        assert obs.reward == pytest.approx(-0.05, abs=0.01)

    def test_clearing_fp_reduces_penalty_track(self, env_bug):
        """Clearing a FP should give positive reward."""
        env_bug.step(ReviewAction(
            action_type="flag_issue", line_number=200, filename="utils.py",
            issue_type="bug", severity="low", description="fp"
        ))
        obs = env_bug.step(ReviewAction(
            action_type="clear_flag", line_number=200, filename="utils.py",
        ))
        assert obs.reward is not None and obs.reward > 0


# ---------------------------------------------------------------------------
# Unfound issue types in progress
# ---------------------------------------------------------------------------

class TestUnfoundIssueTypes:
    def test_unfound_types_present_at_start(self, env_bug):
        """Before flagging anything, all GT issue types should be in unfound_issue_types."""
        obs = env_bug.step(ReviewAction(action_type="request_hint"))
        unfound = obs.progress.get("unfound_issue_types", [])
        assert "bug" in unfound

    def test_unfound_types_shrinks_when_issue_found(self, env_bug):
        """Finding a bug should remove 'bug' from unfound_issue_types."""
        obs_before = env_bug.step(ReviewAction(action_type="request_hint"))
        unfound_before = set(obs_before.progress.get("unfound_issue_types", []))

        env_bug.step(ReviewAction(
            action_type="flag_issue", line_number=6, filename="utils.py",
            issue_type="bug", severity="high", description="found a bug"
        ))
        obs_after = env_bug.step(ReviewAction(action_type="request_hint"))
        unfound_after = set(obs_after.progress.get("unfound_issue_types", []))

        # bug should now be gone from unfound
        assert "bug" not in unfound_after or len(unfound_after) < len(unfound_before)

    def test_unfound_types_is_list(self, env_bug):
        obs = env_bug.step(ReviewAction(action_type="request_hint"))
        assert isinstance(obs.progress.get("unfound_issue_types", []), list)


# ---------------------------------------------------------------------------
# API security task
# ---------------------------------------------------------------------------

class TestApiSecurityTask:
    def test_api_security_task_exists(self, env):
        obs = env.reset(task_id="api-security")
        assert obs.task_id == "api-security"
        assert obs.done is False

    def test_api_security_has_api_py(self, env):
        obs = env.reset(task_id="api-security")
        assert "api.py" in obs.code_files

    def test_api_security_has_8_issues(self):
        from tasks.data import ALL_TASKS
        task = ALL_TASKS["api-security"]
        assert len(task["ground_truth_issues"]) == 8

    def test_api_security_has_critical_issues(self):
        from tasks.data import ALL_TASKS
        task = ALL_TASKS["api-security"]
        severities = {i["severity"] for i in task["ground_truth_issues"]}
        assert "critical" in severities

    def test_api_security_tp_reward(self, env):
        env.reset(task_id="api-security")
        obs = env.step(ReviewAction(
            action_type="flag_issue", line_number=38, filename="api.py",
            issue_type="security", severity="critical",
            description="SQL injection via f-string"
        ))
        assert obs.reward is not None and obs.reward > 0

    def test_api_security_keyword_baseline_finds_issues(self):
        from tasks.data import ALL_TASKS
        from server.graders import run_keyword_baseline
        task = ALL_TASKS["api-security"]
        findings = run_keyword_baseline(task)
        assert len(findings) >= 2

    def test_api_security_difficulty_hard(self):
        from tasks.data import ALL_TASKS
        task = ALL_TASKS["api-security"]
        assert task["difficulty"] == "hard"


# ---------------------------------------------------------------------------
# Auto-end gives full score (not 0.5x)
# ---------------------------------------------------------------------------

class TestAutoEndFullScore:
    def test_auto_end_uses_full_grade(self, env_bug):
        """Auto-end should give full grade_episode score, not a penalized value."""
        # Flag all 3 correct bugs first
        for line, sev in [(6, "high"), (13, "medium"), (33, "low")]:
            env_bug.step(ReviewAction(
                action_type="flag_issue", line_number=line, filename="utils.py",
                issue_type="bug", severity=sev, description=f"bug at {line}"
            ))
        # Exhaust remaining steps with hints
        max_steps = 15
        for _ in range(max_steps - 3 - 1):
            obs = env_bug.step(ReviewAction(action_type="request_hint"))
            if obs.done:
                break

        obs = env_bug.step(ReviewAction(action_type="request_hint"))
        if obs.done and obs.reward_breakdown.get("auto_end_grade") is not None:
            # If auto-ended, score should be >= 0.7 since all 3 bugs found
            assert obs.reward >= 0.7, f"Auto-end gave {obs.reward} instead of full grade"


# ---------------------------------------------------------------------------
# Function ranges in code_metadata
# ---------------------------------------------------------------------------

class TestFunctionRanges:
    def test_reset_has_function_ranges(self, env):
        obs = env.reset(task_id="bug-detection")
        assert "function_ranges" in obs.code_metadata

    def test_function_ranges_is_list(self, env):
        obs = env.reset(task_id="bug-detection")
        assert isinstance(obs.code_metadata["function_ranges"], list)

    def test_function_ranges_have_required_fields(self, env):
        obs = env.reset(task_id="bug-detection")
        for fr in obs.code_metadata["function_ranges"]:
            assert "name" in fr
            assert "file" in fr
            assert "start" in fr
            assert "end" in fr

    def test_function_ranges_nonempty_for_python(self, env):
        obs = env.reset(task_id="bug-detection")
        assert len(obs.code_metadata["function_ranges"]) > 0
