"""
Tests for the grading logic.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from models import Issue
from server.graders import grade_episode, match_issue, run_keyword_baseline
from tasks.data import ALL_TASKS, TASK_IDS


def _issue(line, filename, itype="bug", severity="medium", desc=""):
    return Issue(line_number=line, filename=filename, issue_type=itype,
                 severity=severity, description=desc)


# ---------------------------------------------------------------------------
# match_issue()
# ---------------------------------------------------------------------------

class TestMatchIssue:
    def test_exact_match(self):
        f = _issue(6, "utils.py", "bug", "high")
        gt = _issue(6, "utils.py", "bug", "high")
        assert match_issue(f, gt) is True

    def test_line_within_tolerance(self):
        f = _issue(7, "utils.py", "bug", "high")
        gt = _issue(6, "utils.py", "bug", "high")
        assert match_issue(f, gt, line_tolerance=2) is True

    def test_line_outside_tolerance(self):
        f = _issue(10, "utils.py", "bug", "high")
        gt = _issue(6, "utils.py", "bug", "high")
        assert match_issue(f, gt, line_tolerance=2) is False

    def test_wrong_filename(self):
        f = _issue(6, "other.py", "bug", "high")
        gt = _issue(6, "utils.py", "bug", "high")
        assert match_issue(f, gt) is False

    def test_bug_logic_interchangeable(self):
        f = _issue(6, "utils.py", "logic", "high")
        gt = _issue(6, "utils.py", "bug", "high")
        assert match_issue(f, gt) is True

    def test_logic_bug_interchangeable(self):
        f = _issue(6, "utils.py", "bug", "high")
        gt = _issue(6, "utils.py", "logic", "high")
        assert match_issue(f, gt) is True

    def test_wrong_type_no_match(self):
        f = _issue(6, "utils.py", "performance", "high")
        gt = _issue(6, "utils.py", "bug", "high")
        assert match_issue(f, gt) is False


# ---------------------------------------------------------------------------
# grade_episode()
# ---------------------------------------------------------------------------

class TestGradeEpisode:
    def test_empty_both_is_perfect(self):
        assert grade_episode([], []) == 1.0

    def test_empty_flagged_is_zero(self):
        gt = [_issue(6, "utils.py")]
        assert grade_episode([], gt) == 0.0

    def test_false_positives_only_is_zero(self):
        flagged = [_issue(100, "utils.py"), _issue(200, "utils.py")]
        gt = [_issue(6, "utils.py")]
        score = grade_episode(flagged, gt)
        assert score == 0.0

    def test_perfect_match_is_near_one(self):
        gt = [
            _issue(6, "utils.py", "bug", "high"),
            _issue(13, "utils.py", "bug", "medium"),
        ]
        score = grade_episode(gt, gt)
        assert score >= 0.9

    def test_partial_match(self):
        gt = [
            _issue(6, "utils.py", "bug", "high"),
            _issue(13, "utils.py", "bug", "medium"),
            _issue(33, "utils.py", "bug", "low"),
        ]
        flagged = [_issue(6, "utils.py", "bug", "high")]  # only 1 of 3
        score = grade_episode(flagged, gt)
        # recall = 1/3, precision = 1/1, F1 = 0.5
        assert 0.3 < score < 0.6

    def test_false_positives_lower_score(self):
        gt = [_issue(6, "utils.py", "bug", "high")]
        perfect = [_issue(6, "utils.py", "bug", "high")]
        with_fp = [_issue(6, "utils.py", "bug", "high"), _issue(100, "utils.py")]
        assert grade_episode(perfect, gt) > grade_episode(with_fp, gt)

    def test_severity_mismatch_lowers_score(self):
        gt = [_issue(6, "utils.py", "bug", "critical")]
        exact = [_issue(6, "utils.py", "bug", "critical")]
        wrong_sev = [_issue(6, "utils.py", "bug", "low")]
        assert grade_episode(exact, gt) > grade_episode(wrong_sev, gt)

    def test_score_is_always_in_0_1(self):
        import random
        random.seed(0)
        gt = [_issue(i * 10, "f.py") for i in range(5)]
        for _ in range(20):
            n = random.randint(0, 10)
            flagged = [_issue(random.randint(1, 100), "f.py") for _ in range(n)]
            score = grade_episode(flagged, gt)
            assert 0.0 <= score <= 1.0, f"Score {score} out of range"

    def test_multifile_match(self):
        gt = [
            _issue(21, "views.py", "performance", "high"),
            _issue(8, "models.py", "security", "critical"),
        ]
        flagged = [
            _issue(21, "views.py", "performance", "high"),
            _issue(8, "models.py", "security", "critical"),
        ]
        score = grade_episode(flagged, gt)
        assert score >= 0.85

    def test_multifile_wrong_file_no_match(self):
        gt = [_issue(21, "views.py", "performance", "high")]
        flagged = [_issue(21, "models.py", "performance", "high")]  # wrong file
        assert grade_episode(flagged, gt) == 0.0


# ---------------------------------------------------------------------------
# run_keyword_baseline()
# ---------------------------------------------------------------------------

class TestKeywordBaseline:
    def test_baseline_returns_list(self):
        from tasks.data import TASK_BUG_DETECTION
        findings = run_keyword_baseline(TASK_BUG_DETECTION)
        assert isinstance(findings, list)

    def test_baseline_issues_have_correct_types(self):
        from tasks.data import TASK_BUG_DETECTION
        findings = run_keyword_baseline(TASK_BUG_DETECTION)
        for f in findings:
            assert isinstance(f, Issue)
            assert f.issue_type in ("bug", "security", "performance", "logic")
            assert f.severity in ("low", "medium", "high", "critical")

    def test_baseline_finds_some_security_issues(self):
        from tasks.data import TASK_SECURITY_AUDIT
        findings = run_keyword_baseline(TASK_SECURITY_AUDIT)
        security_finds = [f for f in findings if f.issue_type == "security"]
        assert len(security_finds) >= 2

    def test_baseline_score_in_range(self):
        for task_id in TASK_IDS:
            task = ALL_TASKS[task_id]
            findings = run_keyword_baseline(task)
            gt = [Issue.from_dict(i) for i in task["ground_truth_issues"]]
            score = grade_episode(findings, gt)
            assert 0.0 <= score <= 1.0, f"Task {task_id}: score={score} out of range"

    def test_baseline_score_is_nonzero(self):
        """Heuristic should find at least something in most tasks."""
        for task_id in TASK_IDS:
            task = ALL_TASKS[task_id]
            findings = run_keyword_baseline(task)
            gt = [Issue.from_dict(i) for i in task["ground_truth_issues"]]
            score = grade_episode(findings, gt)
            # Not every task may have regex hits, but security-audit should
            if task_id == "security-audit":
                assert score > 0.0, f"Heuristic found nothing in {task_id}"


# ---------------------------------------------------------------------------
# Ground truth sanity checks
# ---------------------------------------------------------------------------

class TestGroundTruth:
    def test_all_tasks_have_3_plus_issues(self):
        for task_id, task in ALL_TASKS.items():
            assert len(task["ground_truth_issues"]) >= 3, (
                f"Task {task_id} has fewer than 3 issues"
            )

    def test_all_tasks_have_valid_difficulties(self):
        difficulties = {t["difficulty"] for t in ALL_TASKS.values()}
        assert "easy" in difficulties
        assert "medium" in difficulties
        assert "hard" in difficulties

    def test_all_issues_have_required_fields(self):
        for task_id, task in ALL_TASKS.items():
            for i, issue in enumerate(task["ground_truth_issues"]):
                assert "line_number" in issue, f"{task_id}[{i}] missing line_number"
                assert "filename" in issue, f"{task_id}[{i}] missing filename"
                assert "issue_type" in issue, f"{task_id}[{i}] missing issue_type"
                assert "severity" in issue, f"{task_id}[{i}] missing severity"

    def test_bug_detection_issues_in_utils_py(self):
        task = ALL_TASKS["bug-detection"]
        for issue in task["ground_truth_issues"]:
            assert issue["filename"] == "utils.py"

    def test_comprehensive_has_multifile_issues(self):
        task = ALL_TASKS["comprehensive-review"]
        files = {i["filename"] for i in task["ground_truth_issues"]}
        assert "views.py" in files
        assert "models.py" in files
