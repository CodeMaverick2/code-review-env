"""
Tests for the grading logic.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from models import Issue
from server.graders import (
    grade_episode, match_issue, run_keyword_baseline,
    match_quality, compute_code_metadata, grade_episode_detailed,
    NEAR_TOLERANCE,
)
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

    def test_near_tolerance_param_accepted(self):
        """match_issue should accept near_tolerance param without error."""
        f = _issue(6, "utils.py", "bug", "high")
        gt = _issue(6, "utils.py", "bug", "high")
        result = match_issue(f, gt, line_tolerance=2, near_tolerance=5)
        assert result is True


# ---------------------------------------------------------------------------
# match_quality()
# ---------------------------------------------------------------------------

class TestMatchQuality:
    def test_exact_match_within_2_lines(self):
        f = _issue(7, "utils.py", "bug", "high")
        gt = _issue(6, "utils.py", "bug", "high")
        assert match_quality(f, gt) == "exact"

    def test_near_match_3_to_5_lines(self):
        # 4 lines away from GT at 6 → near
        f = _issue(10, "utils.py", "bug", "high")
        gt = _issue(6, "utils.py", "bug", "high")
        assert match_quality(f, gt) == "near"

    def test_near_match_exactly_3_lines(self):
        f = _issue(9, "utils.py", "bug", "high")
        gt = _issue(6, "utils.py", "bug", "high")
        assert match_quality(f, gt) == "near"

    def test_near_match_exactly_5_lines(self):
        f = _issue(11, "utils.py", "bug", "high")
        gt = _issue(6, "utils.py", "bug", "high")
        assert match_quality(f, gt) == "near"

    def test_no_match_beyond_5_lines(self):
        f = _issue(12, "utils.py", "bug", "high")
        gt = _issue(6, "utils.py", "bug", "high")
        assert match_quality(f, gt) == "none"

    def test_no_match_wrong_file(self):
        f = _issue(6, "other.py", "bug", "high")
        gt = _issue(6, "utils.py", "bug", "high")
        assert match_quality(f, gt) == "none"

    def test_near_requires_compatible_type(self):
        """Near match requires compatible issue type (not just proximity)."""
        f = _issue(10, "utils.py", "performance", "high")
        gt = _issue(6, "utils.py", "bug", "high")
        # 4 lines away but wrong type → none
        assert match_quality(f, gt) == "none"

    def test_near_with_compatible_type(self):
        """Near match works with compatible type (bug/logic)."""
        f = _issue(10, "utils.py", "logic", "high")
        gt = _issue(6, "utils.py", "bug", "high")
        # 4 lines away, compatible type → near
        assert match_quality(f, gt) == "near"

    def test_near_tolerance_constant(self):
        assert NEAR_TOLERANCE == 5


# ---------------------------------------------------------------------------
# compute_code_metadata()
# ---------------------------------------------------------------------------

class TestComputeCodeMetadata:
    def test_returns_dict(self):
        code = {"test.py": "def foo(): pass\n"}
        result = compute_code_metadata(code)
        assert isinstance(result, dict)

    def test_total_lines(self):
        code = {"test.py": "line1\nline2\nline3\n"}
        result = compute_code_metadata(code)
        assert result["total_lines"] == 3

    def test_num_functions(self):
        code = {"test.py": "def foo():\n    pass\n\ndef bar():\n    pass\n"}
        result = compute_code_metadata(code)
        assert result["num_functions"] == 2

    def test_function_names(self):
        code = {"test.py": "def foo():\n    pass\n\ndef bar():\n    pass\n"}
        result = compute_code_metadata(code)
        assert "foo" in result["function_names"]
        assert "bar" in result["function_names"]

    def test_num_classes(self):
        code = {"test.py": "class Foo:\n    pass\n\nclass Bar:\n    pass\n"}
        result = compute_code_metadata(code)
        assert result["num_classes"] == 2

    def test_class_names(self):
        code = {"test.py": "class Foo:\n    pass\n"}
        result = compute_code_metadata(code)
        assert "Foo" in result["class_names"]

    def test_imports(self):
        code = {"test.py": "import os\nimport sys\nfrom typing import List\n"}
        result = compute_code_metadata(code)
        assert "os" in result["imports"]
        assert "sys" in result["imports"]
        assert "typing" in result["imports"]

    def test_complexity_low(self):
        code = {"test.py": "def foo():\n    return 1\n"}
        result = compute_code_metadata(code)
        assert result["complexity_estimate"] == "low"

    def test_complexity_medium(self):
        # 6-15 branches — each if is top-level so indent is fine
        lines = ["def foo(x):"]
        for i in range(8):
            lines.append(f"    if x > {i}:")
            lines.append("        pass")
        code = {"test.py": "\n".join(lines) + "\n"}
        result = compute_code_metadata(code)
        assert result["complexity_estimate"] in ("medium", "high")

    def test_complexity_high(self):
        # 16+ branches
        lines = ["def foo(x):"]
        for i in range(20):
            lines.append(f"    if x > {i}:")
            lines.append("        pass")
        code = {"test.py": "\n".join(lines) + "\n"}
        result = compute_code_metadata(code)
        assert result["complexity_estimate"] == "high"

    def test_issue_categories_passed_through(self):
        code = {"test.py": "x = 1\n"}
        result = compute_code_metadata(code, issue_categories=["bug", "security", "bug"])
        # Should deduplicate
        cats = result["issue_categories"]
        assert "bug" in cats
        assert "security" in cats

    def test_syntax_error_no_crash(self):
        """Non-parseable code should not raise."""
        code = {"bad.py": "this is not valid python !!!\n   def broken("}
        result = compute_code_metadata(code)
        assert "total_lines" in result
        assert result["total_lines"] >= 1

    def test_multi_file(self):
        code = {
            "a.py": "def foo():\n    pass\n",
            "b.py": "def bar():\n    pass\n",
        }
        result = compute_code_metadata(code)
        assert result["num_functions"] == 2
        assert result["total_lines"] == 4

    def test_utils_task_metadata(self):
        from tasks.data import ALL_TASKS
        task = ALL_TASKS["bug-detection"]
        result = compute_code_metadata(task["code_files"])
        assert result["total_lines"] > 0
        assert result["num_functions"] >= 4  # utils.py has 4 functions


# ---------------------------------------------------------------------------
# grade_episode_detailed()
# ---------------------------------------------------------------------------

class TestGradeEpisodeDetailed:
    def test_returns_dict(self):
        gt = [_issue(6, "utils.py", "bug", "high")]
        result = grade_episode_detailed(gt, gt)
        assert isinstance(result, dict)

    def test_required_keys(self):
        gt = [_issue(6, "utils.py", "bug", "high")]
        result = grade_episode_detailed(gt, gt)
        for key in ("score", "f1", "precision", "recall", "severity_accuracy",
                    "true_positives", "false_positives", "false_negatives",
                    "near_misses", "per_file"):
            assert key in result, f"Missing key: {key}"

    def test_perfect_match(self):
        gt = [_issue(6, "utils.py", "bug", "high")]
        result = grade_episode_detailed(gt, gt)
        assert result["true_positives"] == 1
        assert result["false_positives"] == 0
        assert result["false_negatives"] == 0

    def test_false_positive_counted(self):
        gt = [_issue(6, "utils.py", "bug", "high")]
        flagged = [_issue(6, "utils.py", "bug", "high"),
                   _issue(100, "utils.py", "bug", "low")]
        result = grade_episode_detailed(flagged, gt)
        assert result["false_positives"] >= 1

    def test_near_miss_counted(self):
        gt = [_issue(6, "utils.py", "bug", "high")]
        # 4 lines away = near miss
        flagged = [_issue(10, "utils.py", "bug", "high")]
        result = grade_episode_detailed(flagged, gt)
        assert result["near_misses"] >= 1

    def test_per_file_breakdown(self):
        gt = [
            _issue(6, "utils.py", "bug", "high"),
            _issue(10, "other.py", "security", "critical"),
        ]
        flagged = [_issue(6, "utils.py", "bug", "high")]
        result = grade_episode_detailed(flagged, gt)
        assert "utils.py" in result["per_file"]

    def test_score_matches_grade_episode(self):
        """Detailed score should match grade_episode for simple cases."""
        gt = [
            _issue(6, "utils.py", "bug", "high"),
            _issue(13, "utils.py", "bug", "medium"),
        ]
        flagged = [_issue(6, "utils.py", "bug", "high")]
        simple_score = grade_episode(flagged, gt)
        detailed = grade_episode_detailed(flagged, gt)
        # Scores may differ slightly (near_miss handling), but should be close
        assert abs(detailed["score"] - simple_score) <= 0.15

    def test_empty_ground_truth_perfect(self):
        result = grade_episode_detailed([], [])
        assert result["score"] == 1.0

    def test_empty_flagged_zero(self):
        gt = [_issue(6, "utils.py")]
        result = grade_episode_detailed([], gt)
        assert result["score"] == 0.0
        assert result["false_negatives"] == 1


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

    def test_baseline_finds_md5_in_pipeline(self):
        """Keyword baseline should find the MD5 issue in data-pipeline."""
        from tasks.data import ALL_TASKS
        task = ALL_TASKS["data-pipeline"]
        findings = run_keyword_baseline(task)
        md5_finds = [f for f in findings if "md5" in f.description.lower() or "MD5" in f.description]
        assert len(md5_finds) >= 1

    def test_baseline_finds_sql_injection_in_pipeline(self):
        """Keyword baseline should find SQL injection via f-string in pipeline.py."""
        from tasks.data import ALL_TASKS
        task = ALL_TASKS["data-pipeline"]
        findings = run_keyword_baseline(task)
        sql_finds = [f for f in findings if f.issue_type == "security"
                     and "sql" in f.description.lower()]
        assert len(sql_finds) >= 1


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

    def test_async_review_has_6_issues(self):
        task = ALL_TASKS["async-review"]
        assert len(task["ground_truth_issues"]) == 6

    def test_data_pipeline_has_7_issues(self):
        task = ALL_TASKS["data-pipeline"]
        assert len(task["ground_truth_issues"]) == 7

    def test_async_review_issues_in_async_py(self):
        task = ALL_TASKS["async-review"]
        for issue in task["ground_truth_issues"]:
            assert issue["filename"] == "async.py"

    def test_data_pipeline_issues_in_pipeline_py(self):
        task = ALL_TASKS["data-pipeline"]
        for issue in task["ground_truth_issues"]:
            assert issue["filename"] == "pipeline.py"

    def test_data_pipeline_has_security_and_performance(self):
        task = ALL_TASKS["data-pipeline"]
        types = {i["issue_type"] for i in task["ground_truth_issues"]}
        assert "security" in types
        assert "performance" in types

    def test_async_review_has_bug_and_performance(self):
        task = ALL_TASKS["async-review"]
        types = {i["issue_type"] for i in task["ground_truth_issues"]}
        assert "bug" in types
        assert "performance" in types

    def test_all_tasks_count(self):
        assert len(ALL_TASKS) >= 6

    def test_async_review_line_numbers_are_valid(self):
        """GT issue line numbers should be within the code file."""
        from tasks.data import TASK_ASYNC_REVIEW
        code = TASK_ASYNC_REVIEW["code_files"]["async.py"]
        total_lines = len(code.splitlines())
        for issue in TASK_ASYNC_REVIEW["ground_truth_issues"]:
            assert 1 <= issue["line_number"] <= total_lines, (
                f"Line {issue['line_number']} out of range (file has {total_lines} lines)"
            )

    def test_pipeline_line_numbers_are_valid(self):
        """GT issue line numbers should be within the code file."""
        from tasks.data import TASK_DATA_PIPELINE
        code = TASK_DATA_PIPELINE["code_files"]["pipeline.py"]
        total_lines = len(code.splitlines())
        for issue in TASK_DATA_PIPELINE["ground_truth_issues"]:
            assert 1 <= issue["line_number"] <= total_lines, (
                f"Line {issue['line_number']} out of range (file has {total_lines} lines)"
            )

    def test_api_security_has_8_issues(self):
        from tasks.data import ALL_TASKS
        task = ALL_TASKS["api-security"]
        assert len(task["ground_truth_issues"]) == 8

    def test_api_security_line_numbers_are_valid(self):
        from tasks.data import ALL_TASKS
        task = ALL_TASKS["api-security"]
        code = task["code_files"]["api.py"]
        total_lines = len(code.splitlines())
        for issue in task["ground_truth_issues"]:
            assert 1 <= issue["line_number"] <= total_lines, (
                f"Line {issue['line_number']} out of range (file has {total_lines} lines)"
            )

    def test_api_security_has_security_issues(self):
        from tasks.data import ALL_TASKS
        task = ALL_TASKS["api-security"]
        types = {i["issue_type"] for i in task["ground_truth_issues"]}
        assert "security" in types


# ---------------------------------------------------------------------------
# compute_function_map and function_ranges in metadata
# ---------------------------------------------------------------------------

class TestFunctionRangesMetadata:
    def test_function_ranges_in_metadata(self):
        code = {"test.py": "def foo():\n    return 1\n\ndef bar(x):\n    return x\n"}
        result = compute_code_metadata(code)
        assert "function_ranges" in result
        assert len(result["function_ranges"]) == 2

    def test_function_ranges_have_correct_fields(self):
        code = {"test.py": "def foo():\n    return 1\n"}
        result = compute_code_metadata(code)
        fr = result["function_ranges"][0]
        assert fr["name"] == "foo"
        assert fr["file"] == "test.py"
        assert "start" in fr
        assert "end" in fr
        assert fr["start"] <= fr["end"]

    def test_function_ranges_empty_for_no_functions(self):
        code = {"test.py": "x = 1\ny = 2\n"}
        result = compute_code_metadata(code)
        assert result["function_ranges"] == []

    def test_function_ranges_multifile(self):
        code = {
            "a.py": "def foo():\n    pass\n",
            "b.py": "def bar():\n    pass\n\ndef baz():\n    pass\n",
        }
        result = compute_code_metadata(code)
        names = {fr["name"] for fr in result["function_ranges"]}
        assert names == {"foo", "bar", "baz"}

    def test_function_ranges_correct_line_numbers(self):
        code = {"test.py": "x = 1\n\ndef foo():\n    return 1\n"}
        result = compute_code_metadata(code)
        assert len(result["function_ranges"]) == 1
        assert result["function_ranges"][0]["start"] == 3  # line 3


# ---------------------------------------------------------------------------
# New keyword patterns
# ---------------------------------------------------------------------------

class TestNewKeywordPatterns:
    def test_baseline_finds_hardcoded_admin_token(self):
        from server.graders import run_keyword_baseline
        from tasks.data import ALL_TASKS
        task = ALL_TASKS["api-security"]
        findings = run_keyword_baseline(task)
        token_finds = [f for f in findings if "ADMIN_TOKEN" in f.description or "token" in f.description.lower()]
        assert len(token_finds) >= 1

    def test_baseline_finds_pickle_loads(self):
        from server.graders import run_keyword_baseline
        from tasks.data import ALL_TASKS
        task = ALL_TASKS["api-security"]
        findings = run_keyword_baseline(task)
        pickle_finds = [f for f in findings if "pickle" in f.description.lower()]
        assert len(pickle_finds) >= 1

    def test_baseline_finds_os_system(self):
        from server.graders import run_keyword_baseline
        from tasks.data import ALL_TASKS
        task = ALL_TASKS["api-security"]
        findings = run_keyword_baseline(task)
        sys_finds = [f for f in findings if "os.system" in f.description.lower() or "command" in f.description.lower()]
        assert len(sys_finds) >= 1

    def test_baseline_api_security_score_nonzero(self):
        from server.graders import run_keyword_baseline, grade_episode
        from models import Issue
        from tasks.data import ALL_TASKS
        task = ALL_TASKS["api-security"]
        findings = run_keyword_baseline(task)
        gt = [Issue.from_dict(i) for i in task["ground_truth_issues"]]
        score = grade_episode(findings, gt)
        assert score > 0.0, "Keyword baseline should find at least 1 issue in api-security"
