"""
Grading logic for the Code Review Environment.
"""
from __future__ import annotations

import re
from typing import List, Tuple, Set

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Issue

_SEV_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}

_TYPE_COMPAT = {
    "bug": {"bug", "logic"},
    "logic": {"bug", "logic"},
    "security": {"security"},
    "performance": {"performance"},
}


def match_issue(flagged: Issue, gt: Issue, line_tolerance: int = 2) -> bool:
    if flagged.filename != gt.filename:
        return False
    if abs(flagged.line_number - gt.line_number) > line_tolerance:
        return False
    compat = _TYPE_COMPAT.get(gt.issue_type, {gt.issue_type})
    if flagged.issue_type not in compat:
        return False
    return True


def grade_episode(
    flagged: List[Issue],
    ground_truth: List[Issue],
    line_tolerance: int = 2,
) -> float:
    """Compute a 0.0–1.0 score: 0.70 * F1 + 0.30 * severity_accuracy."""
    if not ground_truth:
        return 1.0 if not flagged else 0.0

    tp = 0
    fp = 0
    matched_gt_indices: Set[int] = set()
    severity_scores: List[float] = []

    for flag in flagged:
        matched = False
        for i, gt in enumerate(ground_truth):
            if i in matched_gt_indices:
                continue
            if match_issue(flag, gt, line_tolerance):
                tp += 1
                matched_gt_indices.add(i)
                matched = True
                flag_rank = _SEV_RANK.get(flag.severity, 1)
                gt_rank = _SEV_RANK.get(gt.severity, 1)
                distance = abs(flag_rank - gt_rank)
                severity_scores.append(max(0.0, 1.0 - distance * 0.34))
                break
        if not matched:
            fp += 1

    fn = len(ground_truth) - len(matched_gt_indices)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    if severity_scores:
        severity_accuracy = sum(severity_scores) / len(ground_truth)
    else:
        severity_accuracy = 0.0

    final = 0.70 * f1 + 0.30 * severity_accuracy
    return round(min(1.0, max(0.0, final)), 4)


def compute_live_score(flagged: List[Issue], ground_truth: List[Issue]) -> float:
    """F1-only score for per-step feedback (no severity bonus)."""
    if not ground_truth:
        return 1.0 if not flagged else 0.0

    tp = 0
    fp = 0
    matched: Set[int] = set()

    for flag in flagged:
        hit = False
        for i, gt in enumerate(ground_truth):
            if i not in matched and match_issue(flag, gt):
                tp += 1
                matched.add(i)
                hit = True
                break
        if not hit:
            fp += 1

    fn = len(ground_truth) - len(matched)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return round(f1, 4)


_PATTERNS = [
    (r"range\(len\(\w+\)\s*\+\s*1\)", None, "bug", "high",
     "Off-by-one error: range(len(x) + 1) iterates one past the end"),
    (r"left,\s*right\s*=\s*0,\s*len\(", None, "bug", "medium",
     "Binary search upper bound should be len(arr) - 1"),
    (r"counts\[word\]\s*=\s*0\b", None, "bug", "low",
     "Counter initialized to 0 instead of 1"),

    (r'SECRET_KEY\s*=\s*["\']', None, "security", "high",
     "Hardcoded SECRET_KEY in source code"),
    (r'PASSWORD\s*=\s*["\']', None, "security", "high",
     "Hardcoded password in source code"),
    (r"f['\"].*SELECT.*\{", None, "security", "critical",
     "SQL injection via f-string query construction"),
    (r"f['\"].*DELETE.*\{", None, "security", "critical",
     "SQL injection via f-string DELETE query"),
    (r"render_template_string\(f['\"]", None, "security", "high",
     "XSS: unsanitized user input in render_template_string"),
    (r"shell\s*=\s*True", None, "security", "critical",
     "Command injection risk: shell=True with user input"),
    (r"hashlib\.md5\(", None, "security", "medium",
     "MD5 is cryptographically broken, use SHA-256 or HMAC-SHA256"),
    (r"expected\s*==\s*\w+_hash", None, "security", "medium",
     "Timing attack: use hmac.compare_digest() for constant-time comparison"),
    (r"password\s*=\s*models\.CharField", None, "security", "critical",
     "Plaintext password storage in database"),
    (r"os\.path\.join\(['\"]\/", None, "security", "high",
     "Path traversal: os.path.join with absolute prefix doesn't prevent traversal"),

    (r"\.objects\.get\(id=item\.", None, "performance", "high",
     "N+1 query: database lookup inside a loop"),

    (r"FloatField\(\)", None, "bug", "medium",
     "FloatField for monetary values causes precision errors, use DecimalField"),
    (r"BinaryField\(\)", None, "security", "high",
     "BinaryField with pickled data is a deserialization vulnerability"),
]


def run_keyword_baseline(task: dict) -> List[Issue]:
    findings: List[Issue] = []
    seen_lines: set = set()

    for filename, code in task.get("code_files", {}).items():
        lines = code.splitlines()
        for line_idx, line in enumerate(lines, start=1):
            for pattern, fname_hint, itype, severity, desc in _PATTERNS:
                # Optional filename filter
                if fname_hint and fname_hint not in filename:
                    continue
                if re.search(pattern, line):
                    key = (filename, line_idx)
                    if key not in seen_lines:
                        seen_lines.add(key)
                        findings.append(Issue(
                            line_number=line_idx,
                            filename=filename,
                            issue_type=itype,
                            severity=severity,
                            description=desc,
                        ))
    return findings
