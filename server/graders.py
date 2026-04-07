"""
Grading logic for the Code Review Environment.

Reward design is grounded in:
- Potential-Based Reward Shaping (PBRS): Ng et al. 1999
  R_shaped(s,a,s') = R(s,a,s') + γ·Φ(s') - Φ(s)
  where Φ(s) = (tp_found / total_gt) · POTENTIAL_SCALE
- Graduated line-proximity rewards: exponential decay over line distance
  reward = BASE_TP · exp(-DECAY · max(0, line_diff - EXACT_TOLERANCE))
  for 0 < line_diff ≤ NEAR_TOLERANCE
- F1-based terminal scoring: 0.70·F1 + 0.30·severity_accuracy
"""
from __future__ import annotations

import ast
import math
import re
from typing import List, Tuple, Set, Dict, Optional

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

# Tolerances
NEAR_TOLERANCE = 5
EXACT_TOLERANCE = 2

# Graduated reward constants (PBRS + smooth near-miss)
BASE_TP_REWARD = 0.10
NEAR_DECAY = 0.6       # exponential decay per line beyond EXACT_TOLERANCE
POTENTIAL_SCALE = 0.5  # Φ(s) = (tp/total_gt) * POTENTIAL_SCALE


def match_issue(flagged: Issue, gt: Issue, line_tolerance: int = EXACT_TOLERANCE, near_tolerance: int = NEAR_TOLERANCE) -> bool:
    """Return True if flagged matches gt within line_tolerance lines and same type."""
    if flagged.filename != gt.filename:
        return False
    if abs(flagged.line_number - gt.line_number) > line_tolerance:
        return False
    compat = _TYPE_COMPAT.get(gt.issue_type, {gt.issue_type})
    if flagged.issue_type not in compat:
        return False
    return True


def match_quality(flagged: Issue, gt: Issue) -> str:
    """
    Return quality of match between flagged and gt:
      "exact"  — within ±2 lines and right issue type
      "near"   — within ±3-5 lines and same file (regardless of type)
      "none"   — no meaningful match
    """
    if flagged.filename != gt.filename:
        return "none"

    line_diff = abs(flagged.line_number - gt.line_number)

    if line_diff <= EXACT_TOLERANCE:
        compat = _TYPE_COMPAT.get(gt.issue_type, {gt.issue_type})
        if flagged.issue_type in compat:
            return "exact"

    if line_diff <= NEAR_TOLERANCE:
        return "near"

    return "none"


def graduated_near_reward(line_diff: int) -> float:
    """
    Graduated reward for near-miss flags using exponential decay.

    Implements continuous reward shaping based on proximity:
      line_diff = 0-2  → 0.10 (full TP, handled separately)
      line_diff = 3    → 0.10 * exp(-0.6*1) ≈ 0.055
      line_diff = 4    → 0.10 * exp(-0.6*2) ≈ 0.033
      line_diff = 5    → 0.10 * exp(-0.6*3) ≈ 0.020

    This gives smooth gradient signal rather than a hard 0.03 step function,
    encouraging the agent to refine line numbers progressively.
    """
    if line_diff <= EXACT_TOLERANCE:
        return BASE_TP_REWARD
    extra = line_diff - EXACT_TOLERANCE
    return round(BASE_TP_REWARD * math.exp(-NEAR_DECAY * extra), 4)


def compute_potential(tp_count: int, total_gt: int) -> float:
    """
    Potential function Φ(s) for Potential-Based Reward Shaping (PBRS).

    Φ(s) = (tp_found / total_gt) * POTENTIAL_SCALE

    The shaped reward R_shaped = r + Φ(s') - Φ(s) ensures policy invariance
    (Ng et al. 1999): the optimal policy under shaped rewards is the same as
    under the original rewards, but with better intermediate gradient signal.

    Here we compute just Φ(s); the caller computes Φ(s') - Φ(s).
    """
    if total_gt <= 0:
        return 0.0
    return (tp_count / total_gt) * POTENTIAL_SCALE


def compute_function_map(code: str) -> Dict[int, str]:
    """
    Map each line number to the name of its enclosing function (or class method).
    Lines outside any function map to "module". Non-parseable code returns empty dict.
    """
    result: Dict[int, str] = {}
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                end = getattr(node, "end_lineno", node.lineno)
                for lineno in range(node.lineno, end + 1):
                    result[lineno] = node.name
    except SyntaxError:
        pass
    return result


def compute_code_metadata(code_files: Dict[str, str], issue_categories: Optional[List[str]] = None) -> Dict:
    """
    Extract code structure metadata using Python's ast module.

    Returns:
        total_lines, num_functions, function_names, num_classes, class_names,
        imports, complexity_estimate, issue_categories, function_ranges
    """
    total_lines = 0
    num_functions = 0
    function_names: List[str] = []
    num_classes = 0
    class_names: List[str] = []
    imports: List[str] = []
    branch_count = 0
    function_ranges: List[Dict] = []  # [{name, file, start, end}]

    for filename, code in code_files.items():
        lines = code.splitlines()
        total_lines += len(lines)
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    num_functions += 1
                    function_names.append(node.name)
                    end = getattr(node, "end_lineno", node.lineno)
                    function_ranges.append({
                        "name": node.name,
                        "file": filename,
                        "start": node.lineno,
                        "end": end,
                    })
                elif isinstance(node, ast.ClassDef):
                    num_classes += 1
                    class_names.append(node.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module.split(".")[0])
                elif isinstance(node, (ast.If, ast.For, ast.While, ast.Try,
                                        ast.ExceptHandler, ast.With)):
                    branch_count += 1
        except SyntaxError:
            # If ast can't parse (e.g. non-Python file), just count lines
            pass

    # Deduplicate imports
    imports = list(dict.fromkeys(imports))

    # Complexity estimate
    if branch_count <= 5:
        complexity_estimate = "low"
    elif branch_count <= 15:
        complexity_estimate = "medium"
    else:
        complexity_estimate = "high"

    return {
        "total_lines": total_lines,
        "num_functions": num_functions,
        "function_names": function_names,
        "num_classes": num_classes,
        "class_names": class_names,
        "imports": imports,
        "complexity_estimate": complexity_estimate,
        "issue_categories": list(set(issue_categories)) if issue_categories else [],
        "function_ranges": function_ranges,
    }


def compute_code_state_features(
    code_metadata: Dict,
    progress: Optional[Dict] = None,
) -> List[float]:
    """
    Compute a normalized 12-dimensional feature vector for RL training.

    Based on state representation research (code2vec, GraphCodeBERT, 2023-2024),
    combining AST-derived structural features with episode progress metrics.
    This vector is suitable as input to a policy network or value estimator.

    Dimensions:
      0: total_lines / 200          — code size (normalized)
      1: num_functions / 20         — function count
      2: num_classes / 10           — class count
      3: complexity_score           — 0=low, 0.5=medium, 1.0=high
      4: has_bug_issues             — 1 if "bug" in issue_categories
      5: has_security_issues        — 1 if "security" in issue_categories
      6: has_performance_issues     — 1 if "performance" in issue_categories
      7: has_logic_issues           — 1 if "logic" in issue_categories
      8: progress_recall            — tp / total_gt (0 if no progress yet)
      9: progress_precision         — precision so far
     10: steps_used_frac            — steps_used / max_steps
     11: fp_pressure               — false_positives / max(total_flagged, 1)
    """
    if progress is None:
        progress = {}

    complexity_map = {"low": 0.0, "medium": 0.5, "high": 1.0}
    cats = set(code_metadata.get("issue_categories", []))

    total_gt = float(progress.get("total_ground_truth", 1.0)) or 1.0
    tp = float(progress.get("true_positives", 0.0))
    fp = float(progress.get("false_positives", 0.0))
    total_flagged = tp + fp
    steps_used = float(progress.get("steps_used", 0.0))
    steps_rem = float(progress.get("steps_remaining", 1.0))
    max_steps = steps_used + steps_rem or 1.0

    features = [
        min(1.0, code_metadata.get("total_lines", 0) / 200.0),
        min(1.0, code_metadata.get("num_functions", 0) / 20.0),
        min(1.0, code_metadata.get("num_classes", 0) / 10.0),
        complexity_map.get(code_metadata.get("complexity_estimate", "low"), 0.0),
        1.0 if "bug" in cats else 0.0,
        1.0 if "security" in cats else 0.0,
        1.0 if "performance" in cats else 0.0,
        1.0 if "logic" in cats else 0.0,
        min(1.0, tp / total_gt),
        min(1.0, tp / total_flagged) if total_flagged > 0 else 0.0,
        min(1.0, steps_used / max_steps),
        min(1.0, fp / total_flagged) if total_flagged > 0 else 0.0,
    ]
    return [round(f, 4) for f in features]


class RewardNormalizer:
    """
    Variable-Length Return Normalizer for multi-task RL training.

    Based on VL Norm (2025) and Return-based Scaling (2021):
    Normalizes episode returns accounting for variable episode lengths,
    preventing long episodes from dominating gradient computation.

    Usage:
        normalizer = RewardNormalizer(window_size=100)
        # After each episode:
        normalizer.update(episode_return, episode_length)
        normalized_r = normalizer.normalize(episode_return, episode_length)
    """

    def __init__(self, window_size: int = 100, eps: float = 1e-8) -> None:
        self.window_size = window_size
        self.eps = eps
        self._returns: List[float] = []
        self._lengths: List[int] = []
        self.mean: float = 0.0
        self.std: float = 1.0

    def update(self, episode_return: float, episode_length: int) -> None:
        """Record a completed episode for running statistics."""
        self._returns.append(episode_return)
        self._lengths.append(max(1, episode_length))
        if len(self._returns) > self.window_size:
            self._returns.pop(0)
            self._lengths.pop(0)
        self._recompute()

    def _recompute(self) -> None:
        if len(self._returns) < 2:
            return
        returns = [r for r in self._returns]
        lengths = [l for l in self._lengths]
        mean_len = sum(lengths) / len(lengths)
        # Length-adjusted std: longer episodes have proportionally less weight
        self.mean = sum(returns) / len(returns)
        raw_std = (sum((r - self.mean) ** 2 for r in returns) / len(returns)) ** 0.5
        length_factors = [(l / mean_len) ** 0.5 for l in lengths]
        avg_lf = sum(length_factors) / len(length_factors)
        self.std = max(self.eps, raw_std * avg_lf)

    def normalize(self, episode_return: float, episode_length: int) -> float:
        """Return the length-adjusted normalized return."""
        if len(self._returns) < 2:
            return episode_return
        mean_len = sum(self._lengths) / len(self._lengths)
        length_factor = (max(1, episode_length) / mean_len) ** 0.5
        return round((episode_return - self.mean) / (self.std * length_factor + self.eps), 4)

    def to_dict(self) -> Dict:
        return {
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "n_episodes": len(self._returns),
            "window_size": self.window_size,
        }


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


def grade_episode_detailed(
    flagged: List[Issue],
    ground_truth: List[Issue],
    line_tolerance: int = 2,
) -> Dict:
    """
    Full breakdown of grading results.

    Returns:
        score, f1, precision, recall, severity_accuracy,
        true_positives, false_positives, false_negatives,
        near_misses, per_file
    """
    if not ground_truth:
        score = 1.0 if not flagged else 0.0
        return {
            "score": score,
            "f1": score,
            "precision": score,
            "recall": score,
            "severity_accuracy": score,
            "true_positives": 0,
            "false_positives": len(flagged),
            "false_negatives": 0,
            "near_misses": 0,
            "per_file": {},
        }

    tp = 0
    fp = 0
    near_misses = 0
    matched_gt_indices: Set[int] = set()
    severity_scores: List[float] = []
    per_file: Dict[str, Dict] = {}

    for flag in flagged:
        fname = flag.filename
        if fname not in per_file:
            per_file[fname] = {"tp": 0, "fp": 0, "near_miss": 0}

        matched = False
        for i, gt in enumerate(ground_truth):
            if i in matched_gt_indices:
                continue
            if match_issue(flag, gt, line_tolerance):
                tp += 1
                matched_gt_indices.add(i)
                matched = True
                per_file[fname]["tp"] += 1
                flag_rank = _SEV_RANK.get(flag.severity, 1)
                gt_rank = _SEV_RANK.get(gt.severity, 1)
                distance = abs(flag_rank - gt_rank)
                severity_scores.append(max(0.0, 1.0 - distance * 0.34))
                break

        if not matched:
            # Check for near miss (3-5 lines off, same file)
            is_near = False
            for i, gt in enumerate(ground_truth):
                if i in matched_gt_indices:
                    continue
                q = match_quality(flag, gt)
                if q == "near":
                    is_near = True
                    break
            if is_near:
                near_misses += 1
                per_file[fname]["near_miss"] += 1
            else:
                fp += 1
                per_file[fname]["fp"] += 1

    fn = len(ground_truth) - len(matched_gt_indices)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    if severity_scores:
        severity_accuracy = sum(severity_scores) / len(ground_truth)
    else:
        severity_accuracy = 0.0

    score = round(min(1.0, max(0.0, 0.70 * f1 + 0.30 * severity_accuracy)), 4)

    return {
        "score": score,
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "severity_accuracy": round(severity_accuracy, 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "near_misses": near_misses,
        "per_file": per_file,
    }


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
    # --- Bug patterns ---
    (r"range\(len\(\w+\)\s*\+\s*1\)", None, "bug", "high",
     "Off-by-one error: range(len(x) + 1) iterates one past the end"),
    (r"left,\s*right\s*=\s*0,\s*len\(", None, "bug", "medium",
     "Binary search upper bound should be len(arr) - 1"),
    (r"counts\[word\]\s*=\s*0\b", None, "bug", "low",
     "Counter initialized to 0 instead of 1"),

    # --- Hardcoded secrets ---
    (r'SECRET_KEY\s*=\s*["\']', None, "security", "high",
     "Hardcoded SECRET_KEY in source code"),
    (r'ADMIN_TOKEN\s*=\s*["\']', None, "security", "high",
     "Hardcoded ADMIN_TOKEN in source code"),
    (r'PASSWORD\s*=\s*["\']', None, "security", "high",
     "Hardcoded password in source code"),

    # --- Injection attacks ---
    (r"f['\"].*SELECT.*\{", None, "security", "critical",
     "SQL injection via f-string query construction"),
    (r"f['\"].*INSERT.*\{", None, "security", "critical",
     "SQL injection via f-string INSERT query"),
    (r"f['\"].*DELETE.*\{", None, "security", "critical",
     "SQL injection via f-string DELETE query"),
    (r"f['\"].*LIKE.*%\{", None, "security", "critical",
     "SQL injection via f-string LIKE clause"),
    (r"LIMIT\s*\{", None, "security", "critical",
     "SQL injection: LIMIT clause uses unparameterized variable"),
    (r"render_template_string\(f['\"]", None, "security", "high",
     "XSS: unsanitized user input in render_template_string"),
    (r"shell\s*=\s*True", None, "security", "critical",
     "Command injection risk: shell=True with user input"),
    (r"os\.system\(", None, "security", "critical",
     "Command injection risk: os.system() executes shell commands"),
    (r"os\.path\.join\(['\"]\/", None, "security", "high",
     "Path traversal: os.path.join with absolute prefix doesn't prevent traversal"),

    # --- Broken cryptography ---
    (r"hashlib\.md5\(", None, "security", "high",
     "MD5 is cryptographically broken for security use; use SHA-256 or bcrypt"),
    (r"hashlib\.sha1\(", None, "security", "medium",
     "SHA-1 is deprecated for security use; use SHA-256 or better"),
    (r"expected\s*==\s*\w+_hash", None, "security", "medium",
     "Timing attack: use hmac.compare_digest() for constant-time comparison"),

    # --- Dangerous deserialization ---
    (r"pickle\.loads\(", None, "security", "critical",
     "Unsafe deserialization: pickle.loads() on untrusted data allows remote code execution"),
    (r"yaml\.load\(", None, "security", "high",
     "Unsafe YAML deserialization: use yaml.safe_load() instead"),

    # --- Auth / access control ---
    (r"password\s*=\s*models\.CharField", None, "security", "critical",
     "Plaintext password storage in database"),

    # --- Async / concurrency bugs ---
    (r"aiohttp\.ClientSession\(\)", None, "bug", "high",
     "ClientSession created outside 'async with' — may not be closed (resource leak)"),
    (r"timeout\s*=\s*\d+\b", None, "bug", "medium",
     "aiohttp timeout should be aiohttp.ClientTimeout(total=N), not a bare integer"),
    (r"attempt\s*==\s*retries\b", None, "bug", "high",
     "Off-by-one: range(retries) yields 0..retries-1, so attempt==retries is never true"),
    (r"for\s+\w+\s+in\s+\w+_ids\s*:", None, "performance", "high",
     "Sequential loop over IDs — consider asyncio.gather() for concurrent fetching"),

    # --- Performance ---
    (r"\.objects\.get\(id=item\.", None, "performance", "high",
     "N+1 query: database lookup inside a loop"),

    # --- JavaScript-specific patterns ---
    (r"new\s+Function\(", None, "security", "critical",
     "Unsafe dynamic code execution: new Function() with user input is equivalent to eval()"),
    (r"\beval\(", None, "security", "critical",
     "eval() with user-supplied input allows arbitrary code execution"),
    (r"execSync\(", None, "security", "critical",
     "Command injection risk: execSync() with user-supplied data"),
    (r"jwt\.sign\(.*\{(?!.*expiresIn)", None, "security", "medium",
     "JWT issued without expiry (expiresIn) — tokens are valid forever"),
    (r"JWT_SECRET\s*=\s*['\"]", None, "security", "high",
     "Hardcoded JWT secret in source code"),
    (r"res\.send\(`.*\$\{", None, "security", "high",
     "XSS: template literal with user input sent directly in response"),

    # --- Data model bugs ---
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
