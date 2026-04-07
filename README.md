---
title: Code Review Environment
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
tags:
  - openenv
  - code-review
  - security-audit
  - reinforcement-learning
---

# Code Review Environment

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compatible environment for training and evaluating AI agents on code review and security auditing tasks.

The agent inspects code files, flags bugs and vulnerabilities with precise line numbers and severity ratings, and receives graded feedback — enabling reinforcement learning from human-quality code review signal.

## Why This Environment

Code review is one of the highest-value tasks in software engineering. Every professional software team does it daily. Training AI agents to perform thorough, accurate code reviews is commercially valuable and technically challenging:

- **Precise reasoning required**: agent must count lines, understand language semantics, reason about control flow
- **Real impact**: bugs found → prevented production incidents; vulnerabilities found → prevented security breaches
- **Natural difficulty progression**: obvious logic errors → subtle security vulnerabilities → complex architectural issues
- **Clear grading**: issues exist at specific lines with specific types — objective F1-based scoring

## Action Space

```json
{
  "action_type": "flag_issue | clear_flag | request_hint | submit_review",
  "line_number": 6,
  "filename": "utils.py",
  "issue_type": "bug | security | performance | logic",
  "severity": "low | medium | high | critical",
  "description": "Description of the issue",
  "fix_suggestion": "How to fix it (optional)"
}
```

| Action | Description | Reward |
|--------|-------------|--------|
| `flag_issue` | Mark a line as containing an issue | +0.10 if correct, −0.05 if wrong |
| `clear_flag` | Remove a previously flagged issue | +0.03 if was FP, −0.03 if was TP |
| `request_hint` | Get a hint about what to look for | −0.01 |
| `submit_review` | Finalize and receive graded score | Final F1 score |

## Observation Space

```json
{
  "task_id": "bug-detection",
  "task_description": "Review this Python utility module...",
  "code_files": {"utils.py": "def calculate_average(numbers):\n..."},
  "language": "python",
  "flagged_issues": [...],
  "step_count": 3,
  "max_steps": 15,
  "hints_remaining": 2,
  "feedback": "Good catch! Issue flagged at utils.py:6 [+0.10 reward]",
  "current_score": 0.333,
  "done": false,
  "reward": 0.1
}
```

Note: `code_files` is only populated in the first observation (after `reset()`). Subsequent step observations omit it to keep payloads small.

## Tasks

### Task 1: `bug-detection` — Easy

Identify 3 logical bugs in a Python utility module (`utils.py`).

| Line | Issue | Severity |
|------|-------|----------|
| 6 | Off-by-one error: `range(len(numbers) + 1)` causes `IndexError` | High |
| 13 | Binary search upper bound: `len(arr)` should be `len(arr) - 1` | Medium |
| 33 | Word count initializes new entries to `0` instead of `1` | Low |

**Max steps:** 15

### Task 2: `security-audit` — Medium

Audit a Flask web application (`app.py`) for OWASP Top-10 vulnerabilities.

| Line | Issue | Severity |
|------|-------|----------|
| 8 | Hardcoded `SECRET_KEY` in source | High |
| 9 | Hardcoded `DB_PASSWORD` in source | High |
| 19 | SQL injection via f-string query | Critical |
| 27 | XSS via unsanitized `render_template_string` | High |
| 34 | Path traversal via `os.path.join` | High |
| 40 | Missing authentication on admin endpoint | Critical |
| 51 | Command injection via `shell=True` | Critical |

**Max steps:** 20

### Task 3: `comprehensive-review` — Hard

Comprehensive review of a Django e-commerce API across two files (`views.py`, `models.py`).

| File | Line | Issue | Severity |
|------|------|-------|----------|
| views.py | 21 | N+1 query in order creation loop | High |
| views.py | 26 | Race condition — stock check not atomic | Critical |
| views.py | 29 | Order created outside transaction | High |
| views.py | 47 | No max cap on `per_page` parameter | Medium |
| views.py | 66 | MD5 for payment verification (broken crypto) | Medium |
| views.py | 67 | Timing attack in payment hash comparison | Medium |
| models.py | 8 | Plaintext password storage | Critical |
| models.py | 16 | `FloatField` for monetary values | Medium |
| models.py | 18 | `BinaryField` with pickled data (RCE risk) | High |

**Max steps:** 30

### Task 4: `async-review` — Medium-Hard

Review an async Python module (`async.py`) for concurrency bugs, resource leaks, and performance issues with `asyncio` and `aiohttp`.

| Line | Issue | Severity |
|------|-------|----------|
| 5 | Shared mutable cache dict without `asyncio.Lock` — race condition | High |
| 9 | `timeout=5` wrong type for aiohttp; requires `ClientTimeout(total=5)` | Medium |
| 22 | `ClientSession` created but never closed — resource leak | High |
| 24 | Sequential `await` in loop — use `asyncio.gather()` for concurrency | High |
| 37 | Off-by-one in retry condition: `attempt == retries` never true | High |
| 48 | Tasks awaited sequentially; `self.results` accumulates across calls | Medium |

**Max steps:** 20

### Task 5: `data-pipeline` — Hard

Security and correctness audit of a SQLite data pipeline module (`pipeline.py`).

| Line | Issue | Severity |
|------|-------|----------|
| 20 | MD5 for password hashing — cryptographically broken | High |
| 27 | SQL injection via f-string in `INSERT` query | Critical |
| 35 | SQL injection via f-string in `LIKE` query | Critical |
| 41 | One transaction per row in `bulk_load` — severe performance issue | High |
| 46 | `float()` conversion without error handling — crashes on bad input | Medium |
| 52 | `export_records` leaks `password_hash` field in JSON output | High |
| 59 | SQL injection: `limit` interpolated into `LIMIT` clause | Critical |

**Max steps:** 25

### Task 6: `api-security` — Hard

Security audit of a FastAPI REST API (`api.py`) with authentication, authorization, and injection vulnerabilities.

| Line | Issue | Severity |
|------|-------|----------|
| 12 | Hardcoded `SECRET_KEY` in source | High |
| 13 | Hardcoded `ADMIN_TOKEN` in source | High |
| 16 | MD5 for password hashing | High |
| 27 | JWT issued without `exp` expiry claim | Medium |
| 33 | IDOR — any user can fetch any other user's data | Critical |
| 38 | SQL injection via f-string in `SELECT` query | Critical |
| 47 | Command injection via `os.system()` with env-interpolated path | Critical |
| 53 | `pickle.loads()` on untrusted user bytes — RCE | Critical |

**Max steps:** 25

### Task 7: `js-security` — Hard

Security audit of an Express.js REST API (`server.js`) in JavaScript/Node.js.

| Line | Issue | Severity |
|------|-------|----------|
| 11 | Hardcoded `JWT_SECRET` in source | High |
| 16 | SQL injection via template literal in `prepare()` | Critical |
| 18 | JWT issued without `expiresIn` — tokens valid forever | Medium |
| 25 | IDOR + SQL injection: unauthenticated user access + unparameterized query | Critical |
| 31 | XSS: user query param reflected directly in HTML response | High |
| 36 | Command injection via `execSync()` with user-supplied filename | Critical |
| 42 | Path traversal: `path.join` with user-supplied filename | High |
| 48 | `new Function()` with user template — arbitrary code execution | Critical |

**Max steps:** 25

## Scoring

```
final_score = 0.70 × F1 + 0.30 × severity_accuracy

where:
  F1 = 2 × precision × recall / (precision + recall)
  precision = correct_flags / total_flags
  recall = correct_flags / total_gt_issues
  severity_accuracy = avg(1 − |flag_sev_rank − gt_sev_rank| × 0.34) for matched issues

Matching tolerance: ±2 lines, same filename, compatible issue type
Near-miss (±3-5 lines): graduated partial credit via exponential decay
```

## Reward Design

### Per-step rewards

| Event | Reward |
|-------|--------|
| True positive (TP) | +0.10 base |
| TP + severity exact match | +0.02 bonus |
| TP + early (first 40% of steps) | +0.02 bonus |
| TP + high confidence (≥0.7) | +0.01 bonus |
| PBRS potential shaping (Φ(s')−Φ(s)) | +0.03–0.08 |
| Diversity bonus (first TP in new issue category) | +0.02 |
| Exploration bonus (first TP in new file, multi-file tasks) | +0.01 |
| Near-miss (±3-5 lines, compatible type, exp decay) | +0.020–0.055 |
| False positive | −0.05 |
| False positive flood (4th+ FP) | escalating −0.03 extra |
| High-confidence FP | −0.03 extra |
| Clear TP | −0.03 |
| Clear FP | +0.03 |
| Hint | −0.01 |
| Submit / auto-end | Final F1 score |

### Reward shaping foundations

- **Potential-Based Reward Shaping** (Ng et al. 1999): Φ(s) = (tp/total_gt) × 0.5. Policy-invariant shaping that improves sample efficiency without changing the optimal policy.
- **Graduated near-miss** (exponential decay): reward = 0.10 × e^(−0.6 × (line_diff − 2)) for lines 3-5 off with compatible issue type. Gives smooth gradient signal for line-number refinement.
- **Diversity bonus**: +0.02 for first TP in a new issue category (security/bug/performance). Encourages covering all issue types instead of spamming one.
- **Exploration bonus**: +0.01 for first TP in a new file (multi-file tasks only). Encourages cross-file coverage.
- **Variable-Length Return Normalization** (VL Norm 2025): normalized_return = cumulative_reward / steps_used. Makes return comparable across tasks of different lengths.
- **Flood protection**: escalating FP penalty prevents reward hacking via flag-spamming.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/reset` | Start new episode. Body: `{"task_id": "bug-detection", "seed": 42}` |
| `POST` | `/step` | Take action. Body: ReviewAction JSON |
| `GET` | `/state` | Get current episode state |
| `GET` | `/health` | Health check → `{"status": "healthy"}` |
| `GET` | `/tasks` | List all tasks + action schema |
| `POST` | `/grader` | Grade findings: `{"task_id": "...", "flagged_issues": [...]}` |
| `POST` | `/baseline` | Run keyword heuristic on all tasks |
| `WS` | `/ws` | WebSocket session (OpenEnv standard) |
| `GET` | `/docs` | Swagger UI |

## Setup & Usage

### Local (uvicorn)

```bash
git clone https://github.com/CodeMaverick2/code-review-env
cd code-review-env
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t code-review-env .
docker run -p 7860:7860 code-review-env
```

### Quick test

```bash
curl http://localhost:7860/health

curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "bug-detection"}'

curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "flag_issue", "line_number": 6, "filename": "utils.py", "issue_type": "bug", "severity": "high", "description": "Off-by-one"}'

curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "submit_review"}'
```

### Python client

```python
from client import CodeReviewEnv, ReviewAction

with CodeReviewEnv("http://localhost:7860").sync() as env:
    result = env.reset(task_id="bug-detection")
    print(result.observation.code_files["utils.py"])

    result = env.step(ReviewAction(
        action_type="flag_issue",
        line_number=6,
        filename="utils.py",
        issue_type="bug",
        severity="high",
        description="Off-by-one error in range()"
    ))
    print(result.observation.feedback)

    result = env.step(ReviewAction(action_type="submit_review"))
    print(f"Final score: {result.reward:.3f}")
```

### Inference script

```bash
# No API key needed — uses built-in keyword heuristic
python inference.py

# With LLM (OpenAI-compatible API)
export API_BASE_URL=https://openrouter.ai/api/v1
export MODEL_NAME=openai/gpt-4o-mini
export HF_TOKEN=sk-...
python inference.py
```

### Demo

```bash
python demo.py
python demo.py --task security-audit
python demo.py --task comprehensive-review
```

### Tests

```bash
pip install pytest
pytest tests/ -v
```

## Baseline Scores

| Task | Keyword heuristic |
|------|-------------------|
| bug-detection | 1.00 |
| security-audit | 0.75 |
| async-review | 0.71 |
| comprehensive-review | 0.66 |
| api-security | 0.83 |
| js-security | 0.70 |
| data-pipeline | 0.55 |
| **Overall (7 tasks)** | **0.74** |

Keyword heuristic runs via `inference.py` with no API key (uses `/baseline` endpoint). LLM scores use `API_BASE_URL` + `HF_TOKEN`.

## Project Structure

```
code-review-env/
├── README.md
├── openenv.yaml          ← OpenEnv manifest
├── Dockerfile            ← Container (HF Spaces, port 7860)
├── pyproject.toml        ← Package config + entry points
├── requirements.txt
├── uv.lock
├── inference.py          ← Inference script
├── demo.py               ← Demo script (no API key needed)
├── client.py             ← HTTP client
├── models.py             ← ReviewAction, ReviewObservation, ReviewState, Issue
├── tasks/
│   └── data.py           ← 5 task definitions + ground truth
│                            (bug-detection, security-audit, comprehensive-review,
│                             async-review, data-pipeline)
├── server/
│   ├── app.py            ← FastAPI application
│   ├── environment.py    ← Core environment logic (adaptive hints, rich rewards)
│   └── graders.py        ← F1 grading + detailed grading + keyword baseline
└── tests/
    ├── test_environment.py
    └── test_graders.py
```
