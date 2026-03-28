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

## Scoring

```
final_score = 0.70 × F1 + 0.30 × severity_accuracy

where:
  F1 = 2 × precision × recall / (precision + recall)
  precision = correct_flags / total_flags
  recall = correct_flags / total_gt_issues
  severity_accuracy = avg(1 − |flag_sev_rank − gt_sev_rank| × 0.34) for matched issues

Matching tolerance: ±2 lines, same filename, compatible issue type
```

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

| Task | Keyword heuristic | GPT-4o-mini |
|------|-------------------|-------------|
| bug-detection | 1.00 | ~0.52 |
| security-audit | 0.75 | ~0.59 |
| comprehensive-review | 0.67 | ~0.17 |
| **Overall** | **0.81** | **~0.43** |

Keyword heuristic runs via `inference.py` with no API key. LLM scores use `API_BASE_URL` + `HF_TOKEN`.

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
│   └── data.py           ← 3 task definitions + ground truth
├── server/
│   ├── app.py            ← FastAPI application
│   ├── environment.py    ← Core environment logic
│   └── graders.py        ← F1 grading + keyword baseline
└── tests/
    ├── test_environment.py
    └── test_graders.py
```
