"""
Task definitions for the Code Review Environment.
"""
from __future__ import annotations
from typing import List, Dict, Any

def _issue(line: int, filename: str, itype: str, severity: str, desc: str, fix: str = "") -> dict:
    return {
        "line_number": line,
        "filename": filename,
        "issue_type": itype,
        "severity": severity,
        "description": desc,
        "fix_suggestion": fix,
    }


_UTILS_CODE = """\
def calculate_average(numbers):
    \"\"\"Calculate the average of a list of numbers.\"\"\"
    if not numbers:
        return 0
    total = 0
    for i in range(len(numbers) + 1):
        total += numbers[i]
    return total / len(numbers)


def binary_search(arr, target):
    \"\"\"Search for target in sorted array. Returns index or -1.\"\"\"
    left, right = 0, len(arr)
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1


def count_words(text):
    \"\"\"Count word frequency in a text string.\"\"\"
    words = text.lower().split()
    counts = {}
    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 0
    return counts


def reverse_string(s):
    \"\"\"Return the reversed version of a string (no bug here).\"\"\"
    return s[::-1]
"""

TASK_BUG_DETECTION: Dict[str, Any] = {
    "task_id": "bug-detection",
    "difficulty": "easy",
    "description": (
        "Review this Python utility module for logical bugs and errors.\n"
        "The code contains several functions with subtle bugs that would cause\n"
        "incorrect results or crashes. Identify all issues with exact line numbers,\n"
        "issue type, severity, and a clear description of the problem.\n\n"
        "File to review: utils.py"
    ),
    "language": "python",
    "code_files": {
        "utils.py": _UTILS_CODE,
    },
    "ground_truth_issues": [
        _issue(
            6, "utils.py", "bug", "high",
            "Off-by-one error: range(len(numbers) + 1) iterates one past the end, "
            "causing IndexError on the last iteration.",
            "Change to: range(len(numbers))"
        ),
        _issue(
            13, "utils.py", "bug", "medium",
            "Binary search upper bound is wrong: right = len(arr) causes IndexError "
            "when accessing arr[mid] on a full array.",
            "Change to: right = len(arr) - 1"
        ),
        _issue(
            33, "utils.py", "bug", "low",
            "Word count initializes new entries to 0 instead of 1, so every word's "
            "count is underreported by 1.",
            "Change to: counts[word] = 1"
        ),
    ],
    "max_steps": 15,
    "hints": [
        "Look carefully at loop boundary conditions — are they off by one?",
        "The binary_search function has an issue with its initial right bound.",
        "Check how new keys are initialized in the word count dictionary.",
    ],
}


_APP_CODE = """\
import sqlite3
import os
import subprocess
from flask import Flask, request, render_template_string

app = Flask(__name__)

SECRET_KEY = "hardcoded_secret_key_123"
DB_PASSWORD = "admin123"


def get_db():
    return sqlite3.connect('users.db')


@app.route('/user/<username>')
def get_user(username):
    db = get_db()
    query = f"SELECT * FROM users WHERE username = '{username}'"
    result = db.execute(query).fetchone()
    return str(result)


@app.route('/search')
def search():
    term = request.args.get('term', '')
    template = f"<h1>Results for: {term}</h1>"
    return render_template_string(template)


@app.route('/file')
def read_file():
    filename = request.args.get('name', '')
    filepath = os.path.join('/data', filename)
    with open(filepath, 'r') as f:
        return f.read()


@app.route('/admin/delete', methods=['POST'])
def admin_delete():
    user_id = request.form.get('user_id')
    db = get_db()
    db.execute(f"DELETE FROM users WHERE id = {user_id}")
    db.commit()
    return "Deleted"


@app.route('/ping')
def ping():
    host = request.args.get('host', '')
    result = subprocess.run(f"ping -c 1 {host}", shell=True, capture_output=True)
    return result.stdout.decode()
"""

TASK_SECURITY_AUDIT: Dict[str, Any] = {
    "task_id": "security-audit",
    "difficulty": "medium",
    "description": (
        "Perform a security audit on this Flask web application.\n"
        "The code contains multiple OWASP Top-10 security vulnerabilities.\n"
        "Identify all security issues with their exact line numbers, severity ratings,\n"
        "and recommended fixes. Consider: injection attacks, broken authentication,\n"
        "sensitive data exposure, and improper input handling.\n\n"
        "File to review: app.py"
    ),
    "language": "python",
    "code_files": {
        "app.py": _APP_CODE,
    },
    "ground_truth_issues": [
        _issue(
            8, "app.py", "security", "high",
            "Hardcoded SECRET_KEY in source code. Anyone with repo access can forge sessions.",
            "Use: SECRET_KEY = os.environ.get('SECRET_KEY') and set it as an env var."
        ),
        _issue(
            9, "app.py", "security", "high",
            "Hardcoded database password in source code. Credentials should never be in code.",
            "Use: DB_PASSWORD = os.environ.get('DB_PASSWORD')"
        ),
        _issue(
            19, "app.py", "security", "critical",
            "SQL injection: username is interpolated directly into the query string. "
            "An attacker can supply username = \\' OR 1=1 -- to dump the database.",
            "Use parameterized queries: db.execute('SELECT * FROM users WHERE username = ?', (username,))"
        ),
        _issue(
            27, "app.py", "security", "high",
            "Cross-site scripting (XSS): user-supplied 'term' is rendered directly in an "
            "HTML template via render_template_string without escaping.",
            "Use flask.escape(term) or Markup.escape(term) before interpolating into HTML."
        ),
        _issue(
            34, "app.py", "security", "high",
            "Path traversal: os.path.join('/data', filename) does not prevent filenames "
            "like '../etc/passwd' from escaping the /data directory.",
            "Use: filename = os.path.basename(filename) and validate against an allowlist."
        ),
        _issue(
            40, "app.py", "security", "critical",
            "Missing authentication: the /admin/delete endpoint has no access control. "
            "Any unauthenticated user can delete records.",
            "Add @login_required decorator and check that request.user.is_admin is True."
        ),
        _issue(
            51, "app.py", "security", "critical",
            "Command injection: user-supplied 'host' is interpolated into a shell command "
            "with shell=True. Attacker can supply 'x; rm -rf /' to execute arbitrary commands.",
            "Use: subprocess.run(['ping', '-c', '1', host], shell=False) after validating host."
        ),
    ],
    "max_steps": 20,
    "hints": [
        "Look for hardcoded credentials and secrets at the top of the file.",
        "Check every place user input (request.args, request.form) touches a database query, "
        "template, file path, or shell command.",
        "The admin endpoint is missing an authorization check.",
    ],
}


_VIEWS_CODE = """\
import threading
from django.db import transaction
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from .models import Order, Product, Cart
import hashlib

_lock = threading.Lock()


@login_required
def place_order(request):
    user = request.user
    cart_items = Cart.objects.filter(user=user)

    if not cart_items.exists():
        return JsonResponse({'error': 'Cart is empty'}, status=400)

    total = 0
    for item in cart_items:
        product = Product.objects.get(id=item.product_id)
        total += product.price * item.quantity

    for item in cart_items:
        product = Product.objects.get(id=item.product_id)
        if product.stock < item.quantity:
            return JsonResponse({'error': f'Insufficient stock for {product.name}'}, status=400)

    order = Order.objects.create(
        user=user,
        total=total,
        status='pending'
    )

    for item in cart_items:
        product = Product.objects.get(id=item.product_id)
        product.stock -= item.quantity
        product.save()

    cart_items.delete()
    return JsonResponse({'order_id': order.id, 'total': float(total)})


@login_required
def get_order_history(request):
    page = int(request.GET.get('page', 1))
    per_page = int(request.GET.get('per_page', 10))

    orders = Order.objects.filter(user=request.user)[
        (page - 1) * per_page: page * per_page
    ]

    result = []
    for order in orders:
        result.append({
            'id': order.id,
            'total': order.total,
            'status': order.status,
        })

    return JsonResponse({'orders': result})


def verify_payment(order_id, payment_hash):
    order = Order.objects.get(id=order_id)
    expected = hashlib.md5(f"{order_id}{order.total}".encode()).hexdigest()
    return expected == payment_hash
"""

_MODELS_CODE = """\
from django.db import models
import pickle


class User(models.Model):
    username = models.CharField(max_length=150)
    email = models.CharField(max_length=255)
    password = models.CharField(max_length=255)

    class Meta:
        db_table = 'users'


class Product(models.Model):
    name = models.CharField(max_length=255)
    price = models.FloatField()
    stock = models.IntegerField(default=0)
    metadata = models.BinaryField()

    class Meta:
        db_table = 'products'


class Order(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    total = models.FloatField()
    status = models.CharField(max_length=50)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'orders'


class Cart(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    product_id = models.IntegerField()
    quantity = models.IntegerField()

    class Meta:
        db_table = 'cart'
"""

TASK_COMPREHENSIVE: Dict[str, Any] = {
    "task_id": "comprehensive-review",
    "difficulty": "hard",
    "description": (
        "Perform a comprehensive code review of this Django e-commerce API.\n"
        "The code spans two files and contains bugs, security vulnerabilities,\n"
        "performance issues, and data modeling problems.\n"
        "Find ALL issues across BOTH files. This is a hard task — look carefully\n"
        "for subtle architectural problems, not just surface-level issues.\n\n"
        "Files to review: views.py, models.py"
    ),
    "language": "python",
    "code_files": {
        "views.py": _VIEWS_CODE,
        "models.py": _MODELS_CODE,
    },
    "ground_truth_issues": [
        _issue(
            21, "views.py", "performance", "high",
            "N+1 query: Product.objects.get() is called inside a loop, issuing one SQL "
            "query per cart item. With 100 items this means 100 DB roundtrips.",
            "Use: Product.objects.filter(id__in=[i.product_id for i in cart_items]) "
            "then build a dict for O(1) lookup."
        ),
        _issue(
            26, "views.py", "bug", "critical",
            "Race condition: the stock check and stock decrement are not atomic. "
            "Two concurrent requests can both pass the check and oversell the product.",
            "Wrap in transaction.atomic() and use Product.objects.select_for_update() "
            "to lock rows during the check."
        ),
        _issue(
            29, "views.py", "bug", "high",
            "Order is created outside a database transaction. If stock decrement fails "
            "after the order is created, the database is left in an inconsistent state.",
            "Wrap the entire order creation flow in: with transaction.atomic():"
        ),
        _issue(
            47, "views.py", "security", "medium",
            "No maximum cap on per_page: an attacker can request per_page=1000000 "
            "to dump the entire orders table in one request, causing DoS or data leak.",
            "Add: per_page = min(int(request.GET.get('per_page', 10)), 100)"
        ),
        _issue(
            66, "views.py", "security", "medium",
            "MD5 is a cryptographically broken hash function and should not be used "
            "for payment verification. Collisions can be manufactured.",
            "Use HMAC-SHA256: hmac.new(SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()"
        ),
        _issue(
            67, "views.py", "security", "medium",
            "Timing attack: string comparison with == leaks timing information that "
            "allows an attacker to forge valid hashes byte-by-byte.",
            "Use: hmac.compare_digest(expected, payment_hash) for constant-time comparison."
        ),
        _issue(
            8, "models.py", "security", "critical",
            "Plaintext password storage: passwords are stored as raw strings in the "
            "database. Any DB breach immediately exposes all user passwords.",
            "Use Django's built-in: from django.contrib.auth.hashers import make_password, check_password"
        ),
        _issue(
            16, "models.py", "bug", "medium",
            "FloatField for monetary values causes floating-point precision errors "
            "(e.g., 0.1 + 0.2 != 0.3). This will produce wrong totals over time.",
            "Use: DecimalField(max_digits=10, decimal_places=2) for all monetary fields."
        ),
        _issue(
            18, "models.py", "security", "high",
            "BinaryField storing pickled data is dangerous: pickle.loads() on untrusted "
            "data can execute arbitrary code. Anyone who can write to this field can RCE.",
            "Use: JSONField() instead. If binary storage is required, validate/sign the data."
        ),
    ],
    "max_steps": 30,
    "hints": [
        "Look for database queries inside for loops — this is a classic N+1 problem.",
        "Check whether stock checks and order creation happen inside a database transaction.",
        "Look at models.py: how are passwords and monetary values stored?",
    ],
}


_ASYNC_CODE = """\
import asyncio
import aiohttp
from typing import List, Optional

_cache: dict = {}


async def fetch_json(url: str, session: aiohttp.ClientSession) -> dict:
    async with session.get(url, timeout=5) as resp:
        return await resp.json()


async def get_user(user_id: int, session: aiohttp.ClientSession) -> dict:
    if user_id in _cache:
        return _cache[user_id]
    data = await fetch_json(f"https://api.example.com/users/{user_id}", session)
    _cache[user_id] = data
    return data


async def process_users(user_ids: List[int]) -> List[dict]:
    session = aiohttp.ClientSession()
    results = []
    for uid in user_ids:
        result = await get_user(uid, session)
        results.append(result)
    return results


async def run_with_retry(url: str, retries: int = 3) -> Optional[str]:
    for attempt in range(retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    return await resp.text()
        except Exception:
            if attempt == retries:
                raise
    return None


class TaskRunner:
    def __init__(self, concurrency: int = 5):
        self.concurrency = concurrency
        self.results = []

    async def run_all(self, tasks: List) -> List:
        for task in tasks:
            result = await task
            self.results.append(result)
        return self.results
"""

TASK_ASYNC_REVIEW: Dict[str, Any] = {
    "task_id": "async-review",
    "difficulty": "medium-hard",
    "description": (
        "Review this async Python module for concurrency bugs, resource leaks,\n"
        "and performance issues with asyncio and aiohttp.\n"
        "The code has subtle async-specific bugs that would cause failures or\n"
        "degraded performance in production. Identify all issues with exact\n"
        "line numbers, types, and severity.\n\n"
        "File to review: async.py"
    ),
    "language": "python",
    "code_files": {
        "async.py": _ASYNC_CODE,
    },
    "ground_truth_issues": [
        _issue(
            5, "async.py", "bug", "high",
            "Shared mutable dict without asyncio.Lock; concurrent coroutines can read "
            "stale data or overwrite each other's writes. Use async with _lock: around "
            "cache check and write.",
            "Add _lock = asyncio.Lock() and use: async with _lock: around cache check and write."
        ),
        _issue(
            9, "async.py", "bug", "medium",
            "timeout=5 is wrong type for aiohttp; requires aiohttp.ClientTimeout(total=5). "
            "Passing an int raises TypeError at runtime.",
            "Use: timeout=aiohttp.ClientTimeout(total=5)"
        ),
        _issue(
            22, "async.py", "bug", "high",
            "ClientSession created but never closed, causing resource leak. "
            "Use: async with aiohttp.ClientSession() as session: and pass it in.",
            "Replace with: async with aiohttp.ClientSession() as session:"
        ),
        _issue(
            24, "async.py", "performance", "high",
            "Sequential for loop with await serializes all requests. "
            "Use asyncio.gather(*[get_user(uid, session) for uid in user_ids]) "
            "for true concurrency.",
            "Replace loop with: results = await asyncio.gather(*[get_user(uid, session) for uid in user_ids])"
        ),
        _issue(
            37, "async.py", "bug", "high",
            "Off-by-one: range(retries) yields 0..retries-1, so attempt==retries is never true. "
            "Exception is never re-raised. Fix: attempt == retries - 1.",
            "Change: if attempt == retries - 1: raise"
        ),
        _issue(
            48, "async.py", "performance", "medium",
            "Tasks awaited sequentially instead of concurrently. "
            "Use asyncio.gather(*tasks). Also self.results accumulates across multiple run_all calls.",
            "Replace loop with: self.results.extend(await asyncio.gather(*tasks))"
        ),
    ],
    "max_steps": 20,
    "hints": [
        "Check all places where ClientSession is created — are they properly closed?",
        "Look for sequential awaits inside loops where gather() would be more appropriate.",
        "The retry function has an off-by-one error in its condition.",
    ],
}


_PIPELINE_CODE = """\
import csv
import json
import hashlib
import sqlite3
from typing import List, Dict, Optional


def init_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS records "
        "(id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL, "
        "email TEXT NOT NULL, password_hash TEXT, score REAL DEFAULT 0)"
    )
    conn.commit()
    return conn


def hash_password(password: str) -> str:
    return hashlib.md5(password.encode()).hexdigest()


def insert_record(conn: sqlite3.Connection, username: str,
                  email: str, password: str, score: float) -> None:
    pwd = hash_password(password)
    conn.execute(
        f"INSERT INTO records (username, email, password_hash, score) "
        f"VALUES ('{username}', '{email}', '{pwd}', {score})"
    )
    conn.commit()


def search_records(conn: sqlite3.Connection, query: str) -> List[Dict]:
    cursor = conn.execute(
        f"SELECT id, username, email, score FROM records WHERE username LIKE '%{query}%'"
    )
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def bulk_load(conn: sqlite3.Connection, filepath: str) -> int:
    count = 0
    with open(filepath, newline='') as f:
        for row in csv.DictReader(f):
            insert_record(conn, row['username'], row['email'],
                          row.get('password', ''), float(row.get('score', 0)))
            count += 1
    return count


def export_records(conn: sqlite3.Connection, out_path: str) -> None:
    rows = search_records(conn, '')
    with open(out_path, 'w') as f:
        json.dump(rows, f, indent=2)


def get_top_scores(conn: sqlite3.Connection, limit: int) -> List[Dict]:
    cursor = conn.execute(
        f"SELECT username, score FROM records ORDER BY score DESC LIMIT {limit}"
    )
    return [{'username': r[0], 'score': r[1]} for r in cursor.fetchall()]
"""

TASK_DATA_PIPELINE: Dict[str, Any] = {
    "task_id": "data-pipeline",
    "difficulty": "hard",
    "description": (
        "Perform a security and correctness review of this data pipeline module.\n"
        "The module handles user records in SQLite. It contains multiple critical\n"
        "security vulnerabilities, a performance issue, and an error handling gap.\n"
        "Find ALL issues across the file.\n\n"
        "File to review: pipeline.py"
    ),
    "language": "python",
    "code_files": {
        "pipeline.py": _PIPELINE_CODE,
    },
    "ground_truth_issues": [
        _issue(
            20, "pipeline.py", "security", "high",
            "MD5 is cryptographically broken for password hashing. "
            "Use bcrypt, argon2, or hashlib.pbkdf2_hmac instead.",
            "Use: hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)"
        ),
        _issue(
            27, "pipeline.py", "security", "critical",
            "SQL injection: username, email, and pwd interpolated directly into query string. "
            "Use parameterized queries: conn.execute('INSERT INTO records ... VALUES (?,?,?,?)', "
            "(username, email, pwd, score))",
            "Use: conn.execute('INSERT INTO records (username, email, password_hash, score) VALUES (?,?,?,?)', (username, email, pwd, score))"
        ),
        _issue(
            35, "pipeline.py", "security", "critical",
            "SQL injection in LIKE clause: user-supplied query interpolated directly. "
            "Use: conn.execute('... WHERE username LIKE ?', (f'%{query}%',))",
            "Use: conn.execute('SELECT ... WHERE username LIKE ?', (f'%{query}%',))"
        ),
        _issue(
            41, "pipeline.py", "performance", "high",
            "bulk_load commits one transaction per row via insert_record. "
            "Wrap entire loop in with conn: for a single transaction — 10-100x faster for large imports.",
            "Wrap loop body with: with conn: conn.executemany(...)"
        ),
        _issue(
            46, "pipeline.py", "bug", "medium",
            "float() conversion has no error handling. A single malformed score field "
            "crashes the entire import. Wrap in try/except ValueError.",
            "Use: float(row.get('score', 0) or 0) inside try/except ValueError"
        ),
        _issue(
            52, "pipeline.py", "security", "high",
            "export_records calls search_records(conn, '') which returns all records including "
            "password_hash field. Strip sensitive fields before export.",
            "Filter out password_hash: rows = [{k: v for k, v in r.items() if k != 'password_hash'} for r in rows]"
        ),
        _issue(
            59, "pipeline.py", "security", "critical",
            "SQL injection: limit value interpolated into query. Although limit is an int here, "
            "use parameterized query: conn.execute('... LIMIT ?', (limit,))",
            "Use: conn.execute('SELECT username, score FROM records ORDER BY score DESC LIMIT ?', (limit,))"
        ),
    ],
    "max_steps": 25,
    "hints": [
        "Look for every place user-supplied values touch a SQL query string — are they parameterized?",
        "The bulk_load function has both a performance issue and an error handling gap.",
        "Check what fields export_records includes in its output — are any sensitive?",
    ],
}


_API_SECURITY_CODE = """\
from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import jwt
import hashlib
import pickle
import os
import sqlite3

app = FastAPI()
security = HTTPBasic()

SECRET_KEY = "dev-secret-do-not-use-in-prod"
ADMIN_TOKEN = "admin-hardcoded-token-123"

users_db = {
    "admin": hashlib.md5(b"password123").hexdigest(),
    "user": hashlib.md5(b"user123").hexdigest(),
}


@app.post("/login")
def login(credentials: HTTPBasicCredentials = Depends(security)):
    username = credentials.username
    stored = users_db.get(username, "")
    if stored != hashlib.md5(credentials.password.encode()).hexdigest():
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = jwt.encode({"user": username, "admin": username == "admin"},
                       SECRET_KEY, algorithm="HS256")
    return {"token": token}


@app.get("/users/{user_id}")
def get_user(user_id: str, authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing token")
    payload = jwt.decode(authorization, SECRET_KEY, algorithms=["HS256"])
    conn = sqlite3.connect("app.db")
    cursor = conn.execute(f"SELECT * FROM users WHERE id = '{user_id}'")
    return {"user": cursor.fetchone()}


@app.post("/admin/export")
def admin_export(authorization: str = Header(None)):
    if authorization != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")
    path = os.environ.get("EXPORT_PATH", "/tmp/export")
    os.system(f"mysqldump mydb > {path}/dump.sql")
    return {"status": "export complete", "path": path}


@app.post("/import")
def import_data(payload: bytes):
    data = pickle.loads(payload)
    return {"records": len(data)}


@app.get("/search")
def search_users(q: str, limit: int = 100):
    conn = sqlite3.connect("app.db")
    rows = conn.execute(
        f"SELECT id, name, email FROM users WHERE name LIKE '%{q}%' LIMIT {limit}"
    ).fetchall()
    return {"results": rows}
"""

TASK_API_SECURITY: Dict[str, Any] = {
    "task_id": "api-security",
    "difficulty": "hard",
    "description": (
        "Perform a security audit on this FastAPI REST API.\n"
        "The service handles user authentication and data operations.\n"
        "It contains multiple critical security flaws across authentication,\n"
        "authorization, injection attacks, and cryptography.\n"
        "Find ALL issues with exact line numbers and severity ratings.\n\n"
        "File to review: api.py"
    ),
    "language": "python",
    "code_files": {
        "api.py": _API_SECURITY_CODE,
    },
    "ground_truth_issues": [
        _issue(
            12, "api.py", "security", "high",
            "Hardcoded SECRET_KEY in source code. Any developer with repo access can forge "
            "JWT tokens and impersonate any user.",
            "Use: SECRET_KEY = os.environ.get('SECRET_KEY') and rotate it as a secret."
        ),
        _issue(
            13, "api.py", "security", "high",
            "Hardcoded ADMIN_TOKEN in source code. Static tokens in code are trivially "
            "leaked via version control, logs, or error messages.",
            "Use: ADMIN_TOKEN = os.environ.get('ADMIN_TOKEN') and generate it securely."
        ),
        _issue(
            16, "api.py", "security", "high",
            "MD5 used for password hashing. MD5 is cryptographically broken; precomputed "
            "rainbow tables can reverse any MD5 hash in seconds.",
            "Use bcrypt, argon2, or hashlib.pbkdf2_hmac with a random salt."
        ),
        _issue(
            27, "api.py", "security", "medium",
            "JWT token issued without an expiry claim ('exp'). Tokens are valid forever; "
            "a stolen token can never be invalidated without rotating the secret.",
            "Add: {'exp': datetime.utcnow() + timedelta(hours=1)} to the JWT payload."
        ),
        _issue(
            33, "api.py", "security", "critical",
            "Missing authorization check: any authenticated user can fetch any user_id. "
            "This is an Insecure Direct Object Reference (IDOR) — user A can read user B's data.",
            "Check: if payload.get('user') != user_id and not payload.get('admin'): raise 403."
        ),
        _issue(
            38, "api.py", "security", "critical",
            "SQL injection: user_id is interpolated directly into the query string. "
            "An attacker can supply user_id = \"' OR '1'='1\" to dump the users table.",
            "Use parameterized query: conn.execute('SELECT * FROM users WHERE id = ?', (user_id,))"
        ),
        _issue(
            47, "api.py", "security", "critical",
            "Command injection: EXPORT_PATH from environment is interpolated into an "
            "os.system() shell command. A misconfigured env var like '/tmp; rm -rf /' "
            "executes arbitrary commands as the server process.",
            "Use subprocess.run(['mysqldump', 'mydb'], stdout=open(path, 'w'), shell=False)."
        ),
        _issue(
            53, "api.py", "security", "critical",
            "Unsafe deserialization: pickle.loads() on untrusted user-supplied bytes allows "
            "remote code execution. Any client can craft a pickle payload that runs arbitrary code.",
            "Use json.loads() or a schema-validated format. Never unpickle untrusted data."
        ),
    ],
    "max_steps": 25,
    "hints": [
        "Check every hardcoded string assigned to variables like SECRET_KEY, TOKEN, PASSWORD.",
        "Look at every endpoint: which ones verify the caller's identity vs just authentication?",
        "Find all places user-supplied data touches: SQL queries, shell commands, deserialization.",
    ],
}


_JS_CODE = """\
const express = require('express');
const jwt = require('jsonwebtoken');
const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');
const sqlite3 = require('better-sqlite3');

const app = express();
app.use(express.json());

const JWT_SECRET = 'super-secret-key-hardcoded';
const db = new sqlite3('./data.db');

app.post('/login', (req, res) => {
    const { username, password } = req.body;
    const user = db.prepare(`SELECT * FROM users WHERE username = '${username}' AND password = '${password}'`).get();
    if (!user) return res.status(401).json({ error: 'Invalid credentials' });
    const token = jwt.sign({ id: user.id, role: user.role }, JWT_SECRET);
    res.json({ token });
});

app.get('/user/:id', (req, res) => {
    const token = req.headers.authorization;
    const payload = jwt.verify(token, JWT_SECRET);
    const user = db.prepare(`SELECT * FROM users WHERE id = ${req.params.id}`).get();
    res.json(user);
});

app.get('/search', (req, res) => {
    const q = req.query.q;
    res.send(`<h1>Results for: ${q}</h1>`);
});

app.post('/run-report', (req, res) => {
    const { filename } = req.body;
    const output = execSync(`node reports/${filename}`);
    res.send(output.toString());
});

app.get('/files', (req, res) => {
    const name = req.query.name;
    const filePath = path.join(__dirname, 'uploads', name);
    res.send(fs.readFileSync(filePath, 'utf8'));
});

app.post('/template', (req, res) => {
    const { template, data } = req.body;
    const fn = new Function('data', `return \\`${template}\\``);
    res.json({ result: fn(data) });
});

app.listen(3000);
"""

TASK_JS_SECURITY: Dict[str, Any] = {
    "task_id": "js-security",
    "difficulty": "hard",
    "description": (
        "Perform a security audit on this Express.js REST API.\n"
        "The service handles authentication and user data operations in Node.js.\n"
        "It contains critical security vulnerabilities common in JavaScript backends.\n"
        "Identify ALL issues with exact line numbers, types, and severity.\n\n"
        "File to review: server.js"
    ),
    "language": "javascript",
    "code_files": {
        "server.js": _JS_CODE,
    },
    "ground_truth_issues": [
        _issue(
            11, "server.js", "security", "high",
            "Hardcoded JWT secret 'super-secret-key-hardcoded' in source. "
            "Anyone with code access can forge tokens for any user.",
            "Use: const JWT_SECRET = process.env.JWT_SECRET and rotate it as an env secret."
        ),
        _issue(
            16, "server.js", "security", "critical",
            "SQL injection: username and password are interpolated directly into a template "
            "literal inside prepare(). An attacker can bypass authentication with username = ' OR '1'='1'--.",
            "Use parameterized queries: db.prepare('SELECT * FROM users WHERE username = ? AND password = ?').get(username, password)"
        ),
        _issue(
            18, "server.js", "security", "medium",
            "JWT issued without expiry ('expiresIn' option missing). Tokens are valid forever; "
            "a stolen token can never be invalidated without rotating the secret.",
            "Add: jwt.sign({ id: user.id, role: user.role }, JWT_SECRET, { expiresIn: '1h' })"
        ),
        _issue(
            25, "server.js", "security", "critical",
            "Missing authorization + SQL injection: any authenticated user can fetch any "
            "user by changing req.params.id (IDOR). Also id is interpolated directly into SQL.",
            "Check payload.id === req.params.id (or admin role). Use parameterized: db.prepare('SELECT * FROM users WHERE id = ?').get(req.params.id)"
        ),
        _issue(
            31, "server.js", "security", "high",
            "Cross-site scripting (XSS): user-supplied query parameter q is reflected "
            "directly into HTML response without escaping.",
            "Use a templating engine with auto-escaping, or: res.send(`<h1>Results for: ${escapeHtml(q)}</h1>`)"
        ),
        _issue(
            36, "server.js", "security", "critical",
            "Command injection: user-supplied filename is passed directly to execSync() "
            "in a shell command. An attacker can supply 'x; rm -rf /' as filename.",
            "Validate filename against a strict allowlist. Use execFileSync(['node', 'reports/' + sanitizedName]) with shell:false."
        ),
        _issue(
            42, "server.js", "security", "high",
            "Path traversal: user-supplied 'name' is joined to uploads directory with path.join. "
            "An attacker can supply '../../../etc/passwd' to read arbitrary files.",
            "Use: path.resolve(__dirname, 'uploads', path.basename(name)) and validate the result starts with the uploads dir."
        ),
        _issue(
            48, "server.js", "security", "critical",
            "Unsafe dynamic code execution: new Function() with user-supplied template string "
            "is equivalent to eval(). Any client can execute arbitrary JavaScript on the server.",
            "Never use new Function() or eval() with user input. Use a safe template engine like Handlebars or Mustache."
        ),
    ],
    "max_steps": 25,
    "hints": [
        "Check every place user input (req.body, req.params, req.query) touches a database query, shell command, or HTML response.",
        "Look for hardcoded secrets at the top of the file.",
        "The /template and /run-report endpoints have particularly dangerous patterns.",
    ],
}


ALL_TASKS: Dict[str, Dict[str, Any]] = {
    TASK_BUG_DETECTION["task_id"]: TASK_BUG_DETECTION,
    TASK_SECURITY_AUDIT["task_id"]: TASK_SECURITY_AUDIT,
    TASK_COMPREHENSIVE["task_id"]: TASK_COMPREHENSIVE,
    TASK_ASYNC_REVIEW["task_id"]: TASK_ASYNC_REVIEW,
    TASK_DATA_PIPELINE["task_id"]: TASK_DATA_PIPELINE,
    TASK_API_SECURITY["task_id"]: TASK_API_SECURITY,
    TASK_JS_SECURITY["task_id"]: TASK_JS_SECURITY,
}

TASK_IDS: List[str] = list(ALL_TASKS.keys())


def get_task(task_id: str) -> Dict[str, Any]:
    """Return task definition by ID, raising KeyError if not found."""
    if task_id not in ALL_TASKS:
        raise KeyError(f"Unknown task_id '{task_id}'. Valid: {TASK_IDS}")
    return ALL_TASKS[task_id]
