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


ALL_TASKS: Dict[str, Dict[str, Any]] = {
    TASK_BUG_DETECTION["task_id"]: TASK_BUG_DETECTION,
    TASK_SECURITY_AUDIT["task_id"]: TASK_SECURITY_AUDIT,
    TASK_COMPREHENSIVE["task_id"]: TASK_COMPREHENSIVE,
}

TASK_IDS: List[str] = list(ALL_TASKS.keys())


def get_task(task_id: str) -> Dict[str, Any]:
    """Return task definition by ID, raising KeyError if not found."""
    if task_id not in ALL_TASKS:
        raise KeyError(f"Unknown task_id '{task_id}'. Valid: {TASK_IDS}")
    return ALL_TASKS[task_id]
