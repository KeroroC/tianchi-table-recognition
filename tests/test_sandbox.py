"""Tests for the isolated Decimal PythonSandbox and static AST whitelist."""

from __future__ import annotations

import pytest

from tianchi_table_recognition.sandbox import (
    ALLOWED_MODULES,
    PythonSandbox,
    SandboxValidationError,
    extract_python_code,
    validate_python_ast,
)


def test_extract_python_code_strips_fences_and_preserves_plain_code() -> None:
    fenced = "```python\nfrom decimal import Decimal\nRESULT = Decimal('1.23')\n```"
    assert extract_python_code(fenced) == "from decimal import Decimal\nRESULT = Decimal('1.23')"

    generic_fenced = "```\nANSWER = '42'\n```"
    assert extract_python_code(generic_fenced) == "ANSWER = '42'"

    plain = "  RESULT = Decimal('0.1') + Decimal('0.2')  "
    assert extract_python_code(plain) == "RESULT = Decimal('0.1') + Decimal('0.2')"


def test_sandbox_executes_allowed_modules_with_decimal_precision_in_isolated_subprocess() -> None:
    sandbox = PythonSandbox()
    assert sandbox.timeout_seconds == 5.0
    assert ALLOWED_MODULES == frozenset({"decimal", "math", "statistics", "re", "json"})

    code = """
import json
import math
import re
import statistics
from decimal import Decimal, ROUND_HALF_UP

pattern = re.compile(r"[0-9.,-]+")
values = []
for cell in TABLES[0]["cells"]:
    if cell["row"] >= 1 and cell["col"] == 1 and cell["text"]:
        matched = pattern.search(cell["text"])
        if matched:
            values.append(Decimal(matched.group(0).replace(",", "")))

total = sum(values, Decimal("0"))
ratio = (Decimal("0.1") + Decimal("0.2")).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
assert math.isfinite(float(total))
assert statistics.mean([1, 2, 3]) == 2
RESULT = total + ratio
"""
    context = {
        "document_key": "doc_test_1234",
        "question": "计算合计",
        "page_context": "财务附注",
        "tables": [
            {
                "table_id": "t1",
                "cells": [
                    {"row": 0, "col": 0, "text": "项目"},
                    {"row": 0, "col": 1, "text": "金额"},
                    {"row": 1, "col": 0, "text": "甲"},
                    {"row": 1, "col": 1, "text": "1,000.10"},
                    {"row": 2, "col": 0, "text": "乙"},
                    {"row": 2, "col": 1, "text": "200.20"},
                ],
            }
        ],
    }

    res = sandbox.execute(code, context=context)

    assert res.ok is True
    assert res.answer == "1200.60"
    assert res.error is None
    assert res.timed_out is False


def test_sandbox_avoids_float_drift_and_supports_solve_fn_and_json_output() -> None:
    sandbox = PythonSandbox()

    res_decimal = sandbox.execute(
        "from decimal import Decimal\nRESULT = Decimal('0.1') + Decimal('0.2')"
    )
    assert res_decimal.ok is True
    assert res_decimal.answer == "0.3"

    res_solve = sandbox.execute(
        """
from decimal import Decimal

def solve(data):
    return {"sum": Decimal("10.00") + Decimal("2.50"), "ok": True}
"""
    )
    assert res_solve.ok is True
    assert res_solve.answer == '{"sum":"12.50","ok":true}'


@pytest.mark.parametrize(
    "bad_code",
    [
        "import os\nRESULT = os.getcwd()",
        "import sys\nRESULT = sys.version",
        "import subprocess\nRESULT = 1",
        "from pathlib import Path\nRESULT = str(Path('.'))",
        "import socket\nRESULT = 1",
        "import urllib.request\nRESULT = 1",
        "import http.client\nRESULT = 1",
        "import importlib\nRESULT = 1",
        "import builtins\nRESULT = 1",
        "import io\nRESULT = 1",
        "from decimal import *\nRESULT = Decimal('1')",
        "from . import helper\nRESULT = 1",
    ],
)
def test_sandbox_ast_rejects_disallowed_and_unsafe_imports(bad_code: str) -> None:
    with pytest.raises(SandboxValidationError):
        validate_python_ast(bad_code)

    res = PythonSandbox().execute(bad_code)
    assert res.ok is False
    assert res.error_type == "ast_rejected"
    assert res.timed_out is False


@pytest.mark.parametrize(
    "bad_code",
    [
        "RESULT = eval('1 + 1')",
        "exec('RESULT = 1')",
        "RESULT = __import__('os').name",
        "f = open('/etc/passwd')\nRESULT = 1",
        "RESULT = compile('1+1', '<x>', 'eval')",
        "RESULT = getattr(1, '__class__')",
        "RESULT = globals()",
        "RESULT = locals()",
        "RESULT = vars()",
        "breakpoint()\nRESULT = 1",
        "fn = eval\nRESULT = fn('2')",
        "RESULT = ().__class__.__bases__[0].__subclasses__()",
        "x = 1\nRESULT = x.__dict__",
        "g = (i for i in range(1))\nRESULT = g.gi_frame",
        "class Hack:\n    pass\nRESULT = 1",
        "global X\nX = 1\nRESULT = X",
    ],
)
def test_sandbox_ast_rejects_dangerous_builtins_dunder_and_frame_access(bad_code: str) -> None:
    with pytest.raises(SandboxValidationError):
        validate_python_ast(bad_code)

    res = PythonSandbox().execute(bad_code)
    assert res.ok is False
    assert res.error_type == "ast_rejected"


def test_sandbox_enforces_hard_timeout_and_reports_runtime_errors() -> None:
    short_sandbox = PythonSandbox(timeout_seconds=0.5)
    timeout_res = short_sandbox.execute("while True:\n    pass")
    assert timeout_res.ok is False
    assert timeout_res.timed_out is True
    assert timeout_res.error_type == "timeout"

    runtime_res = short_sandbox.execute("RESULT = 1 / 0")
    assert runtime_res.ok is False
    assert runtime_res.timed_out is False
    assert runtime_res.error_type == "runtime_error"
    assert runtime_res.error is not None and "ZeroDivisionError" in runtime_res.error
