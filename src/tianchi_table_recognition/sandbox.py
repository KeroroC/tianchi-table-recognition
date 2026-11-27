"""Isolated Decimal Python sandbox with static AST whitelist and 5-second hard timeout."""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

ALLOWED_MODULES: frozenset[str] = frozenset({"decimal", "math", "statistics", "re", "json"})
DEFAULT_TIMEOUT_SECONDS: float = 5.0

_FENCE_RE = re.compile(r"^```(?:python|py)?\s*\n?(.*?)\n?```$", re.DOTALL | re.IGNORECASE)

_FORBIDDEN_NAMES: frozenset[str] = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "input",
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
        "globals",
        "locals",
        "vars",
        "dir",
        "type",
        "object",
        "super",
        "memoryview",
        "breakpoint",
        "help",
        "license",
        "credits",
        "copyright",
        "exit",
        "quit",
        "classmethod",
        "staticmethod",
        "property",
    }
)

_FORBIDDEN_ATTRS: frozenset[str] = frozenset(
    {
        "f_globals",
        "f_locals",
        "f_builtins",
        "f_back",
        "f_code",
        "gi_frame",
        "gi_code",
        "cr_frame",
        "cr_code",
        "ag_frame",
        "ag_code",
        "tb_frame",
        "tb_next",
        "func_globals",
        "func_code",
    }
)

_ALLOWED_AST_NODES: frozenset[type[ast.AST]] = frozenset(
    {
        ast.Module,
        ast.Expr,
        ast.Assign,
        ast.AugAssign,
        ast.AnnAssign,
        ast.If,
        ast.For,
        ast.While,
        ast.Break,
        ast.Continue,
        ast.Pass,
        ast.FunctionDef,
        ast.Return,
        ast.Try,
        ast.ExceptHandler,
        ast.Raise,
        ast.Import,
        ast.ImportFrom,
        ast.alias,
        ast.Assert,
        ast.Delete,
        ast.With,
        ast.withitem,
        ast.Lambda,
        ast.Constant,
        ast.List,
        ast.Tuple,
        ast.Set,
        ast.Dict,
        ast.Name,
        ast.Load,
        ast.Store,
        ast.Del,
        ast.Attribute,
        ast.Subscript,
        ast.Slice,
        ast.Starred,
        ast.UnaryOp,
        ast.UAdd,
        ast.USub,
        ast.Not,
        ast.Invert,
        ast.BinOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.BitAnd,
        ast.BitOr,
        ast.BitXor,
        ast.LShift,
        ast.RShift,
        ast.BoolOp,
        ast.And,
        ast.Or,
        ast.Compare,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.Is,
        ast.IsNot,
        ast.In,
        ast.NotIn,
        ast.IfExp,
        ast.NamedExpr,
        ast.Call,
        ast.keyword,
        ast.arguments,
        ast.arg,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
        ast.comprehension,
        ast.JoinedStr,
        ast.FormattedValue,
    }
)

_SANDBOX_RUNNER = """
import io
import json
import math
import re
import statistics
import sys
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_ALLOWED_MODULES = set()

def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level != 0:
        raise ImportError("Relative imports are disallowed")
    if name not in _ALLOWED_MODULES:
        raise ImportError(f"Module {name!r} is disallowed in sandbox")
    if fromlist:
        for item in fromlist:
            if item == "*" or str(item).startswith("_"):
                raise ImportError(f"Disallowed import symbol {item!r}")
    return __import__(name, globals, locals, fromlist, level)

SAFE_BUILTINS = {
    "__import__": _safe_import,
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "frozenset": frozenset,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "pow": pow,
    "print": print,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "ZeroDivisionError": ZeroDivisionError,
    "ArithmeticError": ArithmeticError,
    "AssertionError": AssertionError,
    "RuntimeError": RuntimeError,
    "StopIteration": StopIteration,
}

def _decimal_to_str(val):
    if not val.is_finite():
        raise ValueError(f"Non-finite Decimal result: {val!r}")
    if val == 0:
        val = Decimal(0)
    return format(val, "f")

def _normalize_json_value(obj):
    if isinstance(obj, Decimal):
        return _decimal_to_str(obj)
    if isinstance(obj, dict):
        return {str(k): _normalize_json_value(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize_json_value(v) for v in obj]
    return obj

def _format_answer(val):
    if val is None:
        return None, "Script produced None result"
    if isinstance(val, Decimal):
        return _decimal_to_str(val), None
    if isinstance(val, bool):
        return "true" if val else "false", None
    if isinstance(val, int):
        return str(val), None
    if isinstance(val, float):
        if not math.isfinite(val):
            return None, f"Non-finite float result: {val!r}"
        return str(val), None
    if isinstance(val, str):
        return val, None
    if isinstance(val, (dict, list)):
        normalized = _normalize_json_value(val)
        dumped = json.dumps(
            normalized, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )
        return dumped, None
    return str(val), None

def _main():
    real_stdout = sys.stdout
    try:
        payload = json.loads(sys.stdin.read())
        code = payload["code"]
        context = payload.get("context") or {}
        _ALLOWED_MODULES.update(payload.get("allowed_modules") or ())
    except Exception as exc:
        err_payload = {"ok": False, "error_type": "invalid_output", "error": str(exc)}
        real_stdout.write(json.dumps(err_payload, ensure_ascii=False))
        return

    safe_globals = {
        "__builtins__": SAFE_BUILTINS,
        "DATA": context,
        "TABLES": context.get("tables", []),
        "PAGE_CONTEXT": context.get("page_context", ""),
        "QUESTION": context.get("question", ""),
        "Decimal": Decimal,
        "ROUND_HALF_UP": ROUND_HALF_UP,
        "InvalidOperation": InvalidOperation,
    }
    captured = io.StringIO()
    try:
        sys.stdout = captured
        exec(compile(code, "<sandbox>", "exec"), safe_globals)
        raw_stdout = captured.getvalue()
        solve_fn = safe_globals.get("solve")
        value = None
        found = False
        if callable(solve_fn):
            value = solve_fn(context)
            found = True
        else:
            for key in ("RESULT", "ANSWER", "result", "answer"):
                if key in safe_globals:
                    value = safe_globals[key]
                    found = True
                    break
        if not found and raw_stdout.strip():
            lines = [ln.strip() for ln in raw_stdout.strip().splitlines() if ln.strip()]
            last_line = lines[-1]
            try:
                parsed_line = json.loads(last_line)
                if isinstance(parsed_line, dict) and (
                    "answer" in parsed_line or "result" in parsed_line
                ):
                    value = parsed_line.get("answer", parsed_line.get("result"))
                else:
                    value = raw_stdout.strip()
            except Exception:
                value = raw_stdout.strip()
            found = True

        if not found:
            out = {
                "ok": False,
                "error_type": "invalid_output",
                "error": "Script did not define solve(), RESULT, ANSWER, or print output",
                "raw_stdout": raw_stdout,
            }
        else:
            rendered, err = _format_answer(value)
            if err is not None:
                out = {
                    "ok": False,
                    "error_type": "invalid_output",
                    "error": err,
                    "raw_stdout": raw_stdout,
                }
            else:
                out = {"ok": True, "answer": rendered, "raw_stdout": raw_stdout}
    except Exception as exc:
        out = {
            "ok": False,
            "error_type": "runtime_error",
            "error": f"{type(exc).__name__}: {exc}",
            "raw_stdout": captured.getvalue(),
        }
    finally:
        sys.stdout = real_stdout

    real_stdout.write(json.dumps(out, ensure_ascii=False))

_main()
"""


class SandboxValidationError(ValueError):
    """Raised when a generated Python script fails static AST whitelist checks."""


class SandboxResult(BaseModel):
    """Outcome of validating and executing a script inside PythonSandbox."""

    model_config = ConfigDict(frozen=True)

    ok: bool
    answer: str | None = None
    raw_stdout: str = ""
    stderr: str = ""
    error_type: (
        Literal["ast_rejected", "syntax_error", "timeout", "runtime_error", "invalid_output"] | None
    ) = None
    error: str | None = None
    timed_out: bool = False


def extract_python_code(raw_text: str) -> str:
    """Strip surrounding whitespace and optional markdown python code fences."""
    stripped = raw_text.strip()
    match = _FENCE_RE.match(stripped)
    if match:
        return match.group(1).strip()
    return stripped


def _reject_dunder(name: str | None, kind: str) -> None:
    if name is not None and name.startswith("__"):
        raise SandboxValidationError(f"Disallowed dunder {kind} {name!r}")


def validate_python_ast(code: str) -> ast.Module:
    """Parse code and enforce the static AST node, import, builtin, and dunder whitelist."""
    if not code or not code.strip():
        raise SandboxValidationError("Empty Python script")
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise SandboxValidationError(f"Syntax error in script: {exc}") from exc

    for node in ast.walk(tree):
        node_type = type(node)
        if node_type not in _ALLOWED_AST_NODES:
            raise SandboxValidationError(f"Disallowed AST node: {node_type.__name__}")

        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in ALLOWED_MODULES:
                    allowed_str = sorted(ALLOWED_MODULES)
                    raise SandboxValidationError(
                        f"Disallowed module import {alias.name!r}; allowed: {allowed_str}"
                    )
                if alias.asname and alias.asname.startswith("_"):
                    raise SandboxValidationError(
                        f"Disallowed private import alias {alias.asname!r}"
                    )

        elif isinstance(node, ast.ImportFrom):
            if node.level != 0:
                raise SandboxValidationError("Relative imports are not allowed")
            if node.module not in ALLOWED_MODULES:
                allowed_str = sorted(ALLOWED_MODULES)
                raise SandboxValidationError(
                    f"Disallowed module import {node.module!r}; allowed: {allowed_str}"
                )
            for alias in node.names:
                if alias.name == "*" or alias.name.startswith("_"):
                    raise SandboxValidationError(f"Disallowed import symbol {alias.name!r}")
                if alias.asname and alias.asname.startswith("_"):
                    raise SandboxValidationError(
                        f"Disallowed private import alias {alias.asname!r}"
                    )

        elif isinstance(node, ast.Name):
            if node.id in _FORBIDDEN_NAMES:
                raise SandboxValidationError(f"Disallowed identifier {node.id!r}")
            _reject_dunder(node.id, "identifier")

        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr in _FORBIDDEN_ATTRS:
                raise SandboxValidationError(f"Disallowed attribute access {node.attr!r}")

        elif isinstance(node, ast.FunctionDef):
            _reject_dunder(node.name, "function name")

        elif isinstance(node, ast.arg):
            _reject_dunder(node.arg, "argument name")

        elif isinstance(node, ast.keyword):
            _reject_dunder(node.arg, "keyword")

        elif isinstance(node, ast.ExceptHandler):
            _reject_dunder(node.name, "exception name")

    return tree


class PythonSandbox:
    """Executes AST-validated Decimal scripts inside an isolated `python -I` subprocess."""

    def __init__(
        self,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        python_executable: str | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        self._timeout_seconds = timeout_seconds
        self._python_executable = python_executable or sys.executable

    @property
    def timeout_seconds(self) -> float:
        """Hard wall-clock timeout in seconds for each isolated subprocess execution."""
        return self._timeout_seconds

    def execute(self, code: str, *, context: dict[str, Any] | None = None) -> SandboxResult:
        """Validate AST and execute code in a `python -I` subprocess via stdin/stdout JSON."""
        try:
            validate_python_ast(code)
        except SandboxValidationError as exc:
            error_type: Literal["syntax_error", "ast_rejected"] = (
                "syntax_error" if isinstance(exc.__cause__, SyntaxError) else "ast_rejected"
            )
            return SandboxResult(ok=False, error_type=error_type, error=str(exc))

        stdin_payload = json.dumps(
            {
                "code": code,
                "context": context or {},
                "allowed_modules": sorted(ALLOWED_MODULES),
            },
            ensure_ascii=False,
        )
        try:
            proc = subprocess.run(
                [self._python_executable, "-I", "-c", _SANDBOX_RUNNER],
                input=stdin_payload,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=self._timeout_seconds,
                env={},
                check=False,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                ok=False,
                error_type="timeout",
                error=f"Sandbox execution timed out after {self._timeout_seconds:g}s",
                timed_out=True,
            )

        if proc.returncode != 0 and not proc.stdout.strip():
            return SandboxResult(
                ok=False,
                stderr=proc.stderr,
                error_type="runtime_error",
                error=proc.stderr.strip() or f"Sandbox exited with status {proc.returncode}",
            )

        try:
            parsed = json.loads(proc.stdout)
        except Exception as exc:
            return SandboxResult(
                ok=False,
                raw_stdout=proc.stdout,
                stderr=proc.stderr,
                error_type="invalid_output",
                error=f"Malformed sandbox JSON output: {exc}",
            )

        if not isinstance(parsed, dict):
            return SandboxResult(
                ok=False,
                raw_stdout=proc.stdout,
                stderr=proc.stderr,
                error_type="invalid_output",
                error="Sandbox output must be a JSON object",
            )

        return SandboxResult(
            ok=bool(parsed.get("ok")),
            answer=parsed.get("answer") if isinstance(parsed.get("answer"), str) else None,
            raw_stdout=str(parsed.get("raw_stdout") or ""),
            stderr=proc.stderr,
            error_type=parsed.get("error_type"),
            error=parsed.get("error"),
            timed_out=False,
        )
