"""Typed reversible workspace tools and effect-based verification."""
from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import math
import operator
import os
from pathlib import Path
import stat
import time
import uuid

from .sandbox import run_python, SandboxUnavailable

MAX_FILE = 1024 * 1024


def describe_tools() -> list[dict]:
    return [
        {"name": "filesystem.list", "args": {"path": "relative directory, default '.'"},
         "effect": "List at most 200 entries, never follow links."},
        {"name": "filesystem.read", "args": {"path": "relative file"},
         "effect": "Read UTF-8 file, at most 1MiB, returning content/text and sha256."},
        {"name": "filesystem.write", "args": {"path": "relative file", "content": "UTF-8 text",
          "overwrite": "boolean, default false", "expected_sha256": "optional prior hash"},
         "effect": "Atomic write; parents must exist. Overwrites retain an owner recovery copy."},
        {"name": "filesystem.mkdir", "args": {"path": "relative directory"},
         "effect": "Create directories recursively, rejecting links and conflicting files."},
        {"name": "calculator.eval", "args": {"expression": "bounded arithmetic expression"},
         "effect": "Numeric arithmetic with sqrt,abs,round,min,max; no Python eval."},
        {"name": "data.profile", "args": {"path": "CSV or JSON array of records"},
         "effect": "Observed row count, column types, missing counts and numeric statistics."},
        {"name": "data.query", "args": {"path": "CSV or JSON array of records",
          "where": "optional {column,value,op:eq|ne|gt|ge|lt|le|contains}",
          "select": "optional column list", "aggregate": "optional count|sum|mean|min|max",
          "column": "numeric column for aggregation", "limit": "1..200"},
         "effect": "Bounded structured filtering and aggregation; no arbitrary SQL."},
        {"name": "code.run", "args": {"code": "Python source <=100KB",
          "inputs": "optional list of workspace file paths", "timeout": "0.1..30 seconds"},
         "effect": "Landlock/seccomp isolated stdlib Python. Returns stdout and bounded text artifacts; no direct workspace writes."},
        {"name": "inspect", "args": {"path": "optional workspace path"},
         "effect": "List a directory or read a file."},
        {"name": "finish", "args": {"answer": "proposed final answer"},
         "effect": "Declare proposed completion. Runtime must independently verify goal criteria."},
    ]


TOOL_NAMES = frozenset(x["name"] for x in describe_tools())


def path_parts(path: str, *, allow_root=False) -> list[str]:
    if not isinstance(path, str) or not path or len(path) > 2048 or "\x00" in path:
        raise ValueError("invalid workspace path")
    if allow_root and path == ".":
        return []
    parts = path.split("/")
    if path.startswith("/") or "\\" in path or any(x in {"", ".", ".."} for x in parts):
        raise ValueError("path must be relative, without traversal or empty components")
    return parts


class SafeWorkspace:
    """All operations traverse pinned directory handles with O_NOFOLLOW."""
    def __init__(self, root):
        self.root = Path(root).absolute()
        self.root.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def directory(self, parts: list[str], create=False) -> int:
        fd = os.dup(self.fd)
        try:
            for name in parts:
                if create:
                    try:
                        os.mkdir(name, mode=0o700, dir_fd=fd)
                    except FileExistsError:
                        pass
                nxt = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd)
                fd = nxt
            return fd
        except BaseException:
            os.close(fd)
            raise

    def read(self, path: str) -> bytes:
        parts = path_parts(path)
        parent = self.directory(parts[:-1])
        try:
            fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
            with os.fdopen(fd, "rb") as f:
                st = os.fstat(f.fileno())
                if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
                    raise PermissionError("only single-link regular files are readable")
                if st.st_size > MAX_FILE:
                    raise ValueError("file exceeds 1MiB tool limit")
                data = f.read(MAX_FILE + 1)
                if len(data) > MAX_FILE:
                    raise ValueError("file exceeds 1MiB tool limit")
                return data
        finally:
            os.close(parent)

    def list(self, path=".") -> list[dict]:
        fd = self.directory(path_parts(path, allow_root=True))
        try:
            entries = []
            for name in sorted(os.listdir(fd))[:200]:
                st = os.stat(name, dir_fd=fd, follow_symlinks=False)
                kind = "directory" if stat.S_ISDIR(st.st_mode) else "file" if stat.S_ISREG(st.st_mode) else "blocked"
                if st.st_nlink > 1 and stat.S_ISREG(st.st_mode):
                    kind = "blocked"
                entries.append({"name": name, "type": kind, "size": st.st_size})
            return entries
        finally:
            os.close(fd)

    def mkdir(self, path):
        fd = self.directory(path_parts(path), create=True)
        os.close(fd)
        return {"path": path, "is_directory": True}

    def write(self, path, data, *, overwrite=False, expected_sha256=None, backup_dir=None):
        if len(data) > MAX_FILE:
            raise ValueError("write exceeds 1MiB limit")
        parts = path_parts(path)
        parent = self.directory(parts[:-1])
        temporary = ".sage-" + uuid.uuid4().hex
        previous = None
        try:
            try:
                st = os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
                if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
                    raise PermissionError("destination is not a single-link regular file")
                if not overwrite:
                    raise FileExistsError("destination exists; overwrite requires explicit true")
                previous = self.read(path)
            except FileNotFoundError:
                pass
            oldhash = hashlib.sha256(previous).hexdigest() if previous is not None else None
            if expected_sha256 is not None and oldhash != expected_sha256:
                raise ValueError("expected_sha256 precondition failed")
            backup = None
            if previous is not None and backup_dir:
                backup = uuid.uuid4().hex
                Path(backup_dir).mkdir(mode=0o700, parents=True, exist_ok=True)
                recovery = Path(backup_dir, backup)
                recovery.write_bytes(previous)
                recovery.chmod(0o600)
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600, dir_fd=parent)
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            if not overwrite:
                # link gives atomic no-replace publication (rename would overwrite
                # a competing newly created destination). Remove temporary link.
                os.link(temporary, parts[-1], src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
                os.unlink(temporary, dir_fd=parent)
            else:
                os.replace(temporary, parts[-1], src_dir_fd=parent, dst_dir_fd=parent)
            os.fsync(parent)
            observed = self.read(path)
            expected = hashlib.sha256(data).hexdigest()
            actual = hashlib.sha256(observed).hexdigest()
            if expected != actual:
                raise RuntimeError("post-write content verification failed")
            return {"path": path, "sha256": actual, "bytes": len(observed),
                    "previous_sha256": oldhash, "recovery_id": backup, "verified": True}
        finally:
            try:
                os.unlink(temporary, dir_fd=parent)
            except FileNotFoundError:
                pass
            os.close(parent)


def calculate(expression: str):
    if not isinstance(expression, str) or len(expression) > 2000:
        raise ValueError("invalid arithmetic expression")
    tree = ast.parse(expression, mode="eval")
    if len(list(ast.walk(tree))) > 100:
        raise ValueError("expression is too complex")
    binary = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
              ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
              ast.Mod: operator.mod, ast.Pow: operator.pow}
    functions = {"sqrt": math.sqrt, "abs": abs, "round": round, "min": min, "max": max}
    def visit(node):
        if isinstance(node, ast.Expression):
            result = visit(node.body)
        elif isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            result = node.value
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            result = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and type(node.op) in binary:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent exceeds limit")
            result = binary[type(node.op)](left, right)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in functions and not node.keywords:
            result = functions[node.func.id](*(visit(x) for x in node.args))
        else:
            raise ValueError("only bounded numeric arithmetic is allowed")
        if isinstance(result, complex) or abs(result) > 1e100 or not math.isfinite(result):
            raise ValueError("numeric result exceeds limits")
        return result
    return visit(tree)


def _rows(data: bytes, path: str):
    text = data.decode("utf-8-sig")
    rows = json.loads(text) if path.lower().endswith(".json") else list(csv.DictReader(io.StringIO(text)))
    if not isinstance(rows, list) or len(rows) > 20000 or not all(isinstance(x, dict) for x in rows):
        raise ValueError("expected <=20000 tabular records")
    return rows


def _numeric(x):
    try:
        n = float(x)
        return n if math.isfinite(n) else None
    except (TypeError, ValueError):
        return None


def profile(rows):
    columns = sorted({k for row in rows for k in row})
    result = {}
    for column in columns[:100]:
        values = [r.get(column) for r in rows if r.get(column) not in {None, ""}]
        numbers = [_numeric(v) for v in values]
        numeric = bool(values) and all(x is not None for x in numbers)
        result[column] = {"missing": len(rows) - len(values), "type": "number" if numeric else "text"}
        if numeric:
            result[column].update(min=min(numbers), max=max(numbers), mean=sum(numbers) / len(numbers))
    return {"row_count": len(rows), "columns": result}


def query(rows, args):
    where = args.get("where")
    if where:
        ops = {"eq": operator.eq, "ne": operator.ne, "gt": operator.gt,
               "ge": operator.ge, "lt": operator.lt, "le": operator.le,
               "contains": lambda a, b: str(b) in str(a)}
        if where.get("op", "eq") not in ops:
            raise ValueError("unknown query comparison")
        selected = []
        for row in rows:
            a, b = row.get(where["column"]), where["value"]
            if _numeric(a) is not None and _numeric(b) is not None:
                a, b = _numeric(a), _numeric(b)
            try:
                keep = ops[where.get("op", "eq")](a, b)
            except TypeError:
                keep = False
            if keep:
                selected.append(row)
        rows = selected
    aggregate = args.get("aggregate")
    if aggregate:
        if aggregate == "count":
            value = len(rows)
        else:
            numbers = [_numeric(r.get(args["column"])) for r in rows]
            numbers = [n for n in numbers if n is not None]
            if not numbers:
                raise ValueError("no numeric values for aggregation")
            funcs = {"sum": sum, "mean": lambda ns: sum(ns) / len(ns), "min": min, "max": max}
            if aggregate not in funcs:
                raise ValueError("unknown aggregation")
            value = funcs[aggregate](numbers)
        return {"matched_rows": len(rows), "aggregate": aggregate, "value": value}
    columns = args.get("select")
    limit = min(200, max(1, int(args.get("limit", 50))))
    return {"matched_rows": len(rows), "rows": [{k: row.get(k) for k in columns} if columns else row for row in rows[:limit]]}


class Executor:
    def __init__(self, workspace, state_dir):
        self.workspace = SafeWorkspace(workspace)
        self.backups = Path(state_dir) / "recovery"

    def execute(self, action, *, cancelled=None, deadline=None):
        start = time.monotonic()
        receipt = {"id": uuid.uuid4().hex, "action_id": action.get("id", ""),
                   "status": "CONFIRMED_FAILURE", "evidence": {}, "side_effects": []}
        tool, args = action.get("tool"), action.get("args", {})
        try:
            if cancelled and cancelled():
                raise PermissionError("authority was revoked")
            if deadline is not None and time.time() >= deadline:
                raise PermissionError("goal deadline exceeded")
            if tool in {"filesystem.list", "inspect"}:
                path = args.get("path", ".")
                try:
                    evidence = {"entries": self.workspace.list(path), "path": path}
                except NotADirectoryError:
                    if tool != "inspect":
                        raise
                    data = self.workspace.read(path)
                    evidence = {"path": path, "content": data.decode("utf-8"), "sha256": hashlib.sha256(data).hexdigest()}
            elif tool == "filesystem.read":
                data = self.workspace.read(args["path"])
                evidence = {"path": args["path"], "content": data.decode("utf-8"),
                            "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
            elif tool == "filesystem.write":
                content = args.get("content", args.get("text"))
                if not isinstance(content, str):
                    raise ValueError("content must be text")
                evidence = self.workspace.write(args["path"], content.encode(), overwrite=args.get("overwrite") is True,
                           expected_sha256=args.get("expected_sha256"), backup_dir=self.backups)
                receipt["side_effects"] = [{"type": "workspace_write", "path": args["path"], "sha256": evidence["sha256"]}]
            elif tool == "filesystem.mkdir":
                evidence = self.workspace.mkdir(args["path"])
                receipt["side_effects"] = [{"type": "workspace_mkdir", "path": args["path"]}]
            elif tool == "calculator.eval":
                evidence = {"expression": args["expression"], "value": calculate(args["expression"]), "verified": True}
            elif tool in {"data.profile", "data.query"}:
                data = self.workspace.read(args["path"])
                rows = _rows(data, args["path"])
                evidence = profile(rows) if tool == "data.profile" else query(rows, args)
                evidence.update(source_path=args["path"], source_sha256=hashlib.sha256(data).hexdigest())
            elif tool == "code.run":
                inputs = {p: self.workspace.read(p) for p in args.get("inputs", [])}
                if len(inputs) > 16 or sum(map(len, inputs.values())) > MAX_FILE:
                    raise ValueError("code inputs exceed 16 files / 1MiB")
                timeout = float(args.get("timeout", 10))
                if deadline is not None:
                    timeout = min(timeout, max(0.1, deadline - time.time()))
                evidence = run_python(args["code"], inputs=inputs, timeout=timeout, cancelled=cancelled)
                if evidence["exit_code"] != 0:
                    receipt.update(status="BLOCKED" if evidence["stopped"] == "revoked" else "CONFIRMED_FAILURE", evidence=evidence)
                    return receipt
            elif tool == "finish":
                evidence = {"answer": args.get("answer", ""), "completion_verified": False}
            else:
                raise PermissionError("unsupported tool")
            receipt.update(status="CONFIRMED_SUCCESS", evidence=evidence)
        except (PermissionError, SandboxUnavailable) as exc:
            receipt.update(status="BLOCKED", evidence={"error": type(exc).__name__, "message": str(exc)})
        except (ValueError, OSError, KeyError, TypeError, SyntaxError, ArithmeticError) as exc:
            receipt.update(status="CONFIRMED_FAILURE", evidence={"error": type(exc).__name__, "message": str(exc)})
        except Exception as exc:
            receipt.update(status="UNKNOWN", evidence={"error": type(exc).__name__, "message": str(exc)})
        finally:
            receipt["elapsed"] = time.monotonic() - start
        return receipt

    def close(self):
        self.workspace.close()
