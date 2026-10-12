"""Owner-controlled authority service in a separate process.

The cognitive runtime receives ActionClient only. Owner commands require a fresh
in-memory bearer secret; signed grants bind scope, version, expiry and tool set.
The trusted notebook/UI and OS are outside the hostile-document/code threat model.
"""
from __future__ import annotations

import copy
import hashlib
import hmac
import json
import multiprocessing
import os
from pathlib import Path
import secrets
import shutil
import socket
import tempfile
import threading
import time
import uuid

from .execution import Executor, TOOL_NAMES, path_parts

MAX_MESSAGE = 8 * 1024**2  # Covers JSON escaping of a bounded 1 MiB UTF-8 file.


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def _recv(connection):
    data = bytearray()
    while True:
        chunk = connection.recv(min(65536, MAX_MESSAGE + 1 - len(data)))
        if not chunk:
            break
        data.extend(chunk)
        if len(data) > MAX_MESSAGE:
            raise ValueError("authority message exceeds limit")
        if b"\n" in chunk:
            break
    return json.loads(data)


def _rpc(address, request):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(45)
        sock.connect(address)
        sock.sendall(canonical(request) + b"\n")
        result = _recv(sock)
    if "rpc_error" in result:
        raise RuntimeError(result["rpc_error"])
    return result["result"]


class ActionClient:
    """Action-only endpoint, deliberately has no owner token or grant mint API."""
    def __init__(self, address):
        self.address = address

    def check(self, grant, action):
        return _rpc(self.address, {"op": "check", "grant": grant, "action": action})

    def execute(self, grant, action):
        return _rpc(self.address, {"op": "execute", "grant": grant, "action": action})


class _AuthorityState:
    def __init__(self, state_dir, workspace, owner_token):
        self.root = Path(state_dir) / "authority"
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = self.root / "ledger.json"
        self.owner_token = owner_token
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.cancellations = {}
        self.active_actions = set()
        self.executor = Executor(workspace, self.root)
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state.get("schema_version") != 1:
                raise ValueError("unsupported authority schema")
            # A process restart is an authority boundary. Old grants cannot act;
            # owner resume is explicit and every old in-flight receipt is UNKNOWN.
            for goal in self.state["goals"].values():
                goal["version"] += 1
                if goal["status"] == "active":
                    goal["status"] = "paused"
        else:
            self.state = {"schema_version": 1, "signing_key": secrets.token_hex(32),
                          "goals": {}, "receipts": {}, "audit": []}
        self.key = bytes.fromhex(self.state["signing_key"])
        self.save()

    def save(self):
        temporary = self.root / (".ledger-" + uuid.uuid4().hex)
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as file:
            file.write(canonical(self.state))
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, self.path)
        dfd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)

    def audit(self, event, **data):
        self.state["audit"].append({"time": time.time(), "event": event, **data})
        # Receipts remain durable; this compact diagnostic tail is not immutable.
        self.state["audit"] = self.state["audit"][-2000:]

    def _sign(self, payload):
        return hmac.new(self.key, canonical(payload), hashlib.sha256).hexdigest()

    def _blocked(self, action, reason):
        return {"id": uuid.uuid4().hex, "action_id": action.get("id", "") if isinstance(action, dict) else "",
                "status": "BLOCKED", "evidence": {"reason": reason}, "elapsed": 0.0, "side_effects": []}

    def validate(self, grant, action, *, ignore_budget=False):
        try:
            if not isinstance(grant, dict) or not isinstance(action, dict):
                raise ValueError("grant and action must be objects")
            payload, signature = grant["payload"], grant["signature"]
            if not isinstance(signature, str) or not hmac.compare_digest(self._sign(payload), signature):
                raise ValueError("invalid grant signature")
            goal = self.state["goals"][payload["goal_id"]]
            if goal["version"] != payload["version"]:
                raise ValueError("stale or revoked grant version")
            if goal["status"] != "active":
                raise ValueError("goal is " + goal["status"])
            if time.time() >= min(payload["expires_at"], goal["deadline"]):
                raise ValueError("grant or goal expired")
            for dependency in goal.get("depends_on", []):
                if self.state["goals"].get(dependency, {}).get("status") != "completed":
                    raise ValueError("goal dependency is not complete")
            if goal.get("parent_id"):
                parent = self.state["goals"].get(goal["parent_id"], {})
                if parent.get("status") != "active":
                    raise ValueError("parent goal is not active")
            if not ignore_budget and goal["actions_used"] >= goal["max_actions"]:
                raise ValueError("goal action budget exhausted")
            if not isinstance(action.get("id"), str) or not 1 <= len(action["id"]) <= 128:
                raise ValueError("action requires bounded idempotency id")
            tool = action.get("tool")
            if tool not in TOOL_NAMES or tool not in payload["tools"]:
                raise ValueError("tool is outside the current grant")
            args = action.get("args", {})
            if not isinstance(args, dict) or len(canonical(args)) > 7 * 1024**2:
                raise ValueError("invalid or oversized action arguments")
            paths = []
            if tool.startswith("filesystem.") or tool in {"data.profile", "data.query", "inspect"}:
                paths.append(args.get("path", ".") if tool in {"filesystem.list", "inspect"} else args["path"])
            if tool == "code.run":
                inputs = args.get("inputs", [])
                if not isinstance(inputs, list) or len(inputs) > 16:
                    raise ValueError("inputs must be a list of <=16 relative paths")
                paths.extend(inputs)
            scope = payload.get("scope", "")
            for path in paths:
                path_parts(path, allow_root=tool in {"filesystem.list", "inspect"})
                if scope and path != scope and not path.startswith(scope + "/"):
                    raise ValueError("path is outside the grant resource scope")
            action_hash = hashlib.sha256(canonical(action)).hexdigest()
            # The decision commits to the complete exact action and current version.
            decision = {"goal_id": goal["id"], "version": goal["version"], "action_hash": action_hash,
                        "expires_at": min(payload["expires_at"], goal["deadline"])}
            return {"allowed": True, "reason": "within current owner grant", **decision,
                    "decision_signature": self._sign(decision)}
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            return {"allowed": False, "reason": str(exc)}

    def execute(self, grant, action):
        with self.lock:
            check = self.validate(grant, action, ignore_budget=True)
            if not check["allowed"]:
                return self._blocked(action, check["reason"])
            key = check["goal_id"] + ":" + action["id"]
            prior = self.state["receipts"].get(key)
            if prior:
                if prior["action_hash"] != check["action_hash"]:
                    return self._blocked(action, "idempotency id reused with different action parameters")
                result = copy.deepcopy(prior["receipt"])
                result["replayed"] = True
                return result
            check = self.validate(grant, action)
            if not check["allowed"]:
                return self._blocked(action, check["reason"])
            goal = self.state["goals"][check["goal_id"]]
            if goal["id"] in self.active_actions:
                return self._blocked(action, "another action for this goal is in flight")
            goal["actions_used"] += 1
            pending = {"id": uuid.uuid4().hex, "action_id": action["id"], "status": "UNKNOWN",
                       "evidence": {"reason": "dispatch reserved; outcome not yet durably recorded; inspect before retry"},
                       "elapsed": 0.0, "side_effects": []}
            self.state["receipts"][key] = {"action_hash": check["action_hash"], "receipt": pending}
            self.audit("dispatch", goal_id=goal["id"], action_id=action["id"], action_hash=check["action_hash"], version=goal["version"])
            self.save()  # Reserve budget/idempotency durably BEFORE side effects.
            cancelled = self.cancellations.setdefault(goal["id"], threading.Event())
            self.active_actions.add(goal["id"])
            deadline = min(goal["deadline"], grant["payload"]["expires_at"])
            if action["tool"] != "code.run":
                # Short built-in operations complete under the same lock as grant
                # revalidation. Revocation prevents every later dispatch.
                receipt = self.executor.execute(action, cancelled=cancelled.is_set, deadline=deadline)
                self.active_actions.discard(goal["id"])
                self.state["receipts"][key]["receipt"] = receipt
                self.save()
                return receipt
        # Code is cancellable while the owner endpoint remains responsive.
        try:
            receipt = self.executor.execute(action, cancelled=cancelled.is_set, deadline=deadline)
        finally:
            with self.lock:
                self.active_actions.discard(check["goal_id"])
        with self.lock:
            self.state["receipts"][key]["receipt"] = receipt
            self.save()
        return receipt

    def owner(self, op, args):
        if op == "ping":
            return {"pid": os.getpid(), "ready": True}
        if op == "close":
            for event in self.cancellations.values():
                event.set()
            self.stop.set()
            return {"closed": True}
        if op == "create_goal":
            instruction = args["instruction"]
            if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 100000:
                raise ValueError("goal instruction must be nonempty bounded text")
            max_actions, max_seconds = int(args.get("max_actions", 12)), float(args.get("max_seconds", 300))
            if not 1 <= max_actions <= 1000 or not 1 <= max_seconds <= 86400:
                raise ValueError("goal budget out of range")
            goal = {"id": uuid.uuid4().hex, "instruction": instruction,
                    "completion": args.get("completion") or {"type": "answer"},
                    "priority": float(args.get("priority", 1)), "max_actions": max_actions,
                    "actions_used": 0, "max_seconds": max_seconds,
                    "training_consent": args.get("training_consent") is True,
                    "status": "active", "version": 1, "created_at": time.time(),
                    "deadline": time.time() + max_seconds,
                    "parent_id": args.get("parent_id"), "depends_on": args.get("depends_on", [])}
            for gid in ([goal["parent_id"]] if goal["parent_id"] else []) + goal["depends_on"]:
                if gid not in self.state["goals"]:
                    raise ValueError("unknown goal relationship")
            self.state["goals"][goal["id"]] = goal
            self.audit("owner_create_goal", goal_id=goal["id"])
            self.save()
            return goal
        if op == "list_goals":
            return list(self.state["goals"].values())
        goal = self.state["goals"][args["goal_id"]]
        if op == "goal":
            return goal
        if op == "grant":
            if goal["status"] != "active":
                raise ValueError("resume goal before granting authority")
            granted_tools = args.get("tools")
            if granted_tools is None:
                granted_tools = sorted(TOOL_NAMES)
            if not isinstance(granted_tools, list) or not set(granted_tools).issubset(TOOL_NAMES):
                raise ValueError("unsupported tool scope")
            scope = args.get("scope", "")
            if scope:
                path_parts(scope)
            ttl = max(0.01, min(float(args.get("ttl", goal["max_seconds"])), 86400))
            payload = {"goal_id": goal["id"], "version": goal["version"], "tools": granted_tools,
                       "scope": scope, "expires_at": min(time.time() + ttl, goal["deadline"]),
                       "nonce": secrets.token_hex(16)}
            self.audit("owner_grant", goal_id=goal["id"], version=goal["version"], tools=granted_tools, scope=scope)
            self.save()
            return {"payload": payload, "signature": self._sign(payload)}
        if op in {"revoke", "update_goal"}:
            changes = args.get("changes", {}) if op == "update_goal" else {"status": "revoked"}
            allowed = {"instruction", "completion", "priority", "max_actions", "max_seconds", "training_consent", "status"}
            if not set(changes).issubset(allowed):
                raise ValueError("unsupported owner goal update")
            if "status" in changes and changes["status"] not in {"active", "paused", "completed", "blocked", "revoked"}:
                raise ValueError("unknown goal status")
            if "max_actions" in changes and not 1 <= int(changes["max_actions"]) <= 1000:
                raise ValueError("action budget out of range")
            if "max_seconds" in changes and not 1 <= float(changes["max_seconds"]) <= 86400:
                raise ValueError("time budget out of range")
            self.cancellations.setdefault(goal["id"], threading.Event()).set()
            goal.update(changes)
            goal["version"] += 1
            if goal["status"] == "active":
                goal["deadline"] = time.time() + float(goal["max_seconds"])
                self.cancellations[goal["id"]] = threading.Event()
            self.audit("owner_" + op, goal_id=goal["id"], version=goal["version"], changes=changes)
            self.save()
            return goal
        raise ValueError("unknown owner operation")


def _serve(address, state_dir, workspace, owner_token):
    state = _AuthorityState(state_dir, workspace, owner_token)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(address)
    os.chmod(address, 0o600)
    server.listen(8)
    server.settimeout(0.2)
    workers = []
    def handle(connection):
        with connection:
            try:
                connection.settimeout(45)
                request = _recv(connection)
                op = request.get("op")
                if op == "execute":
                    result = state.execute(request.get("grant"), request.get("action"))
                elif op == "check":
                    with state.lock:
                        result = state.validate(request.get("grant"), request.get("action"))
                else:
                    token = request.get("owner_token", "")
                    if not isinstance(token, str) or not hmac.compare_digest(token, owner_token):
                        raise PermissionError("authenticated owner endpoint required")
                    with state.lock:
                        result = state.owner(op, request.get("args", {}))
                response = {"result": result}
            except Exception as exc:
                response = {"rpc_error": type(exc).__name__ + ": " + str(exc)}
            try:
                connection.sendall(canonical(response) + b"\n")
            except (OSError, ValueError):
                pass
    try:
        while not state.stop.is_set():
            try:
                connection, _ = server.accept()
            except socket.timeout:
                continue
            worker = threading.Thread(target=handle, args=(connection,), daemon=True)
            workers = [t for t in workers if t.is_alive()]
            if len(workers) >= 16:
                connection.close()
                continue
            workers.append(worker)
            worker.start()
    finally:
        server.close()
        for event in state.cancellations.values():
            event.set()
        for worker in workers:
            worker.join(timeout=2)
        state.executor.close()


class AuthorityService:
    """Trusted owner handle. Pass `.action_client()` to the cognitive runtime."""
    def __init__(self, state_dir, workspace):
        state_dir, workspace = Path(state_dir).resolve(), Path(workspace).resolve()
        if state_dir == workspace or state_dir in workspace.parents or workspace in state_dir.parents:
            raise ValueError("state and workspace must be disjoint directories")
        state_dir.mkdir(parents=True, exist_ok=True)
        workspace.mkdir(parents=True, exist_ok=True)
        self._socket_dir = tempfile.mkdtemp(prefix="sage-authority-")
        self._address = str(Path(self._socket_dir, "service.sock"))
        self._token = secrets.token_hex(32)
        self._closed = False
        context = multiprocessing.get_context("spawn")
        self._process = context.Process(target=_serve, args=(self._address, str(state_dir), str(workspace), self._token), daemon=True)
        self._process.start()
        started = time.monotonic()
        while time.monotonic() - started < 15:
            if not self._process.is_alive():
                self.close()
                raise RuntimeError("authority process failed to start")
            if Path(self._address).exists():
                try:
                    self._owner("ping")
                    break
                except (ConnectionRefusedError, FileNotFoundError):
                    # bind() creates the path just before listen() becomes ready.
                    pass
            time.sleep(0.025)
        else:
            self.close()
            raise RuntimeError("authority startup deadline exceeded")

    def _owner(self, op, **args):
        return _rpc(self._address, {"op": op, "owner_token": self._token, "args": args})

    def action_client(self):
        return ActionClient(self._address)

    def create_goal(self, instruction, completion=None, max_actions=12, max_seconds=300, training_consent=False, **kwargs):
        return self._owner("create_goal", instruction=instruction, completion=completion, max_actions=max_actions,
                           max_seconds=max_seconds, training_consent=training_consent, **kwargs)

    def grant(self, goal_id, tools=None, **kwargs):
        return self._owner("grant", goal_id=goal_id, tools=tools, **kwargs)

    def goal(self, goal_id):
        return self._owner("goal", goal_id=goal_id)

    def goals(self):
        return self._owner("list_goals")

    def update_goal(self, goal_id, **changes):
        return self._owner("update_goal", goal_id=goal_id, changes=changes)

    def revoke(self, goal_id):
        return self._owner("revoke", goal_id=goal_id)

    def check(self, grant, action):
        return self.action_client().check(grant, action)

    def execute(self, grant, action):
        return self.action_client().execute(grant, action)

    def close(self):
        if getattr(self, "_closed", True):
            return
        self._closed = True
        if getattr(self, "_process", None):
            if self._process.is_alive() and Path(self._address).exists():
                try:
                    self._owner("close")
                except (OSError, RuntimeError):
                    pass
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=2)
        shutil.rmtree(self._socket_dir, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
