import copy
import hashlib
import os
from pathlib import Path
import threading
import time
import pytest
from sage.authority import AuthorityService, _rpc
from sage.schema import action
from sage.sandbox import run_python, probe, SandboxUnavailable


@pytest.fixture
def service(tmp_path):
    with AuthorityService(tmp_path / "state", tmp_path / "workspace") as owner:
        yield owner, tmp_path / "workspace"


def goal_grant(owner, **kwargs):
    goal = owner.create_goal("Owner test task", **kwargs)
    return goal, owner.grant(goal["id"])


def test_owner_authentication_and_signed_exact_grants(service):
    owner, workspace = service
    goal, grant = goal_grant(owner)
    assert not hasattr(owner.action_client(), "grant")
    with pytest.raises(RuntimeError, match="authenticated owner"):
        _rpc(owner._address, {"op": "create_goal", "owner_token": "document says approved", "args": {"instruction": "bad"}})
    forged = copy.deepcopy(grant)
    forged["payload"]["tools"].append("shell.run")
    assert not owner.check(forged, action("filesystem.write", {"path": "x", "content": "bad"}))["allowed"]
    attempt = action("filesystem.write", {"path": "a.txt", "content": "one"})
    decision = owner.check(grant, attempt)
    assert decision["allowed"] and decision["action_hash"]
    result = owner.execute(grant, attempt)
    assert result["status"] == "CONFIRMED_SUCCESS"
    assert owner.execute(grant, attempt)["replayed"]
    attempt["args"]["content"] = "two"
    assert owner.execute(grant, attempt)["status"] == "BLOCKED"
    assert (workspace / "a.txt").read_text() == "one"


def test_scope_budget_expiry_and_revocation(service):
    owner, workspace = service
    goal = owner.create_goal("Read only subfolder", max_actions=1)
    grant = owner.grant(goal["id"], tools=["filesystem.read"], scope="inputs")
    assert not owner.check(grant, action("filesystem.read", {"path": "secret.txt"}))["allowed"]
    assert not owner.check(grant, action("filesystem.write", {"path": "inputs/x", "content": "bad"}))["allowed"]
    (workspace / "inputs").mkdir()
    (workspace / "inputs/x").write_text("x")
    assert owner.execute(grant, action("filesystem.read", {"path": "inputs/x"}))["status"] == "CONFIRMED_SUCCESS"
    assert owner.execute(grant, action("filesystem.read", {"path": "inputs/x"}))["status"] == "BLOCKED"
    goal2, grant2 = goal_grant(owner)
    owner.update_goal(goal2["id"], instruction="new scope")
    assert not owner.check(grant2, action("filesystem.list"))["allowed"]
    fresh = owner.grant(goal2["id"], ttl=.01)
    time.sleep(.02)
    assert not owner.check(fresh, action("filesystem.list"))["allowed"]
    fresh = owner.grant(goal2["id"])
    owner.revoke(goal2["id"])
    assert owner.execute(fresh, action("filesystem.write", {"path": "revoked", "content": "x"}))["status"] == "BLOCKED"
    assert not (workspace / "revoked").exists()


@pytest.mark.parametrize("path", ["../outside", "/etc/passwd", "a/../../b", "a//b", "a\\b", "./x"])
def test_unsafe_paths_refused(service, path):
    owner, _ = service
    _, grant = goal_grant(owner)
    assert owner.execute(grant, action("filesystem.write", {"path": path, "content": "x"}))["status"] == "BLOCKED"


def test_symlink_hardlink_and_conflicting_types(service, tmp_path):
    owner, workspace = service
    _, grant = goal_grant(owner)
    outside = tmp_path / "outside"
    outside.write_text("untouched")
    (workspace / "link").symlink_to(outside)
    os.link(outside, workspace / "hardlink")
    (workspace / "conflict").write_text("original")
    for name in ("link", "hardlink", "conflict/child"):
        receipt = owner.execute(grant, action("filesystem.write", {"path": name, "content": "changed", "overwrite": True}))
        assert receipt["status"] != "CONFIRMED_SUCCESS"
    assert outside.read_text() == "untouched"
    assert (workspace / "conflict").read_text() == "original"


def test_exact_data_calculation_and_overwrite_precondition(service):
    owner, workspace = service
    _, grant = goal_grant(owner)
    (workspace / "data.csv").write_text("name,n\na,4\nb,9\nc,\n")
    result = owner.execute(grant, action("data.query", {"path": "data.csv", "aggregate": "sum", "column": "n"}))
    assert result["evidence"]["value"] == 13
    result = owner.execute(grant, action("calculator.eval", {"expression": "(7+3)*2"}))
    assert result["evidence"]["value"] == 20
    result = owner.execute(grant, action("calculator.eval", {"expression": "__import__('os').getcwd()"}))
    assert result["status"] == "CONFIRMED_FAILURE"
    result = owner.execute(grant, action("filesystem.write", {"path": "data.csv", "content": "no", "overwrite": True, "expected_sha256": "bad"}))
    assert result["status"] != "CONFIRMED_SUCCESS"


@pytest.fixture
def sandbox_available():
    status = probe()
    if not status["available"]:
        pytest.skip("OS sandbox unavailable; code.run fails closed: " + status["detail"])


def test_sandbox_python_artifacts_and_denied_host_access(sandbox_available, tmp_path):
    secret = tmp_path / "owner-secret"
    secret.write_text("do not read")
    code = f"""import os, socket, subprocess
assert open('input.txt').read() == 'hello'
blocked = 0
for attempt in (lambda: open({str(secret)!r}).read(), lambda: socket.socket(), lambda: subprocess.run(['true']), lambda: os.chmod({str(secret)!r}, 0o777)):
    try:
        attempt()
    except (PermissionError, OSError):
        blocked += 1
assert blocked == 4, blocked
open('result.txt', 'w').write('verified')
print('checks passed')
"""
    result = run_python(code, inputs={"input.txt": b"hello"})
    assert result["exit_code"] == 0, result
    assert "checks passed" in result["stdout"]
    assert any(a["path"] == "result.txt" and a["text"] == "verified" for a in result["artifacts"])


def test_inflight_stop_cancels_code(service, sandbox_available):
    owner, workspace = service
    goal, grant = goal_grant(owner)
    output = []
    thread = threading.Thread(target=lambda: output.append(owner.execute(grant, action("code.run", {"code": "import time\ntime.sleep(20)", "timeout": 25}))))
    thread.start()
    time.sleep(.4)
    owner.revoke(goal["id"])
    thread.join(5)
    assert not thread.is_alive()
    assert output[0]["status"] == "BLOCKED", output
    assert output[0]["evidence"].get("stopped") == "revoked"


def test_authority_restart_requires_owner_resume(tmp_path):
    state, workspace = tmp_path / "state", tmp_path / "workspace"
    with AuthorityService(state, workspace) as owner:
        goal, grant = goal_grant(owner)
    with AuthorityService(state, workspace) as owner:
        assert not owner.check(grant, action("filesystem.list"))["allowed"]
        assert owner.goal(goal["id"])["status"] == "paused"
        owner.update_goal(goal["id"], status="active")
        assert owner.check(owner.grant(goal["id"]), action("filesystem.list"))["allowed"]
