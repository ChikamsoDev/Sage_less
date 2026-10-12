"""Fail-closed Linux Python sandbox; no network, subprocesses or host file access.

The trusted host/kernel is outside the threat model. Landlock restricts filesystem
access, seccomp restricts syscalls, and resource limits bound each separate process.
This is a small stdlib execution tool, not a container or an arbitrary package VM.
"""
from __future__ import annotations

import ctypes
import errno
import json
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import tempfile
import time


class SandboxUnavailable(RuntimeError):
    pass


def _landlock(root: str) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    abi = libc.syscall(444, 0, 0, 1)
    if abi < 1:
        raise SandboxUnavailable("Linux Landlock is unavailable or disabled")

    class Ruleset(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64)]

    class PathRule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]

    handled = (1 << 13) - 1
    if abi >= 2:
        handled |= 1 << 13  # REFER
    if abi >= 3:
        handled |= 1 << 14  # TRUNCATE
    rule = Ruleset(handled)
    fd = libc.syscall(444, ctypes.byref(rule), ctypes.sizeof(rule), 0)
    if fd < 0:
        raise SandboxUnavailable("Cannot create Landlock ruleset")
    try:
        # Python libraries are read-only. No /proc, /home, /content or Drive.
        read_only = (1 << 2) | (1 << 3)
        candidates = {str(Path(os.__file__).parent), "/usr/lib", "/lib", "/lib64"}
        for path, access in [(p, read_only) for p in candidates] + [(root, handled & ~1)]:
            if not os.path.exists(path):
                continue
            pfd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                attr = PathRule(access, pfd)
                if libc.syscall(445, fd, 1, ctypes.byref(attr), 0) != 0:
                    raise SandboxUnavailable("Cannot install Landlock path rule")
            finally:
                os.close(pfd)
        if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
            raise SandboxUnavailable("Cannot set no_new_privs")
        if libc.syscall(446, fd, 0) != 0:
            raise SandboxUnavailable("Cannot enforce Landlock")
    finally:
        os.close(fd)


def _seccomp() -> None:
    if platform.machine() not in {"x86_64", "AMD64"}:
        raise SandboxUnavailable("seccomp allowlist currently supports x86_64 only")
    libc = ctypes.CDLL(None, use_errno=True)

    class Filter(ctypes.Structure):
        _fields_ = [("code", ctypes.c_ushort), ("jt", ctypes.c_ubyte),
                    ("jf", ctypes.c_ubyte), ("k", ctypes.c_uint32)]

    class Program(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(Filter))]

    # Minimal CPython filesystem/memory/time process-local operations. In
    # particular: no exec, fork/clone, sockets, signals to other processes,
    # ptrace, process_vm, mount, namespace changes, bpf or io_uring.
    allowed = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20,
        21, 23, 24, 25, 28, 32, 33, 35, 39, 60, 63, 72, 74, 75, 77, 78, 79,
        80, 81, 82, 83, 84, 85, 86, 87, 89, 95, 96, 97, 98, 99,
        102, 104, 107, 108, 110, 158, 186, 202, 204, 217, 218, 228, 230,
        231, 257, 258, 262, 263, 264, 265, 267, 269, 270, 271, 273, 292, 302,
        318, 332, 334,
    }
    # Architecture check also rejects the x32 ABI via unmatched syscall numbers.
    ins = [(0x20, 0, 0, 4), (0x15, 1, 0, 0xC000003E), (0x06, 0, 0, 0x80000000),
           (0x20, 0, 0, 0)]
    for nr in sorted(allowed):
        ins += [(0x15, 0, 1, nr), (0x06, 0, 0, 0x7FFF0000)]
    ins += [(0x06, 0, 0, 0x00050000 | errno.EPERM)]
    arr = (Filter * len(ins))(*(Filter(*x) for x in ins))
    prog = Program(len(ins), arr)
    if libc.prctl(22, 2, ctypes.byref(prog), 0, 0) != 0:
        raise SandboxUnavailable("Cannot enforce seccomp")


def _child() -> int:
    request = json.loads(sys.stdin.buffer.read(200_000))
    root = os.getcwd()
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (3, 4))
        resource.setrlimit(resource.RLIMIT_AS, (256 * 1024**2, 256 * 1024**2))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1024**2, 1024**2))
        resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        # Warm required trusted modules before filesystem restrictions.
        import math, json as _json, csv, random, statistics, decimal, fractions
        if os.geteuid() == 0:
            # Colab runs root. Irreversibly discard root before executing code.
            os.chown(root, 65534, 65534)
            os.setgroups([])
            os.setgid(65534)
            os.setuid(65534)
        _landlock(root)
        _seccomp()
    except BaseException as exc:
        print("SAGE_SANDBOX_UNAVAILABLE: " + str(exc), file=sys.stderr)
        return 125
    try:
        exec(compile(request["code"], "<sandbox>", "exec"), {"__name__": "__main__"})
        return 0
    except BaseException as exc:
        print(type(exc).__name__ + ": " + str(exc), file=sys.stderr)
        return 1


def run_python(code: str, *, inputs: dict[str, bytes] | None = None,
               timeout: float = 10, cancelled=None) -> dict:
    """Execute in isolated scratch space; return bounded stdout and text artifacts."""
    if sys.platform != "linux" or len(code.encode()) > 100_000:
        raise SandboxUnavailable("Linux and <=100KB source are required")
    timeout = max(0.1, min(float(timeout), 30.0))
    with tempfile.TemporaryDirectory(prefix="sage-sandbox-") as temp:
        root = Path(temp)
        scratch = root / "work"
        scratch.mkdir(mode=0o755)
        # Root child drops to nobody; ancestors must permit traversal.
        if os.geteuid() == 0:
            root.chmod(0o755)
        for name, data in (inputs or {}).items():
            parts = Path(name).parts
            if not parts or Path(name).is_absolute() or any(x in {".", ".."} for x in parts):
                raise ValueError("unsafe input path")
            dest = scratch.joinpath(*parts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            dest.chmod(0o444)
        outpath, errpath = root / "stdout", root / "stderr"
        with outpath.open("wb") as out, errpath.open("wb") as err:
            proc = subprocess.Popen(
                [sys.executable, "-I", "-S", str(Path(__file__).resolve()), "--child"],
                cwd=scratch, stdin=subprocess.PIPE, stdout=out, stderr=err,
                close_fds=True, start_new_session=True,
                env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0", "LANG": "C.UTF-8"},
            )
            proc.stdin.write(json.dumps({"code": code}).encode())
            proc.stdin.close()
            started = time.monotonic()
            stopped = None
            while proc.poll() is None:
                if cancelled and cancelled():
                    stopped = "revoked"
                elif time.monotonic() - started >= timeout:
                    stopped = "deadline"
                if stopped:
                    os.killpg(proc.pid, signal.SIGKILL)
                    break
                time.sleep(0.025)
            proc.wait()
        stdout = outpath.read_bytes()[:65536].decode("utf-8", "replace")
        stderr = errpath.read_bytes()[:16384].decode("utf-8", "replace")
        if proc.returncode == 125:
            raise SandboxUnavailable(stderr.strip())
        artifacts = []
        import hashlib
        for parent, dirs, files in os.walk(scratch, followlinks=False):
            dirs[:] = sorted(d for d in dirs if not Path(parent, d).is_symlink())[:32]
            for name in sorted(files)[:32]:
                path = Path(parent, name)
                if path.is_symlink() or len(artifacts) >= 16:
                    continue
                try:
                    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                    with os.fdopen(fd, "rb") as f:
                        import stat
                        st = os.fstat(f.fileno())
                        if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1 or st.st_size > 65536:
                            continue
                        data = f.read(65537)
                    artifacts.append({"path": str(path.relative_to(scratch)), "size": len(data),
                                      "sha256": hashlib.sha256(data).hexdigest(),
                                      "text": data.decode("utf-8", "replace")})
                except OSError:
                    continue
        return {"exit_code": proc.returncode, "stdout": stdout, "stderr": stderr,
                "stopped": stopped, "artifacts": artifacts,
                "isolation": "Linux Landlock + seccomp + rlimits; separate scratch process"}


def probe() -> dict:
    try:
        result = run_python("print('sandbox-ready')", timeout=5)
        return {"available": result["exit_code"] == 0,
                "detail": result["stdout"] or result["stderr"]}
    except Exception as exc:
        return {"available": False, "detail": str(exc)}


if __name__ == "__main__" and "--child" in sys.argv:
    raise SystemExit(_child())
