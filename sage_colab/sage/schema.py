"""Small, versioned JSON contracts shared by the acting loop and persistence."""
from __future__ import annotations
import hashlib
import json
import os
import time
import uuid
from pathlib import Path

SCHEMA_VERSION = 1
STATUSES = {"CONFIRMED_SUCCESS", "CONFIRMED_FAILURE", "BLOCKED", "UNKNOWN"}


def uid() -> str:
    return uuid.uuid4().hex


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uid() + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(canonical(value))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def action(tool, args=None, **extra):
    return {"id": uid(), "tool": tool, "args": args or {}, **extra}


def observation(goal_id, step, evidence, missing=None):
    return {"id": uid(), "timestamp": time.time(), "source_env": "workspace",
            "episode_id": goal_id, "step": step, "entities": evidence,
            "observed_fields": list(evidence), "missing_fields": missing or [],
            "evidence_class": "OBSERVED", "provenance": "live", "schema_version": 1}


def validate_proposal(value):
    """Untrusted model output becomes data, never executable Python or authority."""
    if not isinstance(value, dict):
        raise ValueError("proposal must be an object")
    proposals = value.get("candidates", [])
    if not isinstance(proposals, list) or len(proposals) > 6:
        raise ValueError("expected at most six candidate actions")
    cleaned = []
    for p in proposals:
        if not isinstance(p, dict) or not isinstance(p.get("tool"), str) or not isinstance(p.get("args", {}), dict):
            raise ValueError("invalid typed action")
        if len(canonical(p)) > 100_000:
            raise ValueError("action too large")
        cleaned.append(action(p["tool"], p.get("args", {}), reason=str(p.get("reason", ""))[:500],
                              expected=p.get("expected", {})))
    return {"candidates": cleaned, "answer": str(value.get("answer", ""))[:16000],
            "plan": value.get("plan", [])[:8] if isinstance(value.get("plan", []), list) else [],
            "blocked": str(value.get("blocked", ""))[:2000]}
