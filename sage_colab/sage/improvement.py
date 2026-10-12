"""Bounded planner-configuration experiments, isolated from release authority.

Candidate space is strictly numeric configuration, never executable code. A
trusted subprocess evaluates it; candidate inputs cannot name evaluators, tests,
filesystem locations, credentials or arbitrary Python. The notebook owner is
trusted and can inspect/approve exact bytes. This is not a hostile-host boundary.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import random
import secrets
import subprocess
import sys
import time
import uuid

DEFAULT_CONFIG = {"success_weight": .5, "completion_weight": 1.0, "uncertainty_weight": .3,
                  "delay_weight": .05, "policy_weight": .15, "retrieval_limit": 5, "beam_width": 3}
LIMITS = {"success_weight": (0., 2.), "completion_weight": (.1, 3.), "uncertainty_weight": (0., 2.),
          "delay_weight": (0., .5), "policy_weight": (0., 1.), "retrieval_limit": (1, 10), "beam_width": (1, 6)}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as f:
        f.write(_json(value))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def validate_config(config):
    if not isinstance(config, dict) or set(config) != set(DEFAULT_CONFIG):
        raise ValueError("Only the fixed planner configuration keys may change")
    result = {}
    for key, (low, high) in LIMITS.items():
        v = config[key]
        if isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or not low <= v <= high:
            raise ValueError("Invalid planner coefficient: " + key)
        if key in {"retrieval_limit", "beam_width"} and int(v) != v:
            raise ValueError("Integer configuration required")
        result[key] = v
    return result


def active_config(state_dir):
    path = Path(state_dir) / "improvement" / "active.json"
    if not path.exists():
        return dict(DEFAULT_CONFIG)
    record = json.loads(path.read_text())
    if record.get("config_hash") != _hash(record["config"]):
        raise ValueError("Active planner configuration integrity failure")
    return validate_config(record["config"])


def score_action(prediction, config, policy_score=0, action=None):
    c = validate_config(config)
    def bounded(key, default=0):
        value = float(prediction.get(key, default))
        return min(1., max(0., value)) if math.isfinite(value) else default
    delay = float(prediction.get("delay", 0))
    delay = min(30., max(0., delay)) if math.isfinite(delay) else 30.
    policy = float(policy_score)
    policy = min(1., max(-1., policy)) if math.isfinite(policy) else 0.
    return (c["success_weight"] * bounded("success_probability") + c["completion_weight"] * bounded("completion_probability")
            - c["uncertainty_weight"] * bounded("uncertainty", 1.) - c["delay_weight"] * delay
            + c["policy_weight"] * policy)


def _evaluate(config, baseline, seeds):
    """Protected labels generated only in the evaluator, not in candidate input.

    A small ranking diagnostic, deliberately not presented as an agent benchmark.
    """
    scores = {"candidate": [], "baseline": []}
    families = {}
    for family in ("delay", "uncertainty", "cost"):
        paired = []
        for seed in seeds:
            rng = random.Random(seed + sum(map(ord, family)))
            for _ in range(40):
                choices = []
                for _ in range(4):
                    success, progress, uncertainty = rng.random(), rng.random(), rng.random()
                    delay = rng.uniform(0, 4)
                    uncertainty_cost = .75 if family == "uncertainty" else .3
                    delay_cost = .15 if family == "delay" else .05
                    observed = progress * success - uncertainty_cost * uncertainty - delay_cost * delay
                    prediction = {"success_probability": success, "completion_probability": progress,
                                  "uncertainty": uncertainty, "delay": delay}
                    choices.append((prediction, observed))
                values = {}
                for name, cfg in (("candidate", config), ("baseline", baseline)):
                    index = max(range(len(choices)), key=lambda i: score_action(choices[i][0], cfg))
                    values[name] = choices[index][1]
                    scores[name].append(values[name])
                paired.append(values["candidate"] - values["baseline"])
        families[family] = sum(paired) / len(paired)
    deltas = [a - b for a, b in zip(scores["candidate"], scores["baseline"])]
    mean = sum(deltas) / len(deltas)
    variance = sum((x - mean)**2 for x in deltas) / max(1, len(deltas) - 1)
    halfwidth = 1.96 * math.sqrt(variance / len(deltas))
    return {"mean_paired_gain": mean, "approximate_95_percent_interval": [mean-halfwidth, mean+halfwidth],
            "by_family": families, "tasks": len(deltas), "unauthorized_effects": 0,
            "interpretation": "Synthetic ranking utility, not task completion; normal approximation, correlated-generator limits apply"}


class ImprovementManager:
    def __init__(self, state_dir):
        self.state_dir = Path(state_dir)
        self.root = self.state_dir / "improvement"
        self.archive = self.root / "archive"
        self.archive.mkdir(parents=True, exist_ok=True)

    def propose(self, failures):
        causes = {}
        for item in failures:
            cause = str(item.get("cause", item.get("status", "unknown")))
            causes[cause] = causes.get(cause, 0) + 1
        config = active_config(self.state_dir)
        config["uncertainty_weight"] = min(2., config["uncertainty_weight"] + .25)
        config["delay_weight"] = min(.5, config["delay_weight"] + .025)
        candidate = {"id": uuid.uuid4().hex[:16], "config": config, "failure_clusters": causes,
                     "hypothesis": "Increasing uncertainty and delay penalties reduces costly low-confidence choices",
                     "acceptance": {"minimum_mean_gain": .01, "maximum_family_regression": .02,
                                    "require_positive_interval": True}, "budget": {"evaluation_seconds": 30, "seeds": 3},
                     "kind": "planner_configuration", "created_at": time.time()}
        _write(self.archive / candidate["id"] / "proposal.json", candidate)
        return candidate

    def run(self, candidate):
        identifier = candidate.get("id", "")
        if len(identifier) != 16 or any(c not in "0123456789abcdef" for c in identifier):
            raise ValueError("Invalid candidate id")
        directory = self.archive / identifier
        original = json.loads((directory / "proposal.json").read_text())
        if original != candidate:
            raise ValueError("Candidate changed after preregistration")
        if (directory / "report.json").exists():
            return json.loads((directory / "report.json").read_text())
        config = validate_config(candidate["config"])
        baseline = active_config(self.state_dir)
        evaluator_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        # No final-task outcomes are passed to the candidate proposer.
        seeds = [secrets.randbelow(2**30) for _ in range(3)]
        request = {"config": config, "baseline": baseline, "seeds": seeds}
        _write(directory / "evaluation_input.json", request)
        start = time.monotonic()
        command = [sys.executable, "-I", str(Path(__file__).resolve()), "--evaluate",
                   str((directory / "evaluation_input.json").resolve()), str((directory / "evaluation_output.json").resolve())]
        env = {key: os.environ[key] for key in ("PATH", "LANG", "SYSTEMROOT") if key in os.environ}
        completed = subprocess.run(command, cwd=directory, env=env, capture_output=True, text=True, timeout=30)
        if completed.returncode:
            raise RuntimeError("Isolated evaluator failed: " + completed.stderr[-2000:])
        metrics = json.loads((directory / "evaluation_output.json").read_text())
        gate = candidate["acceptance"]
        passed = (metrics["mean_paired_gain"] >= gate["minimum_mean_gain"] and
                  metrics["approximate_95_percent_interval"][0] > 0 and
                  min(metrics["by_family"].values()) >= -gate["maximum_family_regression"])
        report = {"candidate_id": identifier, "status": "awaiting_owner_approval" if passed else "rejected_no_replicated_gain",
                  "passes_gates": passed, "promoted": False, "metrics": metrics, "seconds": time.monotonic()-start,
                  "config": config, "config_hash": _hash(config), "baseline_hash": _hash(baseline),
                  "proposal_hash": _hash(candidate), "evaluator_hash": evaluator_hash,
                  "seed_manifest_hash": _hash(seeds), "diff": {k: {"before": baseline[k], "after": v} for k, v in config.items() if v != baseline[k]},
                  "limitations": ["Configuration-only search; no arbitrary self-written code is executed.",
                                  "Evaluator is a fixed trusted process but its source is public to the notebook owner.",
                                  "Synthetic ranking gains require live-task replication before any capability claim."]}
        _write(directory / "report.json", report)
        return report

    def promote(self, candidate_id, approved=False):
        if not approved:
            raise PermissionError("An explicit owner release decision is required")
        if not isinstance(candidate_id, str) or len(candidate_id) != 16 or any(c not in "0123456789abcdef" for c in candidate_id):
            raise ValueError("Invalid candidate id")
        directory = self.archive / candidate_id
        report = json.loads((directory / "report.json").read_text())
        proposal = json.loads((directory / "proposal.json").read_text())
        if not report["passes_gates"] or report["proposal_hash"] != _hash(proposal) or report["config_hash"] != _hash(report["config"]):
            raise ValueError("Candidate failed gates or changed after evaluation")
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != report["evaluator_hash"]:
            raise ValueError("Evaluator changed after the candidate was tested")
        if _hash(active_config(self.state_dir)) != report["baseline_hash"]:
            raise ValueError("Active baseline changed; evaluate a new candidate")
        previous = {"config": active_config(self.state_dir), "config_hash": _hash(active_config(self.state_dir))}
        _write(self.root / "previous.json", previous)
        release = {"candidate_id": candidate_id, "config": validate_config(report["config"]), "config_hash": report["config_hash"],
                   "evaluation_hash": _hash(report), "released_at": time.time(), "owner_approved": True}
        _write(self.root / "active.json", release)
        return release

    def rollback(self, approved=False):
        if not approved:
            raise PermissionError("Owner approval is required")
        previous = json.loads((self.root / "previous.json").read_text())
        if previous["config_hash"] != _hash(previous["config"]):
            raise ValueError("Rollback integrity failure")
        _write(self.root / "active.json", previous)
        return {"status": "rolled_back", **previous}


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--evaluate":
        raise SystemExit("Internal evaluator only")
    item = json.loads(Path(sys.argv[2]).read_text())
    _write(Path(sys.argv[3]), _evaluate(validate_config(item["config"]), validate_config(item["baseline"]), item["seeds"]))
