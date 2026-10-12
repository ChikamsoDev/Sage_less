"""Resettable, mechanically checked digital environments for small-scale learning.

Adapters implement mechanics, not a goal-solving policy. Labels are computed from
actual cloned simulator transitions and are kept outside observations. A simulated
observation is never evidence that an operation happened in the real workspace.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass

GENERATOR_VERSION = "digital-curriculum-v2"
TRAIN_FAMILIES = ("files", "inventory", "workflow")
HOLDOUT_FAMILY = "orders"


@dataclass
class DigitalEnvironment:
    family: str
    seed: int
    variant: int = 1

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        self.step_count = 0
        self.revoked = False
        self.complete = False
        self.known = False
        self.pending = 0
        self.committed = False
        self.duplicates = 0
        self.latency = self.rng.randint(0, 2 if self.variant == 1 else 4)
        self.allowed = self.rng.random() > .12
        self.dependency_ready = self.rng.random() > .5
        self.prerequisites_remaining = 0 if self.dependency_ready else self.rng.randint(1, 3)
        self.stock = self.rng.randint(0, 3)
        self.required = self.rng.randint(1, 3)
        self.total_cost = 0.0
        self.retrieved = False
        self.last_status = "UNKNOWN"
        self.goal_name = f"target_{self.rng.randrange(100000)}"

    def observe(self):
        facts = {"permission": self.allowed and not self.revoked,
                 "goal_verified": self.complete, "last_status": self.last_status}
        if self.known:
            facts.update({"dependency_ready": self.dependency_ready, "stock": self.stock,
                          "prerequisites_remaining": self.prerequisites_remaining,
                          "required": self.required, "committed": self.committed,
                          "pending": self.pending})
        return {"schema_version": 1, "source_env": self.family,
                "episode_id": f"{self.family}-{self.seed}-{self.variant}", "step": self.step_count,
                "entities": {"target": self.goal_name}, "observed_fields": facts,
                "missing_fields": [] if self.known else ["dependency_ready", "stock", "committed", "pending"],
                "evidence_class": "OBSERVED", "provenance": "validated_simulation"}

    def describe_actions(self):
        operations = {
            "files": ("filesystem.write", "filesystem.mkdir"),
            "inventory": ("inventory.allocate", "inventory.restock"),
            "workflow": ("workflow.complete", "workflow.prerequisite"),
            "orders": ("orders.submit", "orders.prepare"),
        }
        act, prepare = operations[self.family]
        return [
            {"tool": "inspect", "args": {"target": self.goal_name}, "description": "inspect actual state and pending effects", "kind": "inspect"},
            {"tool": "memory.retrieve", "args": {"query": "dependency and permission"}, "description": "retrieve prerequisite information", "kind": "retrieve"},
            {"tool": act, "args": {"target": self.goal_name, "quantity": self.required}, "description": "commit goal effect requires ready dependency available stock and permission", "kind": "commit"},
            {"tool": prepare, "args": {"target": self.goal_name}, "description": "prepare prerequisite resource dependency", "kind": "prepare"},
            {"tool": "wait", "args": {}, "description": "wait for delayed pending effect", "kind": "wait"},
            {"tool": "finish", "args": {}, "description": "stop without further effects", "kind": "stop"},
        ]

    def execute(self, action):
        before = copy.deepcopy(self.__dict__)
        kind = action.get("kind", "invalid")
        self.step_count += 1
        elapsed = .05 if kind in {"inspect", "retrieve", "stop"} else .2
        self.total_cost += elapsed
        # Delayed effects only advance with observed interactions.
        if self.pending:
            self.pending -= 1
            if not self.pending:
                self.committed = True
        status, error = "CONFIRMED_SUCCESS", ""
        if kind == "stop":
            if not self.complete:
                status, error = "CONFIRMED_FAILURE", "goal_not_verified"
        elif self.revoked or not self.allowed:
            status, error = "BLOCKED", "insufficient_authority"
        elif kind == "inspect":
            self.known = True
            if self.committed:
                self.complete = True
        elif kind == "retrieve":
            self.retrieved = True
            self.known = True
        elif kind == "prepare":
            if self.family == "inventory":
                self.stock += 1
            elif self.family == "workflow":
                self.prerequisites_remaining = max(0, self.prerequisites_remaining - 1)
                self.dependency_ready = self.prerequisites_remaining == 0
            elif self.family == "orders":
                # Held-out composition: both readiness and consumable stock.
                self.dependency_ready = True
                self.stock += 1
            else:
                self.dependency_ready = True
        elif kind == "commit":
            ready = self.stock >= self.required if self.family == "inventory" else self.dependency_ready
            if self.family == "orders":
                ready = self.dependency_ready and self.stock >= self.required
            if not ready:
                status, error = "CONFIRMED_FAILURE", "missing_prerequisite"
            elif self.pending or self.committed:
                self.duplicates += 1
                status, error = "CONFIRMED_FAILURE", "duplicate_effect"
            else:
                if self.family in {"inventory", "orders"}:
                    self.stock -= self.required
                if self.latency:
                    self.pending = self.latency
                    status, error = "UNKNOWN", "acknowledgement_delayed"
                else:
                    self.committed = True
                if self.family == "workflow" and self.latency == 0:
                    self.complete = True
        elif kind != "wait":
            status, error = "CONFIRMED_FAILURE", "invalid_action"
        self.last_status = status
        changed = any(before[key] != getattr(self, key) for key in
                      ("pending", "committed", "complete", "dependency_ready", "prerequisites_remaining", "stock", "known", "retrieved"))
        receipt = {"status": status, "elapsed": elapsed, "evidence": {"error": error, "goal_verified": self.complete},
                   "side_effects": ["simulated_state_change"] if changed else []}
        return receipt

    def verify(self):
        return {"verified": self.complete, "evidence_class": "OBSERVED", "provenance": "validated_simulation"}


def _transition(env, history, action):
    trial = copy.deepcopy(env)
    receipt = trial.execute(action)
    after = trial.observe()
    kind = action["kind"]
    new_completion = trial.complete and not env.complete
    information = .22 if not env.known and trial.known else 0.0
    # Mechanically measured useful changes and costs; no LLM reward judgement.
    prerequisite = .2 if ((trial.dependency_ready and not env.dependency_ready) or trial.stock > env.stock or trial.prerequisites_remaining < env.prerequisites_remaining) else 0.0
    effect = .35 if ((trial.committed and not env.committed) or (trial.pending and not env.pending)) else 0.0
    utility = 1.0 * new_completion + information + prerequisite + effect - receipt["elapsed"] * .15
    if receipt["status"] in {"CONFIRMED_FAILURE", "BLOCKED"}:
        utility -= .6
    if kind == "stop" and (env.complete or not env.allowed or env.revoked):
        utility = .8
    return {"history": copy.deepcopy(history), "action": copy.deepcopy(action), "observation_after": after,
            "receipt": receipt, "verification_result": trial.verify(), "evidence_class": "OBSERVED",
            "provenance": "validated_simulation", "training_consent": True,
            "labels": {"outcome": receipt["status"], "completion": trial.complete,
                       "changed": bool(receipt["side_effects"]), "delay": receipt["elapsed"] + trial.pending * .2,
                       "utility": utility, "control": "act"},
            "family": env.family, "episode_id": f"{env.family}-{env.seed}-{env.variant}",
            "generator_version": GENERATOR_VERSION, "variant": env.variant}


def collect_dataset(split="train", episodes_per_family=28, seed=17):
    """Whole episodes/families/time variants are disjoint; no row-level split."""
    if split not in {"train", "development", "time_holdout", "family_holdout", "retention"}:
        raise ValueError("Unknown split")
    offsets = {"train": 0, "development": 100000, "time_holdout": 200000, "family_holdout": 300000, "retention": 400000}
    families = (HOLDOUT_FAMILY,) if split == "family_holdout" else TRAIN_FAMILIES
    variant = 2 if split in {"time_holdout", "family_holdout"} else 1
    dataset = []
    for family in families:
        for episode in range(episodes_per_family):
            episode_seed = seed + offsets[split] + episode
            env = DigitalEnvironment(family, episode_seed, variant)
            rng = random.Random(episode_seed + 37)
            history = [env.observe()]
            for step in range(6):
                candidates = [_transition(env, history, action) for action in env.describe_actions()]
                # These are resettable simulator outcomes, not imagined live receipts.
                best = max(candidates, key=lambda record: record["labels"]["utility"])
                control = best["action"]["kind"]
                control = control if control in {"inspect", "retrieve", "stop"} else "act"
                for record in candidates:
                    record["labels"]["control"] = control
                    record["split"] = split
                    record["collection_time_index"] = offsets[split] + step
                    dataset.append(record)
                # Diverse exploration, including interruption and unsuccessful actions.
                choice = rng.choice(candidates) if rng.random() < .55 else best
                receipt = env.execute(choice["action"])
                history.extend([{"action": choice["action"], "receipt": receipt}, env.observe()])
                history = history[-8:]
                if step == 3 and rng.random() < .1:
                    env.revoked = True
    return dataset
