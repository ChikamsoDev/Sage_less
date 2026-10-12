"""Bounded and resumable neural learning with immutable candidate checkpoints."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import uuid

import numpy as np
import torch
from torch.nn import functional as F

from .models import (CONTROLS, ENCODER_VERSION, INPUT_DIM, OUTCOMES, WorldPolicyModel,
                     canonical_hash, encode_history, encode_record, parameter_hash)
from .simulators import GENERATOR_VERSION, collect_dataset


def _json_write(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False))
    os.replace(temporary, path)


def _source_hashes():
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("models.py", "simulators.py", "training.py", "evaluation.py")}


def _tensor_dataset(records, device="cpu"):
    histories, actions, states, outcomes, binaries, delays, utilities, controls, masks = [], [], [], [], [], [], [], [], []
    for record in records:
        histories.append(encode_history(record.get("history") or [record.get("observation_before", {})]))
        actions.append(encode_record(record["action"]))
        after = record.get("observation_after") or {}
        states.append(encode_record(after))
        labels = record.get("labels", {})
        receipt = record.get("receipt", {})
        status = labels.get("outcome", receipt.get("status", "UNKNOWN"))
        is_blocked = status == "BLOCKED"
        outcomes.append(OUTCOMES.index(status) if status in OUTCOMES else OUTCOMES.index("UNKNOWN"))
        verified = record.get("verification_result", {})
        if not isinstance(verified, dict):
            # A receipt status is operation evidence, not a goal-completion label.
            verified = {"verified": verified if type(verified) is bool else None}
        completion = labels.get("completion", verified.get("verified", verified.get("complete")))
        changed = labels.get("changed", bool(receipt.get("side_effects")) if "side_effects" in receipt else None)
        binaries.append([float(bool(changed)), float(bool(completion))])
        delay = labels.get("delay", receipt.get("elapsed"))
        delay = float(delay) if isinstance(delay, (float, int)) and math.isfinite(delay) else None
        delays.append(math.log1p(max(0.0, delay or 0.0)))
        success = status == "CONFIRMED_SUCCESS"
        utilities.append(float(labels.get("utility", float(bool(completion)) + .1 * success - .2 * (not success))))
        control = labels.get("control")
        controls.append(CONTROLS.index(control) if control in CONTROLS else 2)
        masks.append([float(bool(after) and not is_blocked), float(not is_blocked),
                      float(changed is not None and not is_blocked), float(completion is not None and not is_blocked),
                      float(delay is not None and not is_blocked), float(control in CONTROLS)])
    return {"history": torch.tensor(np.asarray(histories), device=device),
            "action": torch.tensor(np.asarray(actions), device=device),
            "state": torch.tensor(np.asarray(states), device=device),
            "outcome": torch.tensor(outcomes, dtype=torch.long, device=device),
            "binary": torch.tensor(binaries, dtype=torch.float32, device=device),
            "delay": torch.tensor(delays, dtype=torch.float32, device=device),
            "utility": torch.tensor(utilities, dtype=torch.float32, device=device),
            "control": torch.tensor(controls, dtype=torch.long, device=device),
            "mask": torch.tensor(masks, dtype=torch.float32, device=device)}


def _masked_mean(value, mask):
    return (value * mask).sum() / mask.sum().clamp_min(1)


class LearningSystem:
    """User-private experiences require explicit training_consent=True.

    Initial synthetic bootstrap can activate its own newly initialized weights.
    Later consolidation always creates a candidate; promotion is an owner API.
    """
    def __init__(self, state_dir, device="cpu"):
        self.state_dir = Path(state_dir)
        self.directory = self.state_dir / "learning"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no CUDA device is available")
        if self.device.type == "cpu":
            torch.set_num_threads(min(2, torch.get_num_threads()))
        with torch.random.fork_rng():
            torch.manual_seed(17)
            self.model = WorldPolicyModel().to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=.003, weight_decay=.0001)
        self.bootstrap_steps = 0
        self.total_steps = 0
        self.seed = 17
        self.replay = []
        self._datasets = {}
        self.active_path = None
        manifest = self.directory / "active.json"
        if manifest.exists():
            self._load_manifest(manifest)
        self.model.eval()

    @property
    def checkpoint_id(self):
        return parameter_hash(self.model)[:16]

    def _dataset(self, split):
        if split not in self._datasets:
            self._datasets[split] = collect_dataset(split, episodes_per_family=20 if split == "train" else 8, seed=self.seed)
        return self._datasets[split]

    def _save(self, label, activate=False, extra=None):
        digest = parameter_hash(self.model)
        filename = f"{label}-{self.total_steps:06d}-{digest[:12]}.pt"
        target = self.directory / filename
        temporary = target.with_suffix(".pt.tmp")
        data = {"schema_version": 1, "encoder_version": ENCODER_VERSION, "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(), "bootstrap_steps": self.bootstrap_steps,
                "total_steps": self.total_steps, "seed": self.seed, "replay": self.replay,
                "parameter_hash": digest, "metadata": self.model.metadata(), "source_hashes": _source_hashes(),
                "generator_version": GENERATOR_VERSION, "extra": extra or {}}
        torch.save(data, temporary)
        os.replace(temporary, target)
        manifest = {"checkpoint": filename, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                    "parameter_hash": digest, "bootstrap_steps": self.bootstrap_steps, "total_steps": self.total_steps,
                    "encoder_version": ENCODER_VERSION, "source_hashes": data["source_hashes"], "extra": extra or {}}
        _json_write(target.with_suffix(".json"), manifest)
        if activate:
            _json_write(self.directory / "active.json", manifest)
            self.active_path = target
        return manifest

    def _load_manifest(self, manifest_path):
        manifest = json.loads(Path(manifest_path).read_text())
        filename = manifest["checkpoint"]
        if Path(filename).name != filename:
            raise ValueError("Invalid checkpoint manifest path")
        path = self.directory / filename
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
            raise ValueError("Learning checkpoint integrity check failed")
        data = torch.load(path, map_location=self.device, weights_only=True)
        if data["encoder_version"] != ENCODER_VERSION:
            raise ValueError("Encoder version changed; a deliberate migration is required")
        if data.get("generator_version") != GENERATOR_VERSION:
            raise ValueError("Curriculum generator changed; preserve this checkpoint and start a separate training state")
        self.model.load_state_dict(data["model"])
        self.optimizer.load_state_dict(data["optimizer"])
        self.bootstrap_steps = data["bootstrap_steps"]
        self.total_steps = data["total_steps"]
        self.seed = data["seed"]
        self.replay = data.get("replay", [])
        self.active_path = path
        self._datasets.clear()
        return manifest

    def _fit(self, records, steps, start_index=0, checkpoint_callback=None):
        if not records or steps <= 0:
            return []
        tensors = _tensor_dataset(records, self.device)
        size = len(records)
        losses = []
        self.model.train()
        for offset in range(steps):
            rng = torch.Generator(device="cpu").manual_seed(self.seed + start_index + offset)
            indices = torch.randint(size, (min(64, size),), generator=rng).to(self.device)
            batch = {key: value[indices] for key, value in tensors.items()}
            output = self.model(batch["history"], batch["action"])
            mask = batch["mask"]
            loss = torch.zeros((), device=self.device)
            for head_index, head in enumerate(output["heads"]):
                # Deterministic bootstrap masks give independently fitted heads.
                included = ((indices + head_index * 3 + offset) % (len(output["heads"]) + 2) != head_index).float()
                state_loss = (head["state"] - batch["state"]).square().mean(-1) * INPUT_DIM
                outcome_loss = F.cross_entropy(head["outcome"], batch["outcome"], reduction="none")
                binary_loss = F.binary_cross_entropy_with_logits(head["binary"], batch["binary"], reduction="none")
                delay_loss = (head["delay"] - batch["delay"]).square()
                loss = loss + .3 * _masked_mean(state_loss, mask[:, 0] * included)
                loss = loss + _masked_mean(outcome_loss, mask[:, 1] * included)
                loss = loss + .5 * _masked_mean(binary_loss[:, 0], mask[:, 2] * included)
                loss = loss + .5 * _masked_mean(binary_loss[:, 1], mask[:, 3] * included)
                loss = loss + .25 * _masked_mean(delay_loss, mask[:, 4] * included)
            loss /= len(output["heads"])
            loss = loss + .8 * F.mse_loss(output["policy"], batch["utility"])
            loss = loss + .4 * _masked_mean(F.cross_entropy(output["control"], batch["control"], reduction="none"), mask[:, 5])
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite learning loss; candidate training aborted")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
            self.optimizer.step()
            self.total_steps += 1
            losses.append(float(loss.detach().cpu()))
            if checkpoint_callback:
                checkpoint_callback(offset + 1)
        self.model.eval()
        return losses

    def bootstrap(self, max_steps=120, seed=17, checkpoint_callback=None):
        """Reach max_steps total synthetic-bootstrap steps, checkpointing every 20.

        Re-running after completion performs zero optimization steps. Increasing
        max_steps explicitly extends the authorized initial training budget.
        """
        max_steps = max(0, min(int(max_steps), 2000))
        if self.bootstrap_steps and seed != self.seed:
            raise ValueError("Cannot change the seed of a resumable bootstrap")
        self.seed = int(seed)
        self._datasets.clear()
        before = parameter_hash(self.model)
        steps = max(0, max_steps - self.bootstrap_steps)
        started = time.monotonic()
        if not (self.directory / "reference.json").exists():
            reference = self._save("reference")
            _json_write(self.directory / "reference.json", reference)
        records = self._dataset("train")
        start_step = self.bootstrap_steps
        def checkpoint(completed):
            self.bootstrap_steps = start_step + completed
            if completed % 20 == 0 or completed == steps:
                self._save("bootstrap", activate=True, extra={"training_data_hash": canonical_hash(records),
                           "training_records": len(records), "split": "train", "learning_kind": "initial_synthetic_bootstrap"})
                if checkpoint_callback:
                    checkpoint_callback()
        losses = self._fit(records, steps, start_index=self.total_steps, checkpoint_callback=checkpoint)
        report = {"status": "trained" if steps else "already_complete", "steps_this_call": steps,
                  "bootstrap_steps": self.bootstrap_steps, "weights_changed": before != parameter_hash(self.model),
                  "parameter_hash": parameter_hash(self.model), "parameter_count": self.model.metadata()["parameter_count"],
                  "first_loss": losses[0] if losses else None, "last_loss": losses[-1] if losses else None,
                  "seconds": time.monotonic() - started, "training_records": len(records),
                  "training_data_hash": canonical_hash(records), "checkpoint": str(self.active_path) if self.active_path else None,
                  "trained_components": ["GRU history", "action projection", "world ensemble", "policy value", "compute controller"],
                  "not_trained": ["LLM backbone/adapters", "visual perception", "general-purpose causal reasoning"]}
        _json_write(self.directory / "bootstrap_report.json", report)
        return report

    def _forward(self, history, action):
        self.model.eval()
        with torch.inference_mode():
            return self.model(torch.tensor(encode_history(history)[None], device=self.device),
                              torch.tensor(encode_record(action)[None], device=self.device))

    def predict(self, history, action):
        output = self._forward(history, action)
        outcomes = torch.stack([head["outcome"].softmax(-1)[0] for head in output["heads"]])
        binaries = torch.stack([head["binary"].sigmoid()[0] for head in output["heads"]])
        success = outcomes[:, 0]
        mean = outcomes.mean(0)
        entropy = -(mean * mean.clamp_min(1e-8).log()).sum() / math.log(len(OUTCOMES))
        disagreement = float(success.std(unbiased=False).cpu())
        simulated = any(event.get("source_env") in {"files", "inventory", "workflow", "orders"} for event in history)
        uncertainty = max(float(entropy.cpu()) * .55 + disagreement, 0.15 if simulated else .65)
        return {"success_probability": float(success.mean().cpu()),
                "completion_probability": float(binaries[:, 1].mean().cpu()),
                "changed_probability": float(binaries[:, 0].mean().cpu()),
                "delay": max(0.0, float(torch.stack([head["delay"][0] for head in output["heads"]]).mean().expm1().cpu())),
                "uncertainty": min(1., uncertainty), "ensemble_disagreement": disagreement,
                "outcome_probabilities": {key: float(mean[index].cpu()) for index, key in enumerate(OUTCOMES)},
                "next_state_embedding": torch.stack([head["state"][0] for head in output["heads"]]).mean(0).cpu().tolist(),
                "checkpoint_id": self.checkpoint_id, "trained_steps": self.total_steps,
                "domain_status": "simulation_distribution" if simulated else "live_transfer_unvalidated",
                "evidence_class": "INFERRED", "no_action_supported": True}

    def policy_score(self, history, action):
        return float(self._forward(history, action)["policy"][0].cpu())

    def control(self, history):
        scores = self._forward(history, {})["control"][0].softmax(-1).cpu().tolist()
        return {"decision": CONTROLS[int(np.argmax(scores))], "scores": dict(zip(CONTROLS, scores)),
                "evidence_class": "INFERRED", "trained_steps": self.total_steps}

    def learn(self, experiences, max_steps=20):
        """Train an isolated candidate, never silently replace a deployed model."""
        usable = []
        for original in experiences:
            record = original.get("payload", original)
            if not isinstance(record, dict) or not record.get("training_consent", original.get("training_consent", False)):
                continue
            if record.get("evidence_class", original.get("evidence_class")) != "OBSERVED":
                continue
            if record.get("split") in {"development", "time_holdout", "family_holdout", "retention"}:
                continue
            if record.get("receipt", {}).get("status") == "BLOCKED" or not isinstance(record.get("action"), dict):
                continue
            # A live caller cannot smuggle arbitrary supervised targets into training.
            record = copy.deepcopy(record)
            if record.get("provenance") != "validated_simulation":
                record.pop("labels", None)
            usable.append(record)
        if not usable:
            return {"status": "no_consented_verified_experience", "accepted_experiences": 0, "promoted": False}
        max_steps = max(0, min(int(max_steps), 500))
        if not max_steps:
            return {"status": "zero_budget", "accepted_experiences": len(usable), "promoted": False}
        from .evaluation import evaluate_records
        before_model = copy.deepcopy(self.model.state_dict())
        before_optimizer = copy.deepcopy(self.optimizer.state_dict())
        old_total, old_replay = self.total_steps, self.replay
        retention = self._dataset("retention")
        before_metrics = evaluate_records(self.model, retention, self.device)
        rng = random.Random(self.seed + self.total_steps)
        old = old_replay or self._dataset("train")
        failures = [record for record in old + usable if record.get("receipt", {}).get("status") == "CONFIRMED_FAILURE"]
        n = max(64, min(len(usable), 256))
        # Bounded replay: 50% recent, 30% older domains, 20% verified failures.
        records = rng.choices(usable, k=n * 5) + rng.choices(old, k=n * 3) + rng.choices(failures or old, k=n * 2)
        candidate_id = uuid.uuid4().hex[:12]
        try:
            losses = self._fit(records, max_steps, start_index=self.total_steps)
            after_metrics = evaluate_records(self.model, retention, self.device)
            acceptable = (after_metrics["success_brier"] <= before_metrics["success_brier"] + .03 and
                          after_metrics["completion_brier"] <= before_metrics["completion_brier"] + .03)
            self.replay = (old_replay + usable)[-512:]
            report = {"candidate_id": candidate_id, "status": "awaiting_owner_approval" if acceptable else "rejected_retention_regression",
                      "accepted_experiences": len(usable), "steps": max_steps, "promoted": False,
                      "passes_retention": acceptable, "retention_before": before_metrics, "retention_after": after_metrics,
                      "training_data_hash": canonical_hash(records), "first_loss": losses[0], "last_loss": losses[-1],
                      "evaluation_limit": "internal fixed diagnostic holdout, not an independent external audit"}
            manifest = self._save(f"candidate-{candidate_id}", extra=report)
            _json_write(self.directory / f"candidate-{candidate_id}.json", manifest)
            return report
        finally:
            self.model.load_state_dict(before_model)
            self.optimizer.load_state_dict(before_optimizer)
            self.total_steps, self.replay = old_total, old_replay
            self.model.eval()

    def promote_checkpoint(self, candidate_id, approved=False):
        if not approved:
            raise PermissionError("Only an explicit owner release decision can promote learned parameters")
        if not isinstance(candidate_id, str) or len(candidate_id) != 12 or not all(c in "0123456789abcdef" for c in candidate_id):
            raise ValueError("Invalid candidate id")
        path = self.directory / f"candidate-{candidate_id}.json"
        manifest = json.loads(path.read_text())
        if not manifest.get("extra", {}).get("passes_retention"):
            raise ValueError("Candidate failed retention checks")
        previous = self.directory / "active.json"
        previous_manifest = json.loads(previous.read_text()) if previous.exists() else None
        self._load_manifest(path)
        if previous_manifest:
            _json_write(self.directory / "previous.json", previous_manifest)
        _json_write(previous, manifest)
        return {"status": "promoted", "candidate_id": candidate_id, "checkpoint_id": self.checkpoint_id}

    def rollback(self, approved=False):
        if not approved:
            raise PermissionError("Owner approval is required for rollback")
        previous = self.directory / "previous.json"
        manifest = self._load_manifest(previous)
        _json_write(self.directory / "active.json", manifest)
        return {"status": "rolled_back", "checkpoint_id": self.checkpoint_id}

    def evaluate(self):
        from .evaluation import evaluate_system
        report = evaluate_system(self)
        _json_write(self.directory / "evaluation_report.json", report)
        return report
