"""Mechanically scored prediction and one-step decision diagnostics.

Held-out family/time sets are not used in fitting. These fixed local diagnostics
are inspectable and repeatedly reusable, so they are not a protected independent
AGI audit. A hosted LLM comparison requires running that backbone in Colab.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .models import CONTROLS, INPUT_DIM, OUTCOMES, WorldPolicyModel, canonical_hash


def evaluate_records(model, records, device="cpu"):
    from .training import _tensor_dataset
    tensors = _tensor_dataset(records, device)
    results = {key: [] for key in ("outcome", "binary", "delay", "state", "policy", "control", "belief")}
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(records), 128):
            output = model(tensors["history"][start:start + 128], tensors["action"][start:start + 128])
            for key in ("outcome", "binary", "delay", "state"):
                values = [head[key] for head in output["heads"]]
                if key == "outcome":
                    values = [value.softmax(-1) for value in values]
                elif key == "binary":
                    values = [value.sigmoid() for value in values]
                results[key].append(torch.stack(values).mean(0).cpu())
            for key in ("policy", "control", "belief"):
                results[key].append(output[key].cpu())
    result = {key: torch.cat(values).numpy() for key, values in results.items()}
    truth = {key: value.cpu().numpy() for key, value in tensors.items() if key not in {"history", "action"}}
    allowed = truth["mask"][:, 1].astype(bool)
    state_valid = truth["mask"][:, 0].astype(bool)
    probabilities = result["outcome"][allowed, 0]
    success = (truth["outcome"][allowed] == 0).astype(float)
    brier = float(np.mean((probabilities - success) ** 2))
    # Expected calibration error is descriptive; Brier/log loss are proper scores.
    ece = 0.
    bins = []
    for low in np.arange(0., 1., .2):
        chosen = (probabilities >= low) & (probabilities < low + .2 + 1e-9)
        count = int(chosen.sum())
        if count:
            confidence, observed = float(probabilities[chosen].mean()), float(success[chosen].mean())
            ece += count / len(probabilities) * abs(confidence - observed)
            bins.append({"lower": float(low), "count": count, "forecast": confidence, "observed": observed})
    outcome_prob = result["outcome"][np.arange(len(records)), truth["outcome"]]
    delay_valid = truth["mask"][:, 4].astype(bool)
    completion_valid = truth["mask"][:, 3].astype(bool)
    changed_valid = truth["mask"][:, 2].astype(bool)
    control_valid = truth["mask"][:, 5].astype(bool)
    # Every six adjacent simulator records are separately executed alternatives
    # from an identical state. The oracle is only a scoring upper bound.
    learned, world, random_control, oracle = [], [], [], []
    if len(records) % 6 == 0 and all(record.get("provenance") == "validated_simulation" for record in records):
        for start in range(0, len(records), 6):
            utility = truth["utility"][start:start + 6]
            index = int(np.argmax(result["policy"][start:start + 6]))
            forecasts = (result["binary"][start:start + 6, 1] + .25 * result["binary"][start:start + 6, 0]
                         + .2 * result["outcome"][start:start + 6, 0]
                         - .1 * np.expm1(result["delay"][start:start + 6]).clip(0))
            learned.append(float(utility[index]))
            world.append(float(utility[int(np.argmax(forecasts))]))
            random_control.append(float(utility.mean()))
            oracle.append(float(utility.max()))
    report = {"records": len(records), "episodes": len({record["episode_id"] for record in records}),
              "success_brier": brier,
              "completion_brier": float(np.mean((result["binary"][completion_valid, 1] - truth["binary"][completion_valid, 1]) ** 2)),
              "changed_brier": float(np.mean((result["binary"][changed_valid, 0] - truth["binary"][changed_valid, 0]) ** 2)),
              "outcome_log_loss": float(-np.log(np.clip(outcome_prob[allowed], 1e-8, 1)).mean()),
              "outcome_accuracy": float(np.mean(result["outcome"][allowed].argmax(-1) == truth["outcome"][allowed])),
              "state_mse": float(np.mean((result["state"][state_valid] - truth["state"][state_valid]) ** 2)),
              "delay_mae_seconds": float(np.abs(np.expm1(result["delay"][delay_valid]).clip(0) - np.expm1(truth["delay"][delay_valid])).mean()),
              "success_ece": float(ece), "calibration_bins": bins,
              "policy_utility_mse": float(np.mean((result["policy"] - truth["utility"]) ** 2)),
              "control_accuracy": float(np.mean(result["control"][control_valid].argmax(-1) == truth["control"][control_valid])),
              "representation_mean_variance": float(result["belief"].var(axis=0).mean()),
              "dataset_sha256": canonical_hash(records),
              "blocked_world_labels_excluded": int((~allowed).sum())}
    if learned:
        report.update({"learned_policy_utility": float(np.mean(learned)), "world_ranked_utility": float(np.mean(world)),
                       "uniform_random_utility": float(np.mean(random_control)), "oracle_utility_upper_bound": float(np.mean(oracle)),
                       "policy_gain_over_random": float(np.mean(learned) - np.mean(random_control)),
                       "equal_candidate_count": 6, "rollout_horizon": 1})
    return report


def evaluate_system(learner):
    started = time.monotonic()
    splits = ("development", "time_holdout", "family_holdout", "retention")
    scores = {split: evaluate_records(learner.model, learner._dataset(split), learner.device) for split in splits}
    reference_scores = None
    reference_manifest_path = learner.directory / "reference.json"
    if reference_manifest_path.exists():
        manifest = json.loads(reference_manifest_path.read_text())
        filename = manifest["checkpoint"]
        if Path(filename).name != filename:
            raise ValueError("Invalid reference checkpoint path")
        path = learner.directory / filename
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
            raise ValueError("Reference checkpoint integrity check failed")
        data = torch.load(path, map_location=learner.device, weights_only=True)
        reference = WorldPolicyModel().to(learner.device)
        reference.load_state_dict(data["model"])
        reference_scores = {split: evaluate_records(reference, learner._dataset(split), learner.device) for split in splits}
    report = {"schema_version": 1, "checkpoint_id": learner.checkpoint_id, "trained_steps": learner.total_steps,
              "scores": scores, "untrained_reference": reference_scores,
              "data_separation": {"training_families": ["files", "inventory", "workflow"], "family_holdout": ["orders"],
                                  "time_holdout": "later collection index, unseen episode seeds, generator variant 2 with longer delays",
                                  "split_by_complete_episode": True, "holdout_used_for_training": False},
              "baselines": {"B0_strong_llm_tools": "NOT_RUN: requires selected Colab backbone",
                            "B1_llm_memory": "NOT_RUN: requires selected Colab backbone",
                            "B2_llm_planning": "NOT_RUN: requires selected Colab backbone",
                            "B3_frozen_initial_network": "RUN" if reference_scores else "NOT_RUN",
                            "B4_trained_network": "RUN", "random_action_control": "RUN"},
              "claims": {"agi": False, "asi": False, "live_transfer_validated": False,
                         "independent_audit": False, "multistep_world_rollouts_validated": False},
              "seconds": time.monotonic() - started}
    if reference_scores:
        report["improvement_vs_untrained"] = {
            split: {"success_brier_reduction": reference_scores[split]["success_brier"] - scores[split]["success_brier"],
                    "policy_utility_gain": scores[split]["learned_policy_utility"] - reference_scores[split]["learned_policy_utility"]}
            for split in splits}
    return report
