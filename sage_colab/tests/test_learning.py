"""Actual optimization, split, consent and resumability checks (CPU, no downloads)."""
import copy
import json

import numpy as np
import pytest

from sage.models import encode_record, parameter_hash
from sage.simulators import DigitalEnvironment, collect_dataset
from sage.training import LearningSystem
from sage.training import _tensor_dataset


def test_typed_encoder_preserves_missing_and_excludes_hidden_labels():
    base = {"observed_fields": {"stock": 0}, "missing_fields": ["pending"]}
    assert not np.array_equal(encode_record(base), encode_record({"observed_fields": {"stock": None}}))
    assert np.array_equal(encode_record(base), encode_record({**base, "labels": {"stock": 90}, "hidden_state": {"stock": 100}}))
    assert not np.array_equal(encode_record(base), encode_record({"observed_fields": {"stock": 3}, "missing_fields": ["pending"]}))


def test_simulator_ambiguity_and_family_split():
    env = DigitalEnvironment("files", 12)
    env.allowed, env.dependency_ready, env.latency = True, True, 2
    commit = env.describe_actions()[2]
    assert env.execute(commit)["status"] == "UNKNOWN"
    assert not env.verify()["verified"]
    env.execute(env.describe_actions()[0])
    env.execute(env.describe_actions()[0])
    assert env.verify()["verified"]
    train = collect_dataset("train", 1)
    test = collect_dataset("family_holdout", 1)
    time = collect_dataset("time_holdout", 1)
    assert not ({item["family"] for item in train} & {item["family"] for item in test})
    assert not ({item["episode_id"] for item in train} & {item["episode_id"] for item in time})
    assert all("labels" not in item["history"][-1] for item in train)


def test_real_training_resume_all_heads_and_candidate_authority(tmp_path):
    learning = LearningSystem(tmp_path)
    initial = {name: value.detach().clone() for name, value in learning.model.named_parameters()}
    result = learning.bootstrap(max_steps=24)
    assert result["weights_changed"] and result["steps_this_call"] == 24
    for prefix in ("history", "action", "heads", "policy", "controller"):
        assert any(not value.equal(initial[name]) for name, value in learning.model.named_parameters() if name.startswith(prefix))
    digest = parameter_hash(learning.model)
    restored = LearningSystem(tmp_path)
    assert parameter_hash(restored.model) == digest
    assert restored.bootstrap(max_steps=24)["steps_this_call"] == 0
    assert restored.bootstrap(max_steps=25)["steps_this_call"] == 1
    prediction = restored.predict([{"source_env": "live", "observed_fields": {"file": True}}], {"tool": "filesystem.read", "args": {"path": "x"}})
    assert prediction["uncertainty"] >= .65
    assert prediction["domain_status"] == "live_transfer_unvalidated"
    assert 0 <= prediction["completion_probability"] <= 1
    assert restored.control([])["decision"] in {"inspect", "retrieve", "act", "stop"}
    data = collect_dataset("train", 1)[:24]
    denied = copy.deepcopy(data)
    for row in denied:
        row["training_consent"] = False
    assert restored.learn(denied)["accepted_experiences"] == 0
    assert restored.learn(collect_dataset("family_holdout", 1))["accepted_experiences"] == 0
    digest = parameter_hash(restored.model)
    candidate = restored.learn(data, max_steps=2)
    assert candidate["accepted_experiences"] > 0
    assert not candidate["promoted"]
    assert parameter_hash(restored.model) == digest
    with pytest.raises(PermissionError):
        restored.promote_checkpoint(candidate["candidate_id"])
    if candidate["passes_retention"]:
        restored.promote_checkpoint(candidate["candidate_id"], approved=True)
        assert parameter_hash(restored.model) != digest
        restored.rollback(approved=True)
        assert parameter_hash(restored.model) == digest


def test_corrupt_checkpoint_fails_closed(tmp_path):
    learner = LearningSystem(tmp_path)
    learner.bootstrap(max_steps=1)
    active = json.loads((tmp_path / "learning" / "active.json").read_text())
    (tmp_path / "learning" / active["checkpoint"]).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="integrity"):
        LearningSystem(tmp_path)


def test_operation_receipt_never_becomes_goal_completion_label():
    records = [{"action": {"tool": "read"}, "receipt": {"status": status}, "verification_result": status}
               for status in ("CONFIRMED_FAILURE", "UNKNOWN", "CONFIRMED_SUCCESS")]
    batch = _tensor_dataset(records)
    assert not batch["mask"][:, 3].any()
    assert not batch["binary"][:, 1].any()
