import json
from pathlib import Path
import pytest
from sage.memory import MemoryStore
from sage.persistence import DrivePersistence
from sage.improvement import ImprovementManager, active_config, score_action, DEFAULT_CONFIG


def test_memory_provenance_corrections_scope_and_consent(tmp_path):
    memory = MemoryStore(tmp_path / "memory")
    one = memory.append("observation", {"text": "The stock count is 3"}, goal_id="a", episode_id="one")
    memory.append("observation", {"text": "The stock count is confidential"}, goal_id="b", episode_id="two")
    guessed = memory.append("observation", {"text": "The stock count may be 7"}, goal_id="a", evidence_class="INFERRED")
    claim = memory.claim("Stock is 3", [one])
    two = memory.append("observation", {"text": "Correction: stock is 4"}, goal_id="a", episode_id="three")
    memory.claim("Stock is 4", [two], contradictions=[one], supersedes=claim)
    found = memory.retrieve("stock", scope="a", limit=20)
    assert all(row.get("goal_id", "a") != "b" for row in found)
    assert len(memory.claims()) == 1
    experience = {"action": {"tool": "read", "args": {}}, "evidence_class": "OBSERVED", "receipt": {"status": "CONFIRMED_SUCCESS"}, "token": "SECRET"}
    memory.append("experience", experience, goal_id="a", training_consent=False)
    assert not memory.training_experiences()
    memory.set_training_consent("a", True)
    assert "SECRET" not in json.dumps(memory.training_experiences())
    memory.append("experience", experience, evidence_class="IMAGINED", training_consent=True, provenance="generated_simulation")
    assert len(memory.training_experiences()) == 1
    with pytest.raises(ValueError):
        memory.register_skill({"goal_pattern": "unverified", "evidence_ids": [guessed]})
    memory.delete(goal_id="a")
    assert not memory.claims()
    assert not memory.training_experiences()
    memory.close()


def test_retrieval_encoder_version_rebuild(tmp_path):
    memory = MemoryStore(tmp_path)
    memory.append("observation", {"text": "alpha"}, goal_id="one")
    memory.set_encoder(lambda text: [1., 0., float("alpha" in text)], "test-encoder-1")
    assert memory.retrieve("alpha")
    memory.set_encoder(lambda text: [0., 1.], "test-encoder-2")
    assert memory.retrieve("alpha")
    memory.close()


def test_sqlite_snapshot_restore_integrity_and_corrupt_fallback(tmp_path):
    state, drive = tmp_path / "state", tmp_path / "drive"
    memory = MemoryStore(state / "memory")
    memory.append("observation", {"text": "persisted"}, episode_id="one")
    persistence = DrivePersistence(state, drive)
    first = persistence.sync()
    memory.append("observation", {"text": "second"}, episode_id="two")
    second = persistence.sync()
    memory.close()
    # Force newest lexical order and corrupt its data; older intact copy remains.
    latest = Path(second["path"])
    corrupt_target = next(p for p in (latest / "data").rglob("*") if p.is_file())
    corrupt_target.write_bytes(b"broken")
    restored_dir = tmp_path / "restored"
    other = DrivePersistence(restored_dir, drive)
    result = other.restore(snapshot_id=first["snapshot_id"])
    assert result["status"] == "restored"
    reopened = MemoryStore(restored_dir / "memory")
    assert "persisted" in json.dumps(reopened.events())
    reopened.close()
    assert other.restore()["status"] == "skipped"
    with pytest.raises(PermissionError):
        other.delete_snapshots()
    with pytest.raises(ValueError):
        DrivePersistence(state, state / "inside")


def test_snapshot_symlink_refused(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "link").symlink_to("/etc/passwd")
    with pytest.raises(ValueError, match="symlink"):
        DrivePersistence(state, tmp_path / "drive").sync()


def test_improvement_isolation_archive_and_owner_gate(tmp_path):
    manager = ImprovementManager(tmp_path)
    candidate = manager.propose([{"cause": "uncertainty"}] * 3)
    report = manager.run(candidate)
    assert report["metrics"]["tasks"] == 360
    assert not report["promoted"]
    assert active_config(tmp_path) == DEFAULT_CONFIG
    with pytest.raises(PermissionError):
        manager.promote(candidate["id"])
    if report["passes_gates"]:
        manager.promote(candidate["id"], approved=True)
        assert active_config(tmp_path) == candidate["config"]
        manager.rollback(approved=True)
        assert active_config(tmp_path) == DEFAULT_CONFIG
    changed = dict(candidate, config={**candidate["config"], "command": "arbitrary code"})
    with pytest.raises(ValueError):
        manager.run(changed)
    assert score_action({"success_probability": 1}, DEFAULT_CONFIG) != score_action({"success_probability": 0}, DEFAULT_CONFIG)
