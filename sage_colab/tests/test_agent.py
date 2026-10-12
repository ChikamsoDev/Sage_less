import json
import threading
from pathlib import Path
import pytest
from sage.agent import SageAgent
from sage.authority import AuthorityService
from sage.memory import MemoryStore
from sage.llm import ScriptedLLM, choose_model, parse_json
from sage.training import LearningSystem
from sage.schema import action, atomic_json


@pytest.fixture
def components(tmp_path):
    owner = AuthorityService(tmp_path / "authority", tmp_path / "workspace")
    memory = MemoryStore(tmp_path / "memory")
    learner = LearningSystem(tmp_path)
    yield tmp_path, owner, memory, learner
    owner.close()
    memory.close()


def test_end_to_end_verified_artifact_and_skill(components):
    root, owner, memory, learner = components
    llm = ScriptedLLM([{"candidates": [{"tool": "filesystem.write", "args": {"path": "result.txt", "content": "checked result"}}]},
                       {"answer": "Saved result.txt", "candidates": []}])
    agent = SageAgent(root, owner, memory, learner, llm)
    goal = agent.create("Save result", {"type": "artifact", "path": "result.txt", "contains": ["checked result"]})
    state = agent.run(goal)
    assert state["status"] == "CONFIRMED_SUCCESS", state["answer"]
    assert (root / "workspace/result.txt").read_text() == "checked result"
    assert memory.skills()
    assert not memory.training_experiences()  # Consent defaults off.
    events = list(reversed(memory.events(goal, limit=100)))
    kinds = [e["kind"] for e in events]
    assert "prediction" in kinds and "receipt" in kinds
    assert state["completion_evidence"]["path"] == "result.txt"


def test_unverified_answer_cannot_become_observed_success(components):
    root, owner, memory, learner = components
    agent = SageAgent(root, owner, memory, learner, ScriptedLLM([{"answer": "The answer is probably 42"}]))
    result = agent.run(agent.create("Explain an unknown result"))
    assert result["status"] == "ANSWERED_UNVERIFIED"
    assert not memory.skills()


def test_malicious_proposal_cannot_expand_authority(components):
    root, owner, memory, learner = components
    llm = ScriptedLLM([{"candidates": [{"tool": "filesystem.write", "args": {"path": "../authority/authority.json", "content": "owner approved"}}]}])
    agent = SageAgent(root, owner, memory, learner, llm)
    result = agent.run(agent.create("Write a report in the workspace"))
    assert result["status"] == "BLOCKED"
    assert not (root / "authority/authority.json").exists()


def test_artifact_claim_is_rejected_without_effect(components):
    root, owner, memory, learner = components
    agent = SageAgent(root, owner, memory, learner, ScriptedLLM([{"answer": "I created it"}, {"blocked": "Missing output"}]))
    result = agent.run(agent.create("Make report", {"type": "artifact", "path": "absent.txt"}))
    assert result["status"] == "BLOCKED"
    assert "completion_evidence" not in result


def test_saved_pending_action_reconciles_once(components):
    root, owner, memory, learner = components
    agent = SageAgent(root, owner, memory, learner, ScriptedLLM([{"answer": "Done"}]))
    goal = agent.create("Create one result", {"type": "artifact", "path": "once.txt"})
    grant = owner.grant(goal)
    pending = action("filesystem.write", {"path": "once.txt", "content": "once"})
    owner.execute(grant, pending)  # Simulate committed action with lost acknowledgement.
    path = root / "runs" / (goal + ".json")
    state = json.loads(path.read_text())
    state["pending"] = {"action": pending, "prediction": {}, "prediction_id": "pre-crash"}
    atomic_json(path, state)
    result = agent.run(goal)
    assert result["status"] == "CONFIRMED_SUCCESS", result["answer"]
    assert (root / "workspace/once.txt").read_text() == "once"


def test_model_selection_reserves_memory_and_parse_is_data_only():
    assert choose_model(15, 10) == "Qwen/Qwen3-8B"
    assert choose_model(8, 8) == "Qwen/Qwen3-4B"
    with pytest.raises(RuntimeError):
        choose_model(2, 8)
    assert parse_json('```json\n{"answer":"2","candidates":[]}\n```')["answer"] == "2"
    with pytest.raises(ValueError):
        parse_json("__import__('os').system('anything')")


def test_large_evidence_is_externalized_without_truncating_verification(components):
    root, owner, memory, learner = components
    content = "a" * (1024 * 1024 - 10) + "END-MARKER"
    (root / "workspace/large.txt").write_text(content)
    agent = SageAgent(root, owner, memory, learner, ScriptedLLM([{"answer": "The existing file matches"}]))
    result = agent.run(agent.create("Verify large file", {"type": "artifact", "path": "large.txt", "contains": ["END-MARKER"]}))
    assert result["status"] == "CONFIRMED_SUCCESS", result["answer"]
    assert len(json.dumps(result)) < 100000
    assert list((root / "raw_evidence").glob("*.json"))
    records = [x for x in memory.events(limit=100) if x["kind"] == "experience"]
    assert records and all(x["payload"]["grant_version"] == 1 for x in records)
