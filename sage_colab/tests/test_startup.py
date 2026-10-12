import json
from pathlib import Path
import pytest
from sage.colab import prepare
from sage.llm import ScriptedLLM
from sage.agent import SageAgent


def test_startup_training_drive_restore_and_goal_resume(tmp_path):
    drive = tmp_path / "drive"
    state = tmp_path / "first_runtime"
    app = prepare(state, drive, training_steps=2)
    model_hash = app["learning"].checkpoint_id
    agent = SageAgent(state, app["authority"], app["memory"], app["learning"],
                      ScriptedLLM([{ "blocked": "simulated interruption"}]), app["persistence"])
    goal = agent.create("Save a report", {"type": "artifact", "path": "report.txt"})
    assert agent.run(goal)["status"] == "BLOCKED"
    app["authority"].close()
    app["memory"].close()
    restored = prepare(tmp_path / "new_runtime", drive, training_steps=2)
    try:
        assert restored["learning"].checkpoint_id == model_hash
        assert (restored["workspace"] / "example_stock.csv").exists()
        llm = ScriptedLLM([{"candidates": [{"tool": "filesystem.write", "args": {"path": "report.txt", "content": "resumed"}}]}, {"answer": "Done"}])
        second = SageAgent(restored["state_dir"], restored["authority"], restored["memory"], restored["learning"], llm, restored["persistence"])
        result = second.run(goal, resume=True)
        assert result["status"] == "CONFIRMED_SUCCESS", result["answer"]
        assert (restored["workspace"] / "report.txt").read_text() == "resumed"
    finally:
        restored["authority"].close()
        restored["memory"].close()


def test_cpu_loading_cannot_download_large_backbone(tmp_path, monkeypatch):
    import torch
    from sage.llm import ColabLLM
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CPU weight download"):
        ColabLLM(tmp_path).load()
