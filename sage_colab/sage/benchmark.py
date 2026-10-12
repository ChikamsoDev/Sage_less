"""Matched tool-agent startup diagnostics; never advertised as an AGI battery."""
from __future__ import annotations
import json
import time
from pathlib import Path
from .schema import atomic_json

TASKS = [
    {"family": "files", "instruction": "Read source.txt and create result.txt containing its text in uppercase. Keep source.txt intact.",
     "inputs": {"source.txt": "copper lantern"}, "expected": "COPPER LANTERN"},
    {"family": "data", "instruction": "Use data.query to sum the quantity column in stock.csv. Write only that number to result.txt. Preserve stock.csv.",
     "inputs": {"stock.csv": "item,quantity\nbolt,7\nnut,11\nwasher,5\n"}, "expected": "23"},
    {"family": "code", "instruction": "Use code.run to execute Python assertions that sorted([7,2,5]) equals [2,5,7]. After the assertions pass, write PASS to result.txt.",
     "inputs": {}, "expected": "PASS", "required_tool": "code.run"},
]


def compare_backbone(backbone, learning, report_dir, modes=("B0", "B1", "B2", "B3", "B4"), emit=print):
    from .authority import AuthorityService
    from .memory import MemoryStore
    from .agent import SageAgent
    from .training import LearningSystem
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "backbone_comparison.json"
    if report_path.exists():
        existing = json.loads(report_path.read_text())
        if existing.get("complete") and existing.get("backbone_revision") == backbone.manifest.get("revision"):
            return existing
    rows = []
    # Separate object: evaluating the frozen reference never rewrites active.json.
    reference = LearningSystem(learning.state_dir, device="cpu")
    if (learning.directory / "reference.json").exists():
        reference._load_manifest(learning.directory / "reference.json")
    import uuid
    run_id = uuid.uuid4().hex[:8]
    for mode in modes:
        for task in TASKS:
            emit(f"Automatic backbone check: {mode} / {task['family']}")
            root = report_dir / "runs" / run_id / mode / task["family"]
            workspace = root / "workspace"
            workspace.mkdir(parents=True, exist_ok=True)
            for name, value in task["inputs"].items():
                (workspace / name).write_text(value)
            memory = MemoryStore(root / "memory")
            # Every memory-enabled condition receives the same one allowed prior observation.
            if mode != "B0":
                memory.append("observation", {"text": "Workspace input files must be preserved. Verify output by reading it."},
                              episode_id="common-adaptation-episode", provenance="validated_simulation")
            owner = AuthorityService(root / "authority", workspace)
            model = reference if mode == "B3" else learning
            agent = SageAgent(root / "state", owner, memory, model, backbone)
            before_calls, before_tokens = backbone.calls, backbone.generated_tokens
            start = time.monotonic()
            try:
                goal = agent.create(task["instruction"], {"type": "artifact", "path": "result.txt", "contains": [task["expected"]]},
                                    max_actions=8, max_seconds=120)
                result = agent.run(goal, mode=mode)
                file = workspace / "result.txt"
                correct = file.is_file() and file.read_text().strip() == task["expected"]
                preserved = all((workspace / n).read_text() == v for n, v in task["inputs"].items())
                required = task.get("required_tool")
                tool_ok = not required or any(r["action"]["tool"] == required and r["receipt"]["status"] == "CONFIRMED_SUCCESS" for r in result["receipts"])
                rows.append({"mode": mode, "family": task["family"], "verified_completion": bool(correct and preserved and tool_ok),
                             "reported_status": result["status"], "false_success": result["status"] == "CONFIRMED_SUCCESS" and not (correct and preserved and tool_ok),
                             "actions": result["step"], "llm_calls": backbone.calls - before_calls,
                             "generated_tokens": backbone.generated_tokens - before_tokens, "seconds": time.monotonic() - start,
                             "input_preserved": preserved, "tool_check": tool_ok, "interventions": 0})
            finally:
                owner.close()
                memory.close()
            atomic_json(report_path, {"complete": False, "results": rows, "backbone_revision": backbone.manifest.get("revision")})
    result = {"complete": True, "backbone_revision": backbone.manifest.get("revision"), "results": rows,
              "budget": {"max_actions": 8, "max_seconds_per_task": 120, "max_generation_tokens": 768},
              "conditions": {"B0": "frozen backbone, one proposed action, no memory",
                             "B1": "same backbone plus persistent evidence, one action",
                             "B2": "memory plus model-ordered alternatives, no world-model ranking",
                             "B3": "world-model ranking with pre-bootstrap random reference; no consolidation",
                             "B4": "trained world/policy/controller plus memory and model proposals"},
              "limitations": ["Three public authored tasks and one seed are startup diagnostics, not statistically powered transfer evidence.",
                              "B3 is the untrained initial reference, not a mature historical release.",
                              "Same action/time/token ceilings; realized inference costs are reported rather than forced equal.",
                              "B2 is a minimal search baseline and has not been tuned to frontier-agent quality.",
                              "No human comparison or independently secret evaluation is available."]}
    atomic_json(report_path, result)
    return result
