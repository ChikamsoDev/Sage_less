"""Hosted Colab startup, automatic verification and persistent application wiring."""
from __future__ import annotations
import atexit
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from .schema import atomic_json, action


def core_checks(project_dir, report_dir):
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "pytest", "-q", "tests", "--junitxml=" + str(report_dir / "pytest.xml")]
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="2")
    result = subprocess.run(cmd, cwd=project_dir, env=env, capture_output=True, text=True, timeout=300)
    report = {"returncode": result.returncode, "stdout": result.stdout[-20000:], "stderr": result.stderr[-10000:],
              "python": sys.version, "platform": platform.platform(), "executed_at": time.time(),
              "large_weights_downloaded": False}
    atomic_json(report_dir / "core_checks.json", report)
    print(result.stdout)
    if result.returncode:
        print(result.stderr)
        raise RuntimeError("Automatic core checks failed. Details are in core_checks.json; unsafe startup is stopped.")
    return report


def gpu_check(report_dir):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Select Runtime → Change runtime type → an available GPU, then Run all. No model download was attempted.")
    x = torch.randn(64, 64, device="cuda", requires_grad=True)
    (x @ x.T).square().mean().backward()
    torch.cuda.synchronize()
    if not torch.isfinite(x.grad).all():
        raise RuntimeError("CUDA gradient check produced non-finite values")
    report = {"passed": True, "name": torch.cuda.get_device_name(), "torch": torch.__version__,
              "cuda": torch.version.cuda, "capability": list(torch.cuda.get_device_capability()),
              "free_gib": torch.cuda.mem_get_info()[0] / 2**30}
    atomic_json(Path(report_dir) / "gpu_check.json", report)
    del x
    torch.cuda.empty_cache()
    return report


def prepare(state_dir, drive_dir, training_steps=120):
    from .persistence import DrivePersistence
    from .authority import AuthorityService
    from .memory import MemoryStore
    from .training import LearningSystem
    from .improvement import ImprovementManager
    state_dir = Path(state_dir)
    persistence = DrivePersistence(state_dir, drive_dir)
    if not state_dir.exists() or not any(state_dir.iterdir()):
        restored = persistence.restore()
        print("Restore:", restored)
    state_dir.mkdir(parents=True, exist_ok=True)
    workspace = state_dir / "workspace"
    owner = AuthorityService(state_dir / "authority", workspace)
    atexit.register(owner.close)
    memory = MemoryStore(state_dir / "memory")
    from .sandbox import probe
    sandbox_status = probe()
    atomic_json(state_dir / "reports" / "sandbox.json", sandbox_status)
    print("Python isolation:", sandbox_status)
    learning = LearningSystem(state_dir, device="cpu")
    # Small models stay on CPU to reserve GPU memory for the backbone and adapter.
    print("Training/resuming the world model, policy and computation controller...")
    trained = learning.bootstrap(max_steps=training_steps, seed=17, checkpoint_callback=persistence.sync)
    print(json.dumps(trained, indent=2))
    evaluated = learning.evaluate()
    atomic_json(state_dir / "reports" / "learning_evaluation.json", evaluated)
    if not (workspace / "example_stock.csv").exists():
        g = owner.create_goal("Install a generated example input", max_actions=1)
        grant = owner.grant(g["id"], tools=["filesystem.write"])
        owner.execute(grant, action("filesystem.write", {"path": "example_stock.csv", "content": "item,quantity,unit_price\nbolt,12,1.50\nnut,20,0.75\nwasher,30,0.25\n"}))
    persistence.sync()
    return {"state_dir": state_dir, "workspace": workspace, "authority": owner, "memory": memory,
            "learning": learning, "persistence": persistence, "improvement": ImprovementManager(state_dir)}


def load_backbone(app, model_id="auto", revision=None, adapter_steps=16, run_comparison=True):
    from .llm import ColabLLM
    from .adaptation import train_adapter
    from .agent import SageAgent
    llm = ColabLLM(app["state_dir"], model_id=model_id, revision=revision)
    print("Downloading the selected open-weight model directly into this hosted Colab runtime...")
    print(json.dumps(llm.load(), indent=2))
    smoke = llm.smoke_check()
    atomic_json(app["state_dir"] / "reports" / "llm_smoke.json", smoke)
    if not smoke["passed"] or not smoke["four_bit"]:
        raise RuntimeError("Backbone generation/4-bit startup check failed: " + json.dumps(smoke))
    app["llm"] = llm
    if adapter_steps:
        print("Training a bounded QLoRA candidate; its weights remain disabled pending broader evaluation.")
        try:
            adapter_report = train_adapter(llm, app["state_dir"], max_steps=adapter_steps,
                                           checkpoint_callback=app["persistence"].sync)
        except Exception as exc:
            adapter_report = {"status": "blocked", "complete": False, "active": False,
                              "reason": f"{type(exc).__name__}: {exc}"}
            atomic_json(app["state_dir"] / "lora_candidate" / "report.json", adapter_report)
        # Confirm that inference still works with the experimental adapter disabled.
        try:
            if hasattr(llm.model, "disable_adapter_layers"):
                llm.model.disable_adapter_layers()
            post_training = llm.smoke_check()
            if not post_training["passed"]:
                raise RuntimeError("Post-training inference check failed")
        except Exception:
            import gc, torch
            llm.model = None
            gc.collect()
            torch.cuda.empty_cache()
            print("Restoring a clean pretrained backbone after the adapter experiment.")
            llm = ColabLLM(app["state_dir"], model_id="Qwen/Qwen3-4B" if model_id == "auto" else model_id, revision=revision)
            llm.load()
            if not llm.smoke_check()["passed"]:
                raise RuntimeError("Clean-backbone recovery failed generation check")
            app["llm"] = llm
        print(json.dumps(adapter_report, indent=2))
    # Licence/revision checks occur before pretrained encoder download.
    try:
        from huggingface_hub import HfApi
        from .embeddings import TextEncoder
        info = HfApi().model_info(TextEncoder.model_id)
        if (info.card_data or {}).get("license") != "apache-2.0":
            raise RuntimeError("Retrieval encoder licence changed; review required")
        encoder = TextEncoder(info.sha)
        app["memory"].set_encoder(encoder, encoder.version)
        atomic_json(app["state_dir"] / "retrieval_encoder.json", {"id": TextEncoder.model_id, "revision": info.sha, "license": "apache-2.0"})
    except Exception as exc:
        # Explicitly reported deterministic baseline; core functionality is usable.
        atomic_json(app["state_dir"] / "retrieval_encoder.json", {"status": "hash_baseline", "reason": f"{type(exc).__name__}: {exc}"})
        print("Pretrained retrieval unavailable; using versioned hash retrieval:", exc)
    app["agent"] = SageAgent(app["state_dir"], app["authority"], app["memory"], app["learning"], llm,
                              persistence=app["persistence"], improvement=app["improvement"])
    if run_comparison:
        from .benchmark import compare_backbone
        report = compare_backbone(llm, app["learning"], app["state_dir"] / "reports" / "backbone")
        summary = {mode: sum(r["verified_completion"] for r in report["results"] if r["mode"] == mode) for mode in ("B0", "B1", "B2", "B3", "B4")}
        print("Automatic task checks passed out of 3 in each condition:", summary)
        print("These small checks do not establish broad superiority or transfer.")
    versions = {}
    for name in ("torch", "transformers", "bitsandbytes", "peft", "accelerate", "numpy"):
        versions[name] = importlib.metadata.version(name)
    atomic_json(app["state_dir"] / "runtime_manifest.json", {"python": sys.version, "packages": versions,
                 "schema_version": 1, "backbone": llm.manifest, "learning_checkpoint": app["learning"].checkpoint_id,
                 "policy_version": 1, "permissions": "owner-granted workspace operations only",
                 "known_limits": "See docs/IMPLEMENTATION_STATUS.md in the embedded source."})
    app["persistence"].sync()
    return app
