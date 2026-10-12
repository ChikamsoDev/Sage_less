"""Build the standalone notebook and source archive from checked-in files."""
from __future__ import annotations
import base64
import hashlib
import io
import json
from pathlib import Path
import textwrap
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def source_files():
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if not path.is_file() or any(p.startswith(".") or p == "__pycache__" for p in relative.parts):
            continue
        if path.suffix in {".ipynb", ".zip", ".pyc", ".pt", ".safetensors"} or relative.as_posix() in {"validation/pytest.xml", "validation/notebook.json"}:
            continue
        yield path, relative.as_posix()


def build():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path, relative in source_files():
            info = zipfile.ZipInfo(relative, date_time=(2026, 10, 6, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
    payload = buffer.getvalue()
    checksum = hashlib.sha256(payload).hexdigest()
    (ROOT / "SAGE_Source.zip").write_bytes(payload)
    cells = []
    def markdown(source):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(source).strip() + "\n"})
    def code(source):
        cells.append({"cell_type": "code", "metadata": {}, "source": textwrap.dedent(source).strip() + "\n", "execution_count": None, "outputs": []})

    markdown("""
    # Start SAGE

    Select **Runtime → Change runtime type → an available GPU**, then **Runtime → Run all**.
    Authorise Drive when Google asks. The final cell opens a simple interface inside this notebook.

    This notebook embeds the complete project and tests. Installation and all model downloads run on
    the **hosted Colab machine**, never your local PC. No paid API, API key or purchased compute is used.
    Startup performs local-core tests, actual GPU/model checks, bounded resumable training, and 15 small
    tool-agent diagnostic tasks. Download speed and free GPU availability determine startup time.

    State is saved in `MyDrive/SAGE_Colab_v1`. Keep using that folder to restore memories and trained
    checkpoints. The language adapter is trained as a separate candidate and stays disabled pending
    broader evaluation. Later releases require your approval in the notebook.

    This is an implemented research agent, **not demonstrated AGI/ASI**. Read `docs/IMPLEMENTATION_STATUS.md`
    and the automatically generated reports for exact limits and failed/unverified checks.
    """)
    code('''
    # Optional settings. Defaults require no editing.
    MODEL_ID = "auto"                  # Qwen3-8B or Qwen3-4B, chosen from measured memory
    MODEL_REVISION = None              # Default models have pinned immutable revisions
    WORLD_TRAINING_STEPS = 120         # Total bootstrap steps across restarts, not per session
    ADAPTER_TRAINING_STEPS = 16        # Bounded QLoRA candidate; never silently activated
    RUN_BACKBONE_COMPARISON = True     # Automatic B0–B4 diagnostics, three tasks per condition
    DRIVE_FOLDER = "SAGE_Colab_v1"
    ''')
    encoded = base64.b64encode(payload).decode()
    code(f'''
    # Self-contained source extraction. The source archive is embedded below.
    import base64, hashlib, io, json, os, pathlib, subprocess, sys, zipfile
    try:
        import google.colab
    except ImportError:
        raise RuntimeError("Open this notebook in hosted Google Colab. No local model downloads are supported.")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Choose an available hosted GPU runtime, then Run all. No model weights have been downloaded.")
    ARCHIVE_SHA256 = {checksum!r}
    ARCHIVE_BASE64 = {encoded!r}
    archive_bytes = base64.b64decode(ARCHIVE_BASE64)
    assert hashlib.sha256(archive_bytes).hexdigest() == ARCHIVE_SHA256, "Embedded archive checksum mismatch"
    PROJECT = pathlib.Path("/content") / ("sage_project_" + ARCHIVE_SHA256[:12])
    PROJECT.mkdir(exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        for member in archive.infolist():
            path = pathlib.PurePosixPath(member.filename)
            if path.is_absolute() or ".." in path.parts or member.is_dir():
                raise RuntimeError("Invalid embedded archive member")
            target = PROJECT.joinpath(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(member))
    sys.path.insert(0, str(PROJECT))
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HOME"] = "/content/sage_model_cache"
    # Keep Colab's CUDA torch; install the pinned inference and interface libraries.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
                           "-r", str(PROJECT / "requirements-colab.txt")])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "-e", str(PROJECT)])
    print("Project installed:", PROJECT)
    ''')
    code('''
    # Google requires owner authorisation before access to Drive.
    from google.colab import drive
    drive.mount("/content/drive")
    if not DRIVE_FOLDER or pathlib.Path(DRIVE_FOLDER).name != DRIVE_FOLDER:
        raise ValueError("DRIVE_FOLDER must be a single folder name")
    DRIVE_STATE = pathlib.Path("/content/drive/MyDrive") / DRIVE_FOLDER
    LOCAL_STATE = pathlib.Path("/content/sage_runtime/state")
    print("Persistent storage:", DRIVE_STATE)
    ''')
    code('''
    # Offline core tests exercise actual training, authority, filesystem boundaries and recovery.
    from sage.colab import core_checks, gpu_check
    STARTUP_REPORTS = pathlib.Path("/content/sage_startup_reports")
    CORE_REPORT = core_checks(PROJECT, STARTUP_REPORTS)
    GPU_REPORT = gpu_check(STARTUP_REPORTS)
    print("GPU check:", GPU_REPORT)
    ''')
    code('''
    # Restore an intact snapshot; resume bounded small-model training automatically.
    from sage.colab import prepare
    if "SAGE_APP" in globals():
        if SAGE_APP.get("agent") and SAGE_APP["agent"].active_goal:
            raise RuntimeError("Stop the active goal before rerunning startup cells")
        SAGE_APP["authority"].close()
        SAGE_APP["memory"].close()
        if SAGE_APP.get("llm"):
            SAGE_APP["llm"].model = None
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    SAGE_APP = prepare(LOCAL_STATE, DRIVE_STATE, training_steps=WORLD_TRAINING_STEPS)
    import shutil
    shutil.copytree(STARTUP_REPORTS, LOCAL_STATE / "reports" / "startup", dirs_exist_ok=True)
    ''')
    code('''
    # Download in Colab, verify inference, train a bounded adapter candidate, run tool-agent checks.
    from sage.colab import load_backbone
    SAGE_APP = load_backbone(SAGE_APP, model_id=MODEL_ID, revision=MODEL_REVISION,
                             adapter_steps=ADAPTER_TRAINING_STEPS,
                             run_comparison=RUN_BACKBONE_COMPARISON)
    print("Startup finished. Review reported limitations and use the interface below.")
    ''')
    code('''
    # Notebook-native interface, no public server or tunnel.
    from google.colab import output as colab_output
    colab_output.enable_custom_widget_manager()
    from sage.ui import show_interface
    SAGE_UI = show_interface(SAGE_APP)
    ''')
    markdown('''
    **Use:** enter a goal, optionally specify an output file/content check, then press **Run goal**.
    **Stop / revoke** remains available during work. Use **Add input files** for UTF-8 inputs up to 1 MiB.
    Outputs are in the workspace and the saved Drive snapshot. **Resume saved goal** is an explicit
    owner action; checkpointed initial training resumes automatically on startup.

    Permission records never come from model text. Private goal data is excluded from parameter training
    unless you select its consent checkbox. Review candidate evidence before approving any release.
    A normal prose answer may be useful while remaining `ANSWERED_UNVERIFIED`; only executable checks
    establish the narrower `CONFIRMED_SUCCESS` status.

    **Known limits:** free Colab resources are not guaranteed; hosted GPU/Drive execution was not available
    to the developer. Small synthetic training and public startup diagnostics do not prove broad transfer,
    AGI/ASI, safe arbitrary self-modification, or general superiority. Vision/audio/robotics and external
    account tools are not implemented. Python execution requires Linux Landlock/seccomp and blocks if unavailable.
    ''')
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"colab": {"name": "Start_SAGE.ipynb", "provenance": []},
                             "accelerator": "GPU", "kernelspec": {"display_name": "Python 3", "name": "python3"},
                             "language_info": {"name": "python"}}, "cells": cells}
    for i, cell in enumerate(cells):
        cell["id"] = f"sage-{i:02d}"
    (ROOT / "Start_SAGE.ipynb").write_text(json.dumps(notebook, indent=1))
    print(json.dumps({"notebook": str(ROOT / "Start_SAGE.ipynb"), "archive_bytes": len(payload), "sha256": checksum}))


if __name__ == "__main__":
    build()
