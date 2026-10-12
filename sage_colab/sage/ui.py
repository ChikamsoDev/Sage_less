"""Notebook-native owner interface. No tunnel, public server or paid API."""
from __future__ import annotations
import html
import json
import threading
from pathlib import Path
from .schema import action


def show_interface(app):
    import ipywidgets as w
    from IPython.display import display, Markdown
    agent = app["agent"]
    output = w.Output(layout={"border": "1px solid #ccc", "max_height": "500px", "overflow": "auto"})
    status = w.HTML("<b>SAGE is ready.</b> Workspace files and trained checkpoints persist to your authorised Drive folder.")
    goal = w.Textarea(placeholder="Example: Read example_stock.csv, calculate total quantity, and save a short report as stock_report.txt. Keep the original intact.",
                      layout={"width": "100%", "height": "100px"})
    expected_path = w.Text(value="", description="Output file:", placeholder="Optional exact completion path, e.g. stock_report.txt", layout={"width": "95%"})
    required_text = w.Text(value="", description="Must include:", placeholder="Optional content check", layout={"width": "95%"})
    consent = w.Checkbox(value=False, description="Allow this goal's verified experience to train future model candidates", indent=False)
    run = w.Button(description="Run goal", button_style="success", icon="play")
    stop = w.Button(description="Stop / revoke", button_style="danger", icon="stop")
    save = w.Button(description="Save to Drive", icon="save")
    upload = w.FileUpload(accept=".txt,.md,.csv,.json,.py", multiple=True, description="Add input files")
    saved = w.Dropdown(description="Saved goal:", options=[] , layout={"width": "90%"})
    resume = w.Button(description="Resume saved goal")
    refresh = w.Button(description="Refresh files & goals")
    files = w.HTML()
    file_choice = w.Dropdown(description="Result file:", options=[], layout={"width": "90%"})
    preview = w.Button(description="Preview / download")
    download_link = w.HTML()
    worker = {"thread": None}

    def busy():
        return worker["thread"] is not None and worker["thread"].is_alive()

    def refresh_list(_=None):
        states = agent.saved_runs()
        saved.options = [(s["goal"]["instruction"][:55] + " — " + s["status"], s["goal"]["id"]) for s in states]
        workspace = Path(app["workspace"])
        names = sorted(str(p.relative_to(workspace)) for p in workspace.rglob("*") if p.is_file() and not p.is_symlink())[:80]
        files.value = "<b>Workspace files</b><br>" + "<br>".join(html.escape(n) for n in names)
        file_choice.options = names

    def on_preview(_):
        if not file_choice.value or busy():
            return
        import base64
        owner = app["authority"]
        name = file_choice.value
        g = owner.create_goal("Owner preview: " + name, max_actions=1)
        receipt = owner.execute(owner.grant(g["id"], tools=["filesystem.read"]), action("filesystem.read", {"path": name}))
        if receipt["status"] != "CONFIRMED_SUCCESS":
            emit(str(receipt["evidence"]))
            return
        content = receipt["evidence"]["content"]
        encoded = base64.b64encode(content.encode()).decode()
        download_link.value = '<a download="' + html.escape(Path(name).name, quote=True) + '" href="data:text/plain;base64,' + encoded + '">Download ' + html.escape(name) + '</a>'
        emit(content[:10000])

    def emit(message):
        with output:
            print(message, flush=True)

    def launch(goal_id, is_resume=False):
        run.disabled = resume.disabled = save.disabled = True
        status.value = "<b>Working.</b> Stop / revoke remains available."
        def work():
            try:
                result = agent.run(goal_id, emit=emit, resume=is_resume)
                with output:
                    print("\n" + result["status"])
                    display(Markdown(result.get("answer") or "Result recorded."))
                    if result.get("completion_evidence"):
                        print("Verified:", result["completion_evidence"])
                    if result.get("consolidation"):
                        print("Learning candidate:", json.dumps(result["consolidation"], default=str)[:2500])
                    if result.get("improvement"):
                        print("Isolated improvement:", json.dumps(result["improvement"], default=str)[:2500])
                status.value = "<b>" + html.escape(result["status"]) + "</b> — progress and evidence saved."
            except Exception as exc:
                emit(f"{type(exc).__name__}: {exc}")
                status.value = "<b>Blocked.</b> The exact reason is in the log."
            finally:
                run.disabled = resume.disabled = save.disabled = False
                refresh_list()
        worker["thread"] = threading.Thread(target=work, daemon=True)
        worker["thread"].start()

    def on_run(_):
        if busy():
            return
        try:
            condition = {"type": "artifact", "path": expected_path.value.strip(),
                         "contains": [required_text.value] if required_text.value else []} if expected_path.value.strip() else {"type": "answer"}
            goal_id = agent.create(goal.value, completion=condition, training_consent=consent.value)
            output.clear_output()
            launch(goal_id)
        except Exception as exc:
            emit(str(exc))

    def on_stop(_):
        agent.stop()
        status.value = "<b>Stop requested; grant revoked.</b> Already completed effects remain in the evidence log."

    def on_save(_):
        if busy():
            return
        try:
            emit(json.dumps(app["persistence"].sync()))
        except Exception as exc:
            emit("Drive save failed: " + str(exc))

    def on_upload(change):
        if busy():
            emit("Finish or stop the current goal before adding files.")
            return
        items = change["new"]
        if isinstance(items, dict):
            items = [dict(v, name=k) for k, v in items.items()]
        for item in items:
            try:
                name = item["name"]
                if Path(name).name != name or name in {".", ".."}:
                    raise ValueError("Input name must be a single filename")
                content = bytes(item["content"])
                if len(content) > 1024 * 1024:
                    raise ValueError("Input limit is 1 MiB per file")
                text = content.decode("utf-8")
                owner = app["authority"]
                g = owner.create_goal("Owner upload: " + name, max_actions=1)
                grant = owner.grant(g["id"], tools=["filesystem.write"])
                receipt = owner.execute(grant, action("filesystem.write", {"path": name, "content": text}))
                emit(name + ": " + receipt["status"])
                if receipt["status"] != "CONFIRMED_SUCCESS":
                    emit(str(receipt.get("evidence")))
            except Exception as exc:
                emit("Input file rejected: " + str(exc))
        on_save(None)
        refresh_list()

    def on_resume(_):
        if saved.value and not busy():
            output.clear_output()
            launch(saved.value, True)

    run.on_click(on_run)
    stop.on_click(on_stop)
    save.on_click(on_save)
    resume.on_click(on_resume)
    refresh.on_click(refresh_list)
    preview.on_click(on_preview)
    upload.observe(on_upload, names="value")
    refresh_list()
    review = w.Output()
    candidate_id = w.Text(description="Candidate ID:", layout={"width": "90%"})
    candidate_kind = w.Dropdown(description="Type:", options=["planner", "learning"])
    inspect = w.Button(description="Inspect release")
    approve = w.Button(description="Approve inspected release", button_style="warning")
    rollback = w.Button(description="Roll back selected type")
    inspected = {"key": None}
    def inspect_release(_):
        with review:
            review.clear_output()
            if busy():
                print("Stop or finish the current goal before release operations.")
                return
            identifier = candidate_id.value.strip()
            if not identifier or Path(identifier).name != identifier:
                print("Enter an exact candidate ID from an experiment report.")
                return
            root = Path(app["state_dir"])
            matches = list(root.rglob("*" + identifier + "*.json")) + list(root.rglob(identifier + "/report.json")) + list(root.rglob(identifier + "/manifest.json"))
            if not matches:
                print("No matching release evidence found.")
                return
            for p in matches[:4]:
                print(str(p.relative_to(root)))
                print(p.read_text()[:14000])
            inspected["key"] = (candidate_kind.value, identifier)
    def promote(_):
        with review:
            if busy() or inspected["key"] != (candidate_kind.value, candidate_id.value.strip()):
                print("Inspect this exact release first while the agent is idle.")
                return
            try:
                if candidate_kind.value == "planner":
                    result = app["improvement"].promote(candidate_id.value.strip(), approved=True)
                else:
                    result = app["learning"].promote_checkpoint(candidate_id.value.strip(), approved=True)
                print(result)
                on_save(None)
            except Exception as exc:
                print(type(exc).__name__ + ": " + str(exc))
    def rollback_release(_):
        with review:
            if busy():
                print("Stop or finish the current goal first.")
                return
            target = app["improvement"] if candidate_kind.value == "planner" else app["learning"]
            try:
                print(target.rollback(approved=True))
                on_save(None)
            except Exception as exc:
                print(str(exc))
    inspect.on_click(inspect_release)
    approve.on_click(promote)
    rollback.on_click(rollback_release)
    release = w.Accordion(children=[w.VBox([candidate_kind, candidate_id, w.HBox([inspect, approve, rollback]), review])])
    release.set_title(0, "Review experimental releases (owner only)")
    release.selected_index = None
    display(w.VBox([status, goal, expected_path, required_text, consent, w.HBox([run, stop, save]),
                    upload, w.HBox([resume, refresh]), saved, files, file_choice, preview, download_link, output, release]))
    return {"goal": goal, "run": run, "stop": stop, "output": output, "worker": worker}
