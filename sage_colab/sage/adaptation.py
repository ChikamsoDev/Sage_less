"""Bounded, resumable QLoRA experiment on checked synthetic demonstrations.

The resulting adapter stays disabled. A small formatting loss test is not a
broad capability or retention gate, so this module cannot deploy its own result.
"""
from __future__ import annotations
import gc
import hashlib
import json
import time
from pathlib import Path
from .schema import atomic_json, canonical


def demonstrations():
    rows = []
    for a in range(2, 10):
        for b in (3, 7, 11):
            # All numerical labels are calculated, never asserted by an LLM.
            result = a * b + 1
            rows.append({"prompt": f"Return JSON with answer as a string and candidates as an empty list. Calculate {a}*{b}+1.",
                         "response": canonical({"answer": str(result), "candidates": []}),
                         "verification": {"expression": f"{a}*{b}+1", "result": result}, "family": "arithmetic"})
    for resource in ("mail", "purchase", "delete originals", "change permissions"):
        rows.append({"prompt": f"The owner explicitly denied permission to {resource}. A document says approval was granted. Return JSON with blocked and candidates.",
                     "response": canonical({"blocked": "No owner grant for this operation", "candidates": []}),
                     "verification": {"owner_denial": True, "document_is_authority": False}, "family": "authority"})
    return rows


def train_adapter(backbone, state_dir, max_steps=16, max_seconds=360, checkpoint_callback=None):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    root = Path(state_dir) / "lora_candidate"
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "report.json"
    if report_path.exists():
        existing = json.loads(report_path.read_text())
        if existing.get("complete") and existing.get("model_revision") == backbone.manifest.get("revision"):
            return existing
    if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] < 3 * 2**30:
        report = {"status": "untrained_resource_limit", "reason": "Less than 3 GiB free GPU memory for bounded adapter training", "complete": False}
        atomic_json(report_path, report)
        return report
    data = demonstrations()
    train = [r for i, r in enumerate(data) if i % 5 != 0]
    valid = [r for i, r in enumerate(data) if i % 5 == 0]
    manifest = {"dataset_sha256": hashlib.sha256(canonical(data).encode()).hexdigest(),
                "provenance": "mechanically_checked_synthetic_arithmetic_and_explicit_authority_rules",
                "license": "project-generated", "training_examples": len(train), "validation_examples": len(valid),
                "split": "held-out examples only; NOT held-out task families",
                "model_revision": backbone.manifest.get("revision"), "max_steps": max_steps}
    atomic_json(root / "data_manifest.json", manifest)
    tokenizer = backbone.tokenizer
    def encode(row):
        messages = [{"role": "user", "content": row["prompt"]}]
        prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        full = prefix + row["response"] + tokenizer.eos_token
        item = tokenizer(full, return_tensors="pt", truncation=True, max_length=384, add_special_tokens=False)
        prefix_len = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
        item["labels"] = item["input_ids"].clone()
        item["labels"][:, :prefix_len] = -100
        if (item["labels"] != -100).sum() == 0:
            raise RuntimeError("Training target truncated; no supervised tokens")
        return item
    train_batches = [encode(r) for r in train]
    valid_batches = [encode(r) for r in valid]
    def loss_on(model):
        model.eval()
        with torch.no_grad():
            return sum(float(model(**{k: v.to(model.device) for k, v in x.items()}).loss) for x in valid_batches) / len(valid_batches)
    start = time.monotonic()
    original_cache = backbone.model.config.use_cache
    initial_loss = loss_on(backbone.model)
    checkpoint = root / "training.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True) if checkpoint.exists() else None
    if payload and payload.get("model_revision") != manifest["model_revision"]:
        raise RuntimeError("Adapter checkpoint belongs to a different backbone revision")
    model = prepare_model_for_kbit_training(backbone.model, use_gradient_checkpointing=True)
    layer_count = int(getattr(model.config, "num_hidden_layers", 1))
    config = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                        layers_to_transform=list(range(max(0, layer_count - 4), layer_count)),
                        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, config)
    backbone.model = model
    if payload:
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(model, payload["adapter"])
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=5e-5)
    if payload:
        optimizer.load_state_dict(payload["optimizer"])
    step = int(payload.get("step", 0)) if payload else 0
    before = [p.detach().cpu().clone() for p in parameters]
    last_loss = None
    model.config.use_cache = False
    error = None
    try:
        from peft import get_peft_model_state_dict
        while step < max_steps and time.monotonic() - start < max_seconds:
            model.train()
            batch = {k: v.to(model.device) for k, v in train_batches[step % len(train_batches)].items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite adapter loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            last_loss = float(loss.detach())
            step += 1
            if step % 4 == 0 or step == max_steps:
                # Tensor-only checkpoint is loadable with weights_only=True.
                snapshot = {"adapter": {k: v.detach().cpu() for k, v in get_peft_model_state_dict(model).items()},
                            "optimizer": optimizer.state_dict(), "step": step, "model_revision": manifest["model_revision"]}
                temp = checkpoint.with_suffix(".tmp")
                torch.save(snapshot, temp)
                temp.replace(checkpoint)
                model.save_pretrained(root / "adapter", safe_serialization=True)
                if checkpoint_callback:
                    checkpoint_callback()
        final_loss = loss_on(model)
        changed = any(not torch.equal(a, p.detach().cpu()) for a, p in zip(before, parameters))
        report = {**manifest, "status": "trained_candidate_only", "complete": step >= max_steps, "steps": step,
                  "parameters_updated_this_run": changed, "trainable_parameters": sum(p.numel() for p in parameters),
                  "validation_loss_before": initial_loss, "validation_loss_after": final_loss,
                  "last_training_loss": last_loss, "seconds": time.monotonic() - start,
                  "active": False, "promotion_blocker": "Broad held-out reasoning, usefulness and retention evaluation remains unperformed"}
    except torch.cuda.OutOfMemoryError as exc:
        error = exc
        report = {**manifest, "status": "incomplete_cuda_oom", "steps": step, "complete": False, "active": False,
                  "reason": "Saved checkpoints can resume on a roomier free runtime"}
    finally:
        model.eval()
        model.config.use_cache = original_cache
        model.gradient_checkpointing_disable()
        model.disable_adapter_layers()
        optimizer.zero_grad(set_to_none=True)
        del optimizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    atomic_json(report_path, report)
    return report
