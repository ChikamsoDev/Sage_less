"""Replaceable, Colab-only open-weight language backbone. No paid endpoints."""
from __future__ import annotations
import gc
import json
import os
import re
import shutil
import time
from pathlib import Path
from .schema import atomic_json, validate_proposal

REGISTRY = {
    "Qwen/Qwen3-8B": {"revision": "b968826d9c46dd6066d109eabc6255188de91218", "license": "apache-2.0"},
    "Qwen/Qwen3-4B": {"revision": "1cfa9a7208912126459214e8b04321603b3df60c", "license": "apache-2.0"},
}


def choose_model(free_gpu_gib, available_ram_gib, requested="auto"):
    if free_gpu_gib < 6 or available_ram_gib < 4:
        raise RuntimeError("At least 6 GiB free GPU memory and 4 GiB available host RAM are required. Select a hosted GPU runtime.")
    if requested != "auto":
        return requested
    # Reserve room for activations, KV cache, LoRA and the rest of the runtime.
    return "Qwen/Qwen3-8B" if free_gpu_gib >= 12 and available_ram_gib >= 7 else "Qwen/Qwen3-4B"


def parse_json(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            result, _ = decoder.raw_decode(text[match.start():])
            if isinstance(result, dict) and ("candidates" in result or "answer" in result or "blocked" in result):
                return validate_proposal(result)
        except (ValueError, TypeError):
            continue
    raise ValueError("Model did not return a valid bounded action proposal")


class ColabLLM:
    def __init__(self, state_dir, model_id="auto", revision=None, context_tokens=4096):
        self.state_dir = Path(state_dir)
        self.requested = model_id
        self.requested_revision = revision
        self.context_tokens = max(1024, min(int(context_tokens), 8192))
        self.model = self.tokenizer = None
        self.manifest = {}
        self.calls = self.generated_tokens = 0

    def load(self):
        import torch
        import psutil
        if not torch.cuda.is_available():
            raise RuntimeError("LLM loading requires a hosted Colab GPU; no CPU weight download is attempted.")
        if not (Path("/content").is_dir() and ("COLAB_RELEASE_TAG" in os.environ or "COLAB_BACKEND_VERSION" in os.environ)):
            try:
                import google.colab  # noqa: F401
            except ImportError as e:
                raise RuntimeError("Large weights may only be loaded inside hosted Google Colab.") from e
        from huggingface_hub import HfApi, hf_hub_download
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        free, total = torch.cuda.mem_get_info()
        name = choose_model(free / 2**30, psutil.virtual_memory().available / 2**30, self.requested)
        if shutil.disk_usage("/content").free < 22 * 2**30:
            raise RuntimeError("Need at least 22 GiB free Colab disk for model shards and working files.")
        attempted = []
        while True:
            rev = self.requested_revision or REGISTRY.get(name, {}).get("revision")
            info = HfApi().model_info(name, revision=rev)
            license_name = (info.card_data or {}).get("license")
            if license_name not in {"apache-2.0", "mit", "bsd-3-clause"}:
                raise RuntimeError(f"Replacement model {name} has unreviewed licence {license_name!r}. Review its licence and extend the allowlist explicitly.")
            rev = info.sha
            license_path = hf_hub_download(name, "LICENSE", revision=rev)
            notice_dir = self.state_dir / "model_notices"
            notice_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(license_path, notice_dir / (name.replace("/", "_") + "-LICENSE"))
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                      bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(name, revision=rev, trust_remote_code=False)
                self.model = AutoModelForCausalLM.from_pretrained(
                    name, revision=rev, trust_remote_code=False, use_safetensors=True,
                    quantization_config=quant, device_map={"": 0}, torch_dtype=dtype,
                    low_cpu_mem_usage=True, attn_implementation="sdpa")
                self.model.eval()
                if self.tokenizer.pad_token_id is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self.manifest = {"id": name, "revision": rev, "license": license_name,
                                 "quantization": "bitsandbytes NF4 double quantization", "gpu": torch.cuda.get_device_name(),
                                 "gpu_total_gib": total / 2**30, "context_tokens": self.context_tokens,
                                 "memory_footprint_bytes": self.model.get_memory_footprint(),
                                 "fallback_attempts": attempted, "loaded_at": time.time()}
                atomic_json(self.state_dir / "backbone_manifest.json", self.manifest)
                return self.manifest
            except torch.cuda.OutOfMemoryError:
                attempted.append({"model": name, "reason": "CUDA out of memory"})
                self.model = self.tokenizer = None
                gc.collect()
                torch.cuda.empty_cache()
                if self.requested != "auto" or name.endswith("4B"):
                    raise
                name = "Qwen/Qwen3-4B"

    def generate(self, messages, max_new_tokens=768, cancel=None):
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList
        class Stop(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return bool(cancel and cancel.is_set())
        if self.model is None:
            raise RuntimeError("Backbone has not been loaded")
        if cancel and cancel.is_set():
            return ""
        # Preserve system and current user instruction; discard oldest observations first.
        messages = list(messages)
        while True:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            batch = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            if batch.input_ids.shape[1] <= self.context_tokens - max_new_tokens:
                break
            if len(messages) > 2:
                messages.pop(1)
            else:
                raise ValueError("Instruction exceeds the configured context budget; shorten the goal or use a workspace file")
        batch = batch.to(self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False,
                                             pad_token_id=self.tokenizer.pad_token_id,
                                             eos_token_id=self.tokenizer.eos_token_id,
                                             stopping_criteria=StoppingCriteriaList([Stop()]))
        out = generated[0, batch.input_ids.shape[1]:]
        self.calls += 1
        self.generated_tokens += int(out.numel())
        return self.tokenizer.decode(out, skip_special_tokens=True)

    def propose(self, system, prompt, cancel=None):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        last = ""
        for attempt in range(2):
            raw = self.generate(messages, cancel=cancel)
            if cancel and cancel.is_set():
                return {"candidates": [], "answer": "", "plan": [], "blocked": "Stopped by owner"}
            try:
                return parse_json(raw)
            except ValueError as e:
                last = str(e)
                messages = [messages[0], {"role": "user", "content": prompt + "\nReturn one JSON object only, with candidates as a list of {tool,args}. Previous output could not be parsed."}]
        raise ValueError(last)

    def smoke_check(self):
        import torch
        reply = self.generate([{"role": "system", "content": "Return exactly a JSON object with answer equal to 2 and candidates an empty list."},
                               {"role": "user", "content": "1+1"}], max_new_tokens=64)
        value = parse_json(reply)
        passed = value["answer"].strip() == "2" and not value["candidates"]
        return {"passed": passed, "response": value, "gpu_allocated_gib": torch.cuda.memory_allocated() / 2**30,
                "four_bit": bool(getattr(self.model, "is_loaded_in_4bit", False))}


class ScriptedLLM:
    """Explicit test fixture, never substituted for a real Colab backbone."""
    def __init__(self, proposals):
        self.proposals = iter(proposals)
        self.manifest = {"id": "scripted-test-fixture", "trained": False}
        self.calls = self.generated_tokens = 0

    def propose(self, system, prompt, cancel=None):
        self.calls += 1
        return validate_proposal(next(self.proposals, {"blocked": "Test script exhausted"}))
