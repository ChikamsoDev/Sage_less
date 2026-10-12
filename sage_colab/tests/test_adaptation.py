"""Exercise actual adapter gradients on a tiny random model, never download weights."""
from types import SimpleNamespace
import json
import pytest
import torch


class TinyTokenizer:
    eos_token = "\x00"
    def apply_chat_template(self, messages, **kwargs):
        return "Q:" + messages[-1]["content"] + "\nA:"
    def __call__(self, text, return_tensors=None, max_length=None, **kwargs):
        ids = [ord(c) % 255 for c in text]
        if max_length:
            ids = ids[:max_length]
        if return_tensors:
            return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}
        return {"input_ids": ids}


def test_real_lora_gradient_save_and_resume_without_pretrained_download(tmp_path):
    pytest.importorskip("peft")
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from sage.adaptation import train_adapter
    torch.set_num_threads(2)
    config = Qwen3Config(vocab_size=256, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    backbone = SimpleNamespace(model=Qwen3ForCausalLM(config), tokenizer=TinyTokenizer(), manifest={"revision": "tiny-random-local-test"})
    report = train_adapter(backbone, tmp_path, max_steps=2, max_seconds=30)
    assert report["parameters_updated_this_run"]
    assert report["steps"] == 2
    assert report["active"] is False
    checkpoint = torch.load(tmp_path / "lora_candidate/training.pt", weights_only=True)
    assert checkpoint["step"] == 2 and checkpoint["optimizer"]
    assert (tmp_path / "lora_candidate/adapter/adapter_model.safetensors").exists()
    assert train_adapter(backbone, tmp_path, max_steps=2)["complete"]
