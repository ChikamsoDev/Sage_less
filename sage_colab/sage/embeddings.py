"""Optional pretrained retrieval encoder, downloaded in hosted Colab only."""
import os
from pathlib import Path


class TextEncoder:
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    # Resolved immutable revision is recorded by the startup code.
    def __init__(self, revision):
        if not Path("/content").is_dir():
            raise RuntimeError("Pretrained encoder download is restricted to hosted Colab")
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=revision, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(self.model_id, revision=revision, trust_remote_code=False, use_safetensors=True).cpu().eval()
        self.version = self.model_id + "@" + revision

    def __call__(self, text):
        import torch
        inputs = self.tokenizer(text, truncation=True, max_length=256, return_tensors="pt")
        with torch.inference_mode():
            hidden = self.model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            vector = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            vector = torch.nn.functional.normalize(vector, p=2, dim=1)
        return vector[0].tolist()
