"""Small trainable predictors over extensible typed JSON records.

The hashing input encoder is deliberately fixed and versioned. The GRU, action
projection, ensemble heads, policy and computation controller are learned.
This is structured text grounding, not a pretrained semantic or visual encoder.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

import numpy as np
import torch
from torch import nn

ENCODER_VERSION = "typed-hash-v1"
INPUT_DIM = 192
HIDDEN_DIM = 64
HISTORY_LENGTH = 8
OUTCOMES = ("CONFIRMED_SUCCESS", "CONFIRMED_FAILURE", "UNKNOWN")
CONTROLS = ("inspect", "retrieve", "act", "stop")
# These fields never become model inputs. Labels only enter the loss function.
EXCLUDED = {"id", "episode_id", "timestamp", "provenance", "labels", "hidden_state",
            "training_labels", "verification_result", "source_data_hash", "uuid"}


def _flatten(value: Any, path: str = "", depth: int = 0):
    if depth > 7:
        return
    if isinstance(value, dict):
        for key, item in sorted(value.items()):
            if str(key) not in EXCLUDED:
                yield from _flatten(item, f"{path}.{key}", depth + 1)
    elif isinstance(value, (list, tuple)):
        for item in value[:32]:
            yield from _flatten(item, path + "[]", depth + 1)
    elif isinstance(value, bool):
        yield path + "=" + str(value).lower(), 1.0
    elif isinstance(value, (float, int)):
        if math.isfinite(float(value)):
            yield path + ":present", 1.0
            yield path + ":number", math.copysign(math.log1p(abs(float(value))), float(value)) / 4
    elif value is None:
        yield path + ":unknown", 1.0
    else:
        for word in re.findall(r"[a-z0-9_]+", str(value).lower())[:128]:
            yield path + ":" + word, 1.0


def encode_record(value: dict, dimension: int = INPUT_DIM) -> np.ndarray:
    vector = np.zeros(dimension, dtype=np.float32)
    for token, weight in _flatten(value):
        raw = hashlib.blake2b(token.encode(), digest_size=8).digest()
        index = int.from_bytes(raw[:4], "little") % dimension
        vector[index] += weight if raw[4] % 2 else -weight
    norm = np.linalg.norm(vector)
    return vector / max(float(norm), 1.0)


def encode_history(history: list[dict]) -> np.ndarray:
    events = history[-HISTORY_LENGTH:] or [{}]
    result = np.zeros((HISTORY_LENGTH, INPUT_DIM), dtype=np.float32)
    for index, event in enumerate(events, start=HISTORY_LENGTH - len(events)):
        result[index] = encode_record(event)
    return result


class PredictionHead(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.SiLU())
        self.state = nn.Linear(hidden, INPUT_DIM)
        self.outcome = nn.Linear(hidden, len(OUTCOMES))
        self.binary = nn.Linear(hidden, 2)  # changed, verified completion
        self.delay = nn.Linear(hidden, 1)

    def forward(self, joint):
        z = self.body(joint)
        return {"state": self.state(z), "outcome": self.outcome(z),
                "binary": self.binary(z), "delay": self.delay(z).squeeze(-1)}


class WorldPolicyModel(nn.Module):
    def __init__(self, hidden: int = HIDDEN_DIM, ensemble: int = 3):
        super().__init__()
        self.history = nn.GRU(INPUT_DIM, hidden, batch_first=True)
        self.action = nn.Sequential(nn.Linear(INPUT_DIM, hidden), nn.SiLU())
        self.heads = nn.ModuleList([PredictionHead(hidden) for _ in range(ensemble)])
        self.policy = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.controller = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, 4))
        self.hidden = hidden
        self.ensemble_size = ensemble

    def forward(self, history, action):
        _, encoded = self.history(history)
        belief = encoded[-1]
        joint = torch.cat((belief, self.action(action)), dim=-1)
        return {"heads": [head(joint) for head in self.heads], "belief": belief,
                "policy": self.policy(joint).squeeze(-1), "control": self.controller(belief)}

    def metadata(self):
        return {"encoder_version": ENCODER_VERSION, "input_dim": INPUT_DIM,
                "hidden_dim": self.hidden, "ensemble": self.ensemble_size,
                "parameter_count": sum(parameter.numel() for parameter in self.parameters())}


def parameter_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def canonical_hash(data: Any) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
