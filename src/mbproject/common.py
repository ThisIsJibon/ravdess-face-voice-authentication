from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]


def read_config(path=None):
    path = Path(path or ROOT / "configs/study.yaml").resolve()
    config = yaml.safe_load(path.read_text())
    for key in ("dataset_dir", "data_dir", "model_dir", "results_dir"):
        p = Path(config[key])
        config[key] = str(p if p.is_absolute() else ROOT / p)
    return config


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False, default=str), encoding="utf-8")
    temp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_seed(text):
    return int.from_bytes(hashlib.sha256(str(text).encode()).digest()[:4], "little")


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    if not np.isfinite(x).all() or (norms < 1e-10).any():
        raise ValueError("Invalid or zero biometric embedding")
    return x / norms


def audio_key(snr):
    return "audio_clean" if snr is None else f"audio_snr{snr:g}"


def face_key(sigma):
    return f"face_sigma{sigma:g}"
