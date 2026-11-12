"""Frozen identity features; measurable, repeatable probe-only degradations."""
from __future__ import annotations

import importlib.metadata
import json
import math
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import cv2
import numpy as np
import pandas as pd
import requests
import soundfile as sf
import torch
from scipy.signal import resample_poly

from .common import audio_key, face_key, sha256, stable_seed, unit, write_json

OPENCV_REVISION = "48b10fa505aeb51fbd127fc60a5c660040e823b7"
ECAPA_REVISION = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
MODEL_URLS = {
    "yunet.onnx": f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{OPENCV_REVISION}/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "sface.onnx": f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{OPENCV_REVISION}/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    "ecapa.ckpt": f"https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/{ECAPA_REVISION}/embedding_model.ckpt",
}


def download_models(config):
    folder = Path(config["model_dir"])
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    records = {}
    for name, url in MODEL_URLS.items():
        path = folder / name
        if not path.exists():
            print(f"Downloading {name} from pinned upstream revision", flush=True)
            temp = path.with_suffix(path.suffix + ".tmp")
            with requests.get(url, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                with temp.open("wb") as stream:
                    for block in response.iter_content(1024 * 1024):
                        stream.write(block)
            if temp.stat().st_size < 100000:
                raise ValueError(f"Invalid model payload: {name}")
            temp.replace(path)
        digest = sha256(path)
        if name in previous and previous[name]["sha256"] != digest:
            raise ValueError(f"Model checksum changed: {name}")
        records[name] = dict(url=url, sha256=digest, bytes=path.stat().st_size)
    write_json(manifest_path, records)
    return records


def add_noise(waveform, snr_db, seed):
    waveform = np.asarray(waveform, dtype=np.float32)
    if snr_db is None:
        return waveform.copy(), None
    power = np.mean(waveform.astype(np.float64) ** 2)
    if power <= 1e-12:
        raise ValueError("Cannot set SNR for silent audio")
    noise = np.random.default_rng(seed).normal(size=waveform.shape)
    noise *= np.sqrt(power / (10 ** (snr_db / 10) * np.mean(noise ** 2)))
    noisy = (waveform + noise).astype(np.float32)
    measured = 10 * np.log10(power / np.mean((noisy.astype(np.float64) - waveform) ** 2))
    return noisy, float(measured)


def load_audio(path):
    signal, sr = sf.read(path, dtype="float32", always_2d=True)
    signal = signal.mean(axis=1)
    signal -= signal.mean()
    if sr != 16000:
        divisor = math.gcd(sr, 16000)
        signal = resample_poly(signal, 16000 // divisor, sr // divisor).astype(np.float32)
    # Estimate boundaries on CLEAN audio once, retaining the same samples at all SNRs.
    width = 320
    padded = np.pad(signal, (0, (-len(signal)) % width))
    energy = np.mean(padded.reshape(-1, width) ** 2, axis=1)
    active = np.flatnonzero(energy > max(energy.max() * 10 ** (-35 / 10), 1e-10))
    if not len(active):
        raise ValueError(f"No speech energy in {path}")
    start = max(0, int(active[0] * width) - 1600)
    end = min(len(signal), int((active[-1] + 1) * width) + 1600)
    return signal[start:end], dict(sample_rate=16000, original_seconds=len(signal)/16000,
                                   trim_start_sample=start, trim_end_sample=end,
                                   retained_seconds=(end-start)/16000)


class VoiceEncoder:
    def __init__(self, config):
        # Explicit upstream architecture avoids executing downloaded YAML/Python.
        from speechbrain.lobes.features import Fbank
        from speechbrain.processing.features import InputNormalization
        from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
        requested = config["device"]
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested
        torch.set_num_threads(config["torch_threads"])
        torch.manual_seed(config["split_seed"])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        self.fbank = Fbank(n_mels=80).to(self.device)
        self.norm = InputNormalization(norm_type="sentence", std_norm=False).to(self.device)
        self.model = ECAPA_TDNN(input_size=80, channels=[1024,1024,1024,1024,3072],
            kernel_sizes=[5,3,3,3,1], dilations=[1,2,3,4,1],
            attention_channels=128, lin_neurons=192).to(self.device).eval()
        weights = torch.load(Path(config["model_dir"]) / "ecapa.ckpt", map_location="cpu", weights_only=True)
        self.model.load_state_dict(weights, strict=True)

    @torch.inference_mode()
    def encode(self, waveforms):
        batch = torch.from_numpy(np.stack(waveforms)).to(self.device)
        lengths = torch.ones(len(waveforms), device=self.device)
        features = self.norm(self.fbank(batch), lengths)
        return unit(self.model(features, lengths).squeeze(1).cpu().numpy())


class FaceEncoder:
    def __init__(self, config):
        cv2.setNumThreads(2)
        self.detector = cv2.FaceDetectorYN.create(str(Path(config["model_dir"])/"yunet.onnx"),
                                                 "", (640,360), 0.8, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(str(Path(config["model_dir"])/"sface.onnx"), "")

    def crops(self, path, fractions):
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        crops, frames, probabilities, detection_counts = [], [], [], []
        try:
            for fraction in fractions:
                index = int((total-1) * fraction)
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, image = cap.read()
                if not ok:
                    raise ValueError(f"Cannot decode frame {index} in {path}")
                height, width = image.shape[:2]
                image = cv2.resize(image, (640, round(height * 640/width)))
                self.detector.setInputSize((image.shape[1],image.shape[0]))
                _, faces = self.detector.detect(image)
                if faces is None or len(faces) == 0:
                    raise ValueError(f"No face detected at frame {index} in {path}")
                # RAVDESS clips contain one actor. Smaller false detections can
                # occur on hair; choose the largest area without identity labels.
                selected = faces[int(np.argmax(faces[:,2]*faces[:,3]))]
                crops.append(self.recognizer.alignCrop(image, selected))
                frames.append(index)
                probabilities.append(float(selected[-1]))
                detection_counts.append(len(faces))
        finally:
            cap.release()
        return crops, dict(frame_indices=frames, detector_probabilities=probabilities,
                           detected_face_counts=detection_counts,selection_rule="largest_detected_box_area",
                           aligned_size=[112,112], frame_count=total)

    def encode(self, crops, sigma):
        vectors, sharpness = [], []
        for crop in crops:
            image = crop if sigma == 0 else cv2.GaussianBlur(crop, (0,0), sigmaX=sigma,
                                                           sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
            sharpness.append(float(cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()))
            vectors.append(unit(self.recognizer.feature(image).reshape(-1)))
        return unit(np.mean(vectors, axis=0)), float(np.mean(sharpness))


def feature_signature(config):
    import hashlib
    payload = {key: config[key] for key in ("audio_snr_db", "face_blur_sigma", "frame_fractions", "split_seed")}
    payload["models"] = json.loads((Path(config["model_dir"])/"manifest.json").read_text())
    payload["implementation_sha256"] = sha256(__file__)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def extract(config, limit=None):
    download_models(config)
    data = Path(config["data_dir"])
    folder = data / "features"
    folder.mkdir(exist_ok=True)
    frame = pd.read_csv(data / "manifest.csv", dtype={"recording_id":str})
    if limit:
        frame = frame.head(limit)
    signature = feature_signature(config)
    face, voice = None, None
    failures = []
    start = time.perf_counter()
    for position, row in enumerate(frame.itertuples(),1):
        target = folder / f"{row.recording_id}.npz"
        metadata_path = target.with_suffix(".json")
        if target.exists() and metadata_path.exists():
            meta = json.loads(metadata_path.read_text())
            if meta.get("signature") == signature:
                if position % 100 == 0:
                    print(f"Cached {position}/{len(frame)} recordings",flush=True)
                continue
            raise ValueError(f"Stale feature cache: {target}. Use a new data directory or explicitly remove the old features.")
        try:
            if face is None:
                face, voice = FaceEncoder(config), VoiceEncoder(config)
                print(f"Voice inference device: {voice.device}; face inference: OpenCV CPU",flush=True)
            t0 = time.perf_counter()
            signal, audio_meta = load_audio(data / row.audio_path)
            noisy, snrs = [], {}
            # Same underlying Gaussian realization at each SNR isolates severity.
            for snr in config["audio_snr_db"]:
                wave, measured = add_noise(signal, snr, stable_seed(f"noise:{config['split_seed']}:{row.recording_id}"))
                noisy.append(wave)
                snrs[audio_key(snr)] = measured
            embeddings = {audio_key(s): v for s,v in zip(config["audio_snr_db"],voice.encode(noisy))}
            crops, face_meta = face.crops(data / row.video_path, config["frame_fractions"])
            sharpness = {}
            for sigma in config["face_blur_sigma"]:
                embeddings[face_key(sigma)], sharpness[face_key(sigma)] = face.encode(crops,sigma)
            temp = target.with_suffix(".tmp.npz")
            np.savez_compressed(temp, **embeddings)
            temp.replace(target)
            write_json(metadata_path, dict(recording_id=row.recording_id,signature=signature,
                measured_snr_db=snrs, laplacian_variance=sharpness, audio=audio_meta,
                face=face_meta, extraction_seconds=time.perf_counter()-t0))
        except Exception as exc:
            failures.append(dict(recording_id=row.recording_id,error=str(exc)))
            write_json(data/"extraction_failures.json",failures)
            raise RuntimeError(f"Acquisition failed for {row.recording_id}; no silent sample exclusion") from exc
        if position % 20 == 0 or position == len(frame):
            elapsed = time.perf_counter()-start
            print(f"Extracted {position}/{len(frame)} recordings | elapsed {elapsed/60:.1f} min",flush=True)
    write_json(data / "extraction_run.json",dict(recordings=len(frame),signature=signature,
        seconds=time.perf_counter()-start,failures=failures,python=platform.python_version(),
        cuda_available=torch.cuda.is_available(),gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        versions={name:importlib.metadata.version(name) for name in ["torch","torchaudio","speechbrain","opencv-python-headless","numpy"]}))
    if not limit:
        write_json(data / "extraction_failures.json",failures)
