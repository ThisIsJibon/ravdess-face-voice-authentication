from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import audio_key, face_key, read_config, write_json


def verify(config,recording_id,claim,snr=None,sigma=0,policy="far_0.01",wav=None,video=None):
    from .experiment import cosine_scores,fuse
    from .features import feature_signature
    out = Path(config["results_dir"])
    completed = json.loads((out/"run_complete.json").read_text())
    if completed["feature_signature"] != feature_signature(config):
        raise ValueError("The authenticator configuration differs from the completed experiment")
    saved = json.loads((out/"policies.json").read_text())
    seed = str(config["enrollment_seeds"][0])
    entry = saved[seed]
    with np.load(out/"authenticator_templates.npz",allow_pickle=False) as template:
        matches = np.flatnonzero(template["actors"] == claim)
        if not len(matches):
            raise ValueError(f"Unknown claimed actor {claim}")
        j = int(matches[0])
        if recording_id:
            manifest = pd.read_csv(Path(config["data_dir"])/"manifest.csv",dtype={"recording_id":str})
            row = manifest.loc[manifest.recording_id == recording_id]
            if len(row) != 1 or row.iloc[0].split != "test":
                raise ValueError("Demo probes must be held-out test recordings")
            with np.load(Path(config["data_dir"])/"features"/f"{recording_id}.npz",allow_pickle=False) as features:
                f,v = features[face_key(sigma)], features[audio_key(snr)]
            source = dict(recording_id=recording_id,true_actor=int(row.iloc[0].actor),split="test")
        else:
            from .features import FaceEncoder,VoiceEncoder,add_noise,load_audio
            signal,_ = load_audio(wav)
            signal,_ = add_noise(signal,snr,config["split_seed"])
            v = VoiceEncoder(config).encode([signal])[0]
            encoder = FaceEncoder(config)
            crops,_ = encoder.crops(video,config["frame_fractions"])
            f,_ = encoder.encode(crops,sigma)
            source = dict(audio_path=str(Path(wav).resolve()),video_path=str(Path(video).resolve()))
        face_score = float(cosine_scores(f[None,:],template["face"][j:j+1])[0,0])
        voice_score = float(cosine_scores(v[None,:],template["voice"][j:j+1])[0,0])
    system = entry["selected_fusion"]
    weight = .5 if system == "fusion_fixed" else entry["face_weight"]
    score = float(fuse(face_score,voice_score,weight))
    threshold = entry["policies"][system][policy]
    return dict(**source,claim_actor=claim,system=system,face_score=face_score,voice_score=voice_score,
        fused_score=score,face_weight=weight,threshold=threshold,policy=policy,
        decision="ACCEPT" if score >= threshold else "REJECT",
        audio_snr_db=snr,face_blur_sigma=sigma,
        scope="Research verification demo; no liveness or replay-attack protection")


def main():
    parser = argparse.ArgumentParser(description="RAVDESS face/voice authentication research")
    parser.add_argument("--config",help="YAML configuration path")
    sub = parser.add_subparsers(dest="command",required=True)
    for command in ("prepare","models","evaluate","plots","audit","all"):
        sub.add_parser(command)
    extract_parser = sub.add_parser("extract")
    extract_parser.add_argument("--limit",type=int,help="Extraction smoke test only; cannot evaluate an incomplete dataset")
    t = sub.add_parser("tracking",help="Facial-tracking (AU + landmark) study; run after 'all'")
    t.add_argument("--refresh",action="store_true",help="Re-extract tracking features from the archive")
    v = sub.add_parser("verify")
    v.add_argument("--recording",help="Held-out recording identifier from data/manifest.csv")
    v.add_argument("--audio",help="External WAV probe (use with --video)")
    v.add_argument("--video",help="External face video probe (use with --audio)")
    v.add_argument("--claim",type=int,required=True)
    v.add_argument("--snr",type=float)
    v.add_argument("--sigma",type=float,default=0)
    v.add_argument("--policy",default="far_0.01",choices=["eer","far_0.01","far_0.001","far_0.0001"])
    args = parser.parse_args()
    config = read_config(args.config)
    if args.command in ("prepare","all"):
        from .dataset import prepare
        prepare(config)
    if args.command == "models":
        from .features import download_models
        download_models(config)
    if args.command in ("extract","all"):
        from .features import extract
        extract(config,getattr(args,"limit",None))
    if args.command in ("evaluate","all"):
        from .experiment import run
        run(config)
    if args.command in ("plots","all"):
        from .plots import generate
        generate(config)
    if args.command in ("audit","all"):
        from .audit import audit
        result = audit(config)
        print(json.dumps(result,indent=2))
        if not result["passed"]:
            raise SystemExit(1)
    if args.command == "tracking":
        from .tracking import extract_tracking
        from .tracking_eval import run_tracking
        if args.refresh:
            extract_tracking(config)
        run_tracking(config)
    if args.command == "verify":
        if bool(args.recording) == bool(args.audio and args.video):
            parser.error("Provide either --recording or both --audio and --video")
        print(json.dumps(verify(config,args.recording,args.claim,args.snr,args.sigma,args.policy,
                                args.audio,args.video),indent=2))


if __name__ == "__main__":
    main()
