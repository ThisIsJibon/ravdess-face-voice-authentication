"""Create a code + compact-results handoff without copying raw biometric media."""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("destination",type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    status = root/"results/full/run_complete.json"
    if not status.exists() or not json.loads((root/"results/full/audit.json").read_text())["passed"]:
        raise SystemExit("Complete and audit the full study before packaging")
    args.destination.parent.mkdir(parents=True,exist_ok=True)
    files = [root/name for name in ["README.md","pyproject.toml","requirements-lock.txt",".gitignore"]]
    for folder in ["src","tests","configs","scripts","docs"]:
        files.extend(p for p in (root/folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts and not p.name.endswith(".pyc"))
    for folder in ["results/full","data","models"]:
        for p in (root/folder).rglob("*"):
            if not p.is_file():
                continue
            relative = p.relative_to(root).as_posix()
            if relative.startswith(("data/media/","data/features/","results/full/trial_scores/","results/full/development_scores/")):
                continue
            if p.suffix.lower() in [".csv",".json",".png"]:
                files.append(p)
    with zipfile.ZipFile(args.destination,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p in sorted(set(files)):
            archive.write(p,Path("MBFinalProject")/p.relative_to(root))
        archive.writestr("MBFinalProject/PACKAGE_CONTENTS.txt",
            "Code and compact results from the completed local study.\n"
            "The full working project remains at the original MBFinalProject path.\n"
            "Raw archives/media, model weights, feature caches, full trial matrices,\n"
            "and the Python environment are intentionally kept in that working folder.\n"
            "Reproduction from scratch requires the original RAVDESS speech archives\n"
            "and scripts/setup.ps1. No report or presentation is included.\n")
    with zipfile.ZipFile(args.destination) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Package integrity check failed")
    print(f"Packaged {len(set(files))} files: {args.destination} ({args.destination.stat().st_size/1024**2:.1f} MiB)")


if __name__ == "__main__":
    main()
