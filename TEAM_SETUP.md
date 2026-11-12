# Teammate setup and collaboration

Recommended repository name: `ravdess-face-voice-authentication`.

## 1. Clone the repository

Install Git and Python 3.11. Clone your team's repository, then open a terminal in its root:

```text
git clone https://github.com/YOUR-OWNER/ravdess-face-voice-authentication.git
cd ravdess-face-voice-authentication
```

Do not copy another person's `.venv`. Every teammate creates their own environment.

## 2. Add the dataset

Create `RAVDESS` directly inside the cloned repository. Keep these 25 original ZIP archives in it, without extracting them manually:

```text
ravdess-face-voice-authentication/
  pyproject.toml
  configs/
  scripts/
  src/
  RAVDESS/
    Audio_Speech_Actors_01-24.zip
    Video_Speech_Actor_01.zip
    Video_Speech_Actor_02.zip
    ...
    Video_Speech_Actor_24.zip
```

You need the audio speech archive and all 24 video speech archives. Audio alone cannot run the face experiments. Song archives are not needed. The program extracts only eligible paired speech recordings into `data/media`.

Dataset source: https://doi.org/10.5281/zenodo.1188976

Observe the dataset's CC BY-NC-SA 4.0 terms and attribution requirements. Download data locally instead of uploading it into GitHub. Allow substantial free disk space for the archives, extracted videos, Python environment and models.

## 3. Install on Windows

From PowerShell in the repository root, use the CPU option unless you have a compatible NVIDIA GPU and driver:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1 -Cpu
```

For the CUDA 12.8 NVIDIA build used on the original workstation:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
```

The execution-policy option applies to that process. Activation is unnecessary; invoke the environment's Python directly. If CUDA cannot initialize, use the CPU setup instead. CPU inference takes longer.

## 4. Install on macOS or Linux

The PowerShell setup script targets Windows. For a CPU environment on another OS:

```bash
python3.11 -m venv .venv
# Linux
.venv/bin/python -m pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cpu
# macOS: use the standard PyTorch wheels instead of the CPU index
# .venv/bin/python -m pip install torch==2.7.1 torchaudio==2.7.1
.venv/bin/python -m pip install -e . pytest==8.3.5
.venv/bin/python -m pip check
```

Use `.venv/bin/python` in place of `.venv\Scripts\python.exe` in the commands below. These are installation instructions; the completed study was validated on Windows, not independently on macOS or Linux. Python 3.12 or later is outside this project's declared supported version.

## 5. Run the project

Windows:

```powershell
# Validate the implementation.
& .\.venv\Scripts\python.exe -m pytest -q

# Prepare media, download models, extract features, evaluate, plot and audit.
& .\.venv\Scripts\python.exe -u -m mbproject all
```

Internet access is needed for installation and the first model downloads. The biometric experiments process data locally. Valid existing feature caches are reused.

Run individual stages when troubleshooting:

```powershell
& .\.venv\Scripts\python.exe -m mbproject prepare
& .\.venv\Scripts\python.exe -m mbproject models
& .\.venv\Scripts\python.exe -u -m mbproject extract
& .\.venv\Scripts\python.exe -m mbproject evaluate
& .\.venv\Scripts\python.exe -m mbproject plots
& .\.venv\Scripts\python.exe -m mbproject audit
```

The copied HW3 evaluator is already included under `src/mbproject/vendor`; teammates do not need the homework ZIP archives. Models download automatically. Keep `configs/study.yaml` unchanged for comparable experiments. Its relative directories resolve against each teammate's repository root, so nobody needs the original workstation's absolute path. Changing extraction settings or code can invalidate cached embeddings; use a separate data directory for a different experiment.

Do not install `requirements-lock.txt` indiscriminately on other operating systems: it records the original Windows CUDA environment. The setup commands above use the project's pinned direct dependencies and an appropriate PyTorch build.

## 6. Find results and try authentication

Results are generated in `results/full`. Start with `summary.csv`, `metrics.csv`, `figures/`, and `audit.json`. A complete run produces `run_complete.json`; a partial run must not be treated as completed results.

```powershell
# Genuine claim, then an impostor claim using a held-out actor 1 recording.
powershell -ExecutionPolicy Bypass -File .\scripts\demo.ps1 -TrueActor 1 -ClaimActor 1
powershell -ExecutionPolicy Bypass -File .\scripts\demo.ps1 -TrueActor 1 -ClaimActor 2
```

Floating-point results can differ slightly across hardware. Seeds do not create new independent test recordings. This is an offline research prototype without liveness protection.

## 7. Work together

The repository owner should create an empty GitHub repository, then run these commands from this project folder:

```text
git init
git branch -M main
git add .
git status --short
git diff --cached --stat
git commit -m "Add face and voice authentication research pipeline"
git remote add origin https://github.com/YOUR-OWNER/ravdess-face-voice-authentication.git
git push -u origin main
```

Review the staged files before committing. Dataset archives, extracted data, embeddings, templates, model weights, environments, homework archives and generated results should not appear. Ignore rules do not remove files that were already tracked. If this is an existing Git repository, inspect its current branch and remotes instead of blindly repeating initialization or adding an existing remote.

In GitHub repository settings, invite teammates as collaborators with write access. Each teammate should use a branch and pull request:

```text
git pull origin main
git switch -c feature/short-description
# Make and test changes.
git add PATHS-YOU-CHANGED
git commit -m "Describe the change"
git push -u origin feature/short-description
```

Open a pull request into `main` and request review. Agree on one experiment configuration before comparing results. Share selected result tables through pull requests only if deliberately chosen; generated results are ignored by default. Do not commit credentials, biometric templates or raw recordings.
