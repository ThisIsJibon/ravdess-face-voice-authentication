# Teammate setup and collaboration

Recommended repository name: `ravdess-face-voice-authentication`.

## 1. Choose how to work

Contribute directly on GitHub using a branch and pull request. You do not need to clone the repository or install Git to edit files this way. Follow section 7 for the contribution workflow. The repository owner reviews and merges pull requests.

Sections 2-6 are optional local execution instructions. To run experiments on your computer without cloning, select the branch you want to test on GitHub, download its source ZIP from the repository's Code menu, extract it, and open a terminal in the extracted folder containing `pyproject.toml`. Install Python 3.11 and create your own environment using the instructions below. Do not copy another person's `.venv`.

A downloaded ZIP is a local snapshot. Changes made inside it do not automatically update GitHub. Edit the corresponding files on your GitHub branch or deliberately upload your changed source files to that branch before opening your pull request.

## 2. Add the dataset

Create `RAVDESS` directly inside the extracted project folder. Keep these 25 original ZIP archives in it, without extracting them manually:

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

To also run the AU and landmark study, download `FacialTracking_Actors_01-24.zip` from the tracking release (https://zenodo.org/records/3255102) and place it in the same `RAVDESS/` folder. Keep it zipped. See [docs/TRACKING_STUDY.md](docs/TRACKING_STUDY.md).

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

### Owner setup

After the initial project upload, invite teammates as collaborators with write access through the repository settings. Teammates accept the invitation before contributing. Configure protection for `main` to require pull requests and owner review where your GitHub plan supports it. The team rule is that only the repository owner merges into `main`.

### Teammate workflow in the browser

1. Open the owner's repository on GitHub and select `main` in the branch selector.
2. Create a new branch from the current `main`, using a descriptive name such as `feature/noise-analysis` or `fix/metric-labels`. Use a separate branch for each task.
3. Confirm the branch selector shows your new branch before changing files.
4. Open the source or documentation file you want to change and select the edit action. Make your changes and preview Markdown when appropriate. For new files, use the repository's add-file action. You can also upload deliberately selected source files changed in a downloaded snapshot.
5. Commit the changes to your feature branch with a clear message. Repeat for other files belonging to the same task. Do not commit directly to `main`.
6. Test changes when possible. For local testing without cloning, download the latest ZIP of your feature branch and follow sections 2-6. Do not claim tests passed unless you ran them. State clearly in the pull request if changes were not tested.
7. Open a pull request with `main` as the base and your feature branch as the compare branch, both in the owner's repository.
8. Give the pull request a descriptive title. Explain what changed, why, how you tested it, and any effect on experiment configuration or existing results.
9. Review the Files changed view and request review from the repository owner. Check that the pull request contains only your intended changes.
10. Address review comments by editing and committing to the same branch. The existing pull request updates automatically. The owner handles approval and merging; teammates should not merge their own pull requests.
11. After the owner merges, create your next task branch from the updated `main`.

### Files to keep out of GitHub

Do not upload `RAVDESS/`, `data/`, `models/`, `results/`, `.venv/`, homework archives, raw recordings, biometric embeddings/templates, credentials, or an entire extracted project folder. Upload only the source or documentation files needed for your change.

The web upload workflow requires manual file selection. Do not rely on the local `.gitignore` to screen files you select in the browser. If an unexpected file appears in the pull request, remove it from the branch before requesting review.

Agree on one experiment configuration before comparing results. Share selected result tables only when the owner explicitly wants them included. Coordinate edits to shared configuration files to avoid conflicting experiment settings. If GitHub reports a merge conflict that you cannot resolve confidently, leave the pull request open and ask the owner to help.

### GitHub documentation

- Editing files: https://docs.github.com/en/repositories/working-with-files/managing-files/editing-files
- Creating pull requests: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request
