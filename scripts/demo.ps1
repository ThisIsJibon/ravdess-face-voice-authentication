param([int]$TrueActor=1,[int]$ClaimActor=1)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot
$ProjectPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$ProbeId = & $ProjectPython -c "import pandas as pd; f=pd.read_csv('data/manifest.csv'); print(f.loc[(f.split == 'test') & (f.actor == $TrueActor), 'recording_id'].iloc[0])"
if ($LASTEXITCODE -ne 0) { throw 'Could not find test probe.' }
& $ProjectPython -m mbproject verify --recording $ProbeId --claim $ClaimActor
if ($LASTEXITCODE -ne 0) { throw 'Verification failed.' }
