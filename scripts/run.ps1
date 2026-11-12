param([ValidateSet('all','prepare','models','extract','evaluate','plots','audit','test')][string]$Stage='all')
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot
$ProjectPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $ProjectPython)) { throw 'Run scripts\setup.ps1 first.' }
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
if ($Stage -eq 'test') { & $ProjectPython -m pytest -q }
else { & $ProjectPython -u -m mbproject $Stage }
if ($LASTEXITCODE -ne 0) { throw "Stage failed: $Stage" }
