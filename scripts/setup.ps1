param([switch]$Cpu)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    py -3.11 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 is required.' }
}
$ProjectPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
& $ProjectPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip setup failed.' }
$TorchIndex = if ($Cpu) { 'https://download.pytorch.org/whl/cpu' } else { 'https://download.pytorch.org/whl/cu128' }
& $ProjectPython -m pip install torch==2.7.1 torchaudio==2.7.1 --index-url $TorchIndex
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed.' }
& $ProjectPython -m pip install -e . pytest==8.3.5
if ($LASTEXITCODE -ne 0) { throw 'Project installation failed.' }
& $ProjectPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency conflicts detected.' }
Write-Output 'Ready. Run: .\scripts\run.ps1'
