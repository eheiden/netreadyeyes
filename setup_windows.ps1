# Net Ready Eyes Windows setup helper
# Run from the project folder in PowerShell:
#   Set-ExecutionPolicy -Scope Process Bypass
#   .\setup_windows.ps1

$ErrorActionPreference = "Stop"

Write-Host "Net Ready Eyes setup" -ForegroundColor Cyan

if (!(Test-Path ".venv")) {
    py -3 -m venv .venv
}

.\.venv\Scripts\python.exe -m pip install --upgrade pip

if (Test-Path "requirements.txt") {
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
} else {
    Write-Host "requirements.txt not found; installing common runtime packages." -ForegroundColor Yellow
    .\.venv\Scripts\python.exe -m pip install opencv-python pillow numpy requests pandas onnxruntime collector-vision
}

Write-Host ""
Write-Host "Checking optional Node/nvm tools..." -ForegroundColor Cyan
$nvm = Get-Command nvm -ErrorAction SilentlyContinue
if ($nvm) {
    Write-Host "nvm found. If you use the overlay server tooling, Node 20 LTS is recommended."
    try {
        nvm install 20
        nvm use 20
    } catch {
        Write-Host "nvm command failed; install/use Node manually if needed." -ForegroundColor Yellow
    }
} else {
    Write-Host "nvm not found. Node is optional unless you use Node-based helper tooling." -ForegroundColor Yellow
}

.\.venv\Scripts\python.exe .\check_setup.py

Write-Host ""
Write-Host "Setup finished. Start with:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\python.exe .\netreadyeyes.py"
