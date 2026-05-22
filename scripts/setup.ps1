$ErrorActionPreference = "Stop"

$python = Get-Command py -ErrorAction SilentlyContinue
if ($python) {
    py -3.13 -m venv venv
} else {
    python -m venv venv
}

./venv/Scripts/python.exe -m pip install -r requirements.txt

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example. Please edit .env and set your API keys."
} else {
    Write-Host ".env already exists. Skipped copying .env.example."
}

Write-Host "Setup complete. Run: ./venv/Scripts/python.exe run.py"
