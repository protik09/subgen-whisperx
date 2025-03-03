# Description: Script to initialize the environment for the project

# Check if uv is installed
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv could not be found"
    # Install uv
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
}

# Check if venv exists
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..."
    uv venv
}

if (-not (Test-Path ".venv\Scripts\activate")) {
    Write-Host "Virtual environment activation script not found, recreating..."
    uv venv
}

# Activate venv
if (Test-Path ".venv\Scripts\activate") {
    Write-Host "Activating virtual environment..."
    .venv\Scripts\activate
}

# Install requirements
Write-Host "Installing requirements..."
uv pip install -r requirements.txt