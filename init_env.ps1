# Description: Script to initialize the environment for the project

# Check if uv is installed
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Warning "uv could not be found. Installing ..."
    # Install uv
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
}

# Check if ffmpeg is installed
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Warning "ffmpeg could not be found"
    # Install ffmpeg
    powershell -ExecutionPolicy ByPass -c "winget install --id Gyan.FFmpeg"
}

# Check if venv exists
if (-not (Test-Path ".venv")) {
    Write-Information "Creating virtual environment..."
    uv venv
}

if (-not (Test-Path ".venv\Scripts\activate")) {
    Write-Warning "Virtual environment activation script not found, recreating..."
    uv venv
}

# Activate venv
if (Test-Path ".venv\Scripts\activate") {
    Write-Information "Activating virtual environment..."
    .venv\Scripts\activate
}

# Install requirements
Write-Host "Installing requirements..."
uv pip install --upgrade -r requirements.txt