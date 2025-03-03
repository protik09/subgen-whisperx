# Script configuration
$config = @{
    CondaPath        = "$HOME\miniconda3\condabin"
    CudnnUrl         = "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-8.9.7.29_cuda12-archive.zip"
    RequiredPackages = @{
        Conda  = "Anaconda.Miniconda3"
        FFmpeg = "Gyan.ffmpeg"
    }
}

function Test-CondaInstall {
    try {
        $condaModule = Get-Module -Name (Get-Command conda -CommandType Alias).Source
        return $true
    }
    catch {
        return $false
    }
}

function Test-WhisperXEnv {
    # Test if the WhisperX environment exists in the current folder
    if (Test-Path "$CWD\whisperx") {
        return $true
    }
    return $false
}

function Test-CudnnInstall {
    # Test if CUDNN DLLs are present in the local environment's bin directory
    $whisperxBinPath = "$CWD\whisperx\bin"
    if (Test-Path "$whisperxBinPath\cudnn*.dll") {
        return $true
    }
    return $false
}

function Test-ffmpegInstall {
    if (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
        return $true
    }
    return $false
}

function Install-ffmpeg {
    Write-Host "Installing FFmpeg..."
    winget install -e --id=$config.RequiredPackages.FFmpeg
}

function Install-Conda {
    Write-Host "Installing Conda and FFmpeg..."
    winget install -e --id=$config.RequiredPackages.Conda

    if ($env:Path -split ';' -notcontains $config.CondaPath) {
        $env:Path += ";$config.CondaPath"
        [System.Environment]::SetEnvironmentVariable("Path", $env:Path, [System.EnvironmentVariableTarget]::User)
    }

    # Choco stuff basically there for the magic refreshenv command
    try {
        $env:ChocolateyInstall = Convert-Path "$((Get-Command choco).Path)\..\.."
        Import-Module "$env:ChocolateyInstall\helpers\chocolateyProfile.psm1" -ErrorAction Stop
        refreshenv
        conda init
        refreshenv
        return $true
    }
    catch {
        Write-Error "Error: Chocolatey profile not found. Please install Chocolatey and try again."
        return $false
    }
}

function Install-WhisperX {
    Write-Host "Installing WhisperX environment..."
    # Create the environment in the current folder ($CWD\whisperx)
    conda create --prefix "$CWD\whisperx" python=3.10 -y
    conda activate "$CWD\whisperx"
    conda install -y pytorch==2.0.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
    pip install --upgrade pip  # Upgrade pip to ensure compatibility
    pip install -r requirements.txt
    # The following command is to solve the issue appearing in https://github.com/m-bain/whisperX/issues/983
    pip install torch torchaudio --index-url "https://download.pytorch.org/whl/cu121" --force-reinstall --no-cache-dir
}

# The following function exists to solve the issue appearing in https://github.com/m-bain/whisperX/issues/983
function Install-Cudnn {
    Write-Host "Installing CUDNN..."
    $OUTPUTFILE = Join-Path -Path $CWD -ChildPath "cudnn-windows-x86_64-8.9.7.29_cuda12-archive.zip"

    # If the .zip file already exists, use it; otherwise, download it.
    if (Test-Path $OUTPUTFILE) {
        Write-Host "The .zip file already exists: $OUTPUTFILE"
    }
    else {
        Write-Host "Downloading CUDNN from $($config.CudnnUrl)..."
        Invoke-WebRequest -Uri $config.CudnnUrl -OutFile $OUTPUTFILE
    }

    try {
        Expand-Archive -Path $OUTPUTFILE -DestinationPath "$CWD\cudnn" -Force
        $cudnn_files = Get-ChildItem -Path "$CWD\cudnn\cudnn-windows-x86_64-8.9.7.29_cuda12-archive\bin" -Recurse -Filter "cudnn*.dll"
        
        # Define the local destination path in the local environment
        $localBinPath = "$CWD\whisperx\bin"
        if (-not (Test-Path $localBinPath)) {
            Write-Host "Destination directory does not exist. Creating it: $localBinPath"
            New-Item -Path $localBinPath -ItemType Directory -Force
        }
        
        foreach ($file in $cudnn_files) {
            Write-Host "Copying: $($file.FullName) to $localBinPath"
            Copy-Item -Path $file.FullName -Destination $localBinPath -Force
        }

        # Cleanup: remove the temporary extraction folder
        Remove-Item -Path "$CWD\cudnn" -Recurse -Force
        Remove-Item -Path "$OUTPUTFILE" -Force
        return $true
    }
    catch {
        Write-Error "Error installing CUDNN: $_"
        return $false
    }
}

function Main {
    $FOLDER_HOME = $PSScriptRoot
    Set-Location -Path $FOLDER_HOME
    $CWD = Get-Location
    Write-Host "Working directory: $CWD"

    # Check that winget is installed
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        Write-Error "Error: winget is not installed. Please install winget and try again."
        exit 1
    }

    # Check and install components as needed
    if (-not (Test-CondaInstall)) {
        Write-Host "Conda not found. Installing..."
        if (-not (Install-Conda)) {
            Write-Host "Failed to install Conda. Exiting..."
            exit 1
        }
        Write-Error "Please restart your shell to complete Conda installation."
        exit 0
    }

    if (-not (Test-ffmpegInstall)) {
        Write-Host "FFmpeg not found. Installing..."
        Install-ffmpeg
    }

    if (-not (Test-WhisperXEnv)) {
        Write-Host "WhisperX environment not found. Installing..."
        Install-WhisperX
    }

    if (-not (Test-CudnnInstall)) {
        Write-Host "CUDNN not found. Installing..."
        if (-not (Install-Cudnn)) {
            Write-Host "Failed to install CUDNN. Exiting..."
            exit 1
        }
    }

    # Activate the WhisperX environment if it is not active
    if ($env:CONDA_DEFAULT_ENV -ne "$CWD\whisperx") {
        Write-Host "Activating WhisperX environment..."
        conda activate "$CWD\whisperx"
    }
    else {
        Write-Host "WhisperX environment is already active."
    }
}

# Execute the script
Main
