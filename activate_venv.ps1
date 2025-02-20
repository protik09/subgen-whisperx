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
    if (conda env list | Select-String -Pattern "whisperx") {
        return $true
    }
    return $false
}

function Test-CudnnInstall {
    $whisperxBinPath = "$(conda info --base)\envs\whisperx\bin"
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

    try {
        $env:ChocolateyInstall = Convert-Path "$((Get-Command choco).Path)\..\.."
        Import-Module "$env:ChocolateyInstall\helpers\chocolateyProfile.psm1" -ErrorAction Stop
        refreshenv
        conda init
        refreshenv
        return $true
    }
    catch {
        Write-Host "Error: Chocolatey profile not found. Please install Chocolatey and try again."
        return $false
    }
}

function Install-WhisperX {
    Write-Host "Installing WhisperX environment..."
    conda create -n whisperx python=3.10 -y
    conda activate whisperx
    conda install -y pytorch==2.0.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
    pip install whisperx ffmpeg python-ffmpeg ffmpeg-python coloredlogs halo
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121 --force-reinstall --no-cache-dir
}

function Install-Cudnn {
    Write-Host "Installing CUDNN..."
    $outputFile = Join-Path -Path $CWD -ChildPath "cudnn-windows-x86_64-8.9.7.29_cuda12-archive.zip"

    try {
        Invoke-WebRequest -Uri $config.CudnnUrl -OutFile $outputFile
        Expand-Archive -Path $outputFile -DestinationPath "$CWD\cudnn" -Force
        $cudnn_files = Get-ChildItem -Path "$CWD\cudnn\cudnn-windows-x86_64-8.9.7.29_cuda12-archive\bin" -Recurse -Filter "cudnn*.dll"
        
        foreach ($file in $cudnn_files) {
            Copy-Item -Path $file.FullName -Destination "$(conda info --base)\envs\whisperx\bin" -Force
        }

        # Cleanup
        Remove-Item -Path "$CWD\cudnn" -Recurse -Force
        Remove-Item -Path $outputFile -Force
        return $true
    }
    catch {
        Write-Host "Error installing CUDNN: $_"
        return $false
    }
}

function Main {
    $FOLDER_HOME = $PSScriptRoot
    Set-Location -Path $FOLDER_HOME
    $CWD = Get-Location
    Write-Host "Working directory: $CWD"

    # Check and install components as needed
    if (-not (Test-CondaInstall)) {
        Write-Host "Conda not found. Installing..."
        if (-not (Install-Conda)) {
            Write-Host "Failed to install Conda. Exiting..."
            exit 1
        }
        Write-Host "Please restart your shell to complete Conda installation."
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

    # Activate WhisperX environment if not already active
    if ($env:CONDA_DEFAULT_ENV -ne "whisperx") {
        Write-Host "Activating WhisperX environment..."
        conda activate whisperx
    }
    else {
        Write-Host "WhisperX environment is already active."
    }
}

# Execute the script
Main