# Set the folder to the directory where the script is running from
$FOLDER_HOME = $PSScriptRoot
$VENV_EXIST = 0
Write-Host "Folder set to $FOLDER_HOME"

# Find the location of the conda install
$condaModule = Get-Module -Name (Get-Command conda -CommandType Alias).Source
if ($condaModule) {
    Write-Host "Conda module is available."
    # Check to see if whisperx conda environment is available
    if (conda env list | Select-String -Pattern "whisperx") {
        Write-Host "WhisperX conda environment found."
        # Check to see if the whisperx conda environment is activated
        if ($env:CONDA_DEFAULT_ENV -eq "whisperx") {
            Write-Host "WhisperX conda environment is already activated."
        }
        # If the whisperx conda environment is not activated then activate else do nothing
        else {
            Write-Host "Activating WhisperX conda environment..."
            conda activate whisperx
        }
        $VENV_EXIST = 1
    }
    # Otherwise install the whisperx conda environment
    else {
        Write-Host "WhisperX conda environment not found. Creating...."
        conda create -n whisperx python=3.10 -y
        conda activate whisperx
        conda install -y pytorch==2.0.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
        pip install whisperx ffmpeg python-ffmpeg ffmpeg-python coloredlogs

        # Below is the fix for issues installing whisperx. See (https://github.com/m-bain/whisperX/issues/983) for more details
        pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121 --force-reinstall --no-cache-dir

        # Get zip file from nvidia cudnn
        $url = "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-8.9.7.29_cuda12-archive.zip"
        $outputFile = Join-Path $PWD.Path "cudnn-windows-x86_64-8.9.7.29_cuda12-archive.zip"

        Write-Host "Downloading CUDA cudnn archive..."
        Invoke-WebRequest -Uri $url -OutFile $outputFile

        if (Test-Path $outputFile) {
            Write-Host "Download completed successfully to: $outputFile"
        }
        else {
            Write-Host "Error: Download failed"
            exit 1
        }

        # Unzip the file
        Expand-Archive -Path "$FOLDER_HOME\cudnn-windows-x86_64-8.9.7.29_cuda12-archive.zip" -DestinationPath "$FOLDER_HOME\cudnn" -Force
        Remove-Item -Path "$FOLDER_HOME\cudnn-windows-x86_64-8.9.7.29_cuda12-archive.zip" -Force

        # Locate the files inside the downloaded zip bin folder then extract them to your whisperx environment bin folder
        $cudnn_files = Get-ChildItem -Path "$FOLDER_HOME\cudnn\cudnn-windows-x86_64-8.9.7.29_cuda12-archive\bin" -Recurse -Filter "cudnn*.dll"

        foreach ($file in $cudnn_files) {
            Copy-Item -Path $file.FullName -Destination "$(conda info --base)\envs\whisperx\bin" -Force
        }
        # Remove the extracted folder
        Remove-Item -Path "$FOLDER_HOME\cudnn" -Recurse -Force
        Remove-Item -Path "$FOLDER_HOME\cudnn-windows-x86_64-8.9.7.29_cuda12-archive.zip" -Force

        Write-Host "WhisperX conda environment created and activated."
    }
}
# If the conda module is not available, prompt the user to install it
else {
    Write-Host "Conda module is not available. Do you wish to install it? (y/n)"
    $response = Read-Host

    if ($response -eq 'y' -or $response -eq 'Y') {
        Write-Host "Starting conda installation..."
        winget install -e --id=Anaconda.Miniconda3
        # Refresh shell session and reload environment variables
        # https://stackoverflow.com/questions/46758437/how-to-refresh-the-environment-of-a-powershell-session-after-a-chocolatey-instal

        # Make `refreshenv` available right away, by defining the $env:ChocolateyInstall
        # variable and importing the Chocolatey profile module.
        # Note: Using `. $PROFILE` instead *may* work, but isn't guaranteed to.
        $env:ChocolateyInstall = Convert-Path "$((Get-Command choco).Path)\..\.."
        # Import the Chocolatey profile module and if it fails, exit with an error code.
        try {
            Import-Module "$env:ChocolateyInstall\helpers\chocolateyProfile.psm1" -ErrorAction Stop
        } catch {
            Write-Host "Chocolatey profile module could not be imported. Please install chocolatey and try again."
            exit 1
        }

        # refreshenv is now an alias for Update-SessionEnvironment
        # (rather than invoking refreshenv.cmd, the *batch file* for use with cmd.exe)
        # This should make conda accessible via the refreshed $env:PATH, so that it
        # can be called by name only.
        refreshenv

        # Run conda init
        conda init
        Write-Host "Conda installation completed. Please check for any errors."

    } else {
        Write-Host "Conda installation skipped."
        exit 1
    }
}