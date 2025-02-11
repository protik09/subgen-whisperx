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
        conda create -n whisperx python=3.10 -y pytorch==2.0.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia whisperx gputil

        conda activate whisperx
        Write-Host "WhisperX conda environment created and activated."
    }
} 
# If the conda module is not available, prompt the user to install it
else {
    Write-Host "Conda module is not available. Do you wish to install it? (y/n)"
    $response = Read-Host

    if ($response -eq 'y' -or $response -eq 'Y') {
        Write-Host "Starting conda installation..."
        winget install -e --id=Anaconda.Anaconda3
        # Run conda init
        conda init
        Write-Host "Conda installation completed. Please check for any errors."

    } else {
        Write-Host "Conda installation skipped."
        exit
    }
}