#!/bin/bash

# Set the folder to the directory where the script is running from
FOLDER=$(dirname "$0")
echo "Folder set to $FOLDER"

# Use find to recursively search for a Python venv
VENV_FOUND=$(find "$FOLDER" -type d \( -name ".venv" -o -name "venv" \))

if [ -z "$VENV_FOUND" ]; then
    echo "No Python venv found. Creating a new one..."
    # Create venv with system packages to avoid external management issues
    sudo apt install python3-venv ffmpeg
    python -m venv .venv
    
    # Determine the activation script path based on OS
    if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        ACTIVATE_SCRIPT=".venv/Scripts/activate"
    else
        ACTIVATE_SCRIPT=".venv/bin/activate"
    fi
    
    source "$ACTIVATE_SCRIPT"
    echo "Installing Python requirements..."
    # Add --break-system-packages flag for newer Python versions
    pip install -r requirements.txt --break-system-packages
    echo "Python venv created and requirements installed."
else
    echo "Python venv found at $VENV_FOUND"
    # Determine the activation script path based on OS
    if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        ACTIVATE_PATH="Scripts"
    else
        ACTIVATE_PATH="bin"
    fi
    
    # Activate the venv
    if [ -n "$BASH_VERSION" ] || [ -n "$ZSH_VERSION" ]; then
        source "$VENV_FOUND/$ACTIVATE_PATH/activate"
    else
        echo "Unsupported shell. Please use Bash or Zsh."
    fi
fi
if ! dpkg -l | grep -q nvidia-cuda-toolkit; then
    echo "Installing NVIDIA CUDA Toolkit..."
    sudo apt install nvidia-cuda-toolkit
else
    echo "NVIDIA CUDA Toolkit is already installed."
fi


if command -v nvcc > /dev/null; then
    echo "NVIDIA CUDA Toolkit installation confirmed." > /dev/null
else
    echo "NVIDIA CUDA Toolkit installation failed. Please check the installation."
    exit 1
fi
cd "$FOLDER" || exit
