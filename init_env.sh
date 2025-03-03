# Description: Script to initialize the environment for the project

# Function to check NVIDIA GPU
check_nvidia() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "No NVIDIA GPU detected"
        return 1
    fi
    echo "NVIDIA GPU detected"
    return 0
}

# Function to check CUDA
check_cuda() {
    if ! command -v nvcc &> /dev/null; then
        echo "CUDA not found"
        return 1
    fi
    echo "CUDA $(nvcc --version | grep "release" | awk '{print $5}' | cut -c2-) detected"
    return 0
}

# Check and install NVIDIA components
if ! check_nvidia; then
    echo "Please install NVIDIA drivers from: https://www.nvidia.com/Download/index.aspx"
    exit 1
fi

if ! check_cuda; then
    echo "Please install CUDA toolkit from: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

# Check if cuDNN is installed
if [ ! -d "/usr/local/cuda/include/cudnn.h" ] && [ ! -d "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/*/include/cudnn.h" ]; then
    echo "cuDNN not found. Please install from: https://developer.nvidia.com/cudnn"
    exit 1
fi

# Check if uv is installed
if ! command -v uv &> /dev/null
then
    echo "uv could not be found"
    # Install uv
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Check if venv exists
if [ ! -d ".venv" ]; then
    uv venv
fi

if [ ! -f ".venv/Scripts/activate" ]; then
    uv venv
fi

# Activate venv
if [-f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
fi

# Install requirements
uv pip install -r requirements.txt