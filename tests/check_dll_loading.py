"""
Script to check if all required DLLs can be loaded properly.
Run this script to diagnose DLL loading issues.
"""

import os
import sys
import logging
import ctypes
from ctypes import windll
import platform
import glob

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def check_cuda_environment():
    """Check CUDA environment variables and paths"""
    logger.info("Checking CUDA environment...")

    # Check CUDA_PATH environment variable
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        logger.info(f"CUDA_PATH is set to: {cuda_path}")
        if os.path.exists(cuda_path):
            logger.info(f"✓ CUDA_PATH directory exists")
        else:
            logger.error(f"✗ CUDA_PATH directory does not exist: {cuda_path}")
    else:
        logger.warning("CUDA_PATH environment variable is not set")

    # Check PATH for CUDA directories
    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    cuda_dirs_in_path = [
        d for d in path_dirs if "cuda" in d.lower() or "nvidia" in d.lower()
    ]

    if cuda_dirs_in_path:
        logger.info("CUDA/NVIDIA directories in PATH:")
        for d in cuda_dirs_in_path:
            if os.path.exists(d):
                logger.info(f"  ✓ {d} (exists)")
            else:
                logger.info(f"  ✗ {d} (does not exist)")
    else:
        logger.warning("No CUDA/NVIDIA directories found in PATH")


def check_dll_paths():
    """Check if DLL directories exist and are in PATH"""
    logger.info("\nChecking DLL paths...")

    # Check libs directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    libs_dir = os.path.join(project_root, "libs")

    if os.path.exists(libs_dir):
        logger.info(f"✓ libs directory exists: {libs_dir}")
        dlls = glob.glob(os.path.join(libs_dir, "*.dll"))
        if dlls:
            logger.info(f"  Found {len(dlls)} DLLs in libs directory:")
            for dll in dlls:
                logger.info(f"  - {os.path.basename(dll)}")
        else:
            logger.warning("  No DLLs found in libs directory")
    else:
        logger.error(f"✗ libs directory does not exist: {libs_dir}")

    # Check if libs directory is in PATH
    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    if libs_dir in path_dirs:
        logger.info("✓ libs directory is in PATH")
    else:
        logger.warning("✗ libs directory is not in PATH")


def try_load_dlls():
    """Try to load common CUDA DLLs"""
    logger.info("\nTrying to load CUDA DLLs...")

    # List of common CUDA DLLs to check
    cuda_dlls = [
        "cudart64_12.dll",
        "cublas64_12.dll",
        "cublasLt64_12.dll",
        "cudnn_cnn_infer64_8.dll",
        "cudnn_ops_infer64_8.dll",
        "cudnn64_8.dll",
        "nvcuda.dll",
        "nvrtc64_12_0.dll",
    ]

    for dll_name in cuda_dlls:
        try:
            if sys.platform == "win32":
                dll = windll.LoadLibrary(dll_name)
                logger.info(f"✓ Successfully loaded {dll_name}")
            else:
                logger.info(f"Skipping {dll_name} (not on Windows)")
        except Exception as e:
            logger.error(f"✗ Failed to load {dll_name}: {str(e)}")


def check_torch_cuda():
    """Check if PyTorch can use CUDA"""
    logger.info("\nChecking PyTorch CUDA support...")

    try:
        import torch

        logger.info(f"PyTorch version: {torch.__version__}")

        if torch.cuda.is_available():
            logger.info("✓ CUDA is available for PyTorch")
            logger.info(f"  CUDA version: {torch.version.cuda}")
            logger.info(
                f"  cuDNN version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'Not available'}"
            )
            logger.info(f"  Device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"  Device {i}: {torch.cuda.get_device_name(i)}")
        else:
            logger.error("✗ CUDA is not available for PyTorch")

    except ImportError:
        logger.error("✗ PyTorch is not installed")
    except Exception as e:
        logger.error(f"✗ Error checking PyTorch CUDA support: {str(e)}")


def main():
    """Main function"""
    logger.info(
        f"System: {platform.system()} {platform.release()} {platform.architecture()[0]}"
    )
    logger.info(f"Python: {sys.version}")

    check_cuda_environment()
    check_dll_paths()
    try_load_dlls()
    check_torch_cuda()

    logger.info(
        "\nDLL check complete. If you see errors above, they may indicate why DLLs are not loading properly."
    )


if __name__ == "__main__":
    main()
