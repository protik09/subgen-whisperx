# Setup the DLL/shared lib paths for the program
import os
import sys
import logging
from pathlib import Path


# Improved DLL loading
def setup_dll_paths():
    """Set up paths for DLL loading, especially for PyInstaller bundles."""
    # (Keep existing logic - seems robust enough)
    dll_logger = logging.getLogger("subgen_app.dll_setup")
    dll_logger.debug("Attempting to set up DLL paths...")
    potential_paths = []
    try:
        script_path = Path(os.path.abspath(sys.argv[0]))
        script_dir = script_path.parent
        potential_paths.append(script_dir / "libs")
    except Exception:
        script_dir = None
        dll_logger.warning("Could not reliably determine script directory.")
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).parent.resolve()
        potential_paths.append(exe_dir / "libs")
        potential_paths.append(exe_dir)
        dll_logger.debug(f"Running frozen (PyInstaller?). Executable dir: {exe_dir}")
    else:
        exe_dir = None
    cwd = Path.cwd()
    potential_paths.append(cwd / "libs")
    added_paths_os = set()
    added_paths_env = set(os.environ.get("PATH", "").split(os.pathsep))
    for path in potential_paths:
        if path and path.is_dir():
            path_str = str(path.resolve())
            if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
                if path_str not in added_paths_os:
                    try:
                        os.add_dll_directory(path_str)
                        dll_logger.debug(f"Added to DLL search path (win): {path_str}")
                        added_paths_os.add(path_str)
                    except Exception as e:
                        dll_logger.warning(
                            f"Failed to add path {path_str} via add_dll_directory: {e}"
                        )
            if path_str not in added_paths_env:
                try:
                    os.environ["PATH"] = (
                        path_str + os.pathsep + os.environ.get("PATH", "")
                    )
                    dll_logger.debug(
                        f"Prepended to PATH environment variable: {path_str}"
                    )
                    added_paths_env.add(path_str)
                except Exception as e:
                    dll_logger.warning(
                        f"Failed to prepend path {path_str} to PATH env var: {e}"
                    )
    if sys.platform == "win32":
        cuda_path_env = os.environ.get("CUDA_PATH")
        program_files = os.environ.get("ProgramFiles", "C:\\Program Files")
        nvidia_base = Path(program_files) / "NVIDIA GPU Computing Toolkit" / "CUDA"
        common_cuda_paths = [cuda_path_env]
        if nvidia_base.exists():
            version_dirs = sorted(
                [
                    d
                    for d in nvidia_base.iterdir()
                    if d.is_dir() and d.name.lower().startswith("v")
                ],
                reverse=True,
            )
            common_cuda_paths.extend(version_dirs)
        found_cuda_bin = False
        for cuda_base_path in common_cuda_paths:
            if cuda_base_path:
                cuda_base = Path(cuda_base_path)
                if cuda_base.exists():
                    bin_path = cuda_base / "bin"
                    if bin_path.exists() and bin_path.is_dir():
                        bin_path_str = str(bin_path.resolve())
                        if (
                            hasattr(os, "add_dll_directory")
                            and bin_path_str not in added_paths_os
                        ):
                            try:
                                os.add_dll_directory(bin_path_str)
                                dll_logger.debug(
                                    f"Added CUDA bin path (win): {bin_path_str}"
                                )
                                added_paths_os.add(bin_path_str)
                            except Exception as e:
                                dll_logger.warning(
                                    f"Failed to add CUDA path {bin_path_str} via add_dll_directory: {e}"
                                )
                        if bin_path_str not in added_paths_env:
                            try:
                                os.environ["PATH"] = (
                                    bin_path_str
                                    + os.pathsep
                                    + os.environ.get("PATH", "")
                                )
                                dll_logger.debug(
                                    f"Prepended CUDA bin path to PATH: {bin_path_str}"
                                )
                                added_paths_env.add(bin_path_str)
                            except Exception as e:
                                dll_logger.warning(
                                    f"Failed to prepend CUDA path {bin_path_str} to PATH env var: {e}"
                                )
                        found_cuda_bin = True
                        dll_logger.info(
                            f"Found and added CUDA bin directory: {bin_path_str}"
                        )
                        break
        if not found_cuda_bin:
            dll_logger.debug("Could not find a standard CUDA bin directory.")
    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        dll_logger.debug(
            f"Final os.add_dll_directory paths added: {len(added_paths_os)}"
        )
    dll_logger.debug(
        f"Final PATH (first ~150 chars): {os.environ.get('PATH', '')[:150]}..."
    )
    dll_logger.debug("DLL path setup finished.")
