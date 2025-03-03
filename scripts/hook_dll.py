import os
import sys

# Add a directory to the PATH for dynamic libraries
dll_path = os.path.join(os.path.dirname(sys.executable), "libs")
os.environ["PATH"] += os.pathsep + dll_path

print(f"Runtime hook executed: Added {dll_path} to PATH")

## To be used as a runtime hook, run the following.
# pyinstaller --runtime-hook="scripts/hook_dll.py" subgen_whisperx.py
