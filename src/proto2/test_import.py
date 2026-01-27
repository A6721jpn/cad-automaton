"""Test FreeCAD import."""
import sys
import os

# Print environment info
print(f"Python: {sys.executable}")
print(f"CONDA_PREFIX: {os.environ.get('CONDA_PREFIX', 'NOT SET')}")

# Add FreeCAD path
conda_prefix = os.environ.get("CONDA_PREFIX", r"C:\Users\aokuni\AppData\Local\miniforge3\envs\fcad")
freecad_bin = os.path.join(conda_prefix, "Library", "bin")
print(f"FreeCAD bin path: {freecad_bin}")
print(f"Path exists: {os.path.exists(freecad_bin)}")

# List contents
if os.path.exists(freecad_bin):
    pyd_files = [f for f in os.listdir(freecad_bin) if f.endswith('.pyd')]
    print(f"PYD files: {pyd_files}")

# Add to paths
os.environ["PATH"] = freecad_bin + os.pathsep + os.environ.get("PATH", "")
sys.path.insert(0, freecad_bin)

print(f"\nAttempting FreeCAD import...")
try:
    import FreeCAD
    print(f"SUCCESS! FreeCAD version: {FreeCAD.Version()}")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
