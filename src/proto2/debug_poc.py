"""Debug script to trace PoC execution."""
import sys
import os
import traceback

print("=" * 60)
print("DEBUG: Starting debug_poc.py")
print("=" * 60)

# Step 1: Setup paths
print("\n[Step 1] Setting up paths...")
CONDA_PREFIX = os.environ.get(
    "CONDA_PREFIX",
    r"C:\Users\aokuni\AppData\Local\miniforge3\envs\fcad"
)
print(f"  CONDA_PREFIX: {CONDA_PREFIX}")

freecad_bin = os.path.join(CONDA_PREFIX, "Library", "bin")
print(f"  freecad_bin: {freecad_bin}")
print(f"  freecad_bin exists: {os.path.exists(freecad_bin)}")

os.environ["PATH"] = freecad_bin + os.pathsep + os.environ.get("PATH", "")
sys.path.insert(0, freecad_bin)

# Step 2: Import FreeCAD
print("\n[Step 2] Importing FreeCAD...")
try:
    import FreeCAD
    print(f"  FreeCAD imported successfully: {FreeCAD.Version()[:3]}")
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 3: Import Part
print("\n[Step 3] Importing Part...")
try:
    import Part
    print(f"  Part imported successfully")
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 4: Import Sketcher
print("\n[Step 4] Importing Sketcher...")
try:
    import Sketcher
    print(f"  Sketcher imported successfully")
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 5: Create document
print("\n[Step 5] Creating document...")
try:
    doc = FreeCAD.newDocument("Debug")
    print(f"  Document created: {doc.Name}")
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 6: Create sketch
print("\n[Step 6] Creating sketch object...")
try:
    sketch = doc.addObject("Sketcher::SketchObject", "TestSketch")
    print(f"  Sketch created: {sketch.Name}")
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 7: Add geometry
print("\n[Step 7] Adding geometry (line)...")
try:
    idx = sketch.addGeometry(Part.LineSegment(
        FreeCAD.Vector(0, 0, 0),
        FreeCAD.Vector(10, 0, 0)
    ), False)
    print(f"  Line added at index: {idx}")
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 8: Add constraint
print("\n[Step 8] Adding constraint...")
try:
    cidx = sketch.addConstraint(Sketcher.Constraint("Horizontal", 0))
    print(f"  Constraint added at index: {cidx}")
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 9: Recompute
print("\n[Step 9] Recomputing...")
try:
    doc.recompute()
    print(f"  Recompute successful")
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 10: Check shape
print("\n[Step 10] Checking shape...")
try:
    edges = sketch.Shape.Edges
    print(f"  Number of edges: {len(edges)}")
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("DEBUG: All steps completed successfully!")
print("=" * 60)
