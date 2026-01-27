"""Simplified PoC for FreeCAD Sketcher."""
import sys
import os

# Setup paths
CONDA_PREFIX = os.environ.get(
    "CONDA_PREFIX",
    r"C:\Users\aokuni\AppData\Local\miniforge3\envs\fcad"
)
freecad_bin = os.path.join(CONDA_PREFIX, "Library", "bin")
os.environ["PATH"] = freecad_bin + os.pathsep + os.environ.get("PATH", "")
sys.path.insert(0, freecad_bin)

import FreeCAD
import Part
import Sketcher

print("=" * 60)
print("FreeCAD PoC: Simplified")
print("=" * 60)

# Create document
doc = FreeCAD.newDocument("PoC")

# Create sketch
sketch = doc.addObject("Sketcher::SketchObject", "RectSketch")

width = 2.0
height = 1.5

# Add rectangle (4 lines)
print("\n1. Adding geometry...")
sketch.addGeometry(Part.LineSegment(FreeCAD.Vector(0, 0, 0), FreeCAD.Vector(width, 0, 0)), False)
sketch.addGeometry(Part.LineSegment(FreeCAD.Vector(width, 0, 0), FreeCAD.Vector(width, height, 0)), False)
sketch.addGeometry(Part.LineSegment(FreeCAD.Vector(width, height, 0), FreeCAD.Vector(0, height, 0)), False)
sketch.addGeometry(Part.LineSegment(FreeCAD.Vector(0, height, 0), FreeCAD.Vector(0, 0, 0)), False)
print(f"   Geometry count: {sketch.GeometryCount}")

# Add constraints
print("\n2. Adding constraints...")
sketch.addConstraint(Sketcher.Constraint("Horizontal", 0))
sketch.addConstraint(Sketcher.Constraint("Horizontal", 2))
sketch.addConstraint(Sketcher.Constraint("Vertical", 1))
sketch.addConstraint(Sketcher.Constraint("Vertical", 3))
sketch.addConstraint(Sketcher.Constraint("Coincident", 0, 2, 1, 1))
sketch.addConstraint(Sketcher.Constraint("Coincident", 1, 2, 2, 1))
sketch.addConstraint(Sketcher.Constraint("Coincident", 2, 2, 3, 1))
sketch.addConstraint(Sketcher.Constraint("Coincident", 3, 2, 0, 1))
print(f"   Constraint count: {sketch.ConstraintCount}")

# Recompute
print("\n3. Recomputing...")
doc.recompute()

# Check solve
print("\n4. Checking solve...")
dof = sketch.solve()
print(f"   DOF: {dof}")

# Create wire and face
print("\n5. Creating wire and face...")
edges = sketch.Shape.Edges
print(f"   Edge count: {len(edges)}")
wire = Part.Wire(edges)
print(f"   Wire closed: {wire.isClosed()}")
face = Part.Face(wire)
print(f"   Face area: {face.Area}")

# Export STEP
print("\n6. Exporting STEP...")
output_path = r"c:\github_repo\cad_automaton\output\poc_simple.step"
face.exportStep(output_path)
print(f"   Exported to: {output_path}")

print("\n" + "=" * 60)
print("PoC completed successfully!")
print("=" * 60)
