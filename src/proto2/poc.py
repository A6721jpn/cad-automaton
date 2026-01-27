"""
Proof of Concept: FreeCAD Sketcher with Constraints

This script verifies:
1. FreeCAD Python API works in headless mode
2. Sketcher can create geometry with constraints
3. Constraints can be modified and solver updates geometry
4. STEP export works
"""

import sys
import os
from pathlib import Path

# Add FreeCAD to Python path (conda-forge installation)
CONDA_PREFIX = os.environ.get(
    "CONDA_PREFIX",
    r"C:\Users\aokuni\AppData\Local\miniforge3\envs\fcad"
)
freecad_bin = Path(CONDA_PREFIX) / "Library" / "bin"
freecad_lib = Path(CONDA_PREFIX) / "Library" / "lib"
freecad_mod = Path(CONDA_PREFIX) / "Library" / "Mod"

# Add to PATH for DLL loading
os.environ["PATH"] = str(freecad_bin) + os.pathsep + os.environ.get("PATH", "")

# Add to Python path
sys.path.insert(0, str(freecad_bin))
sys.path.insert(0, str(freecad_lib))

# FreeCAD modules
try:
    import FreeCAD
    import Part
    import Sketcher
    print(f"FreeCAD {FreeCAD.Version()[0]}.{FreeCAD.Version()[1]} loaded successfully")
except ImportError as e:
    print(f"Failed to import FreeCAD: {e}")
    print(f"CONDA_PREFIX: {CONDA_PREFIX}")
    print(f"freecad_bin exists: {freecad_bin.exists()}")
    print(f"sys.path: {sys.path[:5]}")
    sys.exit(1)


def create_rectangle_sketch(doc, width: float, height: float) -> "Sketcher.Sketch":
    """Create a constrained rectangle sketch on YZ plane."""

    # Create sketch on YZ plane (X=0)
    sketch = doc.addObject("Sketcher::SketchObject", "RectangleSketch")
    sketch.MapMode = "Deactivated"

    # YZ plane: X is normal, Y and Z are in-plane
    # In FreeCAD Sketcher, the sketch local X maps to global Y, local Y maps to global Z
    sketch.Placement = FreeCAD.Placement(
        FreeCAD.Vector(0, 0, 0),
        FreeCAD.Rotation(FreeCAD.Vector(0, 1, 0), 90)  # Rotate to YZ plane
    )

    # Add rectangle geometry (4 lines)
    # Points: P0(0,0), P1(w,0), P2(w,h), P3(0,h)
    # Line indices: 0=bottom, 1=right, 2=top, 3=left

    # Bottom line (P0 -> P1)
    sketch.addGeometry(Part.LineSegment(
        FreeCAD.Vector(0, 0, 0),
        FreeCAD.Vector(width, 0, 0)
    ), False)

    # Right line (P1 -> P2)
    sketch.addGeometry(Part.LineSegment(
        FreeCAD.Vector(width, 0, 0),
        FreeCAD.Vector(width, height, 0)
    ), False)

    # Top line (P2 -> P3)
    sketch.addGeometry(Part.LineSegment(
        FreeCAD.Vector(width, height, 0),
        FreeCAD.Vector(0, height, 0)
    ), False)

    # Left line (P3 -> P0)
    sketch.addGeometry(Part.LineSegment(
        FreeCAD.Vector(0, height, 0),
        FreeCAD.Vector(0, 0, 0)
    ), False)

    # Add constraints

    # Fix origin point (line 3, point 2 = end of left line = P0)
    sketch.addConstraint(Sketcher.Constraint("Fixed", 3, 2))

    # Horizontal constraints for bottom and top
    sketch.addConstraint(Sketcher.Constraint("Horizontal", 0))
    sketch.addConstraint(Sketcher.Constraint("Horizontal", 2))

    # Vertical constraints for left and right
    sketch.addConstraint(Sketcher.Constraint("Vertical", 1))
    sketch.addConstraint(Sketcher.Constraint("Vertical", 3))

    # Connect lines (coincident constraints)
    # Line 0 end -> Line 1 start
    sketch.addConstraint(Sketcher.Constraint("Coincident", 0, 2, 1, 1))
    # Line 1 end -> Line 2 start
    sketch.addConstraint(Sketcher.Constraint("Coincident", 1, 2, 2, 1))
    # Line 2 end -> Line 3 start
    sketch.addConstraint(Sketcher.Constraint("Coincident", 2, 2, 3, 1))
    # Line 3 end -> Line 0 start
    sketch.addConstraint(Sketcher.Constraint("Coincident", 3, 2, 0, 1))

    # Width constraint (distance of bottom line)
    width_constraint_idx = sketch.addConstraint(
        Sketcher.Constraint("DistanceX", 0, 1, 0, 2, width)
    )

    # Height constraint (distance of right line)
    height_constraint_idx = sketch.addConstraint(
        Sketcher.Constraint("DistanceY", 1, 1, 1, 2, height)
    )

    doc.recompute()

    return sketch, width_constraint_idx, height_constraint_idx


def sketch_to_face(sketch) -> "Part.Face":
    """Convert closed sketch to face."""
    wire = Part.Wire(sketch.Shape.Edges)
    face = Part.Face(wire)
    return face


def export_step(shape, filepath: Path):
    """Export shape to STEP file."""
    shape.exportStep(str(filepath))
    print(f"Exported STEP to: {filepath}")


def main():
    print("=" * 60)
    print("FreeCAD PoC: Sketcher with Constraints")
    print("=" * 60)

    # Create new document
    doc = FreeCAD.newDocument("PoC")

    # Initial dimensions
    width = 2.0
    height = 1.5

    print(f"\n1. Creating rectangle sketch: {width} x {height}")
    sketch, w_idx, h_idx = create_rectangle_sketch(doc, width, height)

    print(f"   Sketch created with {sketch.GeometryCount} geometry elements")
    print(f"   Constraint count: {sketch.ConstraintCount}")

    # Check if fully constrained
    dof = sketch.solve()
    print(f"   Degrees of freedom: {dof}")

    # Create face from sketch
    print("\n2. Creating face from sketch")
    face = sketch_to_face(sketch)
    print(f"   Face area: {face.Area:.4f}")
    print(f"   Expected area: {width * height:.4f}")

    # Export initial STEP
    output_dir = Path(__file__).parent.parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    step_path = output_dir / "poc_rectangle_initial.step"
    export_step(face, step_path)

    # Modify constraints
    print("\n3. Modifying dimensions via constraints")
    new_width = 3.0
    new_height = 2.0
    print(f"   New dimensions: {new_width} x {new_height}")

    sketch.setDatum(w_idx, FreeCAD.Units.Quantity(f"{new_width} mm"))
    sketch.setDatum(h_idx, FreeCAD.Units.Quantity(f"{new_height} mm"))
    doc.recompute()

    # Verify new geometry
    face2 = sketch_to_face(sketch)
    print(f"   New face area: {face2.Area:.4f}")
    print(f"   Expected area: {new_width * new_height:.4f}")

    # Export modified STEP
    step_path2 = output_dir / "poc_rectangle_modified.step"
    export_step(face2, step_path2)

    print("\n" + "=" * 60)
    print("PoC completed successfully!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        import traceback
        print(f"ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
