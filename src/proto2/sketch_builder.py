"""
FreeCAD Sketcher-based profile builder for TH1 geometry.

This module creates parametric 2D profiles using FreeCAD's Sketcher
with geometric constraints for robust dimension changes.
"""

import sys
import os
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Setup FreeCAD paths
CONDA_PREFIX = os.environ.get(
    "CONDA_PREFIX",
    r"C:\Users\aokuni\AppData\Local\miniforge3\envs\fcad"
)
freecad_bin = os.path.join(CONDA_PREFIX, "Library", "bin")
os.environ["PATH"] = freecad_bin + os.pathsep + os.environ.get("PATH", "")
if freecad_bin not in sys.path:
    sys.path.insert(0, freecad_bin)

import FreeCAD
import Part
import Sketcher


@dataclass
class ProfileParams:
    """Parameters for TH1 profile geometry."""
    # Line lengths
    L1: float = 0.775   # top_inner_y
    L2: float = 0.85    # top_outer_y
    L3: float = 0.943   # top_z
    L4: float = 0.368   # upper_step_y
    L5: float = 0.775   # outer_y
    L6: float = 0.619   # outer_z
    L7: float = 0.631   # lower_step_z
    L8: float = 0.657   # inner_shelf_y
    L9: float = 0.448   # inner_shelf_z
    L10: float = 1.309  # bottom_z

    # Fillet radii
    R1: float = 0.25    # fillet_inner
    R2: float = 0.24    # fillet_bottom

    # Anchor points (fixed reference)
    outer_right_y: float = 2.75
    outer_top_z: float = 2.65
    shelf_inner_y: float = 0.6
    shelf_z: float = 1.744
    origin_y: float = 0.0
    origin_z: float = 0.0


def compute_vertices(params: ProfileParams) -> List[Tuple[float, float]]:
    """
    Compute profile vertices from parameters.

    Returns list of (Y, Z) coordinates for 16 vertices.
    This replicates the logic from the original geometry.py.
    """
    origin_y = params.origin_y
    origin_z = params.origin_z

    outer_top_z = origin_z + params.outer_top_z
    outer_bottom_z = origin_z
    outer_y = origin_y + params.outer_right_y

    # P0: inner top
    p0 = (origin_y + params.L1, outer_top_z)

    # P1: outer top-right
    p1 = (p0[0] + params.L2, outer_top_z)

    # P2: after vertical drop
    p2 = (p1[0], p1[1] - params.L3)

    # P3: diagonal to step
    p3 = (outer_y - params.L4, params.L6)

    # P4: step corner
    p4 = (outer_y, params.L6)

    # P5: outer bottom-right
    p5 = (outer_y, outer_bottom_z)

    # P6: inner bottom-right
    p6 = (outer_y - params.L5, outer_bottom_z)

    # P7: after vertical rise
    p7 = (p6[0], params.L7)

    # P8/P9: inner shelf
    p9 = (origin_y + params.shelf_inner_y, origin_z + params.shelf_z)
    p8 = (p9[0] + params.L8, p9[1])

    # P10: corner before R1 fillet
    p10 = (p9[0], p9[1] - params.L9)

    # P11: tangent point for R1 (horizontal edge)
    # If R1=0, P11 is same as P10 (corner point)
    if params.R1 > 1e-9:
        p11 = (p10[0] - params.R1, p10[1])
    else:
        p11 = p10  # No fillet, sharp corner

    # P12: tangent point for R2 (horizontal edge)
    # If R2=0, P12 is at origin_y (left wall)
    if params.R2 > 1e-9:
        p12 = (origin_y + params.R2, p11[1])
    else:
        p12 = (origin_y, p11[1])  # No fillet, sharp corner

    # P13: tangent point for R2 (vertical edge)
    # If R2=0, P13 is same Y as P12
    if params.R2 > 1e-9:
        p13 = (origin_y, p12[1] - params.R2)
    else:
        p13 = (origin_y, p12[1])  # No fillet, sharp corner

    # P14: inner left
    p14 = (origin_y, p13[1] + params.L10)

    # P15: inner top-left
    p15 = (origin_y + params.L1, p14[1])

    return [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15]


def vec(y: float, z: float) -> FreeCAD.Vector:
    """Create FreeCAD Vector from Y, Z coordinates (in sketch plane)."""
    return FreeCAD.Vector(y, z, 0)


def build_profile_sketch(doc, params: ProfileParams, name: str = "TH1Profile"):
    """
    Build TH1 profile sketch with FreeCAD Sketcher.

    Creates a closed profile with lines and arcs for R1, R2 fillets.
    The sketch is on the XY plane of FreeCAD (which maps to YZ in STEP).

    Returns:
        sketch: FreeCAD Sketcher object
    """
    vertices = compute_vertices(params)
    eps = 1e-9

    # Create sketch
    sketch = doc.addObject("Sketcher::SketchObject", name)

    # Build geometry: lines and arcs
    # We need to track geometry indices for constraints
    geo_indices = []

    # Helper to add line
    def add_line(start_idx: int, end_idx: int):
        p1 = vertices[start_idx]
        p2 = vertices[end_idx]
        idx = sketch.addGeometry(
            Part.LineSegment(vec(p1[0], p1[1]), vec(p2[0], p2[1])),
            False
        )
        geo_indices.append(('line', start_idx, end_idx, idx))
        return idx

    # Helper to add arc (three points)
    def add_arc(p_start: Tuple[float, float], p_mid: Tuple[float, float], p_end: Tuple[float, float], label: str):
        arc = Part.ArcOfCircle(
            Part.Circle(vec(0, 0), FreeCAD.Vector(0, 0, 1), 1),
            0, math.pi
        )
        # Create arc through three points
        arc_geo = Part.ArcOfCircle(
            vec(p_start[0], p_start[1]),
            vec(p_mid[0], p_mid[1]),
            vec(p_end[0], p_end[1])
        )
        idx = sketch.addGeometry(arc_geo, False)
        geo_indices.append(('arc', label, idx))
        return idx

    # Build profile segments
    # Segment: P0 -> P1 -> P2 -> P3 -> P4 -> P5 -> P6 -> P7 -> P8 -> P9
    line_segments_before_r1 = [
        (0, 1),   # P0 -> P1
        (1, 2),   # P1 -> P2
        (2, 3),   # P2 -> P3
        (3, 4),   # P3 -> P4
        (4, 5),   # P4 -> P5
        (5, 6),   # P5 -> P6
        (6, 7),   # P6 -> P7
        (7, 8),   # P7 -> P8
        (8, 9),   # P8 -> P9
    ]

    line_indices = []
    for start, end in line_segments_before_r1:
        idx = add_line(start, end)
        line_indices.append(idx)

    # R1 fillet region: P9 -> P10 -> [R1 arc] -> P11
    if params.R1 > eps:
        # Line P9 -> R1 tangent point (on vertical edge)
        p9 = vertices[9]
        p10 = vertices[10]
        r1_tangent_v = (p10[0], p10[1] + params.R1)  # Tangent on P9-P10 edge
        idx_9_r1 = sketch.addGeometry(
            Part.LineSegment(vec(p9[0], p9[1]), vec(r1_tangent_v[0], r1_tangent_v[1])),
            False
        )
        geo_indices.append(('line', 9, 'r1_tv', idx_9_r1))
        line_indices.append(idx_9_r1)

        # R1 arc
        r1_center = (p10[0] - params.R1, p10[1] + params.R1)
        r1_start = r1_tangent_v
        r1_end = vertices[11]
        r1_mid = (
            r1_center[0] + params.R1 * math.cos(math.radians(-45)),
            r1_center[1] + params.R1 * math.sin(math.radians(-45))
        )
        r1_arc_idx = add_arc(r1_start, r1_mid, r1_end, 'R1')
    else:
        # No R1: line P9 -> P10 (P10 and P11 are same point when R1=0)
        add_line(9, 10)
        # Skip P10 -> P11 since they are the same point

    # Line P11 -> P12 (or to R2 tangent)
    if params.R2 > eps:
        # Line from P11 (or R1 end) to R2 tangent point
        p11 = vertices[11]
        p12 = vertices[12]
        idx_11_12 = sketch.addGeometry(
            Part.LineSegment(vec(p11[0], p11[1]), vec(p12[0], p12[1])),
            False
        )
        geo_indices.append(('line', 11, 12, idx_11_12))
        line_indices.append(idx_11_12)

        # R2 arc
        r2_center = (params.origin_y, p12[1])
        r2_start = p12
        r2_end = vertices[13]
        r2_mid = (
            r2_center[0] + params.R2 * math.cos(math.radians(-45)),
            r2_center[1] + params.R2 * math.sin(math.radians(-45))
        )
        r2_arc_idx = add_arc(r2_start, r2_mid, r2_end, 'R2')
    else:
        # No R2: line P11 -> corner (P12 and P13 are same point when R2=0)
        p11 = vertices[11]
        p12 = vertices[12]  # This is at (origin_y, p11[1]) when R2=0
        # Only add line if P11 and P12 are different
        if abs(p11[0] - p12[0]) > eps or abs(p11[1] - p12[1]) > eps:
            idx_11_12 = sketch.addGeometry(
                Part.LineSegment(vec(p11[0], p11[1]), vec(p12[0], p12[1])),
                False
            )
            geo_indices.append(('line', 11, 12, idx_11_12))
            line_indices.append(idx_11_12)
        # P12 and P13 are same when R2=0, so no line needed between them

    # Line P13 -> P14 -> P15 -> P0 (close)
    line_segments_after_r2 = [
        (13, 14),  # P13 -> P14
        (14, 15),  # P14 -> P15
    ]
    for start, end in line_segments_after_r2:
        idx = add_line(start, end)
        line_indices.append(idx)

    # Close: P15 -> P0
    p15 = vertices[15]
    p0 = vertices[0]
    idx_close = sketch.addGeometry(
        Part.LineSegment(vec(p15[0], p15[1]), vec(p0[0], p0[1])),
        False
    )
    geo_indices.append(('line', 15, 0, idx_close))
    line_indices.append(idx_close)

    # Recompute to finalize geometry
    doc.recompute()

    return sketch


def sketch_to_face(sketch) -> Part.Face:
    """Convert closed sketch to Part.Face."""
    edges = sketch.Shape.Edges
    wire = Part.Wire(edges)
    if not wire.isClosed():
        raise ValueError("Wire is not closed, cannot create face")
    face = Part.Face(wire)
    return face


def export_step(shape, filepath: str):
    """Export shape to STEP file."""
    shape.exportStep(filepath)
    print(f"Exported STEP to: {filepath}")


# For testing
if __name__ == "__main__":
    print("Testing sketch_builder...")

    # Create document
    doc = FreeCAD.newDocument("Test")

    # Use reference parameters
    params = ProfileParams()
    print(f"Building profile with default params...")

    # Build sketch
    sketch = build_profile_sketch(doc, params)
    print(f"Sketch created: {sketch.Name}")
    print(f"  Geometry count: {sketch.GeometryCount}")

    doc.recompute()

    # Create face
    try:
        face = sketch_to_face(sketch)
        print(f"Face created successfully")
        print(f"  Face area: {face.Area:.6f}")

        # Export
        output_path = r"c:\github_repo\cad_automaton\output\th1_test.step"
        export_step(face, output_path)
    except Exception as e:
        print(f"Error creating face: {e}")
        import traceback
        traceback.print_exc()

    print("Done.")
