"""
Geometry module for 2D profile generation using build123d.

This module provides parameterized profile geometry construction
with lines and arcs for robust STEP generation.
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional
import math

from build123d import (
    BuildSketch,
    Polygon,
    make_face,
    Face,
    export_step,
    Plane,
    BuildLine,
    Polyline,
    fillet,
    TangentArc,
)

@dataclass
class EdgeInfo:
    """Edge geometry information for visualization."""
    edge_type: str  # 'LINE', 'ARC', or 'CURVE'
    start: Tuple[float, float]  # (Y, Z) coordinates
    end: Tuple[float, float]  # (Y, Z) coordinates
    # Arc-specific fields
    center: Optional[Tuple[float, float]] = None
    radius: Optional[float] = None
    start_angle: Optional[float] = None
    end_angle: Optional[float] = None
    # Optional discretized points along the edge for plotting (Y, Z)
    samples: Optional[List[Tuple[float, float]]] = None


@dataclass
class ProfileParams:
    """
    Parameters defining the 2D profile geometry.
    
    All lengths in mm. Profile is constructed on Y-Z plane (X=0),
    to match the reference STEP coordinate system.
    """
    # Line lengths (10 parameters)
    L1: float = 0.775   # top_inner_y: P0-P15 Y方向
    L2: float = 0.85    # top_outer_y: P0-P1 Y方向
    L3: float = 0.943   # top_z: P1-P2 Z方向
    L4: float = 0.368   # upper_step_y: P4-P3 Y方向
    L5: float = 0.775   # outer_y: P5-P6 Y方向
    L6: float = 0.619   # outer_z: P5-P4 Z方向
    L7: float = 0.631   # lower_step_z: P6-P7 Z方向
    L8: float = 0.657   # inner_shelf_y: P9-P8 Y方向
    L9: float = 0.448   # inner_shelf_z: P9-P10 Z方向
    L10: float = 1.309  # bottom_z: P14-P13 Z方向
    
    # Fillet radii (2 parameters)
    R1: float = 0.25    # fillet_inner
    R2: float = 0.24    # fillet_bottom
    
    # Constraint
    min_thickness: float = 0.2  # 最小肉厚制約

    # Reference-derived anchors (relative to origin_y/origin_z)
    outer_right_y: float = 2.75  # Outer-most Y (width)
    outer_top_z: float = 2.65    # Top-most Z (height)
    shelf_inner_y: float = 0.6   # Inner shelf start Y
    shelf_z: float = 1.744       # Inner shelf Z
    origin_y: float = 0.0        # Global Y offset
    origin_z: float = 0.0        # Global Z offset
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate parameters are within acceptable ranges."""
        errors = []
        
        # All lengths must be positive
        for name, value in [
            ('L1', self.L1), ('L2', self.L2), ('L3', self.L3),
            ('L4', self.L4), ('L5', self.L5), ('L6', self.L6),
            ('L7', self.L7), ('L8', self.L8), ('L9', self.L9),
            ('L10', self.L10),
        ]:
            if value <= 0:
                errors.append(f"{name} must be positive, got {value}")

        # Fillet radii may be zero (no fillet) depending on geometry
        for name, value in [('R1', self.R1), ('R2', self.R2)]:
            if value < 0:
                errors.append(f"{name} must be >= 0, got {value}")

        # Anchor sanity checks
        if self.outer_right_y <= 0:
            errors.append(f"outer_right_y must be positive, got {self.outer_right_y}")
        if self.outer_top_z <= 0:
            errors.append(f"outer_top_z must be positive, got {self.outer_top_z}")
        if not (0 <= self.shelf_inner_y <= self.outer_right_y):
            errors.append(
                f"shelf_inner_y={self.shelf_inner_y} out of range [0, {self.outer_right_y}]"
            )
        if not (0 <= self.shelf_z <= self.outer_top_z):
            errors.append(f"shelf_z={self.shelf_z} out of range [0, {self.outer_top_z}]")
        
        # Fillet radius constraints
        if self.R1 > 0 and self.R1 > min(self.L9, 0.5):
            errors.append(f"R1={self.R1} exceeds adjacent segment limit")
        if self.R2 > 0 and self.R2 > min(self.L10 * 0.3, 0.5):
            errors.append(f"R2={self.R2} exceeds adjacent segment limit")
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'L1': self.L1, 'L2': self.L2, 'L3': self.L3, 'L4': self.L4,
            'L5': self.L5, 'L6': self.L6, 'L7': self.L7, 'L8': self.L8,
            'L9': self.L9, 'L10': self.L10, 'R1': self.R1, 'R2': self.R2,
            'min_thickness': self.min_thickness,
            'outer_right_y': self.outer_right_y,
            'outer_top_z': self.outer_top_z,
            'shelf_inner_y': self.shelf_inner_y,
            'shelf_z': self.shelf_z,
            'origin_y': self.origin_y,
            'origin_z': self.origin_z,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'ProfileParams':
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def compute_vertices(params: ProfileParams) -> List[Tuple[float, float]]:
    """
    Compute profile vertices (X, Y) from parameters.
    
    The profile is built on Y-Z plane for build123d (Plane.YZ).
    Returned tuples are 2D coordinates in the sketch plane: (Y, Z).
    """
    origin_y = params.origin_y
    origin_z = params.origin_z

    outer_top_y = origin_z + params.outer_top_z
    outer_bottom_y = origin_z
    outer_x = origin_y + params.outer_right_y  # Outer-most Y (width)
    
    # P0: start point (inner top)
    p0_x = origin_y + params.L1
    p0_y = outer_top_y
    
    # P1: outer top-right
    p1_x = p0_x + params.L2
    p1_y = outer_top_y
    
    # P2: after vertical drop
    p2_x = p1_x
    p2_y = p1_y - params.L3
    
    # P3: diagonal to step
    p3_x = outer_x - params.L4
    p3_y = params.L6
    
    # P4: step corner
    p4_x = outer_x
    p4_y = params.L6
    
    # P5: outer bottom-right
    p5_x = outer_x
    p5_y = outer_bottom_y
    
    # P6: inner bottom-right
    p6_x = outer_x - params.L5
    p6_y = outer_bottom_y
    
    # P7: after vertical rise
    p7_x = p6_x
    p7_y = params.L7
    
    # P8/P9: inner shelf (diagonal)
    p9_x = origin_y + params.shelf_inner_y
    p9_y = origin_z + params.shelf_z
    p8_x = p9_x + params.L8
    p8_y = p9_y
    
    # P10: corner before fillet R1
    p10_x = p9_x
    p10_y = p9_y - params.L9
    
    # P11: corner (fillet will smooth this)
    p11_x = p10_x - params.R1
    p11_y = p10_y
    
    # P12: corner before fillet R2
    p12_x = origin_y + params.R2
    p12_y = p11_y
    
    # P13: inner bottom-left
    p13_x = origin_y
    p13_y = p12_y - params.R2
    
    # P14: inner left
    p14_x = origin_y
    p14_y = p13_y + params.L10
    
    # P15: inner top-left
    p15_x = origin_y + params.L1
    p15_y = p14_y
    
    # Return vertices in order (closed loop)
    return [
        (p0_x, p0_y),   # 0
        (p1_x, p1_y),   # 1
        (p2_x, p2_y),   # 2
        (p3_x, p3_y),   # 3
        (p4_x, p4_y),   # 4
        (p5_x, p5_y),   # 5
        (p6_x, p6_y),   # 6
        (p7_x, p7_y),   # 7
        (p8_x, p8_y),   # 8
        (p9_x, p9_y),   # 9
        (p10_x, p10_y), # 10
        (p11_x, p11_y), # 11
        (p12_x, p12_y), # 12
        (p13_x, p13_y), # 13
        (p14_x, p14_y), # 14
        (p15_x, p15_y), # 15
    ]


def build_profile_face(params: ProfileParams) -> Face:
    """
    Build complete profile face from parameters using build123d.
    
    Creates a profile on Y-Z plane (X=0) with proper fillets at R1 and R2 positions.
    Uses ThreePointArc for smooth curved transitions.
    
    Note: compute_vertices already includes fillet offset points:
    - P10 is on the vertical edge, R1 above the corner
    - P11 is on the horizontal edge, R1 left of the corner (already offset)
    - P12 is on the horizontal edge, R2 from the left wall
    - P13 is on the vertical edge, R2 below the corner (already offset)
    """
    from build123d import Line, ThreePointArc
    import math
    
    vertices = compute_vertices(params)
    
    # The vertices already contain the fillet tangent points:
    # P10 = (0.6, 1.296) - tangent point on vertical edge for R1
    # P11 = (0.35, 1.296) - tangent point on horizontal edge for R1
    # The actual corner (without fillet) would be at (0.6, 1.296)
    # So P10 is the corner itself, and P11 is offset by R1 horizontally
    
    # Wait, looking at the code again:
    # p10_y = p9_y - params.L9 = 1.744 - 0.448 = 1.296
    # p10_x = p9_x = 0.6
    # So P10 is where the vertical edge meets the corner
    # p11_x = p10_x - R1 = 0.6 - 0.25 = 0.35
    # p11_y = p10_y = 1.296
    # So P11 is horizontal from P10, offset by R1
    
    # This means the fillet should connect:
    # - A point on P9->P10 (vertical), which is at P10 itself since P10 is ON the corner
    # Actually no - P10 IS the corner. The vertices list doesn't have tangent points yet.
    # The R1 offset in P11 is for the next segment, not a tangent point.
    
    # Let me reconsider:
    # For R1: corner is at P10. Fillet starts R1 ABOVE P10 (on P9-P10 edge) and ends R1 LEFT of P10 (on P10-P11 edge)
    # But P11 is already at (P10_x - R1, P10_y) which would be the tangent point!
    
    # So the vertices already encode:
    # P10 = corner point
    # P11 = tangent point (already offset from corner by R1)
    # Then the arc should go from (P10_x, P10_y + R1) to P11
    
    # Similarly for R2:
    # P12 = (R2, p11_y) - this is the tangent point on horizontal edge
    # P13 = (0, p12_y - R2) - this is the tangent point on vertical edge
    # Corner is at (0, p12_y)
    
    eps = 1e-9

    p10 = vertices[10]  # Corner for R1
    p11 = vertices[11]  # Tangent point on horizontal edge for R1 (if R1>0)
    p12 = vertices[12]  # Tangent point for R2 (horizontal, if R2>0)
    p13 = vertices[13]  # Tangent point for R2 (vertical, if R2>0)

    # Pre-compute arc points when fillets are present
    r1_start = r1_mid = r1_end = None
    if params.R1 > eps:
        r1_start = (p10[0], p10[1] + params.R1)
        r1_end = p11
        r1_cx = p10[0] - params.R1
        r1_cy = p10[1] + params.R1
        r1_mid = (
            r1_cx + params.R1 * math.cos(math.radians(-45)),
            r1_cy + params.R1 * math.sin(math.radians(-45)),
        )

    r2_start = r2_mid = r2_end = None
    if params.R2 > eps:
        r2_cx = params.origin_y
        r2_cy = p12[1]
        r2_start = p12
        r2_end = p13
        r2_mid = (
            r2_cx + params.R2 * math.cos(math.radians(-45)),
            r2_cy + params.R2 * math.sin(math.radians(-45)),
        )
    
    with BuildSketch(Plane.YZ) as sketch:
        with BuildLine():
            current = None

            # Segment 1: P0 -> ... -> P9 (+ optional R1)
            if params.R1 > eps and r1_start and r1_mid and r1_end:
                pts_before_r1 = list(vertices[:10]) + [r1_start]
                Polyline(pts_before_r1)
                ThreePointArc(r1_start, r1_mid, r1_end)
                current = r1_end
            else:
                # No R1 fillet: include the sharp corner at P10
                Polyline(list(vertices[:11]))
                current = p10

            # Segment 2: connect to the bottom-left corner (optional R2)
            if params.R2 > eps and r2_start and r2_mid and r2_end:
                Line(current, r2_start)
                ThreePointArc(r2_start, r2_mid, r2_end)
                current = r2_end
            else:
                # No R2 fillet: connect to the sharp corner on the left wall
                sharp_corner = (params.origin_y, p12[1])
                Line(current, sharp_corner)
                current = sharp_corner

            # Segment 3: P13 -> P14 -> P15 -> P0 (close)
            pts_after_r2 = list(vertices[13:]) + [vertices[0]]
            # Ensure we continue from the current point if it differs from P13
            if pts_after_r2 and (abs(pts_after_r2[0][0] - current[0]) > 1e-9 or abs(pts_after_r2[0][1] - current[1]) > 1e-9):
                pts_after_r2 = [current] + pts_after_r2[1:]
            Polyline(pts_after_r2)
        
        make_face()
    
    return sketch.sketch


def build_profile_face_simple(params: ProfileParams) -> Face:
    """
    Simplified profile face builder (alias for build_profile_face).
    """
    return build_profile_face(params)


# Reference parameters extracted from TH1-ref.stp
REFERENCE_PARAMS = ProfileParams(
    L1=0.775,
    L2=0.85,
    L3=0.943,
    L4=0.368,
    L5=0.775,
    L6=0.619,
    L7=0.631,
    L8=0.657,
    L9=0.448,
    L10=1.309,
    R1=0.25,
    R2=0.24,
    min_thickness=0.2
)
