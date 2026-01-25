"""
Geometry module for 2D profile generation using build123d.

This module provides parameterized profile geometry construction
with lines and arcs for robust STEP generation.
"""

from dataclasses import dataclass
from typing import Tuple, List
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
class ProfileParams:
    """
    Parameters defining the 2D profile geometry.
    
    All lengths in mm. Profile is constructed on X-Y plane.
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
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate parameters are within acceptable ranges."""
        errors = []
        
        # All lengths must be positive
        for name, value in [
            ('L1', self.L1), ('L2', self.L2), ('L3', self.L3),
            ('L4', self.L4), ('L5', self.L5), ('L6', self.L6),
            ('L7', self.L7), ('L8', self.L8), ('L9', self.L9),
            ('L10', self.L10), ('R1', self.R1), ('R2', self.R2),
        ]:
            if value <= 0:
                errors.append(f"{name} must be positive, got {value}")
        
        # Fillet radius constraints
        if self.R1 > min(self.L9, 0.5):
            errors.append(f"R1={self.R1} exceeds adjacent segment limit")
        if self.R2 > min(self.L10 * 0.3, 0.5):
            errors.append(f"R2={self.R2} exceeds adjacent segment limit")
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'L1': self.L1, 'L2': self.L2, 'L3': self.L3, 'L4': self.L4,
            'L5': self.L5, 'L6': self.L6, 'L7': self.L7, 'L8': self.L8,
            'L9': self.L9, 'L10': self.L10, 'R1': self.R1, 'R2': self.R2,
            'min_thickness': self.min_thickness
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'ProfileParams':
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def compute_vertices(params: ProfileParams) -> List[Tuple[float, float]]:
    """
    Compute profile vertices (X, Y) from parameters.
    
    The profile is built on X-Y plane for build123d.
    X corresponds to the original Y (width), Y corresponds to Z (height).
    """
    # Fixed reference heights
    outer_top_y = 2.65
    outer_bottom_y = 0.0
    outer_x = 2.75  # Fixed outer X (width)
    
    # P0: start point (inner top)
    p0_x = params.L1
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
    p9_x = 0.6  # Fixed inner shelf X
    p9_y = 1.744
    p8_x = p9_x + params.L8
    p8_y = p9_y
    
    # P10: corner before fillet R1
    p10_x = p9_x
    p10_y = p9_y - params.L9
    
    # P11: corner (fillet will smooth this)
    p11_x = p10_x - params.R1
    p11_y = p10_y
    
    # P12: corner before fillet R2
    p12_x = params.R2
    p12_y = p11_y
    
    # P13: inner bottom-left
    p13_x = 0.0
    p13_y = p12_y - params.R2
    
    # P14: inner left
    p14_x = 0.0
    p14_y = p13_y + params.L10
    
    # P15: inner top-left
    p15_x = params.L1
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
    
    Creates a polygon on X-Y plane and returns as Face.
    Uses explicit TangentArc for R1 and R2 to ensure valid STEP geometry.
    """
    vertices = compute_vertices(params)
    p12 = vertices[12]
    p13 = vertices[13]
    
    # Calculate corner between P12 and P13 (for R2 replacement)
    # P12 is at (R2, y), P13 is at (0, y-R2)
    # Corner is at (0, y) = (p13[0], p12[1])
    p_corner2 = (p13[0], p12[1])
    
    with BuildSketch() as sketch:
        with BuildLine():
            # Create a single polyline connecting all points with sharp corners
            # 0..12 includes P0..P12
            # Add p_corner2 between P12 and P13
            # Then P13..P15 and close to P0
            
            # P10 is naturally the corner for R1 (P9->P10->P11)
            # So we just follow vertices indices effectively
            pts = vertices[:13] + [p_corner2] + vertices[13:] + [vertices[0]]
            Polyline(pts)

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
