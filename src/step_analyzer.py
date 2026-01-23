"""
STEP analyzer module for extracting parameters from existing STEP files.

Identifies profile vertices and reverse-engineers dimensions (L1-L10, R1, R2).
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional
import math
import logging

from build123d import import_step, Shape, Vertex, Edge, Wire, GeomType
from .geometry import ProfileParams

logger = logging.getLogger(__name__)


def get_vertex_coords(v: Vertex) -> Tuple[float, float, float]:
    """Get vertex coordinates rounded to precision."""
    p = v.center()
    return (round(p.X, 4), round(p.Y, 4), round(p.Z, 4))


def analyze_step_file(file_path: Path, output_image_path: Optional[Path] = None) -> ProfileParams:
    """
    Analyze STEP file and extract profile parameters.
    
    Args:
        file_path: Input STEP file path
        output_image_path: Optional path to save dimensioned image
    """
    logger.info(f"Analyzing STEP file: {file_path}")
    
    try:
        shape = import_step(str(file_path))
    except Exception as e:
        logger.error(f"Failed to import STEP: {e}")
        raise ValueError(f"Could not import STEP file: {file_path}")

    # Extract wire
    if not isinstance(shape, (Wire, Shape)): 
        wires = shape.wires()
        if not wires:
            raise ValueError("No wires found in STEP file")
        wire = wires[0]
    else:
        wire = shape if isinstance(shape, Wire) else shape.wires()[0]

    ordered_edges = wire.order_edges()
    if not ordered_edges:
        raise ValueError("No ordered edges found")

    vertices = []
    for edge in ordered_edges:
        vertices.append(edge.start_point())
    
    # 2. Find Anchor P1 (Top-Right)
    max_z = -float('inf')
    for v in vertices:
        if v.Z > max_z:
            max_z = v.Z
            
    top_candidates = [i for i, v in enumerate(vertices) if abs(v.Z - max_z) < 0.01]
    
    if not top_candidates:
        p1_idx = 0
    else:
        p1_idx = top_candidates[0]
        max_y = vertices[p1_idx].Y
        for idx in top_candidates:
            if vertices[idx].Y > max_y:
                max_y = vertices[idx].Y
                p1_idx = idx

    vp = vertices[p1_idx:] + vertices[:p1_idx]
    
    # 3. Check Direction
    p1 = vp[0]
    p_next = vp[1] if len(vp) > 1 else vp[0]
    vec = p_next - p1
    
    if abs(vec.Z) < abs(vec.Y):
        logger.info("Wire direction seems reversed (P1->TopLeft), flipping...")
        vp = [vp[0]] + list(reversed(vp[1:]))
    
    num_v = len(vp)
    logger.info(f"Identified {num_v} vertices extracted from wire")
    
    params_dict = {}
    
    if 15 <= num_v <= 17:
        try:
            # L3 (P1-P2)
            params_dict['L3'] = abs(vp[0].Z - vp[1].Z)
            # L4 (P4-P3 Y) -> indices [3] and [2]
            params_dict['L4'] = abs(vp[3].Y - vp[2].Y)
            # L6 (P5-P4 Z) -> indices [4] and [3]
            params_dict['L6'] = abs(vp[4].Z - vp[3].Z)
            # L5 (P5-P6 Y) -> indices [4] and [5]
            params_dict['L5'] = abs(vp[4].Y - vp[5].Y)
            # L7 (P6-P7 Z) -> indices [5] and [6]
            params_dict['L7'] = abs(vp[6].Z - vp[5].Z)
            # L8 (P9-P8 Y) -> indices [8] and [7]
            params_dict['L8'] = abs(vp[8].Y - vp[7].Y)
            # L9 (P9-P10 Z) -> indices [8] and [9]
            params_dict['L9'] = abs(vp[8].Z - vp[9].Z)
            # R1 (P10-P11 Y/Z) -> indices [9] and [10]
            params_dict['R1'] = abs(vp[9].Y - vp[10].Y) 
            
            # Adaptive indices for bottom part
            r2_idx1, r2_idx2 = 11, 12
            l10_idx1, l10_idx2 = 13, 12
            
            if num_v == 15:
                # If 15, assume one less vertex somewhere. 
                # If we assume P0-P15 is merged (L1=0), then P15 is last, P14 is -2.
                # Just use relative from end for L10
                l10_idx1 = num_v - 2
                l10_idx2 = num_v - 3
                r2_idx2 = num_v - 3
                r2_idx1 = num_v - 4

            if l10_idx1 < num_v:
                params_dict['L10'] = abs(vp[l10_idx1].Z - vp[l10_idx2].Z)
            
            if r2_idx2 < num_v:
                 params_dict['R2'] = abs(vp[r2_idx1].Z - vp[r2_idx2].Z)

            params_dict['L2'] = abs(vp[0].Y - vp[-1].Y)
            
            if num_v == 16:
                # L1 is P14-P15 (Horizontal Y), which are indices 13 and 14 (-3 and -2)
                params_dict['L1'] = abs(vp[-2].Y - vp[-3].Y)
            else:
                params_dict['L1'] = 0.01 # Non-zero default

        except Exception as e:
            logger.warning(f"Extraction calculation error: {e}")
            raise

    # Validation
    required_keys = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10', 'R1', 'R2']
    for key in required_keys:
        if key not in params_dict:
             params_dict[key] = 0.1 # Default safe value
        
        # Zero check
        if params_dict[key] < 0.001:
             logger.warning(f"Extracted parameter {key} is near zero: {params_dict[key]}")
             # Treat as failure for critical dimensions
             if key not in ['L1', 'R1', 'R2']: # L1 can be small
                 raise ValueError(f"Critical parameter {key} extracted as zero")

    profile_params = ProfileParams(
        L1=round(params_dict['L1'], 4),
        L2=round(params_dict['L2'], 4),
        L3=round(params_dict['L3'], 4),
        L4=round(params_dict['L4'], 4),
        L5=round(params_dict['L5'], 4),
        L6=round(params_dict['L6'], 4),
        L7=round(params_dict['L7'], 4),
        L8=round(params_dict['L8'], 4),
        L9=round(params_dict['L9'], 4),
        L10=round(params_dict['L10'], 4),
        R1=round(params_dict['R1'], 4),
        R2=round(params_dict['R2'], 4),
        min_thickness=0.2
    )

    # Visualization
    if output_image_path and num_v >= 15:
        try:
            from .visualizer import plot_profile_dims
            plot_profile_dims(vp, profile_params, output_image_path)
            logger.info(f"Saved dimension image to: {output_image_path}")
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            # Do not fail extraction just because viz failed

    return profile_params
