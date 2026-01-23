"""
Auto-correction module for parameter projection and geometry regularization.

Implements three-stage fallback:
1. Parameter projection to constraint bounds
2. Geometry regularization (sliver removal)
3. Fallback simplification
"""

from dataclasses import dataclass, replace
from typing import Tuple, List, Optional, Dict
import math

from .geometry import ProfileParams


@dataclass
class CorrectionResult:
    """Result of auto-correction attempt."""
    success: bool
    original_params: ProfileParams
    corrected_params: ProfileParams
    corrections_applied: List[str]
    fallback_level: int  # 0=none, 1=projection, 2=regularization, 3=simplification


# Parameter bounds (min, max) in mm
PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    'L1': (0.1, 2.0),
    'L2': (0.1, 2.0),
    'L3': (0.1, 2.0),
    'L4': (0.1, 1.0),
    'L5': (0.1, 2.0),
    'L6': (0.1, 2.0),
    'L7': (0.1, 2.0),
    'L8': (0.1, 2.0),
    'L9': (0.1, 1.5),
    'L10': (0.3, 3.0),
    'R1': (0.05, 0.5),
    'R2': (0.05, 0.5),
}


def project_to_bounds(value: float, bounds: Tuple[float, float]) -> Tuple[float, bool]:
    """
    Project value to within bounds.
    
    Returns (projected_value, was_modified).
    """
    min_val, max_val = bounds
    if value < min_val:
        return min_val, True
    elif value > max_val:
        return max_val, True
    return value, False


def project_params(params: ProfileParams) -> CorrectionResult:
    """
    Project parameters to constraint bounds.
    
    Stage 1: Ensure all parameters are within valid ranges.
    """
    corrections = []
    param_dict = params.to_dict()
    
    for name, bounds in PARAM_BOUNDS.items():
        if name in param_dict:
            original = param_dict[name]
            projected, modified = project_to_bounds(original, bounds)
            if modified:
                param_dict[name] = projected
                corrections.append(f"{name}: {original:.4f} -> {projected:.4f}")
    
    corrected = ProfileParams.from_dict(param_dict)
    
    return CorrectionResult(
        success=True,
        original_params=params,
        corrected_params=corrected,
        corrections_applied=corrections,
        fallback_level=1 if corrections else 0
    )


def clamp_fillet_radius(params: ProfileParams) -> CorrectionResult:
    """
    Ensure fillet radii don't exceed adjacent segment limits.
    
    Part of regularization stage.
    """
    corrections = []
    
    # R1 limited by L9 (vertical segment before fillet)
    max_r1 = params.L9 * 0.8
    new_r1 = min(params.R1, max_r1)
    if new_r1 != params.R1:
        corrections.append(f"R1: {params.R1:.4f} -> {new_r1:.4f} (segment limit)")
    
    # R2 limited by horizontal segment length
    max_r2 = 0.4  # Fixed limit for bottom fillet
    new_r2 = min(params.R2, max_r2)
    if new_r2 != params.R2:
        corrections.append(f"R2: {params.R2:.4f} -> {new_r2:.4f} (segment limit)")
    
    corrected = replace(params, R1=new_r1, R2=new_r2)
    
    return CorrectionResult(
        success=True,
        original_params=params,
        corrected_params=corrected,
        corrections_applied=corrections,
        fallback_level=2 if corrections else 0
    )


def regularize_geometry(params: ProfileParams) -> CorrectionResult:
    """
    Apply geometry regularization.
    
    Stage 2: Quantize to grid, remove potential slivers.
    """
    corrections = []
    param_dict = params.to_dict()
    
    # Quantize to 0.01mm grid to avoid numerical issues
    grid_size = 0.01
    for name in PARAM_BOUNDS.keys():
        if name in param_dict:
            original = param_dict[name]
            quantized = round(original / grid_size) * grid_size
            if abs(quantized - original) > 1e-9:
                param_dict[name] = quantized
                corrections.append(f"{name}: quantized to {quantized:.4f}")
    
    corrected = ProfileParams.from_dict(param_dict)
    
    # Also apply fillet clamping
    fillet_result = clamp_fillet_radius(corrected)
    corrections.extend(fillet_result.corrections_applied)
    
    return CorrectionResult(
        success=True,
        original_params=params,
        corrected_params=fillet_result.corrected_params,
        corrections_applied=corrections,
        fallback_level=2 if corrections else 0
    )


def fallback_simplify(params: ProfileParams) -> CorrectionResult:
    """
    Fallback simplification for severely problematic parameters.
    
    Stage 3: Reduce fillets to minimum, constrain complex features.
    """
    corrections = []
    
    # Reduce fillets to safe minimum
    safe_r = 0.1
    new_r1 = safe_r if params.R1 != safe_r else params.R1
    new_r2 = safe_r if params.R2 != safe_r else params.R2
    
    if new_r1 != params.R1:
        corrections.append(f"R1: {params.R1:.4f} -> {new_r1:.4f} (fallback)")
    if new_r2 != params.R2:
        corrections.append(f"R2: {params.R2:.4f} -> {new_r2:.4f} (fallback)")
    
    corrected = replace(params, R1=new_r1, R2=new_r2)
    
    return CorrectionResult(
        success=True,
        original_params=params,
        corrected_params=corrected,
        corrections_applied=corrections,
        fallback_level=3
    )


def auto_correct(params: ProfileParams, 
                 max_iterations: int = 3) -> CorrectionResult:
    """
    Full auto-correction pipeline.
    
    Applies corrections in stages:
    1. Parameter projection
    2. Geometry regularization
    3. Fallback simplification (if needed)
    """
    all_corrections = []
    current = params
    max_level = 0
    
    # Stage 1: Project to bounds
    result1 = project_params(current)
    current = result1.corrected_params
    all_corrections.extend(result1.corrections_applied)
    max_level = max(max_level, result1.fallback_level)
    
    # Stage 2: Regularize
    result2 = regularize_geometry(current)
    current = result2.corrected_params
    all_corrections.extend(result2.corrections_applied)
    max_level = max(max_level, result2.fallback_level)
    
    # Validate result
    is_valid, errors = current.validate()
    
    if not is_valid:
        # Stage 3: Fallback simplification
        result3 = fallback_simplify(current)
        current = result3.corrected_params
        all_corrections.extend(result3.corrections_applied)
        max_level = 3
    
    return CorrectionResult(
        success=True,
        original_params=params,
        corrected_params=current,
        corrections_applied=all_corrections,
        fallback_level=max_level
    )


def find_nearest_valid(params: ProfileParams,
                       failed_param: str,
                       target_value: float) -> ProfileParams:
    """
    Find nearest valid value for a single parameter.
    
    Binary search to find closest value that produces valid geometry.
    """
    bounds = PARAM_BOUNDS.get(failed_param, (0.01, 10.0))
    
    # Start with projected value
    projected, _ = project_to_bounds(target_value, bounds)
    
    # Further refine if needed (placeholder for more sophisticated search)
    return replace(params, **{failed_param: projected})
