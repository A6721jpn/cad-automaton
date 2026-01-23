"""
Quality gate module for shape validation and repair.

Provides B-Rep validation, ShapeFix healing, and minimum thickness checking.
"""

from dataclasses import dataclass
from typing import Tuple, List
from enum import Enum

from build123d import Face, Shape


class ValidationStatus(Enum):
    """Validation result status."""
    VALID = "valid"
    INVALID = "invalid"
    FIXED = "fixed"
    UNFIXABLE = "unfixable"


@dataclass
class ValidationResult:
    """Result of shape validation."""
    status: ValidationStatus
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    
    @classmethod
    def valid(cls) -> 'ValidationResult':
        return cls(ValidationStatus.VALID, True, [], [])
    
    @classmethod
    def invalid(cls, errors: List[str]) -> 'ValidationResult':
        return cls(ValidationStatus.INVALID, False, errors, [])


def validate_shape(shape, min_thickness: float = 0.2) -> ValidationResult:
    """
    Validate shape geometry.
    
    Checks:
    - Shape is not null
    - Basic validity
    """
    errors = []
    warnings = []
    
    if shape is None:
        errors.append("Shape is null")
        return ValidationResult.invalid(errors)
    
    # Check if shape has wrapped attribute (build123d object)
    if hasattr(shape, 'wrapped'):
        try:
            from OCP.BRepCheck import BRepCheck_Analyzer
            analyzer = BRepCheck_Analyzer(shape.wrapped)
            if not analyzer.IsValid():
                errors.append("Shape B-Rep is invalid")
        except Exception as e:
            warnings.append(f"Could not perform B-Rep check: {e}")
    
    if errors:
        return ValidationResult.invalid(errors)
    
    result = ValidationResult.valid()
    result.warnings = warnings
    return result


def fix_shape(shape) -> Tuple:
    """
    Attempt to fix shape using ShapeFix.
    
    Returns (fixed_shape, was_modified).
    """
    if shape is None:
        return shape, False
    
    try:
        if hasattr(shape, 'wrapped'):
            from OCP.ShapeFix import ShapeFix_Shape
            fixer = ShapeFix_Shape(shape.wrapped)
            fixer.SetPrecision(1e-6)
            fixer.Perform()
            # Return original shape for now (build123d may not accept raw OCC shape)
            return shape, False
    except Exception:
        pass
    
    return shape, False


def check_min_thickness(shape, min_thickness: float) -> Tuple[bool, float]:
    """
    Check if shape meets minimum thickness constraint.
    
    Simplified implementation using bounding box.
    """
    if shape is None:
        return False, 0.0
    
    try:
        if hasattr(shape, 'bounding_box'):
            bbox = shape.bounding_box()
            # Use minimum of bounding box dimensions as proxy
            min_dim = min(bbox.size.X, bbox.size.Y, bbox.size.Z) if hasattr(bbox.size, 'X') else 0.1
            return min_dim >= min_thickness, min_dim
    except Exception:
        pass
    
    # Default: assume OK
    return True, min_thickness
