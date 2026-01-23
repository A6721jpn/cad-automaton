"""
CAD Automaton - Robust 2D STEP Generator
"""

from .geometry import ProfileParams, build_profile_face, REFERENCE_PARAMS
from .step_generator import generate_step, batch_generate, GenerationStatus, GenerationResult
from .quality_gate import validate_shape, ValidationResult
from .auto_correction import auto_correct, CorrectionResult

__version__ = "0.1.0"

__all__ = [
    'ProfileParams',
    'build_profile_face',
    'REFERENCE_PARAMS',
    'generate_step',
    'batch_generate',
    'GenerationStatus',
    'GenerationResult',
    'validate_shape',
    'ValidationResult',
    'auto_correct',
    'CorrectionResult',
]
