"""
STEP generator module - main pipeline for robust 2D STEP output.

Integrates geometry construction, quality validation, and auto-correction.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
from enum import Enum
import logging
import json
from datetime import datetime

from build123d import export_step

from .geometry import ProfileParams, build_profile_face, REFERENCE_PARAMS
from .quality_gate import validate_shape, fix_shape, ValidationResult, ValidationStatus
from .auto_correction import auto_correct, CorrectionResult


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GenerationStatus(Enum):
    """Status of STEP generation."""
    SUCCESS = "success"
    SUCCESS_WITH_CORRECTION = "success_with_correction"
    FAILED = "failed"


@dataclass
class GenerationResult:
    """Result of STEP generation attempt."""
    status: GenerationStatus
    output_path: Optional[Path]
    params_used: ProfileParams
    correction_result: Optional[CorrectionResult]
    validation_result: Optional[ValidationResult]
    error_message: Optional[str]
    generation_time_ms: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging/serialization."""
        return {
            'status': self.status.value,
            'output_path': str(self.output_path) if self.output_path else None,
            'params_used': self.params_used.to_dict(),
            'corrections_applied': (
                self.correction_result.corrections_applied 
                if self.correction_result else []
            ),
            'fallback_level': (
                self.correction_result.fallback_level 
                if self.correction_result else 0
            ),
            'is_valid': (
                self.validation_result.is_valid 
                if self.validation_result else False
            ),
            'error': self.error_message,
            'generation_time_ms': self.generation_time_ms,
        }


def generate_step(
    params: ProfileParams,
    output_path: Path,
    allow_correction: bool = True,
    max_correction_attempts: int = 3,
) -> GenerationResult:
    """
    Generate STEP file from profile parameters.
    
    Pipeline:
    1. Validate/correct parameters
    2. Build geometry
    3. Validate shape
    4. Fix if needed
    5. Export STEP
    """
    import time
    start_time = time.perf_counter()
    
    current_params = params
    correction_result = None
    
    # Stage 1: Parameter validation and correction
    is_valid, errors = params.validate()
    
    if not is_valid:
        if not allow_correction:
            return GenerationResult(
                status=GenerationStatus.FAILED,
                output_path=None,
                params_used=params,
                correction_result=None,
                validation_result=None,
                error_message=f"Invalid parameters: {errors}",
                generation_time_ms=(time.perf_counter() - start_time) * 1000
            )
        
        # Apply auto-correction
        correction_result = auto_correct(params, max_correction_attempts)
        current_params = correction_result.corrected_params
        logger.info(f"Applied corrections: {correction_result.corrections_applied}")
    
    # Stage 2: Build geometry
    try:
        face = build_profile_face(current_params)
    except Exception as e:
        logger.error(f"Geometry construction failed: {e}")
        
        if allow_correction and correction_result is None:
            correction_result = auto_correct(params, max_correction_attempts)
            current_params = correction_result.corrected_params
            
            try:
                face = build_profile_face(current_params)
            except Exception as e2:
                return GenerationResult(
                    status=GenerationStatus.FAILED,
                    output_path=None,
                    params_used=current_params,
                    correction_result=correction_result,
                    validation_result=None,
                    error_message=f"Geometry construction failed after correction: {e2}",
                    generation_time_ms=(time.perf_counter() - start_time) * 1000
                )
        else:
            return GenerationResult(
                status=GenerationStatus.FAILED,
                output_path=None,
                params_used=current_params,
                correction_result=correction_result,
                validation_result=None,
                error_message=f"Geometry construction failed: {e}",
                generation_time_ms=(time.perf_counter() - start_time) * 1000
            )
    
    # Stage 3: Validate shape
    validation_result = validate_shape(face, current_params.min_thickness)
    
    if not validation_result.is_valid:
        logger.warning(f"Shape validation failed: {validation_result.errors}")
        
        try:
            fixed_face, was_modified = fix_shape(face)
            if was_modified:
                face = fixed_face
                validation_result = validate_shape(face, current_params.min_thickness)
        except Exception as e:
            logger.error(f"Shape fix failed: {e}")
    
    # Stage 4: Export STEP
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        export_step(face, str(output_path))
        logger.info(f"STEP exported to: {output_path}")
    except Exception as e:
        return GenerationResult(
            status=GenerationStatus.FAILED,
            output_path=None,
            params_used=current_params,
            correction_result=correction_result,
            validation_result=validation_result,
            error_message=f"STEP export failed: {e}",
            generation_time_ms=(time.perf_counter() - start_time) * 1000
        )
    
    # Determine final status
    status = (
        GenerationStatus.SUCCESS_WITH_CORRECTION 
        if correction_result and correction_result.corrections_applied 
        else GenerationStatus.SUCCESS
    )
    
    return GenerationResult(
        status=status,
        output_path=output_path,
        params_used=current_params,
        correction_result=correction_result,
        validation_result=validation_result,
        error_message=None,
        generation_time_ms=(time.perf_counter() - start_time) * 1000
    )


def batch_generate(
    param_list: List[ProfileParams],
    output_dir: Path,
    name_prefix: str = "profile",
    allow_correction: bool = True,
) -> List[GenerationResult]:
    """Generate multiple STEP files in batch."""
    results = []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, params in enumerate(param_list):
        output_path = output_dir / f"{name_prefix}_{i:04d}.stp"
        result = generate_step(params, output_path, allow_correction)
        results.append(result)
        
        logger.info(
            f"[{i+1}/{len(param_list)}] {result.status.value} "
            f"({result.generation_time_ms:.1f}ms)"
        )
    
    success = sum(1 for r in results if r.status != GenerationStatus.FAILED)
    corrected = sum(
        1 for r in results 
        if r.status == GenerationStatus.SUCCESS_WITH_CORRECTION
    )
    
    logger.info(
        f"Batch complete: {success}/{len(results)} success "
        f"({corrected} with correction)"
    )
    
    return results


def save_generation_log(
    results: List[GenerationResult],
    log_path: Path
) -> None:
    """Save generation results to JSON log."""
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'total': len(results),
        'success': sum(1 for r in results if r.status != GenerationStatus.FAILED),
        'results': [r.to_dict() for r in results]
    }
    
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
