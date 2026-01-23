"""
CAD Automaton - Robust 2D STEP Generator

CLI entry point for generating parameterized 2D profile STEP files.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .geometry import ProfileParams, REFERENCE_PARAMS
from .step_generator import (
    generate_step, batch_generate, save_generation_log,
    GenerationStatus
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate robust 2D profile STEP files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate with reference parameters
  python -m src.main --output output/test.stp
  
  # Generate with custom parameters
  python -m src.main --output output/custom.stp --L1 0.8 --L2 0.9 --R1 0.3
  
  # Generate from JSON parameter file
  python -m src.main --output output/from_json.stp --params params.json
  
  # Batch generate from JSON array
  python -m src.main --batch params_batch.json --output-dir output/batch/
"""
    )
    
    # Output options
    parser.add_argument(
        '-o', '--output',
        type=Path,
        help='Output STEP file path'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('output'),
        help='Output directory for batch generation'
    )
    
    # Parameter input
    parser.add_argument(
        '--params',
        type=Path,
        help='JSON file with parameters'
    )
    parser.add_argument(
        '--batch',
        type=Path,
        help='JSON file with array of parameter sets for batch generation'
    )
    
    # Individual parameters
    parser.add_argument('--L1', type=float, help='top_inner_y (mm)')
    parser.add_argument('--L2', type=float, help='top_outer_y (mm)')
    parser.add_argument('--L3', type=float, help='top_z (mm)')
    parser.add_argument('--L4', type=float, help='upper_step_y (mm)')
    parser.add_argument('--L5', type=float, help='outer_y (mm)')
    parser.add_argument('--L6', type=float, help='outer_z (mm)')
    parser.add_argument('--L7', type=float, help='lower_step_z (mm)')
    parser.add_argument('--L8', type=float, help='inner_shelf_y (mm)')
    parser.add_argument('--L9', type=float, help='inner_shelf_z (mm)')
    parser.add_argument('--L10', type=float, help='bottom_z (mm)')
    parser.add_argument('--R1', type=float, help='fillet_inner (mm)')
    parser.add_argument('--R2', type=float, help='fillet_bottom (mm)')
    parser.add_argument('--min-thickness', type=float, help='Minimum thickness constraint (mm)')
    
    # Behavior options
    parser.add_argument(
        '--no-correction',
        action='store_true',
        help='Disable auto-correction (fail on invalid parameters)'
    )
    parser.add_argument(
        '--log',
        type=Path,
        help='Path for generation log JSON'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    return parser.parse_args()


def load_params_from_json(json_path: Path) -> ProfileParams:
    """Load parameters from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return ProfileParams.from_dict(data)


def load_batch_params(json_path: Path) -> list:
    """Load batch parameters from JSON array."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("Batch JSON must be an array of parameter objects")
    
    return [ProfileParams.from_dict(item) for item in data]


def build_params_from_args(args: argparse.Namespace) -> ProfileParams:
    """Build ProfileParams from CLI arguments."""
    # Start with reference or loaded params
    if args.params:
        params = load_params_from_json(args.params)
    else:
        params = REFERENCE_PARAMS
    
    # Override with CLI arguments
    param_dict = params.to_dict()
    
    overrides = {
        'L1': args.L1, 'L2': args.L2, 'L3': args.L3, 'L4': args.L4,
        'L5': args.L5, 'L6': args.L6, 'L7': args.L7, 'L8': args.L8,
        'L9': args.L9, 'L10': args.L10, 'R1': args.R1, 'R2': args.R2,
        'min_thickness': args.min_thickness
    }
    
    for key, value in overrides.items():
        if value is not None:
            param_dict[key] = value
    
    return ProfileParams.from_dict(param_dict)


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Batch mode
    if args.batch:
        param_list = load_batch_params(args.batch)
        results = batch_generate(
            param_list,
            args.output_dir,
            allow_correction=not args.no_correction
        )
        
        if args.log:
            save_generation_log(results, args.log)
        
        # Return success if all succeeded
        failed = sum(1 for r in results if r.status == GenerationStatus.FAILED)
        return 1 if failed > 0 else 0
    
    # Single generation mode
    params = build_params_from_args(args)
    
    output_path = args.output or Path('output/profile.stp')
    
    if args.verbose:
        print(f"Parameters: {params.to_dict()}")
        print(f"Output: {output_path}")
    
    result = generate_step(
        params,
        output_path,
        allow_correction=not args.no_correction
    )
    
    if args.log:
        save_generation_log([result], args.log)
    
    # Report result
    if result.status == GenerationStatus.FAILED:
        print(f"FAILED: {result.error_message}", file=sys.stderr)
        return 1
    
    print(f"SUCCESS: {result.output_path}")
    
    if result.status == GenerationStatus.SUCCESS_WITH_CORRECTION:
        print(f"  Corrections applied: {result.correction_result.corrections_applied}")
    
    print(f"  Generation time: {result.generation_time_ms:.1f}ms")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
