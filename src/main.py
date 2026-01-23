"""
CAD Automaton - Robust 2D STEP Generator

CLI entry point with three modes:
  -r : Recognition mode - extract dimensions, generate config YAML
  -e : Edit mode - apply config YAML values to generate STEP
  (none) : Random mode - randomize dimensions within valid ranges
"""

import argparse
import sys
from pathlib import Path
from typing import List
import logging

from .geometry import ProfileParams, build_profile_face, REFERENCE_PARAMS
from .step_generator import generate_step, GenerationStatus
from .config import (
    Config, FileConfig, create_file_config, get_config_path,
    PARAM_RANGES, create_default_config
)
from build123d import export_step


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def get_input_step_files(input_dir: Path) -> List[Path]:
    """Get list of STEP files from input directory."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    step_files = list(input_dir.glob("*.stp")) + list(input_dir.glob("*.step"))
    
    if not step_files:
        raise FileNotFoundError(f"No STEP files found in: {input_dir}")
    
    return sorted(step_files)


def run_recognition_mode(input_dir: Path, config_path: Path, output_dir: Path) -> int:
    """
    Recognition mode (-r): Extract dimensions and generate config YAML.
    Also generates dimensioned images in output_dir.
    """
    logger.info("=== Recognition Mode ===")
    
    try:
        step_files = get_input_step_files(input_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    
    logger.info(f"Found {len(step_files)} STEP file(s)")
    
    # Create config with extracted parameters for each file
    config = Config()
    
    # Ensure output dir exists for images
    output_dir.mkdir(parents=True, exist_ok=True)
    
    from .step_analyzer import analyze_step_file
    
    for step_file in step_files:
        logger.info(f"Processing: {step_file.name}")
        
        try:
            # Extract parameters and generate image
            image_path = output_dir / f"{step_file.stem}_dims.png"
            
            params_obj = analyze_step_file(step_file, image_path)
            params = params_obj.to_dict()
            logger.info("  -> Dimension extraction successful")
        except Exception as e:
            logger.error(f"  -> Extraction failed: {e}")
            logger.warning("  -> Fallback to reference parameters")
            params = REFERENCE_PARAMS.to_dict()
            
        config.add_file(step_file.name, params)
    
    # Save config
    config.save(config_path)
    logger.info(f"Config saved to: {config_path}")
    
    # Print summary
    print(f"\n[Recognition Complete]")
    print(f"  Files processed: {len(step_files)}")
    print(f"  Config file: {config_path}")
    print(f"  Images saved to: {output_dir}")
    print(f"\nEdit the config file to set desired parameter values,")
    print(f"then run with -e option to generate modified STEP files.")
    
    return 0


def run_edit_mode(input_dir: Path, output_dir: Path, config_path: Path) -> int:
    """
    Edit mode (-e): Apply config YAML values to generate STEP files.
    """
    logger.info("=== Edit Mode ===")
    
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        logger.error("Run with -r option first to generate config file.")
        return 1
    
    try:
        step_files = get_input_step_files(input_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    
    # Load config
    config = Config.load(config_path)
    logger.info(f"Loaded config from: {config_path}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    success_count = 0
    
    for step_file in step_files:
        filename = step_file.name
        
        if filename not in config.files:
            logger.warning(f"No config for {filename}, skipping")
            continue
        
        file_config = config.files[filename]
        params_dict = file_config.get_param_dict()
        params_dict['min_thickness'] = file_config.min_thickness
        
        params = ProfileParams.from_dict(params_dict)
        output_path = output_dir / filename
        
        logger.info(f"Generating: {filename}")
        result = generate_step(params, output_path, allow_correction=True)
        
        if result.status != GenerationStatus.FAILED:
            success_count += 1
            logger.info(f"  -> {result.status.value} ({result.generation_time_ms:.1f}ms)")
        else:
            logger.error(f"  -> FAILED: {result.error_message}")
    
    print(f"\n[Edit Complete]")
    print(f"  Success: {success_count}/{len(step_files)}")
    print(f"  Output: {output_dir}")
    
    return 0 if success_count == len(step_files) else 1


def run_random_mode(input_dir: Path, output_dir: Path, config_path: Path) -> int:
    """
    Random mode (no option): Randomize dimensions within valid ranges.
    """
    logger.info("=== Random Mode ===")
    
    try:
        step_files = get_input_step_files(input_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    
    logger.info(f"Found {len(step_files)} STEP file(s)")
    
    # Create config with randomized parameters
    config = Config()
    
    for step_file in step_files:
        params = REFERENCE_PARAMS.to_dict()
        file_config = config.add_file(step_file.name, params)
        file_config.randomize_parameters()
    
    # Save config for reference
    config.save(config_path)
    logger.info(f"Randomized config saved to: {config_path}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    success_count = 0
    
    for step_file in step_files:
        filename = step_file.name
        file_config = config.files[filename]
        
        params_dict = file_config.get_param_dict()
        params_dict['min_thickness'] = file_config.min_thickness
        
        params = ProfileParams.from_dict(params_dict)
        output_path = output_dir / filename
        
        logger.info(f"Generating: {filename}")
        result = generate_step(params, output_path, allow_correction=True)
        
        if result.status != GenerationStatus.FAILED:
            success_count += 1
            status_str = result.status.value
            if result.correction_result and result.correction_result.corrections_applied:
                status_str += " (corrected)"
            logger.info(f"  -> {status_str} ({result.generation_time_ms:.1f}ms)")
        else:
            logger.error(f"  -> FAILED: {result.error_message}")
    
    print(f"\n[Random Generation Complete]")
    print(f"  Success: {success_count}/{len(step_files)}")
    print(f"  Output: {output_dir}")
    print(f"  Config: {config_path}")
    
    return 0 if success_count == len(step_files) else 1


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Robust 2D STEP Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  -r    Recognition mode: Extract dimensions, generate config.yaml + images
  -e    Edit mode: Apply config.yaml values to generate STEP files
  (none) Random mode: Randomize dimensions within valid ranges

Examples:
  python -m src.main -r          # Generate config.yaml & images
  python -m src.main -e          # Edit STEPs using config.yaml
  python -m src.main             # Random edit within valid ranges
"""
    )
    
    parser.add_argument(
        '-r', '--recognize',
        action='store_true',
        help='Recognition mode: extract dimensions and generate config'
    )
    parser.add_argument(
        '-e', '--edit',
        action='store_true',
        help='Edit mode: apply config values to generate STEP files'
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path('input'),
        help='Input directory for STEP files (default: input/)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('output'),
        help='Output directory for generated STEP files (default: output/)'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('config.yaml'),
        help='Config file path (default: config.yaml)'
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    if args.recognize and args.edit:
        logger.error("Cannot use -r and -e together")
        return 1
    
    if args.recognize:
        return run_recognition_mode(args.input_dir, args.config, args.output_dir)
    elif args.edit:
        return run_edit_mode(args.input_dir, args.output_dir, args.config)
    else:
        return run_random_mode(args.input_dir, args.output_dir, args.config)


if __name__ == '__main__':
    sys.exit(main())
