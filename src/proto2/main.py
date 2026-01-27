"""
FreeCAD-based STEP generator main entry point.

Usage:
    python -m src.proto2.main -e              # Generate STEP from config.yaml
    python -m src.proto2.main -e -o out.step  # Specify output file
"""

import sys
import os
import argparse

# Setup FreeCAD paths before any imports
CONDA_PREFIX = os.environ.get(
    "CONDA_PREFIX",
    r"C:\Users\aokuni\AppData\Local\miniforge3\envs\fcad"
)
freecad_bin = os.path.join(CONDA_PREFIX, "Library", "bin")
os.environ["PATH"] = freecad_bin + os.pathsep + os.environ.get("PATH", "")
if freecad_bin not in sys.path:
    sys.path.insert(0, freecad_bin)

import yaml
import FreeCAD

# Handle both module and direct execution
try:
    from .sketch_builder import (
        ProfileParams,
        build_profile_sketch,
        sketch_to_face,
        export_step,
    )
except ImportError:
    from sketch_builder import (
        ProfileParams,
        build_profile_sketch,
        sketch_to_face,
        export_step,
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def params_from_config(config: dict, step_file: str = "TH1-ref.stp") -> ProfileParams:
    """Extract ProfileParams from config dictionary."""
    file_config = config.get('files', {}).get(step_file, {})
    params_dict = file_config.get('parameters', {})
    constraints = file_config.get('constraints', {})

    # Map config parameters to ProfileParams
    kwargs = {}

    # Line lengths
    param_mapping = {
        'L1': 'L1', 'L2': 'L2', 'L3': 'L3', 'L4': 'L4',
        'L5': 'L5', 'L6': 'L6', 'L7': 'L7', 'L8': 'L8',
        'L9': 'L9', 'L10': 'L10',
        'R1': 'R1', 'R2': 'R2',
    }

    for config_key, param_key in param_mapping.items():
        if config_key in params_dict:
            kwargs[param_key] = params_dict[config_key].get('value', ProfileParams.__dataclass_fields__[param_key].default)

    # Anchor values from constraints
    anchors = constraints.get('anchors', {})
    anchor_mapping = {
        'outer_right_y': 'outer_right_y',
        'outer_top_z': 'outer_top_z',
        'shelf_inner_y': 'shelf_inner_y',
        'shelf_z': 'shelf_z',
        'origin_y': 'origin_y',
        'origin_z': 'origin_z',
    }

    for config_key, param_key in anchor_mapping.items():
        if config_key in anchors:
            kwargs[param_key] = anchors[config_key]

    return ProfileParams(**kwargs)


def generate_step(params: ProfileParams, output_path: str) -> bool:
    """Generate STEP file from parameters."""
    print(f"Generating STEP with FreeCAD Sketcher...")
    print(f"Parameters:")
    print(f"  L1={params.L1}, L2={params.L2}, L3={params.L3}, L4={params.L4}")
    print(f"  L5={params.L5}, L6={params.L6}, L7={params.L7}, L8={params.L8}")
    print(f"  L9={params.L9}, L10={params.L10}")
    print(f"  R1={params.R1}, R2={params.R2}")

    try:
        # Create FreeCAD document
        doc = FreeCAD.newDocument("STEP_Generation")

        # Build sketch
        sketch = build_profile_sketch(doc, params, "TH1Profile")
        print(f"Sketch created: {sketch.GeometryCount} geometry elements")

        doc.recompute()

        # Create face
        face = sketch_to_face(sketch)
        print(f"Face created: area = {face.Area:.6f}")

        # Export STEP
        export_step(face, output_path)
        print(f"SUCCESS: STEP exported to {output_path}")

        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="FreeCAD-based STEP generator (proto2)"
    )
    parser.add_argument(
        '-e', '--export',
        action='store_true',
        help='Generate STEP from config.yaml'
    )
    parser.add_argument(
        '-c', '--config',
        default='config.yaml',
        help='Config file path (default: config.yaml)'
    )
    parser.add_argument(
        '-o', '--output',
        default=None,
        help='Output STEP file path'
    )
    parser.add_argument(
        '-f', '--file',
        default='TH1-ref.stp',
        help='Reference STEP file key in config (default: TH1-ref.stp)'
    )

    args = parser.parse_args()

    if not args.export:
        parser.print_help()
        return 1

    # Find config file
    config_path = args.config
    if not os.path.isabs(config_path):
        # Try current directory, then project root
        if not os.path.exists(config_path):
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            config_path = os.path.join(project_root, args.config)

    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found: {args.config}")
        return 1

    print(f"Loading config from: {config_path}")
    config = load_config(config_path)

    # Extract parameters
    params = params_from_config(config, args.file)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # Default: same directory as config, with _generated suffix
        base_name = os.path.splitext(args.file)[0]
        output_dir = os.path.dirname(config_path) or '.'
        output_path = os.path.join(output_dir, f"{base_name}_generated.step")

    # Generate STEP
    success = generate_step(params, output_path)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
