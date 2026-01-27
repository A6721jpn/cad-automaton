from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from freecad_sketch_io import (
    build_constraint_map,
    find_sketch,
    list_constraints,
    open_document,
)


def _dump_data(data: Dict[str, Any], output_path: Path, fmt: str) -> None:
    if fmt == "json":
        output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return
    if fmt == "yaml":
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError("PyYAML is required for YAML output") from exc
        output_path.write_text(
            yaml.dump(data, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        return
    raise ValueError(f"Unsupported format: {fmt}")


def _print_constraints(constraints: list) -> None:
    for info in constraints:
        name = info.get("name") or ""
        value = info.get("value")
        value_text = "-" if value is None else f"{value:.6g}"
        print(f"[{info['index']}] {info.get('type',''):<12} {name:<20} {value_text}")


def build_template(
    fcstd_path: Path,
    sketch_name: str,
    constraints: list,
    include_unnamed: bool,
) -> Dict[str, Any]:
    mapping, ordered = build_constraint_map(constraints, include_unnamed=include_unnamed)
    return {
        "schema_version": 1,
        "source": {
            "fcstd": str(fcstd_path),
            "sketch": sketch_name,
        },
        "constraints": mapping,
        "constraint_order": ordered,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract FreeCAD sketch constraint values.")
    parser.add_argument("fcstd", type=Path, help="Input .FCStd file")
    parser.add_argument("--sketch", dest="sketch", default=None, help="Sketch Name or Label")
    parser.add_argument("--out", dest="out", type=Path, default=None, help="Output file")
    parser.add_argument(
        "--format",
        dest="fmt",
        choices=("yaml", "json"),
        default="yaml",
        help="Output format",
    )
    parser.add_argument("--list", action="store_true", help="List constraints and exit")
    parser.add_argument("--include-unnamed", action="store_true", help="Include unnamed constraints")
    args = parser.parse_args()

    doc = open_document(args.fcstd)
    sketch = find_sketch(doc, args.sketch)
    constraints = list_constraints(sketch)

    if args.list or args.out is None:
        _print_constraints(constraints)
        if args.out is None:
            return

    data = build_template(args.fcstd, sketch.Label, constraints, args.include_unnamed)
    _dump_data(data, args.out, args.fmt)
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
