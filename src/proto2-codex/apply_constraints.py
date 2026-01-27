from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from freecad_sketch_io import (
    find_constraint_in_sketches,
    find_constraint_index,
    find_sketch,
    open_document,
    save_document,
    set_constraint_value,
)


def _load_data(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError("PyYAML is required for YAML input") from exc
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_values(payload: Dict[str, Any]) -> Dict[str, float]:
    raw = payload.get("constraints", {})
    values: Dict[str, float] = {}
    for name, meta in raw.items():
        if isinstance(meta, dict) and "value" in meta:
            try:
                values[name] = float(meta["value"])
            except Exception:
                continue
            continue
        try:
            values[name] = float(meta)
        except Exception:
            continue
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply constraint values to a FreeCAD sketch.")
    parser.add_argument("fcstd", type=Path, help="Input .FCStd file")
    parser.add_argument("--sketch", dest="sketch", default=None, help="Sketch Name or Label")
    parser.add_argument("--in", dest="input_path", type=Path, required=True, help="YAML/JSON constraints file")
    parser.add_argument("--out", dest="out", type=Path, required=True, help="Output .FCStd file")
    parser.add_argument("--dry-run", action="store_true", help="Validate only")
    args = parser.parse_args()

    payload = _load_data(args.input_path)
    values = _extract_values(payload)

    doc = open_document(args.fcstd)
    sketch_order = payload.get("sketch_order", None)
    sketch = find_sketch(doc, args.sketch)

    missing = []
    for name, value in values.items():
        if sketch_order:
            target_sketch, index = find_constraint_in_sketches(doc, sketch_order, name)
            if index is None:
                missing.append(name)
                continue
            if not args.dry_run:
                set_constraint_value(target_sketch, index, value)
            continue

        index = find_constraint_index(sketch, name)
        if index is None:
            missing.append(name)
            continue
        if not args.dry_run:
            set_constraint_value(sketch, index, value)

    if missing:
        print("Missing constraint names:")
        for name in missing:
            print(f"  - {name}")

    if args.dry_run:
        print("Dry-run complete. No changes written.")
        return

    doc.recompute()
    save_document(doc, args.out)
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
