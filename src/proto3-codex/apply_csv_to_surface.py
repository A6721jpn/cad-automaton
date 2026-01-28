from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import FreeCAD  # type: ignore


def _open_document(path: Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    return FreeCAD.openDocument(str(path))


def _find_sketch(doc, sketch_hint: Optional[str] = None):
    sketches = [obj for obj in doc.Objects if obj.TypeId == "Sketcher::SketchObject"]
    if not sketches:
        raise ValueError("No Sketcher::SketchObject found in document")

    if sketch_hint:
        for obj in sketches:
            if obj.Name == sketch_hint or obj.Label == sketch_hint:
                return obj
        hint_lower = sketch_hint.lower()
        for obj in sketches:
            if obj.Name.lower() == hint_lower or obj.Label.lower() == hint_lower:
                return obj
        raise ValueError(f"Sketch not found: {sketch_hint}")

    return sketches[0]


def _find_constraint_index(sketch, name: str) -> Optional[int]:
    for index, constraint in enumerate(getattr(sketch, "Constraints", [])):
        if getattr(constraint, "Name", "") == name:
            return index
    return None


def _set_constraint_value(sketch, index: int, value: float) -> None:
    try:
        quantity = FreeCAD.Units.Quantity(value)
    except Exception:
        quantity = value
    sketch.setDatum(index, quantity)


def _read_csv_values(path: Path) -> Tuple[Dict[str, float], List[str]]:
    values: Dict[str, float] = {}
    missing_name_rows: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("name") or "").strip()
            if not name:
                missing_name_rows.append(str(row))
                continue
            raw_value = row.get("value")
            if raw_value is None:
                continue
            try:
                value = float(raw_value)
            except Exception:
                continue
            values[name] = value
    return values, missing_name_rows


def _find_surface(doc, name: Optional[str], label: Optional[str]):
    if name:
        obj = doc.getObject(name)
        if obj:
            return obj
    if label:
        for obj in doc.Objects:
            if obj.Label == label:
                return obj
    return None


def _check_surface(obj) -> Tuple[bool, List[str]]:
    issues: List[str] = []
    if obj is None:
        return False, ["Surface object not found"]

    shape = getattr(obj, "Shape", None)
    if shape is None:
        issues.append("Surface has no Shape")
        return False, issues

    if hasattr(shape, "isNull") and shape.isNull():
        issues.append("Surface shape is null")

    try:
        if hasattr(shape, "isValid") and not shape.isValid():
            issues.append("Surface shape is invalid")
    except Exception as exc:
        issues.append(f"Surface validity check failed: {exc}")

    try:
        if hasattr(shape, "check"):
            problems = shape.check(True)
            if problems:
                issues.append(f"Shape check reported {len(problems)} issues")
    except Exception as exc:
        issues.append(f"Shape check failed: {exc}")

    try:
        area = getattr(shape, "Area", None)
        if area is not None and area <= 0:
            issues.append(f"Surface area not positive: {area}")
    except Exception as exc:
        issues.append(f"Area check failed: {exc}")

    return len(issues) == 0, issues


def _quit_freecad(doc=None) -> None:
    if doc is not None:
        try:
            FreeCAD.closeDocument(doc.Name)
        except Exception:
            pass
    try:
        FreeCAD.quit()
    except Exception:
        pass


def _hard_exit(code: int) -> None:
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(code)


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply CSV constraint values to a FreeCAD sketch and validate a surface.")
    parser.add_argument("fcstd", type=Path, help="Input .FCStd file")
    parser.add_argument("csv", type=Path, help="CSV file with columns: index,type,name,value")
    parser.add_argument("--out", dest="out", type=Path, required=True, help="Output .FCStd file")
    parser.add_argument("--sketch", dest="sketch", default=None, help="Sketch Name or Label")
    parser.add_argument("--surface-name", dest="surface_name", default="Face", help="Surface object Name")
    parser.add_argument("--surface-label", dest="surface_label", default="SURFACE", help="Surface object Label")
    parser.add_argument("--dry-run", action="store_true", help="Validate only; no write")
    parser.add_argument("--allow-missing", action="store_true", help="Allow missing CSV names or sketch constraints")
    return parser.parse_args(argv)


def _load_args_from_env_or_file() -> List[str]:
    args_file = os.environ.get("PROTO3_ARGS_FILE", "").strip()
    if not args_file:
        args_file = str(Path(__file__).resolve().parent / "proto3_args.json")
    if Path(args_file).exists():
        try:
            data = json.loads(Path(args_file).read_text(encoding="utf-8-sig"))
            if isinstance(data, list):
                return [str(item) for item in data]
        except Exception:
            pass
    return []


def _extract_user_argv(argv: List[str]) -> List[str]:
    if "--" in argv:
        idx = argv.index("--")
        return argv[idx + 1 :]
    if argv and str(argv[0]).lower().endswith(".py"):
        return argv[1:]
    return []


def main() -> None:
    user_argv = _extract_user_argv(sys.argv)
    if not user_argv:
        user_argv = _load_args_from_env_or_file()
    if not user_argv:
        print("No arguments provided. Use run_proto3.py or set PROTO3_ARGS_FILE.")
        _hard_exit(1)

    args = _parse_args(user_argv)

    values, missing_name_rows = _read_csv_values(args.csv)
    if missing_name_rows:
        print("CSV rows missing name:")
        for row in missing_name_rows:
            print(f"  - {row}")
        if not args.allow_missing:
            _quit_freecad(doc=None)
            print("Missing constraint names in CSV")
            _hard_exit(2)

    doc = _open_document(args.fcstd)
    sketch = _find_sketch(doc, args.sketch)

    missing = []
    for name, value in values.items():
        index = _find_constraint_index(sketch, name)
        if index is None:
            missing.append(name)
            continue
        if not args.dry_run:
            _set_constraint_value(sketch, index, value)

    if missing:
        print("Missing constraint names in sketch:")
        for name in missing:
            print(f"  - {name}")
        if not args.allow_missing:
            _quit_freecad(doc=None)
            print("Missing constraints in sketch")
            _hard_exit(3)

    if args.dry_run:
        print("Dry-run complete. No changes written.")
        _quit_freecad(doc)
        _hard_exit(0)

    doc.recompute()

    surface = _find_surface(doc, args.surface_name, args.surface_label)
    ok, issues = _check_surface(surface)
    if not ok:
        print("Surface check failed:")
        for issue in issues:
            print(f"  - {issue}")
        _quit_freecad(doc)
        _hard_exit(4)

    doc.saveAs(str(args.out))
    print(f"Wrote: {args.out}")
    _quit_freecad(doc)
    _hard_exit(0)


if __name__ == "__main__":
    main()
