from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# Attempt FreeCAD import with a fallback to conda layout like proto2/poc.py.
try:
    import FreeCAD  # type: ignore
except ImportError:
    conda_prefix = os.environ.get(
        "CONDA_PREFIX",
        r"C:\Users\aokuni\AppData\Local\miniforge3\envs\fcad",
    )
    env_candidates = []
    if conda_prefix:
        env_candidates.append(Path(conda_prefix))
    for name in ("fcad", "fcad-codex", "b123d"):
        env_candidates.append(Path(r"C:\Users\aokuni\AppData\Local\miniforge3\envs") / name)

    for env_path in env_candidates:
        freecad_bin = env_path / "Library" / "bin"
        freecad_lib = env_path / "Library" / "lib"
        if not freecad_bin.exists():
            continue
        os.environ["PATH"] = str(freecad_bin) + os.pathsep + os.environ.get("PATH", "")
        sys.path.insert(0, str(freecad_bin))
        if freecad_lib.exists():
            sys.path.insert(0, str(freecad_lib))
        try:
            import FreeCAD  # type: ignore
            break
        except ImportError:
            continue
    else:
        raise ImportError(
            "FreeCAD import failed. Use a Python from the FreeCAD conda env "
            "(e.g., conda activate fcad) or run with "
            "C:\\Users\\aokuni\\AppData\\Local\\miniforge3\\envs\\fcad\\python.exe. "
            f"CONDA_PREFIX={conda_prefix}, sys.version={sys.version}"
        )

_PROTO2_IO = None


def _proto2_io():
    global _PROTO2_IO
    if _PROTO2_IO is not None:
        return _PROTO2_IO
    repo = Path(__file__).resolve().parents[2]
    proto2_dir = repo / "src" / "proto2-codex"
    sys.path.insert(0, str(proto2_dir))
    import freecad_sketch_io as proto2_io  # type: ignore

    _PROTO2_IO = proto2_io
    return _PROTO2_IO


@dataclass
class ConstraintSpec:
    index: int
    name: str
    ctype: str
    base_value: float
    sketch: Optional[object] = None
    angle_unit: Optional[str] = None


def _open_document(path: Path):
    proto2_io = _proto2_io()
    return proto2_io.open_document(path)


def _find_sketch(doc, sketch_hint: Optional[str] = None):
    proto2_io = _proto2_io()
    return proto2_io.find_sketch(doc, sketch_hint)


def _list_sketches(doc) -> List[object]:
    return [obj for obj in doc.Objects if obj.TypeId == "Sketcher::SketchObject"]


def _find_constraint_index(sketch, name: str) -> Optional[int]:
    proto2_io = _proto2_io()
    return proto2_io.find_constraint_index(sketch, name)


def _set_constraint_value(
    sketch,
    index: int,
    value: float,
    ctype: str,
    angle_unit: Optional[str],
) -> None:
    if ctype == "Angle":
        if angle_unit == "rad":
            value = math.degrees(value)
        quantity = FreeCAD.Units.Quantity(f"{value} deg")
        sketch.setDatum(index, quantity)
        return
    if ctype in {"Distance", "DistanceX", "DistanceY"}:
        # For distances, plain float is accepted and more robust in this model.
        sketch.setDatum(index, float(value))
        return
    sketch.setDatum(index, value)


def _setdatum_ok(sketch, index: int) -> bool:
    try:
        current = sketch.getDatum(index)
    except Exception:
        return False
    try:
        sketch.setDatum(index, current)
        return True
    except Exception:
        return False


def _build_setdatum_map(sketch) -> dict:
    mapping: dict = {}
    for idx, constraint in enumerate(getattr(sketch, "Constraints", [])):
        name = getattr(constraint, "Name", "") or ""
        if not name:
            continue
        ctype = getattr(constraint, "Type", "") or ""
        if not _setdatum_ok(sketch, idx):
            continue
        mapping.setdefault(name, []).append((idx, ctype))
    return mapping


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


def _check_recompute(doc) -> Tuple[bool, List[str]]:
    issues: List[str] = []
    try:
        objects = list(getattr(doc, "Objects", []))
    except Exception:
        objects = []
    for obj in objects:
        try:
            state = getattr(obj, "State", None)
            if state and any(flag in state for flag in ("Invalid", "RecomputeError")):
                issues.append(f"{obj.Name} state={state}")
        except Exception:
            continue
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


def _read_csv_specs(path: Path) -> List[ConstraintSpec]:
    specs: List[ConstraintSpec] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("name") or "").strip()
            if not name:
                continue
            raw_value = row.get("value")
            if raw_value is None:
                continue
            try:
                value = float(raw_value)
            except Exception:
                continue
            ctype = (row.get("type") or "").strip()
            try:
                index = int(row.get("index") or 0)
            except Exception:
                index = 0
            spec = ConstraintSpec(index=index, name=name, ctype=ctype, base_value=value)
            if ctype == "Angle":
                spec.angle_unit = "deg"
            specs.append(spec)
    return specs


def _load_template(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError("PyYAML is required for YAML templates") from exc
        return yaml.safe_load(text)
    if path.suffix.lower() == ".json":
        import json
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                import yaml  # type: ignore
            except Exception as exc:
                raise RuntimeError("Template looks like YAML but PyYAML is missing") from exc
            return yaml.safe_load(text)
    try:
        import yaml  # type: ignore
    except Exception:
        import json

        return json.loads(text)
    return yaml.safe_load(text)


def _read_template_specs(path: Path, csv_path: Path) -> List[ConstraintSpec]:
    data = _load_template(path)
    constraints = data.get("constraints", {}) if isinstance(data, dict) else {}
    csv_specs = _read_csv_specs(csv_path)
    csv_by_name = {spec.name: spec for spec in csv_specs}
    specs: List[ConstraintSpec] = []
    for name, meta in constraints.items():
        if not isinstance(meta, dict):
            continue
        index = int(meta.get("index", 0) or 0)
        ctype = str(meta.get("type", "") or "")
        base_value = meta.get("value", None)
        if base_value is None and name in csv_by_name:
            base_value = csv_by_name[name].base_value
        if base_value is None:
            continue
        try:
            base_value = float(base_value)
        except Exception:
            continue
        spec = ConstraintSpec(index=index, name=name, ctype=ctype, base_value=base_value)
        if ctype == "Angle":
            spec.angle_unit = "rad" if base_value <= 6.5 else "deg"
        specs.append(spec)
    return specs


def _convert_to_sketch_value(ctype: str, value: float, angle_unit: Optional[str]) -> Optional[float]:
    if not math.isfinite(value):
        return None
    if ctype in {"Distance", "DistanceX", "DistanceY"}:
        if value <= 0:
            return None
        return value
    if ctype == "Angle":
        if angle_unit == "rad":
            if value <= 0 or value >= math.pi:
                return None
            return value
        if value <= 0 or value >= 180:
            return None
        return math.radians(value)
    return value


def _clamp_candidate(ctype: str, value: float, angle_unit: Optional[str]) -> Optional[float]:
    if not math.isfinite(value):
        return None
    if ctype in {"Distance", "DistanceX", "DistanceY"}:
        if value <= 0:
            return None
        return value
    if ctype == "Angle":
        if angle_unit == "rad":
            if value <= 0 or value >= math.pi:
                return None
            return value
        if value <= 0 or value >= 180:
            return None
        return value
    return value


def _latin_hypercube(
    samples: int,
    dims: int,
    low: float,
    high: float,
    rng: random.Random,
) -> List[List[float]]:
    buckets = []
    for _ in range(dims):
        slots = [(i + rng.random()) / samples for i in range(samples)]
        rng.shuffle(slots)
        buckets.append(slots)
    result: List[List[float]] = []
    span = high - low
    for i in range(samples):
        row = [low + buckets[d][i] * span for d in range(dims)]
        result.append(row)
    return result


def _estimate_runtime(
    doc,
    surface,
    specs: List[ConstraintSpec],
    samples: List[List[float]],
    base_values: List[float],
    base_sketch_values: List[Optional[float]],
    warmup: int,
    debug: bool = False,
) -> float:
    if warmup <= 0:
        return 0.0
    warmup = min(warmup, len(samples))
    start = time.perf_counter()
    for i in range(warmup):
        ratios = samples[i]
        _apply_sample(doc, surface, specs, base_values, base_sketch_values, ratios, debug=debug)
    elapsed = time.perf_counter() - start
    return (elapsed / warmup) * len(samples)


def _apply_sample(
    doc,
    surface,
    specs: List[ConstraintSpec],
    base_values: List[float],
    base_sketch_values: List[Optional[float]],
    ratios: List[float],
    debug: bool = False,
) -> bool:
    # Reset to baseline before applying a new sample to avoid cascading invalid states.
    for spec, base_sketch in zip(specs, base_sketch_values):
        if spec.sketch is None or base_sketch is None:
            continue
        try:
            _set_constraint_value(spec.sketch, spec.index, base_sketch, spec.ctype, spec.angle_unit)
        except Exception:
            return False
    for spec, base, ratio in zip(specs, base_values, ratios):
        if spec.sketch is None:
            continue
        idx = spec.index
        candidate = base * ratio
        candidate = _clamp_candidate(spec.ctype, candidate, spec.angle_unit)
        if candidate is None:
            return False
        sketch_value = _convert_to_sketch_value(spec.ctype, candidate, spec.angle_unit)
        if sketch_value is None:
            return False
        try:
            _set_constraint_value(spec.sketch, idx, sketch_value, spec.ctype, spec.angle_unit)
        except Exception as exc:
            if debug:
                sketch_name = getattr(spec.sketch, "Name", "")
                sketch_label = getattr(spec.sketch, "Label", "")
                count = getattr(spec.sketch, "ConstraintCount", None)
                print(
                    "setDatum failed:",
                    f"name={spec.name}",
                    f"type={spec.ctype}",
                    f"idx={idx}",
                    f"value={sketch_value}",
                    f"sketch={sketch_name}/{sketch_label}",
                    f"constraint_count={count}",
                    f"error={exc}",
                    sep=" ",
                )
            return False

    doc.recompute()
    ok_recompute, _ = _check_recompute(doc)
    if not ok_recompute:
        return False
    ok, _ = _check_surface(surface)
    return ok


def _update_ranges(
    mins: List[Optional[float]],
    maxs: List[Optional[float]],
    specs: List[ConstraintSpec],
    base_values: List[float],
    ratios: List[float],
) -> None:
    for i, (spec, base, ratio) in enumerate(zip(specs, base_values, ratios)):
        candidate = base * ratio
        candidate = _clamp_candidate(spec.ctype, candidate, spec.angle_unit)
        if candidate is None:
            continue
        current_min = mins[i]
        current_max = maxs[i]
        mins[i] = candidate if current_min is None else min(current_min, candidate)
        maxs[i] = candidate if current_max is None else max(current_max, candidate)


def _append_ranges_to_csv(
    input_csv: Path,
    output_csv: Path,
    specs: List[ConstraintSpec],
    mins: List[Optional[float]],
    maxs: List[Optional[float]],
    min_col: str,
    max_col: str,
) -> None:
    rows = list(csv.DictReader(input_csv.open("r", encoding="utf-8")))
    index_by_name = {spec.name: i for i, spec in enumerate(specs)}
    fieldnames = list(rows[0].keys()) if rows else ["index", "type", "name", "value"]
    if min_col not in fieldnames:
        fieldnames.append(min_col)
    if max_col not in fieldnames:
        fieldnames.append(max_col)

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            name = (row.get("name") or "").strip()
            idx = index_by_name.get(name)
            if idx is not None:
                row[min_col] = "" if mins[idx] is None else f"{mins[idx]:.6f}".rstrip("0").rstrip(".")
                row[max_col] = "" if maxs[idx] is None else f"{maxs[idx]:.6f}".rstrip("0").rstrip(".")
            else:
                row[min_col] = row.get(min_col, "")
                row[max_col] = row.get(max_col, "")
            writer.writerow(row)


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate feasible ranges via multi-variable sampling on FreeCAD sketch constraints."
    )
    parser.add_argument("fcstd", type=Path, help="Input .FCStd file (template)")
    parser.add_argument("csv", type=Path, help="CSV (deg for Angle) with index,type,name,value")
    parser.add_argument("--out-csv", dest="out_csv", type=Path, default=None, help="Output CSV path")
    parser.add_argument("--sketch", dest="sketch", default=None, help="Sketch Name or Label")
    parser.add_argument("--surface-name", dest="surface_name", default="Face", help="Surface object Name")
    parser.add_argument("--surface-label", dest="surface_label", default="SURFACE", help="Surface object Label")
    parser.add_argument("--samples", type=int, default=5000, help="Number of samples")
    parser.add_argument("--ratio-min", type=float, default=0.7, help="Min ratio for sampling")
    parser.add_argument("--ratio-max", type=float, default=1.3, help="Max ratio for sampling")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup samples for time estimate")
    parser.add_argument("--estimate-only", action="store_true", help="Estimate runtime and exit")
    parser.add_argument("--min-col", type=str, default="feasible_min", help="Output min column name")
    parser.add_argument("--max-col", type=str, default="feasible_max", help="Output max column name")
    parser.add_argument("--allow-missing", action="store_true", help="Allow missing constraints in sketch")
    parser.add_argument("--debug", action="store_true", help="Verbose errors when a constraint update fails")
    parser.add_argument("--template", type=Path, default=None, help="Constraint template (YAML/JSON) from proto2")
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args(sys.argv[1:])
    if args.template:
        specs = _read_template_specs(args.template, args.csv)
    else:
        specs = _read_csv_specs(args.csv)
    if not specs:
        print("No constraints loaded from CSV.")
        _hard_exit(2)

    doc = _open_document(args.fcstd)
    surface = _find_surface(doc, args.surface_name, args.surface_label)

    sketches = _list_sketches(doc)
    if args.sketch:
        sketches = [_find_sketch(doc, args.sketch)]
    elif not sketches:
        print("No Sketcher::SketchObject found in document")
        _quit_freecad(doc)
        _hard_exit(3)

    missing = []
    duplicates = []
    unresolved = []
    for spec in specs:
        found = False
        for sketch in sketches:
            idx = _find_constraint_index(sketch, spec.name)
            if idx is None:
                continue
            if found:
                duplicates.append(spec.name)
                continue
            spec.index = idx
            spec.sketch = sketch
            found = True
        if not found:
            missing.append(spec.name)
    proto2_io = _proto2_io()
    setdatum_maps = {sketch: _build_setdatum_map(sketch) for sketch in sketches}
    for spec in specs:
        if spec.sketch is None:
            continue
        candidates = setdatum_maps.get(spec.sketch, {}).get(spec.name, [])
        idx = None
        if spec.ctype:
            for candidate_idx, candidate_type in candidates:
                if candidate_type == spec.ctype:
                    idx = candidate_idx
                    break
        if idx is None and candidates:
            idx = candidates[0][0]
        if idx is None:
            unresolved.append(spec.name)
            continue
        spec.index = idx
    if duplicates:
        print("Duplicate constraint names found across sketches:")
        for name in sorted(set(duplicates)):
            print(f"  - {name}")
    if unresolved:
        print("Named constraints found but not resolvable as dimensional values:")
        for name in sorted(set(unresolved)):
            print(f"  - {name}")
    if missing and not args.allow_missing:
        for name in missing:
            print(f"Missing constraint in sketches: {name}")
        _quit_freecad(doc)
        _hard_exit(3)

    base_values = [spec.base_value for spec in specs]
    base_sketch_values: List[Optional[float]] = []
    for spec in specs:
        if spec.ctype == "Angle" and spec.angle_unit == "rad":
            base_sketch_values.append(math.degrees(spec.base_value))
        else:
            base_sketch_values.append(spec.base_value)
    rng = random.Random(args.seed)
    samples = _latin_hypercube(args.samples, len(specs), args.ratio_min, args.ratio_max, rng)

    est_seconds = _estimate_runtime(
        doc,
        surface,
        specs,
        samples,
        base_values,
        base_sketch_values,
        args.warmup,
        debug=args.debug,
    )
    print(f"Estimated runtime for {args.samples} samples: {est_seconds:.1f} sec")
    if args.estimate_only:
        _quit_freecad(doc)
        _hard_exit(0)

    mins: List[Optional[float]] = [None] * len(specs)
    maxs: List[Optional[float]] = [None] * len(specs)
    success = 0
    for ratios in samples:
        if _apply_sample(doc, surface, specs, base_values, base_sketch_values, ratios, debug=args.debug):
            success += 1
            _update_ranges(mins, maxs, specs, base_values, ratios)
    print(f"Success samples: {success}/{len(samples)}")

    out_csv = args.out_csv or args.csv
    _append_ranges_to_csv(args.csv, out_csv, specs, mins, maxs, args.min_col, args.max_col)
    print(f"Wrote CSV: {out_csv}")
    _quit_freecad(doc)
    _hard_exit(0)


if __name__ == "__main__":
    main()
