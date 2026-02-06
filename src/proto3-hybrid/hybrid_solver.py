from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

# FreeCAD import fallback
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
        raise


@dataclass
class ConstraintSpec:
    index: int
    name: str
    ctype: str
    base_value: float
    sketch: Optional[object] = None
    angle_unit: Optional[str] = None


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


def _check_surface(obj) -> bool:
    if obj is None:
        return False
    shape = getattr(obj, "Shape", None)
    if shape is None:
        return False
    if hasattr(shape, "isNull") and shape.isNull():
        return False
    try:
        if hasattr(shape, "isValid") and not shape.isValid():
            return False
    except Exception:
        return False
    try:
        if hasattr(shape, "check"):
            problems = shape.check(True)
            if problems:
                return False
    except Exception:
        return False
    try:
        area = getattr(shape, "Area", None)
        if area is not None and area <= 0:
            return False
    except Exception:
        return False
    return True


def _check_recompute(doc) -> bool:
    try:
        objects = list(getattr(doc, "Objects", []))
    except Exception:
        objects = []
    for obj in objects:
        try:
            state = getattr(obj, "State", None)
            if state and any(flag in state for flag in ("Invalid", "RecomputeError")):
                return False
        except Exception:
            continue
    return True


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


def _load_template(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml  # type: ignore
        return yaml.safe_load(text)
    if path.suffix.lower() == ".json":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            import yaml  # type: ignore
            return yaml.safe_load(text)
    try:
        import yaml  # type: ignore
    except Exception:
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
        base_value = float(base_value)
        spec = ConstraintSpec(index=index, name=name, ctype=ctype, base_value=base_value)
        if ctype == "Angle":
            spec.angle_unit = "rad" if base_value <= 6.5 else "deg"
        specs.append(spec)
    return specs


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
            value = float(raw_value)
            ctype = (row.get("type") or "").strip()
            index = int(row.get("index") or 0)
            spec = ConstraintSpec(index=index, name=name, ctype=ctype, base_value=value)
            if ctype == "Angle":
                spec.angle_unit = "deg"
            specs.append(spec)
    return specs


def _set_constraint_value(
    sketch,
    index: int,
    value: float,
    ctype: str,
    angle_unit: Optional[str],
) -> None:
    if ctype == "Angle":
        # angle_unit == "rad" means value is radians; otherwise assume degrees.
        if angle_unit == "rad":
            value = math.degrees(value)
        quantity = FreeCAD.Units.Quantity(f"{value} deg")
        sketch.setDatum(index, quantity)
        return
    if ctype in {"Distance", "DistanceX", "DistanceY"}:
        sketch.setDatum(index, float(value))
        return
    sketch.setDatum(index, value)


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
        return value
    return value


def _apply_sample(
    doc,
    surface,
    specs: List[ConstraintSpec],
    base_values: List[float],
    base_sketch_values: List[Optional[float]],
    ratios,
) -> int:
    # reset baseline
    for spec, base_sketch in zip(specs, base_sketch_values):
        if spec.sketch is None or base_sketch is None:
            continue
        # base_sketch_values are already in sketch units (deg/mm).
        try:
            _set_constraint_value(spec.sketch, spec.index, base_sketch, spec.ctype, None)
        except Exception:
            return 0

    for spec, base, ratio in zip(specs, base_values, ratios):
        candidate = _clamp_candidate(spec.ctype, base * ratio, spec.angle_unit)
        if candidate is None:
            return 0
        sketch_value = _convert_to_sketch_value(spec.ctype, candidate, spec.angle_unit)
        if sketch_value is None:
            return 0
        try:
            _set_constraint_value(spec.sketch, spec.index, sketch_value, spec.ctype, spec.angle_unit)
        except Exception:
            return 0

    doc.recompute()
    if not _check_recompute(doc):
        return 0
    if not _check_surface(surface):
        return 0
    return 1


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
    parser = argparse.ArgumentParser(description="Hybrid feasibility explorer (LHS + active learning).")
    parser.add_argument("fcstd", type=Path, help="Input .FCStd file")
    parser.add_argument("csv", type=Path, help="CSV (deg for Angle) with index,type,name,value")
    parser.add_argument("--template", type=Path, required=True, help="Constraint template from proto2")
    parser.add_argument("--sketch", dest="sketch", default=None, help="Sketch Name or Label")
    parser.add_argument("--surface-name", dest="surface_name", default="Face", help="Surface object Name")
    parser.add_argument("--surface-label", dest="surface_label", default="SURFACE", help="Surface object Label")
    parser.add_argument("--samples", type=int, default=5000, help="Total evaluation budget")
    parser.add_argument("--init-samples", type=int, default=200, help="Initial LHS samples")
    parser.add_argument("--candidate-samples", type=int, default=5000, help="Candidates per iteration")
    parser.add_argument("--batch-size", type=int, default=200, help="Evaluations per iteration")
    parser.add_argument("--iters", type=int, default=10, help="Active learning iterations")
    parser.add_argument("--explore-frac", type=float, default=0.2, help="Fraction of random exploration per iter")
    parser.add_argument("--narrow-ratio", type=float, default=0.15, help="Fallback narrow ratio span when safe=0")
    parser.add_argument(
        "--simple-frac",
        type=float,
        default=0.2,
        help="Fraction of total budget for pure random sampling before active learning",
    )
    parser.add_argument("--auto-shrink", action="store_true", help="Auto-shrink range if safe ratio drops")
    parser.add_argument("--shrink-trigger", type=float, default=0.2, help="Safe ratio threshold to shrink")
    parser.add_argument("--shrink-factor", type=float, default=0.5, help="Range shrink factor (0-1)")
    parser.add_argument("--shrink-window", type=int, default=500, help="Window size for safe ratio check")
    parser.add_argument("--auto-expand", action="store_true", help="Auto-expand range if safe ratio is high")
    parser.add_argument("--expand-trigger", type=float, default=0.8, help="Safe ratio threshold to expand")
    parser.add_argument("--expand-factor", type=float, default=1.5, help="Range expand factor (>1)")
    parser.add_argument("--expand-window", type=int, default=500, help="Window size for expansion check")
    parser.add_argument(
        "--expand-on-zero",
        action="store_true",
        help="Auto-expand if safe ratio is zero for a window",
    )
    parser.add_argument(
        "--zero-expand-factor",
        type=float,
        default=2.0,
        help="Expansion factor when expanding due to safe ratio zero",
    )
    parser.add_argument(
        "--center-on-safe",
        action="store_true",
        help="Center shrink/expand ranges on median of safe samples in the window",
    )
    parser.add_argument(
        "--safe-envelope",
        action="store_true",
        help="Use per-dimension min/max from safe samples as sampling bounds",
    )
    parser.add_argument(
        "--safe-envelope-margin",
        type=float,
        default=0.0,
        help="Margin added around safe envelope (ratio units)",
    )
    parser.add_argument(
        "--envelope-fallback-global",
        action="store_true",
        help="If safe=0 for a window, temporarily sample from global ratio range",
    )
    parser.add_argument(
        "--envelope-grow-on-zero",
        type=float,
        default=0.0,
        help="Increase envelope margin when a window has safe=0 (ratio units)",
    )
    parser.add_argument(
        "--recover-on-zero",
        action="store_true",
        help="If a window has safe=0, force next batch to pure random exploration",
    )
    parser.add_argument("--ratio-min", type=float, default=0.7, help="Min ratio for sampling")
    parser.add_argument("--ratio-max", type=float, default=1.3, help="Max ratio for sampling")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--estimate-only", action="store_true", help="Estimate runtime and exit")
    parser.add_argument("--estimate-warmup", type=int, default=50, help="Warmup evals for estimate-only")
    parser.add_argument("--min-col", type=str, default="feasible_min", help="Output min column name")
    parser.add_argument("--max-col", type=str, default="feasible_max", help="Output max column name")
    parser.add_argument("--out-csv", dest="out_csv", type=Path, default=None, help="Output CSV path")
    parser.add_argument("--log-every", type=int, default=50, help="Log progress every N evaluations")
    return parser.parse_args(argv)


def _lhs_samples_py(n: int, dims: int, low: float, high: float, seed: int) -> List[List[float]]:
    rng = random.Random(seed)
    buckets: List[List[float]] = []
    for _ in range(dims):
        slots = [(i + rng.random()) / n for i in range(n)]
        rng.shuffle(slots)
        buckets.append(slots)
    span = high - low
    return [[low + buckets[d][i] * span for d in range(dims)] for i in range(n)]


def _lazy_np() -> Any:
    import numpy as np  # type: ignore
    return np


def _lazy_pd() -> Any:
    import pandas as pd  # type: ignore
    return pd


def _lazy_rf() -> Any:
    from sklearn.ensemble import RandomForestClassifier  # type: ignore
    return RandomForestClassifier


def _lazy_qmc() -> Any:
    from scipy.stats import qmc  # type: ignore
    return qmc


def _lhs_samples(n: int, dims: int, low: float, high: float, seed: int):
    qmc = _lazy_qmc()
    sampler = qmc.LatinHypercube(d=dims, seed=seed)
    u = sampler.random(n)
    return qmc.scale(u, low, high)


def _sobol_samples(n: int, dims: int, low: float, high: float, seed: int):
    qmc = _lazy_qmc()
    sampler = qmc.Sobol(d=dims, scramble=True, seed=seed)
    # Sobol balance is best with power-of-two sizes; round up.
    pow2 = 1 << (n - 1).bit_length()
    u = sampler.random(pow2)
    if pow2 > n:
        u = u[:n]
    return qmc.scale(u, low, high)


def _range_window(args: argparse.Namespace) -> Optional[int]:
    if args.auto_shrink and args.auto_expand:
        return max(1, min(args.shrink_window, args.expand_window))
    if args.auto_shrink:
        return max(1, args.shrink_window)
    if args.auto_expand:
        return max(1, args.expand_window)
    return None


def _update_range(
    cur_min: float,
    cur_max: float,
    args: argparse.Namespace,
    safe_ratio: float,
    label: str,
    *,
    center_ratio: Optional[float] = None,
    force_expand: bool = False,
    expand_factor: Optional[float] = None,
) -> Tuple[float, float]:
    span = cur_max - cur_min
    max_span = max(1e-9, args.ratio_max - args.ratio_min)
    if args.auto_expand and force_expand:
        factor = args.expand_factor if expand_factor is None else expand_factor
        new_span = span * max(1.0, factor)
        new_span = max(1e-6, min(max_span, new_span))
        center = 1.0 if center_ratio is None else center_ratio
        cur_min = max(args.ratio_min, center - new_span / 2)
        cur_max = min(args.ratio_max, center + new_span / 2)
        print(f"[{label}] auto-expand: safe_ratio=zero -> range=({cur_min:.3f},{cur_max:.3f})")
        return cur_min, cur_max
    if args.auto_shrink and safe_ratio < args.shrink_trigger:
        new_span = span * max(0.05, min(1.0, args.shrink_factor))
        new_span = max(1e-6, min(max_span, new_span))
        center = 1.0 if center_ratio is None else center_ratio
        cur_min = max(args.ratio_min, center - new_span / 2)
        cur_max = min(args.ratio_max, center + new_span / 2)
        print(f"[{label}] auto-shrink: safe_ratio={safe_ratio:.2f} -> range=({cur_min:.3f},{cur_max:.3f})")
        return cur_min, cur_max
    if args.auto_expand and (force_expand or safe_ratio > args.expand_trigger):
        factor = args.expand_factor if expand_factor is None else expand_factor
        new_span = span * max(1.0, factor)
        new_span = max(1e-6, min(max_span, new_span))
        center = 1.0 if center_ratio is None else center_ratio
        cur_min = max(args.ratio_min, center - new_span / 2)
        cur_max = min(args.ratio_max, center + new_span / 2)
        reason = "zero" if force_expand else f"{safe_ratio:.2f}"
        print(f"[{label}] auto-expand: safe_ratio={reason} -> range=({cur_min:.3f},{cur_max:.3f})")
    return cur_min, cur_max


def _center_ratio_from_block(X_block, y_block) -> Optional[float]:
    ratios = []
    for row, ok in zip(X_block, y_block):
        if not ok:
            continue
        # Use per-sample mean ratio as a scalar center.
        try:
            ratios.append(float(sum(row) / len(row)))
        except Exception:
            continue
    if not ratios:
        return None
    ratios.sort()
    mid = len(ratios) // 2
    if len(ratios) % 2 == 1:
        return ratios[mid]
    return (ratios[mid - 1] + ratios[mid]) / 2.0


def _uniform_samples(n: int, low, high, seed: int):
    np = _lazy_np()
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=(n, len(low)))


def _safe_envelope(X, y, ratio_min: float, ratio_max: float, margin: float):
    np = _lazy_np()
    safe_mask = y == 1
    safe_rows = X[safe_mask]
    if safe_rows.size == 0:
        return None, None
    safe_min = safe_rows.min(axis=0)
    safe_max = safe_rows.max(axis=0)
    pad = max(0.0, margin)
    low = np.maximum(ratio_min, safe_min - pad)
    high = np.minimum(ratio_max, safe_max + pad)
    return low, high


def main() -> None:
    args = _parse_args(sys.argv[1:])
    print("[proto3-hybrid] start", flush=True)
    specs = _read_template_specs(args.template, args.csv)
    if not specs:
        print("No constraints loaded.")
        return

    doc = _open_document(args.fcstd)
    sketch = _find_sketch(doc, args.sketch)
    for spec in specs:
        spec.sketch = sketch

    surface = _find_surface(doc, args.surface_name, args.surface_label)

    base_values = [spec.base_value for spec in specs]
    base_sketch_values: List[Optional[float]] = []
    for spec in specs:
        if spec.ctype == "Angle" and spec.angle_unit == "rad":
            base_sketch_values.append(math.degrees(spec.base_value))
        else:
            base_sketch_values.append(spec.base_value)

    dims = len(specs)
    budget = args.samples
    init_n = min(args.init_samples, budget)
    start_time = time.perf_counter()
    # Initial LHS
    if args.estimate_only:
        X = _lhs_samples_py(init_n, dims, args.ratio_min, args.ratio_max, args.seed)
        y = [0] * init_n
    else:
        np = _lazy_np()
        X = _lhs_samples(init_n, dims, args.ratio_min, args.ratio_max, args.seed)
        y = np.zeros(init_n, dtype=int)
    if args.estimate_only:
        warmup = min(args.estimate_warmup, init_n)
        if warmup <= 0:
            print("Estimated runtime for 0 samples: 0.0 sec")
            _quit_freecad(doc)
            return
        for i in range(warmup):
            y[i] = _apply_sample(doc, surface, specs, base_values, base_sketch_values, X[i])
            if (i + 1) % args.log_every == 0 or i + 1 == warmup:
                elapsed = time.perf_counter() - start_time
                print(f"[estimate] {i+1}/{warmup} evals, safe={int(sum(y[:i+1]))}, elapsed={elapsed:.1f}s")
        elapsed = time.perf_counter() - start_time
        est = (elapsed / warmup) * budget
        print(f"Estimated runtime for {budget} samples: {est:.1f} sec")
        _quit_freecad(doc)
        return

    for i in range(init_n):
        y[i] = _apply_sample(doc, surface, specs, base_values, base_sketch_values, X[i])
        if (i + 1) % args.log_every == 0 or i + 1 == init_n:
            elapsed = time.perf_counter() - start_time
            print(f"[init] {i+1}/{init_n} evals, safe={int(y[:i+1].sum())}, elapsed={elapsed:.1f}s")

    envelope_margin = args.safe_envelope_margin
    if args.safe_envelope:
        env_low, env_high = _safe_envelope(X, y, args.ratio_min, args.ratio_max, envelope_margin)

    remaining = budget - init_n

    cur_min = args.ratio_min
    cur_max = args.ratio_max
    env_low = None
    env_high = None

    # Simple (mixed) sampling phase
    simple_n = int(budget * max(0.0, min(1.0, args.simple_frac)))
    simple_n = min(simple_n, remaining)
    if simple_n > 0:
        print(f"[simple] sampling {simple_n} points before active learning")
        window = _range_window(args)
        X_simple = np.zeros((simple_n, dims), dtype=float)
        y_simple = np.zeros(simple_n, dtype=int)
        next_idx = 0
        block_id = 0
        while next_idx < simple_n:
            block = window or (simple_n - next_idx)
            block = min(block, simple_n - next_idx)
            if args.safe_envelope and env_low is not None and env_high is not None:
                X_block = _uniform_samples(block, env_low, env_high, args.seed + 999 + block_id)
            else:
                X_block = _sobol_samples(block, dims, cur_min, cur_max, args.seed + 999 + block_id)
            X_simple[next_idx : next_idx + block] = X_block
            for j in range(block):
                i = next_idx + j
                y_simple[i] = _apply_sample(doc, surface, specs, base_values, base_sketch_values, X_simple[i])
                if (i + 1) % args.log_every == 0 or i + 1 == simple_n:
                    elapsed = time.perf_counter() - start_time
                    print(f"[simple] {i+1}/{simple_n} evals, safe={int(y_simple[:i+1].sum())}, elapsed={elapsed:.1f}s")
            if window:
                window_safe = int(y_simple[next_idx : next_idx + block].sum())
                safe_ratio = window_safe / block
                center_ratio = None
                if args.center_on_safe:
                    center_ratio = _center_ratio_from_block(
                        X_simple[next_idx : next_idx + block],
                        y_simple[next_idx : next_idx + block],
                    )
                if args.expand_on_zero and window_safe == 0:
                    if args.safe_envelope and args.envelope_grow_on_zero > 0:
                        envelope_margin = min(
                            args.ratio_max - args.ratio_min,
                            envelope_margin + args.envelope_grow_on_zero,
                        )
                        env_low, env_high = _safe_envelope(
                            X, y, args.ratio_min, args.ratio_max, envelope_margin
                        )
                    cur_min, cur_max = _update_range(
                        cur_min,
                        cur_max,
                        args,
                        safe_ratio,
                        "simple",
                        center_ratio=center_ratio,
                        force_expand=True,
                        expand_factor=args.zero_expand_factor,
                    )
                    if args.safe_envelope and args.envelope_fallback_global:
                        env_low = None
                        env_high = None
                else:
                    cur_min, cur_max = _update_range(
                        cur_min,
                        cur_max,
                        args,
                        safe_ratio,
                        "simple",
                        center_ratio=center_ratio,
                    )
            next_idx += block
            block_id += 1
        X = np.vstack([X, X_simple])
        y = np.concatenate([y, y_simple])
        remaining -= simple_n

    # Active learning loop
    np = _lazy_np()
    pd = _lazy_pd()
    RandomForestClassifier = _lazy_rf()
    iters = max(1, args.iters)
    batch = max(1, args.batch_size)
    recover_random = False
    for it in range(iters):
        if remaining <= 0:
            break
        safe_total = int(y.sum())
        print(f"[iter {it+1}/{iters}] training model on {len(y)} samples (safe={safe_total})")
        # If only one class is present, boundary learning is not possible.
        if safe_total == 0 or safe_total == len(y):
            print(f"[iter {it+1}] single-class data; skipping active learning")
            break
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            class_weight="balanced",
            random_state=args.seed,
        )
        model.fit(X, y)

        cand_n = min(args.candidate_samples, 50000)
        k = min(batch, remaining)
        window = _range_window(args)
        y_new = np.zeros(k, dtype=int)
        X_new = np.zeros((k, dims), dtype=float)
        next_idx = 0
        block_id = 0
        while next_idx < k:
            block = window or (k - next_idx)
            block = min(block, k - next_idx)
            if recover_random:
                X_block = _sobol_samples(
                    block,
                    dims,
                    args.ratio_min,
                    args.ratio_max,
                    args.seed + it + 5000 + block_id,
                )
                X_new[next_idx : next_idx + block] = X_block
                if (next_idx + block) % args.log_every == 0 or next_idx + block == k:
                    print(f"[iter {it+1}] recover-random block={block}")
                recover_random = False
            else:
                if args.safe_envelope and env_low is not None and env_high is not None:
                    C = _uniform_samples(cand_n, env_low, env_high, args.seed + it + 1 + block_id)
                else:
                    C = _sobol_samples(cand_n, dims, cur_min, cur_max, args.seed + it + 1 + block_id)
                p = model.predict_proba(C)[:, 1]
                idx = np.argsort(np.abs(p - 0.5))
                explore_k = max(1, int(block * args.explore_frac))
                exploit_k = block - explore_k
                X_exploit = C[idx[:exploit_k]] if exploit_k > 0 else np.empty((0, dims))
                if args.safe_envelope and env_low is not None and env_high is not None:
                    X_explore = _uniform_samples(explore_k, env_low, env_high, args.seed + it + 101 + block_id)
                else:
                    X_explore = _sobol_samples(
                        explore_k, dims, cur_min, cur_max, args.seed + it + 101 + block_id
                    )
                X_block = np.vstack([X_exploit, X_explore]) if exploit_k > 0 else X_explore
                if X_block.shape[0] > block:
                    X_block = X_block[:block]
                X_new[next_idx : next_idx + block] = X_block
            
            for j in range(block):
                i = next_idx + j
                y_new[i] = _apply_sample(doc, surface, specs, base_values, base_sketch_values, X_new[i])
                if (i + 1) % args.log_every == 0 or i + 1 == k:
                    elapsed = time.perf_counter() - start_time
                    print(
                        f"[iter {it+1}] {i+1}/{k} evals, "
                        f"safe={int(y_new[:i+1].sum())}, "
                        f"remaining={max(0, remaining - (i+1))}, "
                        f"elapsed={elapsed:.1f}s"
                    )
            if window:
                window_safe = int(y_new[next_idx : next_idx + block].sum())
                safe_ratio = window_safe / block
                center_ratio = None
                if args.center_on_safe:
                    center_ratio = _center_ratio_from_block(
                        X_new[next_idx : next_idx + block],
                        y_new[next_idx : next_idx + block],
                    )
                if args.expand_on_zero and window_safe == 0:
                    if args.safe_envelope and args.envelope_grow_on_zero > 0:
                        envelope_margin = min(
                            args.ratio_max - args.ratio_min,
                            envelope_margin + args.envelope_grow_on_zero,
                        )
                        env_low, env_high = _safe_envelope(
                            X, y, args.ratio_min, args.ratio_max, envelope_margin
                        )
                    cur_min, cur_max = _update_range(
                        cur_min,
                        cur_max,
                        args,
                        safe_ratio,
                        f"iter {it+1}",
                        center_ratio=center_ratio,
                        force_expand=True,
                        expand_factor=args.zero_expand_factor,
                    )
                    if args.safe_envelope and args.envelope_fallback_global:
                        # Drop envelope for next block to re-explore globally.
                        env_low = None
                        env_high = None
                    if args.recover_on_zero:
                        recover_random = True
                else:
                    cur_min, cur_max = _update_range(
                        cur_min,
                        cur_max,
                        args,
                        safe_ratio,
                        f"iter {it+1}",
                        center_ratio=center_ratio,
                    )
                    recover_random = False
            next_idx += block
            block_id += 1
        X = np.vstack([X, X_new])
        y = np.concatenate([y, y_new])
        remaining -= k
        if args.safe_envelope:
            env_low, env_high = _safe_envelope(
                X, y, args.ratio_min, args.ratio_max, args.safe_envelope_margin
            )
        if int(y_new.sum()) == 0:
            # If no safe samples found, tighten sampling around base ratios.
            narrow = max(0.02, args.narrow_ratio)
            low = max(args.ratio_min, 1.0 - narrow)
            high = min(args.ratio_max, 1.0 + narrow)
            if high > low:
                extra = min(batch, remaining)
                if extra > 0:
                    X_extra = _lhs_samples(extra, dims, low, high, args.seed + it + 1000)
                    y_extra = np.zeros(extra, dtype=int)
                    for i in range(extra):
                        y_extra[i] = _apply_sample(doc, surface, specs, base_values, base_sketch_values, X_extra[i])
                        if (i + 1) % args.log_every == 0 or i + 1 == extra:
                            elapsed = time.perf_counter() - start_time
                            print(
                                f"[iter {it+1} fallback] {i+1}/{extra} evals, "
                                f"safe={int(y_extra[:i+1].sum())}, "
                                f"remaining={max(0, remaining - (i+1))}, "
                                f"elapsed={elapsed:.1f}s"
                            )
                    X = np.vstack([X, X_extra])
                    y = np.concatenate([y, y_extra])
                    remaining -= extra

    # Compute per-dimension min/max from safe samples
    safe_mask = y == 1
    mins: List[Optional[float]] = [None] * dims
    maxs: List[Optional[float]] = [None] * dims
    if np.any(safe_mask):
        safe_ratios = X[safe_mask]
        for i, spec in enumerate(specs):
            vals = safe_ratios[:, i] * base_values[i]
            mins[i] = float(np.min(vals))
            maxs[i] = float(np.max(vals))

    out_csv = args.out_csv or args.csv
    _append_ranges_to_csv(args.csv, out_csv, specs, mins, maxs, args.min_col, args.max_col)

    temp_out = Path("temp") / "proto3-hybrid_samples.csv"
    temp_out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(X, columns=[s.name for s in specs])
    df["safe"] = y
    df.to_csv(temp_out, index=False)

    _quit_freecad(doc)


if __name__ == "__main__":
    main()
