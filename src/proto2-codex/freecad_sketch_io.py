from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _import_freecad():
    try:
        import FreeCAD  # type: ignore
        return FreeCAD
    except Exception as exc:
        import os
        import sys
        from pathlib import Path

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
                return FreeCAD
            except Exception:
                continue
        raise RuntimeError(
            "FreeCAD Python module not found. "
            "Run this script with FreeCAD's Python (e.g., FreeCADCmd) "
            "or inside the conda env that provides FreeCAD."
        ) from exc


def _parse_float(text: str) -> Optional[float]:
    if not text:
        return None
    token = ""
    for ch in text:
        if ch.isdigit() or ch in ".-+eE":
            token += ch
        elif token:
            break
    try:
        return float(token)
    except Exception:
        return None


def _constraint_value(constraint: Any) -> Optional[float]:
    if not hasattr(constraint, "Value"):
        return None
    value = getattr(constraint, "Value", None)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        pass
    if hasattr(value, "Value"):
        try:
            return float(value.Value)
        except Exception:
            pass
    if hasattr(value, "UserString"):
        return _parse_float(str(value.UserString))
    return _parse_float(str(value))


def open_document(path: Path):
    freecad = _import_freecad()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    return freecad.openDocument(str(path))


def find_sketch(doc: Any, sketch_hint: Optional[str] = None):
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


def list_constraints(sketch: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for index, constraint in enumerate(getattr(sketch, "Constraints", [])):
        name = getattr(constraint, "Name", "") or ""
        constraint_type = getattr(constraint, "Type", "") or ""
        value = _constraint_value(constraint)
        out.append(
            {
                "index": index,
                "name": name,
                "type": constraint_type,
                "value": value,
                "is_dimensional": value is not None,
            }
        )
    return out


def build_constraint_map(
    constraints: List[Dict[str, Any]],
    include_unnamed: bool = False,
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    result: Dict[str, Dict[str, Any]] = {}
    ordered_keys: List[str] = []

    for info in constraints:
        if not info.get("is_dimensional", False):
            continue
        raw_name = str(info.get("name") or "").strip()
        if not raw_name and not include_unnamed:
            continue
        key = raw_name if raw_name else f"constraint_{info['index']}"
        if key in result:
            key = f"{key}__{info['index']}"
        result[key] = {
            "index": info["index"],
            "type": info.get("type", ""),
            "value": info.get("value"),
            "source_name": raw_name,
        }
        ordered_keys.append(key)

    return result, ordered_keys


def find_constraint_index(sketch: Any, name: str) -> Optional[int]:
    for index, constraint in enumerate(getattr(sketch, "Constraints", [])):
        if getattr(constraint, "Name", "") == name:
            return index
    return None


def find_constraint_in_sketches(doc: Any, sketch_order: List[str], name: str):
    if not sketch_order:
        raise ValueError("sketch_order must include at least one sketch name")
    for sketch_name in sketch_order:
        sketch = find_sketch(doc, sketch_name)
        index = find_constraint_index(sketch, name)
        if index is not None:
            return sketch, index
    return None, None


def set_constraint_value(sketch: Any, index: int, value: float) -> None:
    freecad = _import_freecad()
    quantity = None
    try:
        quantity = freecad.Units.Quantity(value)
    except Exception:
        quantity = value
    sketch.setDatum(index, quantity)


def save_document(doc: Any, output_path: Path) -> None:
    output_path = Path(output_path)
    doc.saveAs(str(output_path))
