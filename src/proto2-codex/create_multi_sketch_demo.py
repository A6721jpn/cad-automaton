from __future__ import annotations

from pathlib import Path


try:
    import FreeCAD  # type: ignore
    import Sketcher  # type: ignore
    import Part  # type: ignore
except Exception as exc:
    raise RuntimeError(
        "Run this script with FreeCAD's Python (e.g., FreeCADCmd)."
    ) from exc


def _add_line_sketch(doc, name: str, length: float, constraint_name: str) -> None:
    sketch = doc.addObject("Sketcher::SketchObject", name)
    sketch.MapMode = "Deactivated"

    # One line segment (X axis) with a distance constraint.
    sketch.addGeometry(
        Part.LineSegment(FreeCAD.Vector(0, 0, 0), FreeCAD.Vector(length, 0, 0)),
        False,
    )
    sketch.addConstraint(Sketcher.Constraint("Horizontal", 0))
    idx = sketch.addConstraint(Sketcher.Constraint("DistanceX", 0, 1, 0, 2, length))
    sketch.renameConstraint(idx, constraint_name)


def main() -> int:
    output_dir = Path(__file__).resolve().parents[2] / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    fcstd_path = output_dir / "multi_sketch_demo.FCStd"
    yaml_path = output_dir / "multi_sketch_demo.yaml"

    doc = FreeCAD.newDocument("MultiSketchDemo")
    _add_line_sketch(doc, "SketchA", 10.0, "L1")
    _add_line_sketch(doc, "SketchB", 20.0, "L2")
    doc.recompute()
    doc.saveAs(str(fcstd_path))
    FreeCAD.Console.PrintMessage("Saved demo file.\n")
    print(f"Wrote: {fcstd_path}")
    print(f"Wrote: {yaml_path}")

    yaml_path.write_text(
        "\n".join(
            [
                "schema_version: 1",
                "source:",
                f"  fcstd: {fcstd_path}",
                "sketch_order:",
                "  - SketchA",
                "  - SketchB",
                "constraints:",
                "  L1: 30.0",
                "  L2: 50.0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Wrote: {fcstd_path}")
    print(f"Wrote: {yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
