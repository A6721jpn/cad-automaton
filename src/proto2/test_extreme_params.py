"""Test with extreme parameters to verify robustness."""
import sys
import os

# Setup FreeCAD paths
CONDA_PREFIX = os.environ.get(
    "CONDA_PREFIX",
    r"C:\Users\aokuni\AppData\Local\miniforge3\envs\fcad"
)
freecad_bin = os.path.join(CONDA_PREFIX, "Library", "bin")
os.environ["PATH"] = freecad_bin + os.pathsep + os.environ.get("PATH", "")
if freecad_bin not in sys.path:
    sys.path.insert(0, freecad_bin)

import FreeCAD
from sketch_builder import ProfileParams, build_profile_sketch, sketch_to_face, export_step


def test_params(name: str, params: ProfileParams) -> bool:
    """Test a parameter set."""
    print(f"\n{'='*60}")
    print(f"Test: {name}")
    print(f"  L9={params.L9}, L10={params.L10}, R1={params.R1}, R2={params.R2}")
    print(f"{'='*60}")

    try:
        doc = FreeCAD.newDocument(f"Test_{name}")
        sketch = build_profile_sketch(doc, params, f"Profile_{name}")
        doc.recompute()
        face = sketch_to_face(sketch)
        output_path = f"c:/github_repo/cad_automaton/output/test_{name}.step"
        export_step(face, output_path)
        print(f"  SUCCESS: area = {face.Area:.6f}")
        FreeCAD.closeDocument(doc.Name)
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def main():
    print("Testing FreeCAD Sketcher with various parameters...")

    results = []

    # Test 1: Default (reference) parameters
    results.append(("default", test_params("default", ProfileParams())))

    # Test 2: config.yaml values (previously failing)
    results.append(("config_yaml", test_params("config_yaml", ProfileParams(
        L9=0.698, L10=0.3, R1=0.25, R2=0.2
    ))))

    # Test 3: R1=0, R2=0 (no fillets)
    results.append(("no_fillets", test_params("no_fillets", ProfileParams(
        R1=0.0, R2=0.0
    ))))

    # Test 4: Large fillets
    results.append(("large_fillets", test_params("large_fillets", ProfileParams(
        R1=0.4, R2=0.4
    ))))

    # Test 5: Minimum L10
    results.append(("min_L10", test_params("min_L10", ProfileParams(
        L10=0.1
    ))))

    # Test 6: Large L10
    results.append(("large_L10", test_params("large_L10", ProfileParams(
        L10=2.0
    ))))

    # Test 7: Small L9
    results.append(("small_L9", test_params("small_L9", ProfileParams(
        L9=0.1
    ))))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    passed = sum(1 for _, r in results if r)
    failed = sum(1 for _, r in results if not r)
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {name}: {status}")
    print(f"\nTotal: {passed} passed, {failed} failed")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
