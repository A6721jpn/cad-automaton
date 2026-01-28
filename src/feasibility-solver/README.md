# feasibility-solver

Multi-variable feasibility range estimator for FreeCAD sketch constraints.

## Usage

Run inside the FreeCAD Python environment (same as proto3):

```bash
python src/feasibility-solver/feasibility_solver.py ^
  ref/TH1-ref.FCStd ^
  input/dims_deg.csv ^
  --template temp/constraints.json ^
  --samples 5000
```

### Options
- `--out-csv`: write results to a different CSV (default: overwrite input)
- `--template`: proto2 constraint template (YAML/JSON) with correct indices
- `--template` prefers indices that are `setDatum`-compatible; fallback is name-only
- `--sketch`: target Sketch Name/Label (default: first sketch)
- `--surface-name` / `--surface-label`: surface validation target
- `--ratio-min` / `--ratio-max`: sampling range around base (default 0.7..1.3)
- `--estimate-only`: estimate runtime and exit
- `--min-col` / `--max-col`: output column names (default `feasible_min` / `feasible_max`)

## Notes
- Angle values in CSV are degrees; internal solver converts to radians.
- Distances must be positive; angles must be in (0, 180).
