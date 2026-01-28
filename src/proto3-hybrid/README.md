# proto3-hybrid

Hybrid feasibility explorer:
- initial LHS sweep to map global safe/fail shape
- active learning loop focusing on boundary (P(safe) ≈ 0.5)

## Quick start
```
# inside fcad-codex
pip install -U numpy pandas scipy scikit-learn

# estimate-only
python src/proto3-hybrid/hybrid_solver.py ^
  ref/TH1-ref.FCStd ^
  input/dims_deg.csv ^
  --template temp/constraints.json ^
  --samples 5000 ^
  --estimate-only
```

## Output
- Appends `feasible_min` / `feasible_max` to the CSV.
- Writes `proto3-hybrid_samples.csv` in `temp/` for inspection.
