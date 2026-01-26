---
description: How to run Python scripts in this project using conda b123d environment
---

# Python Execution in cad_automaton Project

// turbo-all

This project requires the `b123d` conda environment for all Python execution.

## Activating the Environment

Always use the following command pattern to run Python:

```powershell
conda activate b123d && python <script>
```

## Running Tests

```powershell
conda activate b123d && python -m unittest test.test_fidelity -v
```

## Running Main Script

```powershell
conda activate b123d && python -m src.main <args>
```

## Important Notes

- Never use the system Python directly
- Always activate `b123d` before running any Python commands
- This applies to all scripts in `c:\github_repo\cad_automaton`
