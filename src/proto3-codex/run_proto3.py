from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    script = repo / "src" / "proto3-codex" / "apply_csv_to_surface.py"

    globals_dict = {"__name__": "__main__", "__file__": str(script), "sys": sys}
    code = script.read_text(encoding="utf-8")
    exec(compile(code, str(script), "exec"), globals_dict)


if __name__ == "__main__":
    main()
