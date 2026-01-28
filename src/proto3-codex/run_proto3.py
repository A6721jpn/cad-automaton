from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    script = repo / "src" / "proto3-codex" / "apply_csv_to_surface.py"

    globals_dict = {"__name__": "__main__", "__file__": str(script), "sys": sys}
    try:
        code = script.read_text(encoding="utf-8")
        exec(compile(code, str(script), "exec"), globals_dict)
    except SystemExit:
        # The target script handles exits; fall through to hard-exit.
        pass
    except Exception as exc:
        try:
            print(f"proto3 run error: {exc}", file=sys.stderr)
        except Exception:
            pass
        os._exit(1)
    os._exit(0)


if __name__ == "__main__":
    main()
