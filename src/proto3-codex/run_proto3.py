from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    src_dir = repo / "src" / "proto3-codex"
    script = src_dir / "apply_csv_to_surface.py"

    sys.path.insert(0, str(src_dir))

    # Pass through args after `--` to the target script.
    user_argv: list[str] = []
    if "--" in sys.argv:
        idx = sys.argv.index("--")
        user_argv = sys.argv[idx + 1 :]

    # FreeCAD sometimes injects an input file as the last arg (e.g. CSV).
    # Keep a possible leading positional arg (before --) if no -- args were passed.
    if not user_argv:
        tail_args = [arg for arg in sys.argv[1:] if arg != "--"]
        if tail_args:
            user_argv = tail_args

    globals_dict = {"__name__": "__main__", "__file__": str(script), "sys": sys}
    sys.argv = [str(script)] + user_argv
    code = script.read_text(encoding="utf-8")
    exec(compile(code, str(script), "exec"), globals_dict)


if __name__ == "__main__":
    main()
