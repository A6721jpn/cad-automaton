#!/usr/bin/env bash
set -euo pipefail

# Prefer git-bash if available on Windows.
if [ -x "/c/Program Files/Git/bin/bash.exe" ]; then
  "/c/Program Files/Git/bin/bash.exe" "$(realpath "$0")"
  exit 0
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FREECAD_CMD="/c/Program Files/FreeCAD 1.0/bin/freecadcmd.exe"

"$FREECAD_CMD" -c "$REPO_DIR/src/proto3-codex/run_proto3.py"
