#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

python "$PROJECT_DIR/TriangularJ1J2PostProcessRunner.py" "$@"
