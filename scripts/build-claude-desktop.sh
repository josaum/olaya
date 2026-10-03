#!/bin/sh
# Default build now produces the complete native runtime rather than the legacy lexical bundle.
set -eu
repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
exec "$repo/.venv/bin/python" "$repo/scripts/build-claude-desktop-complete.py"
