#!/usr/bin/env bash
# Launch llama-server for the dev tutor config, using tutor/settings.py to
# resolve the binary path and argv (no flag-building logic duplicated here).
#
# Usage: scripts/serve.sh [config.toml]   (default: config/dev.toml)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-"$REPO_ROOT/config/dev.toml"}"

mapfile -t LINES < <(python -m tutor.settings --argv "$CONFIG")

BINARY="${LINES[0]}"
ARGV=("${LINES[@]:1}")

if [ ! -e "$BINARY" ]; then
    echo "server binary not found: $BINARY" >&2
    exit 1
fi

exec "$BINARY" "${ARGV[@]}"
