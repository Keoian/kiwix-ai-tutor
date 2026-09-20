#!/usr/bin/env bash
# Launch the tutor app's HTTP server (FastAPI/uvicorn), binding to the host
# and port from the config's [app] table.
#
# Usage: scripts/run_app.sh [config.toml]   (default: config/dev.toml)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-"$REPO_ROOT/config/dev.toml"}"

cd "$REPO_ROOT"
exec python -m tutor.app.main --config "$CONFIG"
