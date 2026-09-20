#!/usr/bin/env bash
# Launch a SECOND llama-server, started with --embedding, for the dense
# sidecar's embedding model. Runs ALONGSIDE (not instead of) the chat
# server started by serve.sh -- two separate processes on two separate
# ports, both read from the same config file's [runtime]/[embedding]
# tables. Never hard-codes a binary or model path; both come from config
# via tutor/settings.py --argv-embedding.
#
# Usage: scripts/serve_embed.sh [config.toml]   (default: config/dev.toml)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-"$REPO_ROOT/config/dev.toml"}"

mapfile -t LINES < <(python -m tutor.settings --argv-embedding "$CONFIG")

BINARY="${LINES[0]}"
ARGV=("${LINES[@]:1}")

if [ ! -e "$BINARY" ]; then
    echo "server binary not found: $BINARY" >&2
    exit 1
fi

exec "$BINARY" "${ARGV[@]}"
