#!/usr/bin/env bash
# Launch the tutor UI in kiosk (fullscreen, chromeless) mode on Linux.
#
# Tries chromium, then google-chrome, then firefox, in that order, against
# the given URL (defaults to the loopback tutor app).
set -euo pipefail

URL="${1:-http://127.0.0.1:8420/}"

if command -v chromium >/dev/null 2>&1; then
    exec chromium --kiosk "$URL"
elif command -v chromium-browser >/dev/null 2>&1; then
    exec chromium-browser --kiosk "$URL"
elif command -v google-chrome >/dev/null 2>&1; then
    exec google-chrome --kiosk "$URL"
elif command -v firefox >/dev/null 2>&1; then
    exec firefox --kiosk "$URL"
else
    echo "No supported browser (chromium, google-chrome, firefox) found." >&2
    exit 1
fi
