#!/bin/bash
# Compatibility entrypoint. The old host-local sequential dispatcher is no longer
# compatible with the Docker/RAM-assets runtime layout.
set -Eeuo pipefail

WORK_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "dispatch_sequential.sh is deprecated for this Docker-based experiment layout."
echo "Use the resumable 8-GPU Docker dispatcher instead:"
echo "  bash dispatch_all.sh"
echo ""
echo "Forwarding to dispatch_all.sh now."

exec bash "${WORK_DIR}/dispatch_all.sh"
