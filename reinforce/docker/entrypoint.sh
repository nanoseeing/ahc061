#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-/workspace/AHC061}"

exec "$@"
