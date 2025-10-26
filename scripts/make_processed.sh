#!/usr/bin/env bash
set -euo pipefail
python "$(dirname "$0")/make_processed.py" "$@"
