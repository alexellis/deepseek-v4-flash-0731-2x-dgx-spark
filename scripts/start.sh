#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=scripts/lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
ensure_runtime
sync_env

cd "$RUNTIME_DIR"
export DSPARK_TP3=0
exec ./start-deepseek-v4-flash-dspark.sh "$@"
