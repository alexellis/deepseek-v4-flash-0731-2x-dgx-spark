#!/usr/bin/env bash

RECIPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="${RUNTIME_DIR:-$RECIPE_ROOT/.runtime}"
RUNTIME_REPO="https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark.git"
RUNTIME_COMMIT="c444d7032957f5a5437261d5366fd06b27a01760"
RUNTIME_MARKER="$RUNTIME_DIR/.alex-0731-recipe"
export RECIPE_ROOT RUNTIME_DIR RUNTIME_REPO RUNTIME_COMMIT RUNTIME_MARKER

require_env() {
  if [ ! -f "$RECIPE_ROOT/.env" ]; then
    echo "Missing $RECIPE_ROOT/.env; copy .env.example and edit it first." >&2
    exit 2
  fi
}

ensure_runtime() {
  "$RECIPE_ROOT/scripts/bootstrap.sh"
}

sync_env() {
  require_env
  install -m 600 "$RECIPE_ROOT/.env" "$RUNTIME_DIR/.env.dspark"
}
