#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=scripts/lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
require_env

set -a
# shellcheck source=/dev/null
source "$RECIPE_ROOT/.env"
set +a

: "${WORKER_HOST:?WORKER_HOST must be set in .env}"
project_name="${PROJECT_NAME:-deepseek-v4-flash}"
project_filter="label=com.docker.compose.project=$project_name"
service_filter="label=com.docker.compose.service=vllm-dspark"

if docker ps -aq --filter "$project_filter" --filter "$service_filter" | grep -q .; then
  echo "Head rank already exists; stale-worker recovery is not needed."
  exit 0
fi

worker_ready=0
for _attempt in $(seq 1 30); do
  if ssh -o BatchMode=yes -o ConnectTimeout=3 "$WORKER_HOST" true >/dev/null 2>&1; then
    worker_ready=1
    break
  fi
  sleep 2
done

if [ "$worker_ready" != 1 ]; then
  echo "Worker did not become reachable during stale-rank recovery." >&2
  exit 1
fi

worker_ids="$(ssh -o BatchMode=yes -o ConnectTimeout=5 "$WORKER_HOST" \
  "docker ps -aq --filter '$project_filter' --filter '$service_filter'")"

if [ -z "$worker_ids" ]; then
  echo "No stale worker rank found."
  exit 0
fi

echo "Removing stale worker rank before paired launch."
ssh -o BatchMode=yes -o ConnectTimeout=5 "$WORKER_HOST" \
  "docker rm -f $worker_ids >/dev/null"
