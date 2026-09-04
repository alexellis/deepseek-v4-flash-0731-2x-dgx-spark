#!/usr/bin/env bash
set -uo pipefail

# shellcheck source=scripts/lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
require_env

set -a
# shellcheck source=/dev/null
source "$RECIPE_ROOT/.env"
set +a

: "${WORKER_HOST:?WORKER_HOST must be set in .env}"
probe_url="${PROBE_URL:-http://127.0.0.1:${VLLM_PORT:-8888}/v1/chat/completions}"
model="deepseek-v4-flash-0731"
interval="${WATCHDOG_INTERVAL:-30}"
fail_threshold="${WATCHDOG_FAIL_THRESHOLD:-2}"
probe_timeout="${WATCHDOG_PROBE_TIMEOUT:-180}"
crash_dir="${WATCHDOG_CRASH_DIR:-${HOME}/ds4f-crashes}"

log() { echo "$(date -u +%FT%TZ) $*"; }

container() {
  docker ps --format '{{.Names}}' | grep -i vllm | head -1
}

probe() {
  local http
  http="$(curl -sS -m "$probe_timeout" -o /dev/null -w '%{http_code}' \
    "$probe_url" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$model\",\"max_tokens\":1,\"messages\":[{\"role\":\"user\",\"content\":\"Reply OK\"}]}" \
    2>/dev/null || true)"
  [ "$http" = "200" ]
}

capture() {
  local dir current
  dir="$crash_dir/$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$dir"
  current="$(container)"
  {
    echo "== head memory =="
    free -g
    echo "== head kernel memory messages =="
    journalctl -k --since '-10 min' 2>/dev/null | grep -Ei 'NVRM|OOM|Xid' | tail -50 || true
    echo "== head metrics =="
    curl -fsS -m 8 "http://127.0.0.1:${VLLM_PORT:-8888}/metrics" 2>/dev/null \
      | grep -E 'num_requests_(running|waiting)' | grep -v '^#' || true
    docker inspect "$current" --format \
      'OOMKilled={{.State.OOMKilled}} Status={{.State.Status}} Restarts={{.RestartCount}}' 2>/dev/null || true
  } > "$dir/head.txt" 2>&1
  docker logs --tail 500 "$current" > "$dir/head-vllm.log" 2>&1 || true
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$WORKER_HOST" '
    free -g
    journalctl -k --since "-10 min" 2>/dev/null | grep -Ei "NVRM|OOM|Xid" | tail -50 || true
    current=$(docker ps --format "{{.Names}}" | grep -i vllm | head -1)
    docker logs --tail 500 "$current" 2>&1 || true
  ' > "$dir/worker.txt" 2>&1 || true
  log "Forensics saved to $dir"
}

restart_pair() {
  log "Restarting both ranks"
  "$RECIPE_ROOT/scripts/stop.sh" >> "$crash_dir/restart.log" 2>&1 || true
  "$RECIPE_ROOT/scripts/start.sh" >> "$crash_dir/restart.log" 2>&1 || return 1
  for _attempt in $(seq 1 40); do
    if probe; then
      log "Engine is healthy after restart"
      return 0
    fi
    sleep 15
  done
  log "Engine did not recover within ten minutes"
  return 1
}

mkdir -p "$crash_dir"
log "Watchdog started: interval=${interval}s threshold=${fail_threshold} timeout=${probe_timeout}s"
fails=0
while true; do
  if probe; then
    fails=0
  else
    fails=$((fails + 1))
    log "Completion probe failed ($fails/$fail_threshold)"
    if [ "$fails" -ge "$fail_threshold" ]; then
      capture
      restart_pair || true
      fails=0
    fi
  fi
  sleep "$interval"
done
