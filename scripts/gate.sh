#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

for script in scripts/*.sh; do
  bash -n "$script"
done

grep -Fq 'deepseek-ai/DeepSeek-V4-Flash-0731' patches/0731-only.patch \
  || fail "0731 model lock is absent from the runtime patch"
grep -Fq '9e165c30e2704aec5d9d593cce3eebd58bbef1cb' patches/0731-only.patch \
  || fail "checkpoint revision lock is absent"
grep -Fq 'DSPARK_MAX_INFLIGHT_PREFILLS=2' .env.example \
  || fail "qualified prefill profile is absent"

if grep -Eiq '^(ABLITERATED|DSPARK_MODEL|DSPARK_MODEL_ABLITERATED|DSPARK_MODEL_OFFICIAL)=' .env.example; then
  fail ".env.example exposes a checkpoint selector"
fi

# Known private deployment identifiers must never enter the public recipe.
if rg -n --hidden \
  -g '!.git/**' -g '!.runtime/**' -g '!scripts/gate.sh' \
  '192\.168\.1\.(140|152)|/home/alex|alex@192\.168|spark-1c24' .; then
  fail "private deployment identifier found"
fi

# Reject obvious committed credentials without printing their values.
if rg -l --hidden -g '!.git/**' -g '!.runtime/**' \
  '(ghp|gho|github_pat|hf)_[A-Za-z0-9_=-]{20,}|sk-[A-Za-z0-9_-]{20,}' . \
  >/dev/null; then
  fail "possible credential found"
fi

if [ "${CHECK_RUNTIME:-0}" = "1" ]; then
  ./scripts/bootstrap.sh
  test "$(git -C .runtime rev-parse HEAD)" = \
    'c444d7032957f5a5437261d5366fd06b27a01760' \
    || fail "runtime commit differs from the qualified pin"
  grep -Fq 'exec /usr/local/bin/vllm serve deepseek-ai/DeepSeek-V4-Flash-0731' \
    .runtime/docker-compose.dspark.yml \
    || fail "patched runtime is not locked to 0731"
  if grep -Fq 'hotfix-dsv4-vision-exp.py:ro' .runtime/docker-compose.dspark.yml; then
    fail "Vision-Exp hotfix remains mounted in the active runtime"
  fi
fi

echo "Recipe validation passed."
