#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=scripts/lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

if [ -f "$RUNTIME_MARKER" ] \
  && [ "$(<"$RUNTIME_MARKER")" = "$RUNTIME_STAMP" ]; then
  exit 0
fi

# A runtime this recipe built from the same upstream commit but an older patch
# or overlay set is regenerated; the previous copy is kept alongside.
if [ -f "$RUNTIME_MARKER" ] \
  && [ "$(cut -d' ' -f1 "$RUNTIME_MARKER")" = "$RUNTIME_COMMIT" ]; then
  previous="$RUNTIME_DIR.prev-$(date +%Y%m%d%H%M%S)"
  mv "$RUNTIME_DIR" "$previous"
  echo "Recipe overlays changed; previous runtime moved to $previous"
fi

if [ -e "$RUNTIME_DIR" ]; then
  echo "$RUNTIME_DIR exists but is not this recipe's pinned runtime." >&2
  echo "Move it aside, inspect it, or remove it explicitly, then retry." >&2
  exit 2
fi

tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/ds4f-0731-bootstrap.XXXXXX")"
cleanup() {
  rm -rf -- "$tmp_dir"
}
trap cleanup EXIT

git clone --filter=blob:none --no-checkout "$RUNTIME_REPO" "$tmp_dir/runtime"
git -C "$tmp_dir/runtime" checkout --detach "$RUNTIME_COMMIT"
git -C "$tmp_dir/runtime" apply --check "$RECIPE_ROOT/patches/0731-only.patch"
git -C "$tmp_dir/runtime" apply "$RECIPE_ROOT/patches/0731-only.patch"
# Runtime overlays: new opt-in hotfixes, plus the admission-aware adaptive
# chunk (v2) replacing the upstream v1 patcher of the same name.
install -m 644 "$RECIPE_ROOT"/patches/runtime/*.py "$tmp_dir/runtime/patches/"
printf '%s\n' "$RUNTIME_STAMP" > "$tmp_dir/runtime/.alex-0731-recipe"

mv "$tmp_dir/runtime" "$RUNTIME_DIR"
echo "Prepared pinned 0731 runtime at $RUNTIME_DIR"
