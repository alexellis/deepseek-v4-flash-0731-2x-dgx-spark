#!/usr/bin/env python3
"""Adaptive long-prefill chunk cap (v2): big chunks only when nobody is waiting.

``--long-prefill-token-threshold 1024`` (issue #27/#43) keeps mixed-load
decode from starving, but a cold c=1 prefill still pays the per-chunk fixed
cost (collectives, indexer/compressor setup, launches) once per 1024 tokens.

v1 raised the cap to ``max_num_batched_tokens`` whenever no running request
was decode-active. That starved WAITING requests: the long prefill's chunk
consumed the whole token budget, the waiting loop only runs while
``token_budget > 0``, so a short newcomer waited for the entire long prefill
(measured 2026-09-27: 50.0 s newcomer TTFT during a 128K cold prefill), and
two concurrent prefills were serialised FCFS instead of interleaved.

v2 cap, evaluated per request at both chunking sites:

* configured threshold (1024) if any running request is decode-active
  (unchanged mixed-load behaviour);
* configured threshold if another request is waiting and could be admitted
  this step (running below max_num_seqs, in-flight prefills below the #27
  cap), so the newcomer gets budget in the same step;
* configured threshold if any prefilling request is already deeper than
  ``DSPARK_ADAPTIVE_CHUNK_MAX_DEPTH`` computed tokens (default 131072, the
  depth big chunks were validated to; 0 disables the guard). On 2026-09-28 an
  engine died during a ~470K-token cold prefill with big chunks enabled;
  until deeper chunks are qualified, depth falls back to 1024;
* otherwise ``max_num_batched_tokens`` split evenly across the prefilling
  requests (running prefill chunks, plus the one being admitted), rounded
  down to a multiple of the threshold and never below it.

Idempotent. Patches ``vllm/v1/core/sched/scheduler.py``.
"""
from pathlib import Path
import sys

P = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py")
MARK = "# [dspark-adaptive-chunk-v2]"
V1_MARK = "# [dspark-adaptive-chunk] keep the 1024"
if len(sys.argv) > 1 and sys.argv[1] == "--status":
    status_src = P.read_text() if P.is_file() else ""
    print("dspark adaptive prefill chunk v2   :",
          "APPLIED" if MARK in status_src else "NOT APPLIED")
    raise SystemExit(0)
src = P.read_text()
if MARK in src:
    print(f"[dspark-adaptive-chunk-v2] already applied to {P}")
    raise SystemExit(0)
assert V1_MARK not in src, (
    "dspark-adaptive-chunk-v2: v1 is already applied; start from a stock scheduler")

RUNNING_OLD = (
    "            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:\n"
    "                num_new_tokens = self.scheduler_config.long_prefill_token_threshold\n"
)
WAITING_OLD = (
    "                    threshold = self.scheduler_config.long_prefill_token_threshold\n"
    "                    if 0 < threshold < num_new_tokens:\n"
    "                        num_new_tokens = threshold\n"
)
assert src.count(RUNNING_OLD) == 1, "dspark-adaptive-chunk-v2: running-loop anchor not found; refusing to patch"
assert src.count(WAITING_OLD) == 1, "dspark-adaptive-chunk-v2: waiting-loop anchor not found; refusing to patch"

RUNNING_NEW = (
    "            # [dspark-adaptive-chunk-v2] admission-aware, fair chunk cap.\n"
    "            _chunk_cap = _dspark_adaptive_chunk_cap(self, admitting=False)\n"
    "            if 0 < _chunk_cap < num_new_tokens:\n"
    "                num_new_tokens = _chunk_cap\n"
)
WAITING_NEW = (
    "                    # [dspark-adaptive-chunk-v2] same cap as the running loop.\n"
    "                    threshold = _dspark_adaptive_chunk_cap(self, admitting=True)\n"
    "                    if 0 < threshold < num_new_tokens:\n"
    "                        num_new_tokens = threshold\n"
)
HELPER = '''

# [dspark-adaptive-chunk-v2] helper (module level; called from schedule()).
import os as _dspark_ac_os
_DSPARK_ADAPTIVE_MAX_DEPTH = int(_dspark_ac_os.environ.get("DSPARK_ADAPTIVE_CHUNK_MAX_DEPTH", "131072"))


def _dspark_adaptive_chunk_cap(sched, admitting):
    """Chunk cap for one prefill; see hotfix-dsv4-adaptive-prefill-chunk.py."""
    cfg = sched.scheduler_config
    base = cfg.long_prefill_token_threshold
    if base <= 0:
        return base
    running = sched.running
    if any(not r.is_prefill_chunk for r in running):
        return base
    queued = len(getattr(sched, "waiting", ()) or ()) + len(
        getattr(sched, "skipped_waiting", ()) or ())
    if admitting:
        queued -= 1  # the request being admitted is not someone else waiting
    if queued > 0:
        inflight_cap = getattr(sched, "_dspark_max_inflight_prefills", 0)
        inflight = len(getattr(sched, "_inflight_prefills", ()) or ())
        if admitting:
            inflight += 1
        slots = len(running) + (1 if admitting else 0)
        if ((inflight_cap <= 0 or inflight < inflight_cap)
                and slots < sched.max_num_running_reqs):
            return base
    max_depth = _DSPARK_ADAPTIVE_MAX_DEPTH
    if max_depth > 0 and any(
            r.is_prefill_chunk and r.num_computed_tokens >= max_depth
            for r in running):
        return base
    prefilling = sum(1 for r in running if r.is_prefill_chunk)
    if admitting:
        prefilling += 1
    share = cfg.max_num_batched_tokens // max(1, prefilling)
    return max(base, (share // base) * base)
'''
src = src.replace(RUNNING_OLD, RUNNING_NEW, 1)
src = src.replace(WAITING_OLD, WAITING_NEW, 1)
src = src.rstrip("\n") + "\n" + HELPER
compile(src, "scheduler.py", "exec")
P.write_text(src)
print(f"[dspark-adaptive-chunk-v2] patched {P}")
