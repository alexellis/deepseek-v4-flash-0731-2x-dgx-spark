#!/usr/bin/env python3
"""DSA prefill indexer: split each chunk's QUERY ROWS across the TP ranks (opt-in).

The indexer is replicated across TP ranks, so in prefill every rank computes the
same FP8/FP4 MQA logits and the same per-row top-k for every query row of a
chunk. With ``DSPARK_INDEXER_ROW_SPLIT=1``, a chunk with at least
``DSPARK_INDEXER_ROW_SPLIT_MIN_ROWS`` query rows and at least
``DSPARK_INDEXER_ROW_SPLIT_MIN_KEYS`` gathered compressed keys is cut into
contiguous, tile-aligned row ranges, one per rank. Each rank runs the UNCHANGED
kernels (``fp8_fp4_mqa_logits`` + ``top_k_per_row_prefill``) on its rows only,
against the full gathered K, then the ranks all-gather the int32 top-k ids.
No float collective, no merge kernel.

Exactness: a row's logits depend only on that row's query, weights and key
bounds; the top-k kernel works row by row on that row's scores and bounds; the
per-row bounds (cu_seqlen_ks/ke) are sliced, never rebased. The split decision
uses chunk metadata that is identical on every rank, and every rank joins every
gather (a rank without rows sends -1 rows), so the collective cannot
desynchronise.

Differs from hotfix-dsv4-sp-indexer-prefill.py, which splits KEYS and must
merge (score, id) float candidates with a CuTe-DSL top-k (rejected 2026-09-27).
Idea: RiNGSiDE glm53_indexer_rowsplit.py (othexmr, Apache-2.0), after rhys101's
SG18 prefill split; this is an independent implementation for vLLM's
sparse_attn_indexer.

Scope: CUDA, DCP == 1, TP > 1, eager (never under graph capture). Anything
else keeps the stock path byte for byte. Idempotent.
Patches ``vllm/model_executor/layers/sparse_attn_indexer.py``.
"""
from pathlib import Path
import sys

P = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer.py")
MARK = "# [dspark-indexer-row-split]"
if len(sys.argv) > 1 and sys.argv[1] == "--status":
    s = P.read_text() if P.is_file() else ""
    print("dspark indexer row split           :", "APPLIED" if MARK in s else "NOT APPLIED")
    raise SystemExit(0)
src = P.read_text()
if MARK in src:
    print(f"[dspark-indexer-row-split] already applied to {P}")
    raise SystemExit(0)
assert "[dspark-sp-indexer]" not in src, "row split and the SP (key split) indexer are mutually exclusive"

CALL_OLD = """                else:
                    logits = fp8_fp4_mqa_logits(
                        (q_slice_cast, q_scale_slice),
                        (k_quant_cast, k_scale_cast),
                        weights[chunk.token_start : chunk.token_end],
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                        clean_logits=False,
                    )
                num_rows = logits.shape[0]
                ops.top_k_per_row_prefill(
                    logits,
                    cu_seqlen_ks,
                    cu_seqlen_ke,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )
"""
CALL_NEW = """                elif _dspark_row_split_applies(
                    q_slice_cast.shape[0], chunk.local_total_seq_lens, dcp_world_size
                ):
                    # [dspark-indexer-row-split] each rank scores its own rows.
                    logits = _dspark_row_split_topk(
                        q_slice_cast,
                        q_scale_slice,
                        k_quant_cast,
                        k_scale_cast,
                        weights[chunk.token_start : chunk.token_end],
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                        topk_indices,
                        topk_tokens,
                    )
                else:
                    logits = fp8_fp4_mqa_logits(
                        (q_slice_cast, q_scale_slice),
                        (k_quant_cast, k_scale_cast),
                        weights[chunk.token_start : chunk.token_end],
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                        clean_logits=False,
                    )
                    num_rows = logits.shape[0]
                    ops.top_k_per_row_prefill(
                        logits,
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                        topk_indices,
                        num_rows,
                        logits.stride(0),
                        logits.stride(1),
                        topk_tokens,
                    )
"""
assert src.count(CALL_OLD) == 1, "dspark-indexer-row-split: logits/top-k anchor not found; refusing to patch"

HELPER_ANCHOR = "@eager_break_during_capture\ndef sparse_attn_indexer(\n"
assert src.count(HELPER_ANCHOR) == 1, "dspark-indexer-row-split: function anchor not found"
HELPER = '''# [dspark-indexer-row-split] helpers (see patches/hotfix-dsv4-indexer-row-split.py)
import os as _dspark_rs_os

_DSPARK_RS_ON = _dspark_rs_os.environ.get("DSPARK_INDEXER_ROW_SPLIT", "0") == "1"
_DSPARK_RS_MIN_ROWS = int(_dspark_rs_os.environ.get("DSPARK_INDEXER_ROW_SPLIT_MIN_ROWS", "1024"))
_DSPARK_RS_MIN_KEYS = int(_dspark_rs_os.environ.get("DSPARK_INDEXER_ROW_SPLIT_MIN_KEYS", "4096"))
_DSPARK_RS_TILE = max(1, int(_dspark_rs_os.environ.get("DSPARK_INDEXER_ROW_SPLIT_TILE", "64")))
_DSPARK_RS_COUNTS = {"split": 0, "logged": False}


def _dspark_row_split_applies(rows, keys, dcp_world_size):
    if not _DSPARK_RS_ON or dcp_world_size > 1:
        return False
    if rows < _DSPARK_RS_MIN_ROWS or keys < _DSPARK_RS_MIN_KEYS:
        return False
    if current_platform.is_xpu() or torch.cuda.is_current_stream_capturing():
        return False
    from vllm.distributed import get_tensor_model_parallel_world_size
    return get_tensor_model_parallel_world_size() > 1


def _dspark_row_split_topk(q, q_scale, k_quant, k_scale, weights, ks, ke, topk_indices, topk_tokens):
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )
    from vllm.distributed.parallel_state import get_tp_group

    tp = get_tensor_model_parallel_world_size()
    rank = get_tensor_model_parallel_rank()
    rows = q.shape[0]
    tile = _DSPARK_RS_TILE
    per = -(-rows // tp)
    per = -(-per // tile) * tile
    lo = min(rank * per, rows)
    hi = min(lo + per, rows)
    local = torch.full((per, topk_tokens), -1, dtype=topk_indices.dtype, device=topk_indices.device)
    logits = None
    if hi > lo:
        logits = fp8_fp4_mqa_logits(
            (q[lo:hi], q_scale[lo:hi] if q_scale is not None else None),
            (k_quant, k_scale),
            weights[lo:hi],
            ks[lo:hi],
            ke[lo:hi],
            clean_logits=False,
        )
        n = hi - lo
        ops.top_k_per_row_prefill(
            logits,
            ks[lo:hi],
            ke[lo:hi],
            local[:n],
            n,
            logits.stride(0),
            logits.stride(1),
            topk_tokens,
        )
    gathered = get_tp_group().all_gather(local, dim=0)
    topk_indices.copy_(gathered[:rows])
    _DSPARK_RS_COUNTS["split"] += 1
    if not _DSPARK_RS_COUNTS["logged"]:
        _DSPARK_RS_COUNTS["logged"] = True
        import logging
        logging.getLogger(__name__).info(
            "[dspark-indexer-row-split] engaged: rows=%d keys=%d tp=%d per_rank=%d topk=%d",
            rows, k_quant.shape[0], tp, per, topk_tokens)
    if logits is None:
        logits = q.new_empty((0, 0), dtype=torch.float32)
    return logits


'''
src = src.replace(CALL_OLD, CALL_NEW, 1)
src = src.replace(HELPER_ANCHOR, HELPER + HELPER_ANCHOR, 1)
compile(src, str(P), "exec")
P.write_text(src)
print(f"[dspark-indexer-row-split] patched {P}")
