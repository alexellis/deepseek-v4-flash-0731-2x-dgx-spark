#!/usr/bin/env python3
"""FP8 (W8A16, Marlin) copies of DS4F's BF16 aux projections for decode-sized M (opt-in).

Profiled 2026-09-27 (2x GB10, TP2): the checkpoint keeps the compressor
``fused_wkv_wgate`` (main + indexer) and the indexer ``weights_proj`` in BF16,
replicated on every rank (K=4096, N=2048/1024/256/64). In decode they run on
the SM80 WMMA cuBLAS kernel at 50-160 GB/s, ~9 ms of GPU time per step (~13%
of decode kernel time, partly overlapped on aux streams).

With ``DSPARK_FP8_AUX_DENSE=1`` the three call sites in
``DeepseekV4Attention`` route rows <= ``DSPARK_FP8_AUX_DENSE_MAX_M`` (default 64)
through a per-output-channel FP8 copy (half the bytes). The copy is built on the
first EAGER call of each module whatever its M (the profile run), never under
graph capture; a module whose build fails keeps BF16. Larger M (prefill) keeps
the exact stock BF16 path. The two compressor sites asked for fp32 output: the
FP8 path returns BF16 upcast to fp32 (extra rounding on kv scores), so validate
long-context retrieval.

Idea: RiNGSiDE glm53_dual_fp8_dense.py (othexmr, Apache-2.0). Independent
implementation. Idempotent. Patches ``vllm/models/deepseek_v4/attention.py``.
"""
from pathlib import Path
import sys

P = Path("/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/attention.py")
MARK = "# [dspark-fp8-aux-dense]"
if len(sys.argv) > 1 and sys.argv[1] == "--status":
    s = P.read_text() if P.is_file() else ""
    print("dspark fp8 aux dense               :", "APPLIED" if MARK in s else "NOT APPLIED")
    raise SystemExit(0)
src = P.read_text()
if MARK in src:
    print(f"[dspark-fp8-aux-dense] already applied to {P}")
    raise SystemExit(0)

EDITS = [
    ("""                return torch.mm(
                    hidden_states,
                    compressor.fused_wkv_wgate.weight.T,
                    out_dtype=torch.float32,
                )
""", """                return _dspark_aux_mm(  # [dspark-fp8-aux-dense]
                    hidden_states, compressor.fused_wkv_wgate, torch.float32
                )
"""),
    ("""                weights, _ = indexer.weights_proj(hidden_states)
                return weights
""", """                _w = _dspark_aux_fp8(hidden_states, indexer.weights_proj)  # [dspark-fp8-aux-dense]
                if _w is not None:
                    return _w
                weights, _ = indexer.weights_proj(hidden_states)
                return weights
"""),
    ("""                return torch.mm(
                    hidden_states,
                    indexer.compressor.fused_wkv_wgate.weight.T,
                    out_dtype=torch.float32,
                )
""", """                return _dspark_aux_mm(  # [dspark-fp8-aux-dense]
                    hidden_states, indexer.compressor.fused_wkv_wgate, torch.float32
                )
"""),
]
for old, _ in EDITS:
    assert src.count(old) == 1, f"dspark-fp8-aux-dense: anchor not found once:\n{old}"

HELPER = '''
# [dspark-fp8-aux-dense] helpers (see patches/hotfix-dsv4-fp8-aux-dense.py)
import os as _dspark_aux_os

_DSPARK_AUX_ON = _dspark_aux_os.environ.get("DSPARK_FP8_AUX_DENSE", "0") == "1"
_DSPARK_AUX_MAX_M = int(_dspark_aux_os.environ.get("DSPARK_FP8_AUX_DENSE_MAX_M", "64"))
# Marlin has a ~25 us floor: measured DRAM-streaming 2026-09-27, N=2048 1.72x, N=1024 1.4-1.7x, N=256 0.5x.
_DSPARK_AUX_MIN_N = int(_dspark_aux_os.environ.get("DSPARK_FP8_AUX_DENSE_MIN_N", "1024"))
_DSPARK_AUX_LOGGED = set()


def _dspark_aux_build(module):
    from torch.nn import Parameter
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
        prepare_fp8_layer_for_marlin,
    )
    w = module.weight.data
    if w.dim() != 2 or w.dtype != torch.bfloat16:
        return False
    N, K = w.shape
    if N < _DSPARK_AUX_MIN_N:
        return False
    scale = (w.abs().amax(dim=1).float() / 448.0).clamp(min=1e-12)
    w8 = (w.float() / scale[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).contiguous()
    holder = torch.nn.Module()
    holder.input_size_per_partition = K
    holder.output_size_per_partition = N
    holder.orig_dtype = torch.bfloat16
    holder.params_dtype = torch.bfloat16
    holder.weight = Parameter(w8, requires_grad=False)
    holder.weight_scale = Parameter(scale.contiguous(), requires_grad=False)
    prepare_fp8_layer_for_marlin(holder, size_k_first=False)
    key = (N, K)
    if key not in _DSPARK_AUX_LOGGED:
        _DSPARK_AUX_LOGGED.add(key)
        import logging
        logging.getLogger(__name__).info(
            "[dspark-fp8-aux-dense] FP8 Marlin copy N=%d K=%d (max_m=%d)", N, K, _DSPARK_AUX_MAX_M)
    return (holder.weight, holder.weight_scale, holder.workspace, N, K)


def _dspark_aux_fp8(x, module):
    """FP8 Marlin output (BF16) for small M, else None (caller keeps stock path)."""
    if not _DSPARK_AUX_ON:
        return None
    state = module.__dict__.get("_dspark_fp8_aux", None)
    if state is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        try:
            state = _dspark_aux_build(module)
        except Exception:
            import logging
            logging.getLogger(__name__).exception("[dspark-fp8-aux-dense] build failed; BF16 kept")
            state = False
        module.__dict__["_dspark_fp8_aux"] = state
    if not state or not (0 < x.shape[0] <= _DSPARK_AUX_MAX_M):
        return None
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
        apply_fp8_marlin_linear,
    )
    wq, ws, wk, N, K = state
    return apply_fp8_marlin_linear(x, wq, ws, wk, N, K, None)


def _dspark_aux_mm(x, module, out_dtype):
    y = _dspark_aux_fp8(x, module)
    if y is not None:
        return y.to(out_dtype)
    return torch.mm(x, module.weight.T, out_dtype=out_dtype)

'''
ANCHOR = "\nclass "
idx = src.find(ANCHOR)
assert idx > 0, "dspark-fp8-aux-dense: no class found for helper placement"
for old, new in EDITS:
    src = src.replace(old, new, 1)
idx = src.find(ANCHOR)
src = src[:idx] + "\n" + HELPER + src[idx:]
compile(src, str(P), "exec")
P.write_text(src)
print(f"[dspark-fp8-aux-dense] patched {P}")
