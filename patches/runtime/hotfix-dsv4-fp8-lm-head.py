#!/usr/bin/env python3
"""FP8 (W8A16, Marlin) copy of the BF16 lm_head for decode-sized logits (opt-in).

DS4F's lm_head is BF16 129,280 x 4,096 (1.06 GB, ~530 MB per TP2 rank) and is
read every decode step for the target logits AND the DSpark draft base logits.
Decode is weight-streaming bound on GB10, so reading an FP8 copy (half the
bytes) through vLLM's FP8 Marlin kernel saves roughly 1 ms per read.

With ``DSPARK_FP8_LM_HEAD=1`` both lm_head apply sites in LogitsProcessor
(``_get_logits`` and ``get_top_tokens``) route rows <= ``DSPARK_FP8_LM_HEAD_MAX_M``
(default 64) through a per-output-channel FP8 copy built lazily on the first
eager call (never inside a CUDA graph capture; until built, the BF16 path is
used). Larger M and any other module keep the stock BF16 path. The BF16 weight
is kept (it may be shared with the drafter and the embedding).

Numerics: per-channel absmax/448 FP8 weights -> logits differ by ~FP8 rounding;
greedy argmax can flip on near-ties. Validate with greedy output agreement and
draft acceptance. Idea: RiNGSiDE glm53_dual_fp8_dense.py "head" family
(othexmr, Apache-2.0); independent minimal implementation.

Idempotent. Patches ``vllm/model_executor/layers/logits_processor.py``.
"""
from pathlib import Path
import sys

P = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/logits_processor.py")
MARK = "# [dspark-fp8-lm-head]"
if len(sys.argv) > 1 and sys.argv[1] == "--status":
    s = P.read_text() if P.is_file() else ""
    print("dspark fp8 lm_head                 :", "APPLIED" if MARK in s else "NOT APPLIED")
    raise SystemExit(0)
src = P.read_text()
if MARK in src:
    print(f"[dspark-fp8-lm-head] already applied to {P}")
    raise SystemExit(0)

OLD = "        logits = lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)\n"
NEW = ("        logits = _dspark_lm_head_apply(lm_head, hidden_states, embedding_bias)"
       "  # [dspark-fp8-lm-head]\n")
assert src.count(OLD) == 2, f"dspark-fp8-lm-head: expected 2 apply sites, found {src.count(OLD)}"
ANCHOR = "# --8<-- [start:logits_processor]\n"
assert src.count(ANCHOR) == 1, "dspark-fp8-lm-head: class anchor not found"

HELPER = '''# [dspark-fp8-lm-head] helpers (see patches/hotfix-dsv4-fp8-lm-head.py)
import os as _dspark_lmh_os

_DSPARK_LMH_ON = _dspark_lmh_os.environ.get("DSPARK_FP8_LM_HEAD", "0") == "1"
_DSPARK_LMH_MAX_M = int(_dspark_lmh_os.environ.get("DSPARK_FP8_LM_HEAD_MAX_M", "64"))


def _dspark_lm_head_build(lm_head):
    import torch as _t
    from torch.nn import Parameter
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
        prepare_fp8_layer_for_marlin,
    )
    w = lm_head.weight.data
    if w.dim() != 2 or w.dtype != _t.bfloat16:
        return None
    N, K = w.shape
    scale = (w.abs().amax(dim=1).float() / 448.0).clamp(min=1e-12)
    w8 = (w.float() / scale[:, None]).clamp(-448.0, 448.0).to(_t.float8_e4m3fn).contiguous()
    holder = _t.nn.Module()
    holder.input_size_per_partition = K
    holder.output_size_per_partition = N
    holder.orig_dtype = _t.bfloat16
    holder.params_dtype = _t.bfloat16
    holder.weight = Parameter(w8, requires_grad=False)
    holder.weight_scale = Parameter(scale.contiguous(), requires_grad=False)
    prepare_fp8_layer_for_marlin(holder, size_k_first=False)
    del w8
    _t.cuda.empty_cache()
    import logging
    logging.getLogger(__name__).info(
        "[dspark-fp8-lm-head] built FP8 Marlin lm_head copy N=%d K=%d max_m=%d", N, K, _DSPARK_LMH_MAX_M)
    return (holder.weight, holder.weight_scale, holder.workspace, N, K)


def _dspark_lm_head_apply(lm_head, hidden_states, embedding_bias):
    if _DSPARK_LMH_ON and embedding_bias is None:
        import torch as _t
        x2 = hidden_states.reshape(-1, hidden_states.shape[-1])
        state = lm_head.__dict__.get("_dspark_fp8_head", None)
        if state is None and not _t.cuda.is_current_stream_capturing():
            # Build on the first eager call whatever M is (the profile run), so
            # decode graphs captured later already see the FP8 copy.
            try:
                state = _dspark_lm_head_build(lm_head) or False
            except Exception:
                import logging
                logging.getLogger(__name__).exception("[dspark-fp8-lm-head] build failed; BF16 kept")
                state = False
            lm_head.__dict__["_dspark_fp8_head"] = state
        if state and 0 < x2.shape[0] <= _DSPARK_LMH_MAX_M:
            if True:
                from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
                    apply_fp8_marlin_linear,
                )
                wq, ws, wk, N, K = state
                out = apply_fp8_marlin_linear(x2, wq, ws, wk, N, K, None)
                return out.view(*hidden_states.shape[:-1], N)
    return lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)


'''
src = src.replace(OLD, NEW)
src = src.replace(ANCHOR, HELPER + ANCHOR, 1)
compile(src, str(P), "exec")
P.write_text(src)
print(f"[dspark-fp8-lm-head] patched {P}")
