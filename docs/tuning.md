# Tuning and measurement

## Tune for agent traffic

Agent traffic alternates between large prompt ingestion and relatively short
generation bursts. Measure at least:

- cold prefill at 8K, 32K, and 128K;
- warm-prefix time to first token;
- natural prose decode;
- code-oriented decode;
- speculative acceptance;
- two and four staggered sessions;
- tool calls and multi-turn history.

Do not use counting alone. It is useful for measuring an upper bound on draft
acceptance, but it is not representative of prose or novel code.

## Knobs that matter

`LONG_PREFILL_TOKEN_THRESHOLD=1024` bounds each admitted prefill chunk. Larger
chunks can improve isolated prefill throughput but starve active decoders.

`DSPARK_MAX_INFLIGHT_PREFILLS=2` permits two partial prefills in flight. One is
safer under adversarial concurrency; two was the better balance in our live
small-team traffic. Re-test staggered requests if this changes.

`MAX_NUM_BATCHED_TOKENS=8192` is the total scheduler budget. Raising it is not
a free prefill win because decoding and prefilling share the step.

`MTP_NUM_TOKENS=5` matches the 0731 DSpark block. Longer drafts may help highly
predictable output and hurt prose. Compare acceptance and end-to-end latency,
not only emitted tokens per second.

`VLLM_USE_BREAKABLE_CUDAGRAPH=0` selects regular CUDA graphs. The automatic
breakable path was slower on the qualified pair.

`GPU_MEMORY_UTILIZATION_TEXT=0.835` leaves headroom for JIT compilation and
graphs. Raising it may increase KV capacity but makes cold boot and deep
context less reliable.

## Context and concurrency share memory

`MAX_MODEL_LEN` is a per-request ceiling, not a reservation for every active
sequence. Six ordinary sessions can coexist, but six simultaneous 1M-token
requests cannot. Watch the boot-time KV pool and actual live-token sum.

## Acceptance is workload-shaped

The drafter can accept most tokens in structured output and far fewer in
natural language. Always report prompt, temperature, output class, stream
count, TTFT, prefill rate, decode rate, and acceptance together. A single
headline tokens-per-second number conceals the important behaviour.

## Profile-driven speed-ups (2026-09-27)

The first torch-profiler traces of this lane (vLLM `--profiler-config`, see
`DSPARK_PROFILER_ARGS`) changed what we tuned. The model is not torch-compiled
here; it runs eagerly with CUDA graphs. At c=1:

- **Decode is bytes-bound.** The routed MoE (W4A16) and the FP8 block-scaled
  dense projections together take most of a ~63 ms step and run close to the
  GB10's ~220 GB/s. The BF16 `lm_head` (129,280 × 4,096, sharded) was read
  about three times per step (target logits, DSpark base logits, Markov head)
  at ~2.4 ms per read.
- **Prefill was network-bound in one place.** NCCL all-reduce took 17.5% of a
  cold 32K prefill, reaching only ~11 GB/s on a 200 Gb/s cable. MoE prefill
  already runs at roughly the BF16 tensor-core roofline.

Each change below is an opt-in overlay in `patches/runtime` with its own
`.env` switch. The profile in `.env.example` enables all of them except the FP8 compressor
copies (see the incident below).

| Change | Switch | Effect |
|---|---|---|
| Both PCIe functions of the cable | `NCCL_IB_HCA=…f0,roceP2p1s0f0`, `NCCL_CROSS_NIC=0` | 67 MB all-reduce 6.4 → 3.3 ms; prefill all-reduce 2.9 → 1.2 s per 32K |
| FP8 lm_head copy (Marlin W8A16, decode M ≤ 64) | `DSPARK_FP8_LM_HEAD=1` | 2.0–2.65× on that GEMM, ~1.2 ms saved per read |
| FP8 compressor projections (N ≥ 1024 only), **off by default** | `DSPARK_FP8_AUX_DENSE=1` | 1.4–1.7× on those GEMMs (DRAM-streaming); smaller N lose and stay BF16; ~1.5% decode |
| Query-row split of the DSA prefill indexer | `DSPARK_INDEXER_ROW_SPLIT=1` | exact top-k ids, int32 gather only; ~3% at 128K |
| Two-rank all-reduce as one exchange | `DSPARK_TP2_ALLREDUCE=p2p` | one network step instead of two for ≤ 512 KiB messages |
| Admission-aware adaptive prefill chunk (v2) | `DSPARK_ENABLE_ADAPTIVE_CHUNK=1` | full 8,192-token chunks only when nobody is decoding or waiting; −16% at 32K |

The FP8 copies are per-output-channel (absmax / 448). Logits are close to, not
bit-equal with, BF16. Greedy output is not reproducible run to run on this
stack even without them, so they were gated on functional checks instead:
exact long-context retrieval, tool calls, and RigMark's output gates.

The adaptive chunk (v2) replaces the upstream v1 patcher. v1 raised the chunk
whenever no request was decoding, so a long prefill used the whole token
budget and a newly arrived short request waited for all of it: we measured a
50 s time to first token behind a cold 128K prompt. v2 drops back to 1,024
tokens while another request is waiting to be admitted, and splits the budget
between concurrent prefills; the same newcomer now starts in 6.6 s.

### Use both PCIe functions of the cable

Each ConnectX port is exposed as two PCIe functions, `rocep1s0fN` and
`roceP2p1s0fN`, each limited to PCIe Gen5 ×4. With only the first function,
NCCL tops out near 11 GB/s. The second function needs an address and jumbo
frames on both nodes. Use a separate small subnet from the first function, and
make it persistent: NetworkManager removes ad hoc `ip addr` changes on
unconfigured devices.

```bash
# Head (use .11 on the worker). Interface = the second function of the cabled port.
sudo nmcli connection add type ethernet con-name ds4f-fabric-pf2 \
  ifname enP2p1s0f0np0 ipv4.method manual ipv4.addresses 198.51.100.10/24 \
  ipv4.never-default yes ipv6.method disabled 802-3-ethernet.mtu 9000 \
  connection.autoconnect yes
sudo nmcli connection up ds4f-fabric-pf2
ping -c 2 -M do -s 8972 -I enP2p1s0f0np0 198.51.100.11
```

`start.sh` validates that every listed function has a usable RoCE v2 GID and
refuses to start otherwise. With `NCCL_CROSS_NIC=0`, function 0 pairs with
function 0 and function 1 with function 1 across the cable. `NCCL_PROTO`,
`NCCL_IB_QPS_PER_CONNECTION`, `NCCL_BUFFSIZE`, and `NCCL_MIN_NCHANNELS` made no
measurable difference once both functions were in use.

### Measured and rejected

- **Draft length 6**: code +10%, structured +14%, but prose −11.5% and C2
  −6%. The 6th draft position lies beyond the checkpoint's trained
  `dspark_block_size` of 5, and every draft token adds one iteration of the
  serial Markov-head loop. Keep `MTP_NUM_TOKENS=5` unless the traffic is almost
  entirely code.
- **Per-request adaptive draft length**: prose −24%. Batches whose verify length
  differs from `1 + MTP_NUM_TOKENS` fall out of the captured FULL CUDA graphs
  (65,165 vs 3,545 kernel launches per trace). MoE time per call is also nearly
  flat across verify rows, so shortening drafts saves little.
- **SP indexer (key split)**: 128K prefill 3.6% slower. It merges float
  candidates with an extra JIT kernel; the query-row split above does not.
- **W4A8 MoE** (b12x 1.3, MXFP4 × MXFP8): 1.23–1.31× at prefill sizes, flat at
  decode sizes, and ~10× the MoE output error of W4A16. It also needs a newer
  image; not worth it for roughly 5% of prefill.
- **Prefill chunks above 1,024 without admission control**: see the v1
  starvation above.

### Incident: engine crash on 2026-09-28

About 17 minutes into real multi-agent traffic after a power-on, with every
overlay enabled, the head rank died with `CUBLAS_STATUS_INTERNAL_ERROR`. The
failing call was the stock BF16 `torch.mm` of a compressor projection, during a
mixed step: a 1,024-token prefill chunk of a 29K-token, mostly cached prompt,
plus one decoding request. The FP8 aux-dense wrapper was on the stack (its BF16
fallback path). The kernel logged no Xid, and the same call shape runs fine in
isolation.

It did not reproduce in about 1.5 hours of targeted tests on the same pair:
- a cold 470K-token prefill with concurrent chats;
- 25 minutes each of 3-session agent replays (413 and 474 requests, prompts
  to ~100K), including one after a warm-up-less Docker restart matching the
  power-on.

Because the crash is unexplained, the profile ships defensively:
- `DSPARK_FP8_AUX_DENSE=0`: the only overlay on the parallel aux streams and on
  the crash stack, worth only ~1.5% decode;
- `DSPARK_ADAPTIVE_CHUNK_MAX_DEPTH=131072`: keeps big chunks inside the depth
  they were qualified to.

On the guarded profile, 32K / 128K cold TTFT is 16.0 s / 65.9 s (baseline
19.5 s / 82.4 s), with retrieval passing. If you see this error, report it with
both ranks' logs.

