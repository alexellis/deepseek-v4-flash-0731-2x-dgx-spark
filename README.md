# DeepSeek-V4-Flash-0731 — 2× DGX Spark TP2

Serve the **official DeepSeek-V4-Flash-0731 checkpoint** across two NVIDIA
DGX Sparks as one tensor-parallel vLLM engine. The profile uses NVFP4 DS-MLA
KV, the checkpoint's matching DSpark speculative drafter, mixed-prefill
fairness fixes, prefix caching, and a 1M-token context ceiling.

This repository is deliberately narrow: **two Sparks, TP2, and the official
0731 text checkpoint**. It does not expose Vision-Exp or an ablated checkpoint
selector. Every hostname, address, interface, and filesystem path in the
repository is a placeholder.

Built and run in production by **Alex Ellis / OpenFaaS Ltd**. The serving
runtime is derived from MiaAI-Lab's excellent DGX Spark recipe at a pinned,
qualified commit; see [CREDITS.md](CREDITS.md).

## What comes out

- One OpenAI-compatible endpoint on the head Spark, default port `8888`.
- Served model id: `deepseek-v4-flash-0731`.
- Tensor parallelism across two GB10 GPUs.
- Up to 1,048,576 tokens per request, subject to the shared KV pool.
- Six active sequence slots for normal multi-agent traffic.
- Automatic worker synchronisation and optional systemd/watchdog units.

## Real serving behaviour

Synthetic, predictable output flatters speculative decoding. The useful
figures are workload-shaped:

| Workload | Observed decode |
|---|---:|
| Prose and conversational turns | **38–45 tok/s** |
| Code-oriented turns with high draft acceptance | **65–89 tok/s** |
| Older structured single-stream benchmark | **~82 tok/s** |

These are not interchangeable. A counting or rigid JSON test can achieve high
draft acceptance and say little about how quickly an agent writes prose. Long
agent turns are also dominated by prefill: the older published sweep measured
about **1,428–1,713 prompt tok/s** at 8K–32K and **1,665 prompt tok/s** at 131K
for one stream. Warm prefix reuse is much faster and is central to making the
lane feel responsive.

The values above describe the qualified family of configurations, not a
guarantee for another pair, prompt, temperature, or client.

## Why 0731 only?

In extended coding and discussion use, we preferred the 0731 text checkpoint
to Vision-Exp. We also exclude ablated derivatives from this appliance.

The DSpark draft head and serving path are matched to a particular target
checkpoint. (This recipe uses **DSpark**, not GLM's DFlash2, though the same
matching principle applies.) Changing target weights without a correspondingly
qualified drafter can reduce speculative acceptance and erase the speed
benefit — an operational impedance mismatch. Exact speculative verification
should not change the target distribution merely because a draft token is
rejected, so this repository does **not** claim that MTP mathematically damages
output. The practical concern is the combined, unqualified
checkpoint/template/drafter stack: lower acceptance, different behaviour, and
failure modes we do not want in the production lane.

If you want Vision-Exp or ablated weights, use an upstream recipe designed and
tested for those checkpoints. They are intentionally not switches here.

## Hardware and network

- 2× NVIDIA DGX Spark (GB10 / `sm_121`).
- One 100G ConnectX link between the two Sparks.
- A separate management network for SSH is recommended.
- Enough local NVMe on each node for the checkpoint, image, and JIT caches.
- Passwordless SSH from the head to the worker.

```text
management:  operator ───── head ───── worker
                              │ 100G │
RoCE fabric:                  rank 0 ─ rank 1
```

No switch is required for a two-node lane.

## Quick start

Run these commands on the head Spark.

1. Clone and configure:

   ```bash
   git clone https://github.com/alexellis/deepseek-v4-flash-0731-2x-dgx-spark
   cd deepseek-v4-flash-0731-2x-dgx-spark
   cp .env.example .env
   $EDITOR .env
   ```

2. Pull the pinned image on both Sparks:

   ```bash
   docker pull ghcr.io/anemll/dspark-vllm-gx10:0.1.1@sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8
   ssh dgx-spark-worker.example.net docker pull ghcr.io/anemll/dspark-vllm-gx10:0.1.1@sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8
   ```

3. Fetch the pinned runtime and official weights:

   ```bash
   ./scripts/bootstrap.sh
   ./scripts/prepare.sh
   ```

4. Start worker rank 1, then head rank 0:

   ```bash
   ./scripts/start.sh
   ./scripts/status.sh
   ./scripts/smoke.sh
   ```

The first boot may spend a long time compiling kernels. Do not mistake quiet
logs for a dead engine; use `./scripts/logs.sh` and allow the configured
30-minute model-execution timeout.

Stop both ranks with:

```bash
./scripts/stop.sh
```

## The qualified profile

| Setting | Value | Reason |
|---|---:|---|
| Checkpoint | official 0731 @ `9e165c30…` | Immutable, matched target/drafter |
| Runtime | Anemll `0.1.1` @ digest | Reproducible GB10 image |
| Tensor parallel | 2 | One rank per Spark |
| KV cache | `nvfp4_ds_mla` | Makes the 1M ceiling practical |
| Context ceiling | 1,048,576 | Long-horizon agent lane |
| Active sequences | 6 | Small-team concurrency |
| Batch tokens | 8,192 | Prefill/decode balance |
| Long-prefill chunk | 1,024 | Prevents prefill from starving decode |
| In-flight prefills | 2 | Qualified mixed-load setting |
| DSpark draft length | 5 | Matches 0731 block size |
| GPU utilisation | 0.835 | Leaves room for graphs/JIT |
| Breakable graphs | off | Regular graphs were faster here |

See [docs/tuning.md](docs/tuning.md) before changing these together. The
runtime integration and pin are explained in [docs/recipe.md](docs/recipe.md),
and unattended recovery in [docs/reliability.md](docs/reliability.md).

## Client reasoning effort

The appliance default is `max` because this is a capability lane. Clients can
override the request to `low`, `high`, or `max`. `low` is dramatically more
interactive for discussion and short coding loops; `max` is best reserved for
long-horizon work where depth matters more than latency.

## Updating

The upstream runtime commit and our small 0731-only patch are part of the
qualified configuration. Do not casually follow upstream `main`:

1. change `RUNTIME_COMMIT` in `scripts/lib.sh`;
2. make `patches/0731-only.patch` apply cleanly;
3. run `CHECK_RUNTIME=1 ./scripts/gate.sh`;
4. cold-boot both ranks;
5. repeat smoke, long-context, prose, code, tool-call, and mixed-load tests.

This is an appliance recipe, not a promise that every new vLLM or model commit
is automatically an upgrade.

## Licence

MIT. Upstream components retain their own licences and copyrights.
