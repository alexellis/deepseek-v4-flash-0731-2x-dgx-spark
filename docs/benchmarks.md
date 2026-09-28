# Benchmarks and production traffic

There is no single honest tokens-per-second number for a speculative model.
This page keeps controlled endpoint measurements, manually classified work,
and production gateway telemetry separate.

## Matched before/after: profile-driven overlays (2026-09-27)

Standard [RigMark](https://github.com/alexellis/rigmark) protocol 1.1.0 at
`c5a0db01`, low reasoning, 4,096-token decode, 5 decode runs, 3 prefill pairs,
3 concurrency rounds. Both runs used the same Spark pair and the pinned image,
about 30 minutes apart. The baseline is the previous profile (fixed
1,024-token prefill chunks, one PCIe function, no runtime overlays); the tuned
run is the new `.env.example` profile (see
[tuning](tuning.md#profile-driven-speed-ups-2026-09-27)). Receipts:
[`baseline`](../data/rigmark-2026-09-27-baseline.json),
[`tuned`](../data/rigmark-2026-09-27-tuned.json).

| RigMark cell | Baseline | Tuned | Change |
|---|---:|---:|---:|
| Cold prefill 8K | 1,688 tok/s | 2,128 tok/s | **+26.1%** |
| Cold prefill 32K | 1,672 tok/s | 2,010 tok/s | **+20.2%** |
| Cold prefill 64K | 1,652 tok/s | 2,019 tok/s | **+22.2%** |
| Code decode | 70.8 tok/s | 74.4 tok/s | +5.1% |
| Structured decode | 83.5 tok/s | 85.9 tok/s | +2.8% |
| Prose decode | 46.8 tok/s | 47.6 tok/s | +1.7% |
| Short-code aggregate C1 / C2 / C4 | 51.5 / 79.5 / 113.5 | 53.2 / 80.6 / 120.4 | +3.3 / +1.4 / +6.0% |

A separate thinking-off screen on the same boots measured cold 128K time to
first token at 82.4 s before and 65.4 s after (−20.5%). Both profiles passed
15/15 RigMark basic gates; the tuned profile also passed exact three-key
retrieval over 114,614 tokens and a tool call.

## Controlled prompt/concurrency sweep

The raw data is [`data/controlled-sweep.json`](../data/controlled-sweep.json):
the official 0731 qualification run published on 31 July 2026. Each successful
request asked for roughly 512 output tokens. Values below are medians within a
case; aggregate output rate is calculated across all streams.

| Prompt target | Streams | TTFT | Prefill tok/s | Decode tok/s per stream | Aggregate tok/s |
|---:|---:|---:|---:|---:|---:|
| 256 | 1 | 0.63 s | 447 | 75.4 | 69.1 |
| 256 | 2 | 0.81 s | 357 | 58.3 | 104.9 |
| 256 | 4 | 1.26 s | 222 | 46.8 | 164.5 |
| 256 | 6 | 1.42 s | 197 | 36.9 | **191.2** |
| 2K | 1 | 0.81 s | **2,563** | 68.8 | 62.0 |
| 2K | 2 | 1.11 s | 1,911 | 57.0 | 97.6 |
| 2K | 4 | 1.38 s | 1,505 | 44.0 | 154.7 |
| 2K | 6 | 6.06 s | 342 | 34.7 | 143.7 |
| 8K | 1 | 4.80 s | 1,713 | 73.9 | 43.7 |
| 8K | 2 | 7.51 s | 1,176 | 49.8 | 56.2 |
| 8K | 4 | 14.50 s | 578 | 37.4 | 72.3 |
| 8K | 6 | 18.38 s | 454 | 23.6 | 73.1 |
| 32K | 1 | 22.96 s | 1,428 | 64.0 | 16.6 |
| 32K | 2 | 26.82 s | 1,287 | 41.5 | 24.8 |
| 32K | 4 | 44.85 s | 756 | 17.4 | 26.7 |
| 32K | 6 | 60.75 s | 550 | 10.8 | 27.9 |
| 128K | 1 | 78.75 s | 1,665 | 65.2 | 5.9 |
| 128K | 2 | 111.17 s | 1,306 | 30.9 | 6.6 |
| 128K | 4 | 189.49 s median | — | — | — |

The final 128K × 4 case is a failure, not a tiny throughput result: all four
requests were cancelled, two before producing any output. The raw observations
are preserved so this limit cannot disappear from a summary table.

What the sweep shows:

- short-context aggregate throughput scales to six streams, while per-stream
  speed falls;
- isolated 2K–128K prefill is normally around 1.3K–2.6K tok/s;
- simultaneous long cold prefills dominate latency;
- a 1M advertised ceiling does not imply that several 1M cold requests can run
  together—the KV pool and scheduler budget are shared.

This is historical qualification data for official 0731, not a claim that
every subsequent scheduler change was present in this exact run.

## Real OpenCode traffic

[`data/real-traffic-2026-09-04.json`](../data/real-traffic-2026-09-04.json)
contains an anonymised Toilgate aggregate covering 06:43–16:18 UTC on 4
September 2026.

| Metric | Observed |
|---|---:|
| Requests | **982** |
| HTTP 200 | 982 |
| Interrupted / loop-detected | 1 / **0** |
| Logical prompt tokens | **169.2M** |
| Cached prompt tokens | **167.3M (98.9%)** |
| Uncached prompt tokens | 1.87M |
| Completion tokens | **592K** |
| Sustained requests analysed | 974 |
| Generation p10 / p50 / p90 | **37.5 / 57.4 / 81.7 tok/s** |
| TTFT p50 / p90 / p99 | **1.33 / 3.25 / 10.84 s** |
| Maximum TTFT | 145.84 s |

The workload was overwhelmingly one active principal (981 of 982 calls), so
this is a daily-driver trace, not a concurrency test. It is valuable precisely
because it includes real session replay, tool turns, short actions, long
answers, warm cache hits, and occasional cold prefixes.

### Output length changes the number

The sustained distribution excludes output shorter than 50 tokens, errors,
non-streaming calls, and the interrupted call.

| Completion length | Requests | Median tok/s | Mean tok/s | Range |
|---:|---:|---:|---:|---:|
| 50–149 | 195 | **73.7** | 69.3 | 20.6–100.4 |
| 150–499 | 457 | **63.2** | 62.1 | 6.7–89.8 |
| 500–1,499 | 222 | **49.0** | 49.8 | 22.6–79.8 |
| 1,500+ | 100 | **46.8** | 46.4 | 28.7–72.2 |

This is the practical answer to “why did I see 80 tok/s, then 40?” Short tool
actions and predictable code fragments are friendlier to the drafter. Long,
novel prose converges towards the mid-to-high 40s on this trace.

### Cache misses explain responsiveness

Logical context is not the same as work done. The agent repeatedly sent large
histories, but 98.9% of prompt tokens were cache hits.

| Uncached prompt | Requests | Mean uncached tokens | Mean TTFT | Worst TTFT |
|---:|---:|---:|---:|---:|
| <1K | 616 | 531 | **1.15 s** | 3.14 s |
| 1–8K | 339 | 2,617 | **2.63 s** | 7.13 s |
| 8–32K | 15 | 12,790 | **9.82 s** | 14.92 s |
| 32K+ | 4 | 115,083 | **84.24 s** | 145.84 s |

Prefix caching is therefore not a cosmetic optimisation. Without it, the
169.2M logical prompt tokens would make the interactive lane untenable.

## Manually classified observations

During the same bring-up we observed roughly **38–45 tok/s** on conversational
prose and **65–89 tok/s** on a code-oriented agent task. These labels came from
the operator who started the tasks; they are not inferred retrospectively from
private prompts. They agree with the production output-length distribution,
but remain observations rather than a balanced prose/code evaluation set.
