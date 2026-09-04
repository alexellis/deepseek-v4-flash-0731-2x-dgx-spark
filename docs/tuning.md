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
