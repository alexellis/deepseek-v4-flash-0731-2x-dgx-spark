# Published data

## `controlled-sweep.json`

The raw 0731 qualification sweep originally published by MiaAI-Lab at commit
`c90fe1e92e718cc06dc5db8bed986758140d2b3a` on 31 July 2026. It contains 19
prompt-length/concurrency cases and all 59 request observations. The original
private endpoint address is redacted; measurements are otherwise unchanged.

This historical sweep used the official 0731 checkpoint and the then-current
recipe profile. It establishes the shape of TP2 performance, but it should not
be presented as a fresh measurement of every later scheduler setting.

## `real-traffic-2026-09-04.json`

An anonymised aggregate from Toilgate's SQLite usage records for one day of
OpenCode traffic through the official 0731 endpoint. It contains no request or
response bodies and no user, host, network, or credential identifiers.

The logical prompt count includes replayed context. Cached prompt tokens are
therefore shown separately. The trace is overwhelmingly one active user's
daily-driver workload, not a controlled concurrency experiment.

The generation distribution excludes completions shorter than 50 tokens,
non-streaming calls, interrupted calls, errors, and rows without a positive
generation rate.
