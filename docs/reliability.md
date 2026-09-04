# Reliability and unattended recovery

`/health` can remain green when the vLLM engine can no longer complete a
request. The included watchdog therefore sends a real one-token completion.
After consecutive failures it captures state from both ranks, stops and starts
the pair in the correct order, and waits for a successful completion.

Install the example units on the head node after adjusting their
`WorkingDirectory` paths:

```bash
sudo cp systemd/ds4f-0731.service /etc/systemd/system/
sudo cp systemd/ds4f-0731-watchdog.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ds4f-0731.service
sudo systemctl enable --now ds4f-0731-watchdog.service
```

The watchdog is intentionally conservative. A long JIT compile or deep
prefill may occupy the engine for minutes, so its completion timeout and
failure threshold should reflect the workload. Forensics default to
`$HOME/ds4f-crashes` and contain logs and machine state; review them before
sharing publicly.

Disable `earlyoom` on both Sparks for deep-context serving. It can kill the
engine while the unified-memory workload is healthy from vLLM's perspective.
