# Runtime recipe

## Pinning model

The repository is a small appliance layer over this immutable source:

```text
repository: https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark
commit:     c444d7032957f5a5437261d5366fd06b27a01760
```

`scripts/bootstrap.sh` checks out that commit under `.runtime/` and applies
`patches/0731-only.patch`. The runtime directory is generated and ignored by
Git. The patch:

- locks the model id to `deepseek-ai/DeepSeek-V4-Flash-0731`;
- locks the checkpoint revision to `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`;
- locks the served alias to `deepseek-v4-flash-0731`;
- requires a DSpark draft length of at least five;
- removes the Vision-Exp hotfix and multimodal argument from the active path.

The public wrapper only runs TP2 and only prepares official weights. The
upstream checkout remains visible under `.runtime/` for auditability.

## Runtime stack

The pinned Anemll image supplies the GB10 port of vLLM plus DSpark,
`nvfp4_ds_mla`, FlashInfer B12x MoE, and the required CUDA kernels. The Mia
runtime applies the later 0731 fixes used by the qualified deployment,
including:

- long-context NVFP4 DS-MLA correctness;
- partial-prefill admission and mixed prefill/decode fairness;
- sparse sliding-window prefix-cache retention;
- DSML encoder/tool-call compatibility;
- stop-token suppression while reasoning;
- assistant-final continuation;
- shared-memory and GB10 spin-wait reliability fixes.

Some experimental hotfixes remain present upstream but default off. The
values in `.env.example` are the qualified public profile.

## Node lifecycle

The head script synchronises the runtime, patch files, and generated
`.env.dspark` to the worker. It starts worker rank 1 before head rank 0 and
uses the direct ConnectX fabric for the distributed engine.

Docker's `unless-stopped` policy can restore containers after a reboot. An
already-running stack returns status 3 from the upstream launcher; the systemd
unit treats that as success.

## Reproducibility boundary

The following are pinned:

- upstream orchestration commit;
- model repository revision;
- runtime image digest;
- model id, alias, TP degree, KV type, and drafter family.

Kernel caches, host kernel/driver versions, ConnectX firmware, and Docker are
host state. Record those when publishing benchmark results.
