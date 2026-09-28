# Credits and provenance

This appliance profile is built on the work of the local-AI community.

- [MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)
  provides the two-node vLLM orchestration, DSpark integration, scheduler
  fixes, long-context work, and most of the runtime qualification. The
  upstream source is MIT licensed and is fetched at an immutable commit.
- [Anemll/dspark-vllm-gx10](https://github.com/Anemll/dspark-vllm-gx10)
  provides the pinned GB10 vLLM runtime image.
- DeepSeek provides the official
  [`DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
  checkpoint and its matching DSpark draft head.
- [othexmr/GLM-5.3-Flash-NVFP4-2x-4x-DGX-Sparks-RiNGSiDE](https://github.com/othexmr/GLM-5.3-Flash-NVFP4-2x-4x-DGX-Sparks-RiNGSiDE)
  (Apache-2.0) inspired the runtime overlays in `patches/runtime`: the two-rank
  all-reduce exchange is derived from its collectives patch, and the query-row
  indexer split and FP8 dense copies re-implement its ideas. See
  [`patches/runtime/NOTICE`](patches/runtime/NOTICE).
- The vLLM, FlashInfer, DFlash/DSpark, NCCL, and wider DGX Spark communities
  made the underlying serving stack possible.

The 0731-only appliance contract, live profile, bootstrap wrapper, operational
watchdog, and documentation in this repository are maintained by Alex Ellis /
OpenFaaS Ltd.

See [`docs/recipe.md`](docs/recipe.md) for the exact upstream revision and the
small patch applied to it.
