# qwen-qwen3.6-35b-a3b-fp8-sglang

Docker image that runs the **official FP8 checkpoint of Qwen3.6-35B-A3B** as an OpenAI-compatible API server, built for the **NVIDIA GB10 (DGX Spark)**.

The weights are pre-quantized to **FP8 (dynamic, e4m3)** by the Qwen team and served with [SGLang](https://github.com/sgl-project/sglang). Compare with [`nvidia-qwen3.6-35b-a3b-nvfp4-sglang`](https://github.com/picopapaya/nvidia-qwen3.6-35b-a3b-nvfp4-sglang), which serves NVIDIA's NVFP4 (ModelOpt) quantization of the same model.

## What it is

Qwen3.6-35B-A3B is a hybrid-attention Mixture-of-Experts vision-language model:

- 35 billion total parameters, ~3 billion active per token (256 experts per MoE layer, 8 routed + 1 shared active), so it runs with the compute cost of a much smaller model.
- Hybrid attention: 3 of every 4 layers use linear attention (Gated DeltaNet), every 4th layer uses full attention — long contexts are cheap in both compute and cache memory.
- Multimodal: accepts text and images (the FP8 checkpoint keeps the vision encoder in BF16).
- 262,144-token native context, and an MTP layer for speculative decoding.

## FP8 vs NVFP4 on the GB10

FP8 runs on this chip's fast native math path. NVFP4 doesn't — the chip has to convert it back to a bigger format before it can compute anything, which should make it slower in theory.

**Update 2026-07-08:** this section used to say FP8 was faster, based on other people's online benchmarks. We tested it ourselves and found the opposite: **NVFP4 was faster — 65.6 vs 60.5 tokens/sec**, tested one request at a time. That old claim wasn't based on real testing on this machine, so don't trust it. Full numbers in `../nvidia-qwen3.6-35b-a3b-nvfp4-sglang/README.md`.

What we haven't checked: what happens with several requests running at once (only tested one at a time so far), and whether NVFP4's answers are as accurate as FP8's — it's a smaller number format, so it can lose some precision, and we haven't verified how much that matters in practice.

The trade-off is memory: FP8 uses about 35 GB, NVFP4 about 20 GB — pick NVFP4 if you want to run several models on this machine at once. For raw speed, NVFP4 wins per the test above. For "definitely accurate," FP8 is the safer bet until we check NVFP4's output quality.

SGLang **v0.5.13+** is required for the `qwen3_5_moe` architecture; this image uses `lmsysorg/sglang:v0.5.14-cu130` (CUDA 13.x is required for sm_121a).

## MTP speculative decoding (experimental)

Qwen3.6 ships a multi-token-prediction layer, retained in this checkpoint. Decode on the GB10 is memory-bandwidth-bound, so accepted draft tokens are nearly free — community reports show large single-stream speedups. Set `ENABLE_MTP=1` to launch with the SGLang cookbook's recommended flags (`--speculative-algorithm EAGLE`, 3 steps, 4 draft tokens, `SGLANG_ENABLE_SPEC_V2=1`). Off by default: not yet validated on SM_121a.

## Running alongside another ~30B-class model

The packaged defaults (`CONTEXT_LEN=262144`, `MEM_FRACTION=0.85`) assume this is the only large model on the GPU. To run it side by side with another ~30B-class model (e.g. the NVFP4 sibling image) on a roughly 50/50 memory split, set in `.env`:

```
CONTEXT_LEN=131072
MEM_FRACTION=0.5
```

**Verified 2026-07-08:** both this image and the NVFP4 sibling started and ran healthy at the same time with these settings, and 4 concurrent requests at realistic prompt sizes (~7-8K tokens) all completed cleanly with no errors.

**Caveat:** at these settings, SGLang's own computed KV cache pool is **102,798 tokens** — smaller than a single fully-used 131,072-token (128K) request. SGLang auto-reduces `MAX_RUNNING_REQUESTS` from 4 to 3 as a result. This configuration is fine for typical shorter prompts (like the ~7-8K tokens tested above) run concurrently, but it cannot actually serve 3-4 requests that are each independently using the full 128K context ceiling at the same time — treat `CONTEXT_LEN` here as a safety ceiling for your longest realistic prompt, not a guarantee of full-context concurrency.

## Requirements

- NVIDIA GB10 / DGX Spark (SM_121a)
- Docker with NVIDIA Container Toolkit
- The `llm-net` Docker network: `docker network create llm-net`
- A Hugging Face token is **optional** — the model is not gated (Apache-2.0)

## Usage

```bash
# Prod — pull image from Docker Hub
docker compose up

# Dev — build image locally
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

The server starts on port **30000** and exposes an OpenAI-compatible API once the health check passes (allow up to 10 minutes for the first run while the ~35 GB of weights download).

## Configuration

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | *(empty)* | Optional Hugging Face token (avoids anonymous rate limits) |
| `CONTEXT_LEN` | `262144` | Maximum context length in tokens |
| `MEM_FRACTION` | `0.85` | Fraction of VRAM reserved for weights + KV cache |
| `MAX_RUNNING_REQUESTS` | `4` | Maximum concurrent requests |
| `ENABLE_MTP` | `0` | Set to `1` for MTP speculative decoding (experimental) |
| `REASONING_PARSER` | `qwen3` | SGLang reasoning parser |
| `TOOL_CALL_PARSER` | `qwen3_coder` | SGLang tool-call parser (per the SGLang Qwen3.6 cookbook) |
| `ATTENTION_BACKEND` | `triton` | Attention backend for the full-attention layers |
| `EXTRA_ARGS` | *(empty)* | Extra flags passed directly to `sglang.launch_server` |

## LiteLLM router

The model is registered in the shared LiteLLM proxy (`/home/shared/Documents/litellm/config.yaml`) as `qwen-qwen3.6-35b-a3b-fp8-sglang`. Restart the router after config changes:

```bash
docker restart litellm
```

## Publishing

Pushing a `v*.*.*` tag to GitHub builds the linux/arm64 image and publishes it to Docker Hub as `picopapaya/qwen-qwen3.6-35b-a3b-fp8-sglang` (see `.github/workflows/docker-publish.yml`; requires `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN` repo secrets).

## License

MIT — see [LICENSE](LICENSE).
