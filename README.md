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

FP8 takes the GB10's native CUTLASS matmul path (SM_121a), while NVFP4 MoE layers fall back to the Marlin kernel, which dequantizes FP4 → BF16 in-kernel — and in a 35B-A3B MoE, the expert layers are almost all of the weight. Community benchmarks of this model class on GB10-class hardware show FP8 ahead at concurrency 1–4 (e.g. 51 vs 41 tok/s single-stream) with NVFP4 only pulling ahead at 8+ concurrent requests; this deployment caps concurrency at 4. FP8 is also closer to lossless than any 4-bit format.

The trade-off is memory: ~35 GB FP8 vs ~20 GB NVFP4. Prefer the NVFP4 image when colocating many models; prefer this one for latency and quality.

SGLang **v0.5.13+** is required for the `qwen3_5_moe` architecture; this image uses `lmsysorg/sglang:v0.5.14-cu130` (CUDA 13.x is required for sm_121a).

## MTP speculative decoding (experimental)

Qwen3.6 ships a multi-token-prediction layer, retained in this checkpoint. Decode on the GB10 is memory-bandwidth-bound, so accepted draft tokens are nearly free — community reports show large single-stream speedups. Set `ENABLE_MTP=1` to launch with the SGLang cookbook's recommended flags (`--speculative-algorithm EAGLE`, 3 steps, 4 draft tokens, `SGLANG_ENABLE_SPEC_V2=1`). Off by default: not yet validated on SM_121a.

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
