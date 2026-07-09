# qwen-qwen3.6-35b-a3b-fp8-sglang

Docker image that runs the **official FP8 checkpoint of Qwen3.6-35B-A3B** as an OpenAI-compatible API server, built for the **NVIDIA GB10 (DGX Spark)**.

The weights are pre-quantized to **FP8 (dynamic, e4m3)** by the Qwen team and served with [SGLang](https://github.com/sgl-project/sglang). Compare with [`nvidia-qwen3.6-35b-a3b-nvfp4-sglang`](https://github.com/picopapaya/nvidia-qwen3.6-35b-a3b-nvfp4-sglang), which serves NVIDIA's NVFP4 (ModelOpt) quantization of the same model.

## What this image is

Qwen3.6-35B-A3B is a Mixture-of-Experts model that also understands images, not just text:

- 35 billion total parameters, but for any given word it only actually uses about 3 billion of them ("35B-A3B" = 35B total, ~3B active). The rest sit in memory ready to be picked, but don't add to the compute cost — it runs like a much smaller model while still having the knowledge of a much bigger one.
- Most of its layers use a cheaper, memory-light form of attention (a technique called Gated DeltaNet); only 1 in every 4 layers uses the full, more expensive kind. This is why it can handle very long conversations without needing huge amounts of extra memory for each one.
- Accepts both text and images (this checkpoint keeps the image-understanding part in a higher-precision format, BF16, rather than compressing it down to FP8 like the text part).
- Supports up to 262,144 tokens (roughly 200,000 words) of context, and ships an extra "MTP" layer that can speed up generation — see below.

## FP8 vs NVFP4 on the GB10

FP8 runs on this chip's fast native math path. NVFP4 doesn't — the chip has to convert it back to a bigger format before it can compute anything, which should make it slower in theory.

**Update 2026-07-08:** this section used to say FP8 was faster, based on other people's online benchmarks. We tested it ourselves and found the opposite: **NVFP4 was faster — 65.6 vs 60.5 tokens/sec**, tested one request at a time. That old claim wasn't based on real testing on this machine, so don't trust it. Full numbers in `../nvidia-qwen3.6-35b-a3b-nvfp4-sglang/README.md`.

What we haven't checked: what happens with several requests running at once (only tested one at a time so far), and whether NVFP4's answers are as accurate as FP8's — it's a smaller number format, so it can lose some precision, and we haven't verified how much that matters in practice.

The trade-off is memory: FP8 uses about 35 GB, NVFP4 about 20 GB — pick NVFP4 if you want to run several models on this machine at once. For raw speed, NVFP4 wins per the test above. For "definitely accurate," FP8 is the safer bet until we check NVFP4's output quality.

SGLang **v0.5.13+** is required for the `qwen3_5_moe` architecture; this image uses `lmsysorg/sglang:v0.5.14-cu130` (CUDA 13.x is required for sm_121a).

## MTP speculative decoding (experimental)

This chip is usually limited by how fast it can move data in and out of memory, not by how much raw computing it can do. MTP takes advantage of that: instead of generating one word at a time, it guesses a few words ahead and checks them together, which is nearly free when memory movement — not computation — is what's slowing things down. Qwen3.6 ships this extra layer built in.

Set `ENABLE_MTP=1` to turn it on, using the settings the SGLang project recommends. It's off by default because it hasn't been thoroughly tested on this specific chip yet — turn it on, watch for problems, and turn it back off if anything looks wrong.

## Configuration

### Tunable via `.env`

These have a default baked into the image, but you can override them per-deployment by setting them in a `.env` file next to `docker-compose.yml`. Docker Compose reads that file automatically and passes the values into the container when it starts — no image rebuild needed, just edit `.env` and restart.

| Variable | Default | What it does |
|---|---|---|
| `HF_TOKEN` | *(empty)* | Optional Hugging Face token — avoids download rate limits, not required (this model isn't gated) |
| `CONTEXT_LEN` | `262144` | The longest conversation/prompt (in tokens) the server will accept |
| `MEM_FRACTION` | `0.85` | How much of the GPU's memory this server is allowed to claim |
| `ENABLE_MTP` | `0` | Set to `1` to turn on the speed feature described above |
| `ATTENTION_BACKEND` | `triton` | Which kernel library handles the attention math — `flashinfer` measured ~30% faster decode on this box (2026-07-08) |

### Fixed — not overridable via `.env`

These define what this image *is*, not how it's tuned. Changing them means you're describing a different image, not adjusting this one.

| Variable | Value | Why it's fixed |
|---|---|---|
| `MODEL_ID` | `Qwen/Qwen3.6-35B-A3B-FP8` | This is which model the image downloads and runs — that's the image's whole identity |
| `QUANTIZATION` | `fp8` | Matches the checkpoint's actual format |
| `KV_CACHE_DTYPE` | `auto` | Left to SGLang to pick automatically |
| `MAX_RUNNING_REQUESTS` | `4` | Not currently wired up as a `.env` override — could be added if a need for it comes up |
| `REASONING_PARSER` | `qwen3` | Needed so SGLang understands this model's "thinking" output format |
| `TOOL_CALL_PARSER` | `qwen3_coder` | Needed so SGLang understands this model's function-calling output format |

`EXTRA_ARGS` also exists (passed straight through to the underlying server command) but isn't wired to `.env` by default — it's commented out in `docker-compose.yml` as a documented escape hatch. Uncomment it there directly if you need to pass something not covered above.

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

## LiteLLM router

The model is registered in the shared LiteLLM proxy (`/home/shared/Documents/litellm/config.yaml`) as `qwen-qwen3.6-35b-a3b-fp8-sglang`. Restart the router after config changes:

```bash
docker restart litellm
```

## Publishing

Pushing a `v*.*.*` tag to GitHub builds the linux/arm64 image and publishes it to Docker Hub as `picopapaya/qwen-qwen3.6-35b-a3b-fp8-sglang` (see `.github/workflows/docker-publish.yml`; requires `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN` repo secrets).

## License

MIT — see [LICENSE](LICENSE).
