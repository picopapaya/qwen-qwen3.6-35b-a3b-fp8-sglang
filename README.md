# qwen-qwen3.6-35b-a3b-fp8-sglang

Docker image that runs the **official FP8 checkpoint of Qwen3.6-35B-A3B** as an OpenAI-compatible API server, built for the **NVIDIA GB10 (DGX Spark)**.

The weights are pre-quantized to **FP8 (dynamic, e4m3)** by the Qwen team and served with [SGLang](https://github.com/sgl-project/sglang). Compare with [`nvidia-qwen3.6-35b-a3b-nvfp4-sglang`](https://github.com/picopapaya/nvidia-qwen3.6-35b-a3b-nvfp4-sglang), which serves NVIDIA's NVFP4 (ModelOpt) quantization of the same model.

## What this image is

Qwen3.6-35B-A3B is a Mixture-of-Experts model that also understands images, not just text:

- 35 billion total parameters, but for any given word it only actually uses about 3 billion of them ("35B-A3B" = 35B total, ~3B active). The rest sit in memory ready to be picked, but don't add to the compute cost — it runs like a much smaller model while still having the knowledge of a much bigger one.
- Most of its layers use a cheaper, memory-light form of attention (a technique called Gated DeltaNet); only 1 in every 4 layers uses the full, more expensive kind. This is why it can handle very long conversations without needing huge amounts of extra memory for each one.
- Accepts both text and images (this checkpoint keeps the image-understanding part in a higher-precision format, BF16, rather than compressing it down to FP8 like the text part).
- Supports up to 262,144 tokens (roughly 200,000 words) of context, and ships an extra "MTP" layer that can speed up generation — see below.

SGLang **v0.5.13+** is required for the `qwen3_5_moe` architecture; this image uses `lmsysorg/sglang:v0.5.14-cu130` (CUDA 13.x is required for sm_121a).

### FP8 (this image) vs NVFP4

FP8 runs on this chip's fast native math path; NVFP4 has to be converted back to a bigger format before the chip can compute with it. The trade-off is memory: FP8 uses about 35 GB, NVFP4 about 20 GB. See `EXPERIMENT_NOTES.md` for measured speed and accuracy comparisons between the two on Nvidia DGX Spark machine.

### MTP speculative decoding (experimental)

This chip is usually limited by how fast it can move data in and out of memory, not by how much raw computing it can do. MTP takes advantage of that: instead of generating one word at a time, it guesses a few words ahead and checks them together, which is nearly free when memory movement — not computation — is what's slowing things down. Qwen3.6 ships this extra layer built in.

Set `ENABLE_MTP=1` to turn it on, using the settings the SGLang project recommends. It's off by default because it hasn't been thoroughly tested on this specific chip yet.

## Configuration

### Fixed configuration

These define what this image is, not how it's tuned. Changing them means you're describing a different image, not adjusting this one.

| Variable | Value | Why it's fixed |
|---|---|---|
| `MODEL_ID` | `Qwen/Qwen3.6-35B-A3B-FP8` | This is which model the image downloads and runs — that's the image's whole identity |
| `QUANTIZATION` | `fp8` | Matches the checkpoint's actual format |
| `KV_CACHE_DTYPE` | `auto` | Left to SGLang to pick automatically |
| `REASONING_PARSER` | `qwen3` | Needed so SGLang understands this model's "thinking" output format |
| `TOOL_CALL_PARSER` | `qwen3_coder` | Needed so SGLang understands this model's function-calling output format |

### Tunable via `.env`

These have a default baked into the image, but you can override them per-deployment by setting them in a `.env` file next to `docker-compose.yml`. Docker Compose reads that file automatically and passes the values into the container when it starts — no image rebuild needed, just edit `.env` and restart. Copy `.env.example` to `.env` to get started.

| Variable | Default | What it does |
|---|---|---|
| `HF_TOKEN` | *(empty)* | Optional Hugging Face token — avoids download rate limits, not required (this model isn't gated) |
| `CONTEXT_LEN` | `262144` | The longest conversation/prompt (in tokens) the server will accept |
| `MEM_FRACTION` | `0.85` | How much of the GPU's memory this server is allowed to claim |
| `MAX_RUNNING_REQUESTS` | `4` | How many requests SGLang will run concurrently |
| `ENABLE_MTP` | `0` | Set to `1` to turn on the speed feature described above |
| `ATTENTION_BACKEND` | `triton` | Which kernel library handles the attention math |
| `EXTRA_ARGS` | *(empty)* | Extra flags passed straight through to the underlying `sglang.launch_server` command, for anything not covered above |

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

## License

MIT — see [LICENSE](LICENSE).
