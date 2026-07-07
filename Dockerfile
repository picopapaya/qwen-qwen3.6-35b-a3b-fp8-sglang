# Qwen3.6-35B-A3B official FP8 checkpoint served by SGLang, on the NVIDIA GB10 (DGX Spark).
# Compare with ../nvidia-qwen3.6-35b-a3b-nvfp4-sglang which uses NVIDIA's NVFP4 (ModelOpt) weights.
#
# Model architecture — Qwen3.6-35B-A3B is a hybrid-attention Mixture-of-Experts VLM:
#   - 35B total parameters, ~3B active per token ("35B-A3B" = 35B total, 3B active)
#   - 256 experts per MoE layer; router activates 8 per token (+ shared expert)
#   - Hybrid: 3 of every 4 layers use linear attention (Gated DeltaNet), 1 uses full attention
#   - Multimodal: includes a vision encoder (text + image input)
#   - 262144-token native context; ships an MTP layer for speculative decoding
#
# Quantization — why FP8 over NVFP4 on SM121a (GB10):
#   - FP8 (dynamic, e4m3) takes the native CUTLASS matmul path on SM121a.
#   - NVFP4 MoE layers fall back to Marlin (FP4 → BF16 dequant in-kernel) on SM12x,
#     and in a 35B-A3B MoE the expert layers are almost all of the weight.
#   - Measured on this model class: FP8 wins at concurrency 1–4 (this box caps at 4);
#     NVFP4 only pulls ahead at c8+, at the cost of more quantization error.
#   - Official Qwen checkpoint: vision encoder kept in BF16, MTP layer retained.
#   - ~35 GB weights vs ~70 GB BF16 — no quantize-at-load cost, half the download.
#
# Base image: CUDA 13.x is required for sm_121a, and Qwen3.6 (qwen3_5_moe arch)
# modeling support requires SGLang >= v0.5.13.
ARG SGLANG_IMAGE=lmsysorg/sglang:v0.5.14-cu130
FROM --platform=linux/arm64 ${SGLANG_IMAGE}

ENV MODEL_ID="Qwen/Qwen3.6-35B-A3B-FP8" \
    HOST="0.0.0.0" \
    PORT="30000" \
    QUANTIZATION="fp8" \
    KV_CACHE_DTYPE="auto" \
    CONTEXT_LEN="262144" \
    MEM_FRACTION="0.85" \
    MAX_RUNNING_REQUESTS="4" \
    REASONING_PARSER="qwen3" \
    TOOL_CALL_PARSER="qwen3_coder" \
    ATTENTION_BACKEND="triton" \
    ENABLE_MTP="0" \
    EXTRA_ARGS="" \
    HF_HOME="/root/.cache/huggingface" \
    # Point Triton at CUDA 13.0's ptxas instead of PyTorch's bundled one.
    # The bundled ptxas predates SM_121 and rejects --gpu-name=sm_121a, causing
    # JIT compilation failures for attention and other kernels at runtime.
    # /usr/local/cuda/bin/ptxas (from CUDA 13.0 in this image) knows SM_121a natively.
    TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 30000

HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
