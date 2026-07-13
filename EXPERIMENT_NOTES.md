## 2026-07-08

### Is FP8 or NVFP4 faster on the GB10 for this model?

**Findings**

In theory FP8 should win: it runs on this chip's fast native math path, while NVFP4 has to be converted back to a bigger format before the chip can compute with it. Measured on this box, one request at a time, the opposite held: NVFP4 decoded faster — 65.6 tokens/sec vs FP8's 60.5 tokens/sec. Most likely explanation: NVFP4 moves half as much data through memory, and on this workload that saving outweighs the conversion cost. Full numbers in `../nvidia-qwen3.6-35b-a3b-nvfp4-sglang/EXPERIMENT_NOTES.md`.

**Conclusion**

Not yet checked: behavior with several requests running at once, and whether NVFP4's answers are as accurate as FP8's (it's a smaller number format, so it can lose some precision). For "definitely accurate," FP8 remains the safer bet until NVFP4's output quality is checked.

---

### Does switching the attention kernel from `triton` to `flashinfer` help this image?

**Findings**

Measured on this box: `flashinfer` decoded about 30% faster than the `triton` default.

**Conclusion**

Worth using `flashinfer` over the `triton` default for this image on this box.

---

### Can this image run side by side with another ~30B-class model (e.g. the NVFP4 sibling) on a shared GPU?

**Findings**

The packaged defaults (`CONTEXT_LEN=262144`, `MEM_FRACTION=0.85`) assume this is the only large model on the GPU. Setting `CONTEXT_LEN=131072` and `MEM_FRACTION=0.5` in `.env` gives a roughly 50/50 memory split. Verified: both this image and the NVFP4 sibling started and ran healthy at the same time with these settings, and 4 concurrent requests at realistic prompt sizes (~7-8K tokens) all completed cleanly with no errors.

Caveat found: at these settings, SGLang's own computed KV cache pool is 102,798 tokens — smaller than a single fully-used 131,072-token (128K) request. SGLang auto-reduces `MAX_RUNNING_REQUESTS` from 4 to 3 as a result.

**Conclusion**

This split is fine for typical shorter prompts (like the ~7-8K tokens tested) run concurrently, but it cannot actually serve 3-4 requests that are each independently using the full 128K context ceiling at the same time. Treat `CONTEXT_LEN` in a shared-GPU setup as a safety ceiling for your longest realistic prompt, not a guarantee of full-context concurrency.
