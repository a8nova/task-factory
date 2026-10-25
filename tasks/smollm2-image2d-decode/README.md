# SmolLM2-135M-Instruct FP16 Decode Optimization on Adreno

This task benchmarks an agent's ability to optimize steady-state fp16 `M=1` greedy token decoding for `SmolLM2-135M-Instruct` (a 30-layer LLaMA-style GQA transformer with `H=576`, `INTER=1536`, `V=49152`) on a real Android phone (Motorola Razr 2020, Adreno 620 GPU).

## Key Optimizations Tested
- **Adreno Texture-Cache (`image2d_t`) 4-Output GEMV (`gemv_m1_k576_no4_img`):** Wrapping fp16 weight buffers as zero-copy `CL_RGBA`/`CL_HALF_FLOAT` 2D images and computing 4 output rows per workgroup for Q, K, V, residual-fused O-proj, and residual-fused MLP down-proj.
- **Row-Tiled `LmHead` 2D Images:** Partitioning the `[49152, 576]` tied embedding matrix into 128-byte-aligned OpenCL sub-buffers and 2D image tiles to respect Adreno's `CL_DEVICE_IMAGE2D_MAX_HEIGHT = 16384` limit.
- **Vec4 Image-Backed SwiGLU (`fused_gate_up_silu_m1_v4_img`):** Vectorizing the fused gate+up projection inner loop over 2D weight images while keeping 1 output per workgroup (2 fp32 accumulators/thread) to avoid register spilling.
- **Two-Pass GPU Argmax (`kernels/argmax.cl`):** Replacing the per-token 98 KB blocking fp16 logit readback and host scan with an on-GPU reduction.
- **OpenCL Context Tuning:** Adding `-cl-fast-relaxed-math`, persistent FNV-1a-keyed program binary caching, environment-gated command queue profiling, and `clSetPerfHintQCOM`.

## Passing Criteria
- **Throughput:** `>= 1.42x` baseline decode tok/s (`13.12 tok/s` -> `>= 18.63 tok/s`; reference achieves `19.63 tok/s`, `1.50x`), timed externally.
- **Correctness:** Exact token-ID parity with the reference implementation on the 40-token fixed input and held-out evaluation prompts.