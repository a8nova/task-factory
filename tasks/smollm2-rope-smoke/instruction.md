# Optimize Greedy Decode Throughput for SmolLM2-135M on Adreno GPU

## Task

Optimize fp16 greedy autoregressive decode for the SmolLM2-135M-Instruct model port on a Motorola Razr 2020 smartphone (Qualcomm Snapdragon 765G with Adreno 620 GPU) running Android OpenCL.

Your deliverables are:
1. `/workspace/submission/patch.diff`: a unified diff of your changes against the base commit `4f0f2af972fb08361524964f9b826a64fa5e4817` in `/workspace/repo`.
2. `/workspace/submission/summary.md`: a technical summary explaining what you changed, why it improves throughput, and how numerical correctness was maintained.

To generate `patch.diff`, run:
```bash
cd /workspace/repo
git add -N .
git diff --binary 4f0f2af972fb08361524964f9b826a64fa5e4817 > /workspace/submission/patch.diff
```

Only the `/workspace/submission/` directory is copied to the evaluation harness.

## Constraints

- **In-Scope Files**: Only files within `src/models/smollm2-135m-instruct/` may be modified, specifically:
  - `src/models/smollm2-135m-instruct/src/model.cpp`
  - `src/models/smollm2-135m-instruct/src/layers/attention.h`
  - `src/models/smollm2-135m-instruct/src/layers/attention.cpp`
  - `src/models/smollm2-135m-instruct/src/opencl_context.cpp`
  - `src/models/smollm2-135m-instruct/kernels/block_fused.cl`
  - `src/models/smollm2-135m-instruct/kernels/attention.cl`
  - `src/models/smollm2-135m-instruct/kernels/mlp_swiglu.cl`
- **Off-Limits Files**: Do not modify benchmark timing utilities (`src/benchmark.h`), execution scripts (`scripts/run_android.sh`, `scripts/build.sh`, `scripts/deploy_android.sh`), build configuration (`CMakeLists.txt`), reference fixtures (`reference/`), weight conversion/metadata, or other model directories (`src/models/lfm2-5-350m/`, etc.).
- **Environment Constraints**:
  - You cannot compile or run Android binaries locally on the agent container. All builds and phone executions occur remotely via the device gateway.
  - No outbound internet access is permitted other than communicating with the device gateway via `val.sh`.
- **Precision and Weights**:
  - The phone stores only fp16 weights (`weights/model.fp16.bin`). Weight quantization (int8/int4) is not deployed on the test device and would break token-exact output requirements.

## Success Criteria

- **Correctness and Validity**:
  1. Exact token equality: On every hidden evaluation prompt and at all generation lengths (around 8, 40, and 96 tokens, each perturbed by a random 0–8 tokens per grading run), the candidate binary must output greedy tokens identical to the reference implementation.
  2. Clean execution: The binary must return exit code 0 without crashes, memory faults, or OpenCL errors.
  3. Linearity: The measured inference times across token generation lengths must follow a straight line with $R^2 \ge 0.98$.
  4. Short-run latency: The shortest run must not exhibit abnormal latency regressions compared to the baseline.
  5. Timer consistency: The binary's self-reported internal speed must not exceed the externally measured phone speed by more than 25%.
  6. Patch scope: Modifying any off-limits file invalidates the submission.
  7. The unchanged base implementation is valid but achieves 0 reward.
- **Speed Measurement**:
  - Decode speed (tokens/sec) is measured from outside the binary by the phone's clock as the slope of the linear fit between generated token count and elapsed seconds.
  - Speedup is calculated as: $\text{speedup} = \frac{\text{candidate\_tok\_s}}{\text{base\_tok\_s}}$.
  - Baseline decode throughput is **20.35 tok/s**; the reference solution reaches **24.01 tok/s** (a **1.193x** speedup).
- **Reward Function**:
  $$\text{Reward} = \begin{cases} 0 & \text{if not valid} \\ \min\left(1.0, \max\left(0.0, \frac{\text{speedup} - 1.0}{1.193 - 1.0}\right)\right) & \text{if valid} \end{cases}$$
  A speedup of 1.193x or greater earns the maximum score of 1.00.

## Design specification

To achieve the target 24.01 tok/s decode rate while preserving bit-exact token identity over 100+ decode steps, implement the following optimizations:

### 1. Eliminate Per-Token RoPE Table Host Recomputations and Allocations
- **Problem**: In the baseline implementation, `Attention::ensure_rope_tables(queue, seq_len)` is invoked on each decode step with `seq_k = start_pos + 1`. Because `rope_seq_len_` starts initialized only to the prompt length and increases strictly by 1 on each step, the check `rope_seq_len_ >= seq_len` fails on every single decode token across all 30 transformer layers. Consequently, every token generated triggers 60 buffer releases, CPU-side fp32 `std::pow`, `std::sin`, and `std::cos` loops for all 30 layers, and 60 `clCreateBuffer(CL_MEM_COPY_HOST_PTR)` driver calls. This driver and CPU overhead consumes hundreds of milliseconds per generation.
- **Target Design**:
  - Add a public method to `Attention` (e.g., `prewarm_rope_tables(cl_command_queue queue, int max_seq)`) that calls `ensure_rope_tables(queue, max_seq)`.
  - In `Model::initialize()` in `src/models/smollm2-135m-instruct/src/model.cpp`, loop through all 30 layers (`MODEL_CONFIG::NUM_HIDDEN_LAYERS`) and pre-allocate the RoPE tables once to `MODEL_CONFIG::MAX_POSITION_EMBEDDINGS` (8192).
  - Because `rope_seq_len_` is now 8192 from initialization, `ensure_rope_tables` becomes a zero-cost no-op throughout decoding, eliminating all per-token CPU trig computations and OpenCL buffer allocations.

### 2. Enable OpenCL 2.0 Standard Build Flag (`-cl-std=CL2.0`)
- **Problem**: The baseline OpenCL context builds kernels using OpenCL 1.2 rules. On Qualcomm Adreno 6xx/7xx drivers, the OpenCL compiler's instruction scheduler and register allocation algorithms achieve 15–20% higher ALU vectorization and memory bandwidth efficiency for cooperative GEMV workgroups when targeting OpenCL 2.0.
- **Target Design**:
  - In `OpenCLContext::make_effective_options` (`src/models/smollm2-135m-instruct/src/opencl_context.cpp`), append `-cl-std=CL2.0` to the build options if `-cl-std=` is not already specified, unless disabled by the environment flag `NNOPT_NO_CL2_STD=1`.

### 3. Replace Software Math with Native SFU Builtins (`native_exp`)
- **Problem**: In the attention softmax and MLP SwiGLU activation paths, standard `exp()` compiles to multi-instruction software approximations with high register pressure.
- **Target Design**: Replace `exp()` with `native_exp()`, which maps directly to Adreno's Special Function Unit (SFU) hardware instructions. The slight numerical difference remains within the greedy argmax margin across all steps:
  - In `src/models/smollm2-135m-instruct/kernels/attention.cl` within `gqa_softmax`: replace `exp(v - maxv)` with `native_exp(v - maxv)`.
  - In `src/models/smollm2-135m-instruct/kernels/block_fused.cl`:
    - In `silu_f`: replace `exp(-x)` with `native_exp(-x)`.
    - In `fused_gate_up_silu_m1`, `fused_gate_up_silu_m1_v4`, and `fused_gate_up_silu_m1_v4_img`: replace `exp(-g)` with `native_exp(-g)`.
    - In `fused_decode_attn_m1`: replace `exp(ls_scores[t] - gmax)` with `native_exp(ls_scores[t] - gmax)`.
  - In `src/models/smollm2-135m-instruct/kernels/mlp_swiglu.cl` within `sigmoid_f`: replace `exp(-x)` with `native_exp(-x)`.

### 4. Remove 64-bit Pointer Casts in `fused_decode_attn_m1`
- **Problem**: In `kernels/block_fused.cl`, the `fused_decode_attn_m1` kernel performs address arithmetic using explicit `(size_t)` casts (e.g., `(size_t)t * KV_DIM`). Adreno 6xx ALUs are natively 32-bit; 64-bit integer arithmetic in inner reduction loops creates register spill and unnecessary conversion instructions.
- **Target Design**: Change pointer arithmetic in `fused_decode_attn_m1` from `(size_t)t * KV_DIM + kvh * D` to 32-bit integer arithmetic: `t * KV_DIM + kvh * D`.

### 5. Numerical Preservation Guidelines
- Do not change workgroup sizes (`WG_SIZE` must remain 64).
- Do not alter tree-reduction order in `gemv_m1_k576_no4_img` or `fused_decode_attn_m1`.
- Do not use experimental subgroup shuffle operations (`sub_group_reduce_add` / `sub_group_reduce_max`), as hardware subgroup reductions regress throughput on Adreno 620 by up to 90% due to full-wave group size overhead.

## Validation and time

- **Validation Script**:
  - Test your submission file: `/workspace/validation/val.sh`
  - Test uncommitted workspace changes: `/workspace/validation/val.sh --repo`
  - The validation script builds your patch for Android arm64, executes it on the physical phone across 2 development prompts at lengths 8, 40, and 96, checks generated tokens against `/workspace/task/expected_tokens.json`, and reports decode throughput and score.
  - Each device run takes 1–3 minutes. Because the phone processes builds sequentially, validate deliberately after making cohesive changes.
- **Time Tracking**:
  - Check remaining task time: `/workspace/timer.sh`
  - You have a 60-minute time budget. Reserve at least 5–10 minutes before the deadline to generate your final `/workspace/submission/patch.diff` and verify it with `val.sh`.
