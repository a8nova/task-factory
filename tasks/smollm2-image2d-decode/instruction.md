# Optimize SmolLM2-135M-Instruct FP16 Greedy Decode on Adreno 620

## Goal
Optimize steady-state **fp16 `M=1` greedy decode throughput** for the `src/models/smollm2-135m-instruct` OpenCL port on the attached Motorola Razr 2020 (Snapdragon 765G / Adreno 620).

- **Baseline decode speed on this device:** `13.12 tok/s`
- **Reference decode speed on this device:** `19.63 tok/s` (`1.50x` baseline)
- **Required threshold to pass:** `>= 1.42x` baseline decode speed (`>= 18.63 tok/s`) **AND** exact token-for-token output parity with the reference implementation.

---

## How Grading Works & How to Test

- **No local build or execution:** You cannot compile or run Android binaries locally in this container, and there is no general network access.
- **Device gateway (`/app/task/phone_run.sh`):** Run `/app/task/phone_run.sh` to test your changes. It packages your current `git diff` against the base commit and sends it to the device gateway. The gateway cross-compiles the fp16 release binary (`NNOPT_DTYPE=fp16 ./scripts/build.sh --release`), deploys it to the phone, runs inference, and returns:
  1. Compiler or OpenCL build errors (if any).
  2. **External decode `tok/s`**, measured outside the binary by timing a long generation run minus a short generation run on the phone's wall clock (the binary's internal `BenchmarkTimer` printout is ignored for grading).
  3. The **40 generated output tokens** for the fixed evaluation input.
  Each `/app/task/phone_run.sh` invocation takes approximately **1–2 minutes**.
  *Tip:* Because `phone_run.sh` sends `git diff`, if you create a new file (such as `kernels/argmax.cl`), run `git add -N <path>` so `git diff` includes it.
- **Exact Token Parity Rule:** Final grading evaluates your diff on the fixed input (expected tokens provided in `/app/task/expected_tokens.json`) **and** on held-out hidden prompts you never see. Your generated tokens must match the reference implementation **identically** at every position.
  - **Important numerical note:** The base commit and the reference implementation agree only for the first 8 generated tokens on the fixed input because switching from scalar-strided buffer GEMVs to `vec4` `image2d_t` GEMVs and enabling `-cl-fast-relaxed-math` changes floating-point reduction order and rounding. To match `/app/task/expected_tokens.json` and pass hidden prompts, follow the exact kernel layouts, accumulation order, and build options specified below.

---

## Scope

### In-Scope Files (you may edit or create)
- `src/models/smollm2-135m-instruct/kernels/block_fused.cl`
- `src/models/smollm2-135m-instruct/kernels/argmax.cl` *(new file)*
- `src/models/smollm2-135m-instruct/src/layers/attention.h`
- `src/models/smollm2-135m-instruct/src/layers/attention.cpp`
- `src/models/smollm2-135m-instruct/src/layers/mlp.h`
- `src/models/smollm2-135m-instruct/src/layers/mlp.cpp`
- `src/models/smollm2-135m-instruct/src/layers/lm_head.h`
- `src/models/smollm2-135m-instruct/src/layers/lm_head.cpp`
- `src/models/smollm2-135m-instruct/src/model.h`
- `src/models/smollm2-135m-instruct/src/model.cpp`
- `src/models/smollm2-135m-instruct/src/opencl_context.h`
- `src/models/smollm2-135m-instruct/src/opencl_context.cpp`

### Off-Limits Files (do NOT modify)
Do not touch any other files in the repository, including:
- `src/models/smollm2-135m-instruct/src/main.cpp`
- `src/models/smollm2-135m-instruct/src/benchmark.h`
- `src/models/smollm2-135m-instruct/src/debug_utils.h`
- `src/models/smollm2-135m-instruct/src/prof.h` and `src/prof.cpp`
- `src/models/smollm2-135m-instruct/src/sampler.h` and `src/sampler.cpp`
- `src/models/smollm2-135m-instruct/src/tokenizer.h` and `src/tokenizer.cpp`
- `src/models/smollm2-135m-instruct/src/utils.h` and `src/utils.cpp`
- `src/models/smollm2-135m-instruct/src/weights.h`, `src/weights.cpp`, `src/weight_map.h`, `src/model_config.h`
- `src/models/smollm2-135m-instruct/CMakeLists.txt`, `scripts/`, `reference/`, `weights/`
- All other model directories under `src/models/`

---

## Target Design Specification

SmolLM2-135M-Instruct dimensions (`MODEL_CONFIG`): `NUM_HIDDEN_LAYERS = 30`, `HIDDEN_SIZE (H) = 576`, `NUM_ATTENTION_HEADS = 9`, `NUM_KEY_VALUE_HEADS = 3`, `HEAD_DIM = 64` (`Q_DIM = 576`, `KV_DIM = 192`), `INTERMEDIATE_SIZE (INTER) = 1536`, `VOCAB_SIZE (V) = 49152`.

### 1. OpenCL Context: Compiler Flags, Queue Profiling Gate, Perf Hint, and Binary Cache (`opencl_context.h`, `opencl_context.cpp`)

1. **Effective Build Options (`-cl-fast-relaxed-math`):**
   - In `OpenCLContext::build_program` (and a static helper `OpenCLContext::build_cached_program_from_queue(cl_command_queue queue, const std::string& source, const std::string& options)`), normalize the compiler options string before building:
     - Under `#ifdef NNOPT_USE_FP16`, if `"USE_FP16"` is not already present in `options`, append `"-D USE_FP16=1"`.
     - If `"-cl-fast-relaxed-math"` is not already present, append `"-cl-fast-relaxed-math"`.
   - **Critical:** Every OpenCL program built via `OpenCLContext` must be compiled with `-cl-fast-relaxed-math` so prefill and decode math match the reference numerics.
2. **Environment-Gated Queue Profiling:**
   - In `OpenCLContext::initialize`, do not pass `CL_QUEUE_PROFILING_ENABLE` unconditionally to `clCreateCommandQueue` (recording timestamps on ~240 dispatches/token adds measurable driver overhead). Enable `CL_QUEUE_PROFILING_ENABLE` **only** when `std::getenv("NNOPT_PROFILE")` is non-null and starts with `'1'`; otherwise pass `0` for `cl_command_queue_properties`.
3. **Qualcomm High Perf Hint (`cl_qcom_perf_hint`):**
   - In `OpenCLContext::initialize`, unless `NNOPT_NO_PERF_HINT` is set to `'1'`, dynamically resolve `clSetPerfHintQCOM` (signature `cl_int (CL_API_CALL *)(cl_context, cl_uint)`) via `clGetExtensionFunctionAddressForPlatform(platform_, "clSetPerfHintQCOM")`, falling back to `dlsym(RTLD_DEFAULT, "clSetPerfHintQCOM")`. If found, invoke `fn(context_, 0x40C3)` (`CL_PERF_HINT_HIGH_QCOM = 0x40C3`).
4. **Persistent Program Binary Cache (`kernel_cache/`):**
   - Cache compiled OpenCL program binaries in directory `"kernel_cache"` (created with `mkdir("kernel_cache", 0755)` if missing).
   - Compute a 64-bit FNV-1a hash (offset basis `0xcbf29ce484222325ULL`, prime `0x100000001b3ULL`) over `source + "|" + effective_options + "|" + device_name + "|" + driver_version` (`CL_DEVICE_NAME` and `CL_DRIVER_VERSION`), and format the filename as `kernel_cache/%016llx.bin`.
   - On cache hit, load with `clCreateProgramWithBinary` + `clBuildProgram`. On miss or build failure, fall back to `clCreateProgramWithSource` + `clBuildProgram` and save the resulting binary (`CL_PROGRAM_BINARY_SIZES` / `CL_PROGRAM_BINARIES`) when `CL_PROGRAM_NUM_DEVICES == 1`.

---

### 2. Adreno `image2d_t` Texture-Cache GEMV & Vec4 SwiGLU Kernels (`kernels/block_fused.cl`)

Adreno 6xx GPUs have a dedicated texture fetch unit and L1 texture cache with significantly higher streaming bandwidth than global buffer loads. Under `#ifdef USE_FP16`, wrap 2D weight buffers `[N, K]` (row-major fp16, where `K % 4 == 0`) as read-only 2D images:
- `cl_image_format`: `image_channel_order = CL_RGBA`, `image_channel_data_type = CL_HALF_FLOAT` (each pixel holds 4 contiguous `half` values along `K`).
- `cl_image_desc`: `image_type = CL_MEM_OBJECT_IMAGE2D`, `image_width = K / 4` (`K_PIX`), `image_height = N`, `image_row_pitch = 0`, `buffer = weight_buf` (zero-copy buffer-backed image created via `clCreateImage(ctx, CL_MEM_READ_ONLY, &fmt, &desc, nullptr, &err)`).
- In `kernels/block_fused.cl` (inside `#ifdef USE_FP16`), define:
  ```c
  __constant sampler_t kImgSampler =
      CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
  ```

Implement the following kernels in `kernels/block_fused.cl` with `WG_SIZE = 64` (`__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))`):

#### A. `gemv_m1_k576_no4_img`
- **Signature:** `(__global const half* x, __read_only image2d_t W_img, __global half* out, const int N)`
- **Threading:** 4 output rows per workgroup (`n_base = (int)get_group_id(0) * 4`, `tid = (int)get_local_id(0)`). If `n_base >= N`, return. Launch with `gws = (N / 4) * 64`, `lws = 64`.
- **Accumulation (`K = 576 = 144` vec4 pixels = 2 waves of 64 + 16-thread tail):**
  - Declare `float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;`.
  - **2 full waves** (`#pragma unroll` `for (int j = 0; j < 2; ++j)`):
    - `x_off = j * (WG_SIZE * 4) + tid * 4`
    - `pix = j * WG_SIZE + tid`
    - `float4 xv = vload_half4(0, x + x_off);`
    - Read 4 weight pixels from `W_img` at `(int2)(pix, n_base + 0..3)` using `convert_float4(read_imageh(W_img, kImgSampler, (int2)(pix, n_base + r)))`.
    - Accumulate `acc0 += dot(xv, w0); acc1 += dot(xv, w1); acc2 += dot(xv, w2); acc3 += dot(xv, w3);`.
  - **Tail** (`if (tid < 16)`):
    - `x_off = 2 * (WG_SIZE * 4) + tid * 4`
    - `pix = 2 * WG_SIZE + tid` (pixels `128..143`)
    - Same `vload_half4`, four `read_imageh` calls at `(int2)(pix, n_base + 0..3)`, and `acc0..3 += dot(xv, w0..3)`.
- **Reduction & Write:**
  - Store `acc0..3` into `__local float partial[4][WG_SIZE];`, `barrier(CLK_LOCAL_MEM_FENCE);`.
  - Tree reduce for `int s = WG_SIZE >> 1; s > 0; s >>= 1`: when `tid < s`, add `partial[r][tid] += partial[r][tid + s]` for `r = 0..3`, followed by `barrier(CLK_LOCAL_MEM_FENCE);`.
  - When `tid == 0`, write `vstore_half(partial[r][0], 0, out + n_base + r)` for `r = 0..3`.

#### B. `fused_oproj_residual_m1_no4_img`
- **Signature:** `(__global const half* attn_out, __read_only image2d_t Wo_img, __global half* residual, const int H)`
- **Threading & Accumulation:** Identical 4-outputs-per-WG layout (`n_base = get_group_id(0) * 4`, `gws = (H / 4) * 64`, `lws = 64`), identical `K = 576` inner loop (2 full waves + `tid < 16` tail), and identical `__local float partial[4][WG_SIZE]` tree reduction as `gemv_m1_k576_no4_img`.
- **Fused Residual Epilogue:** When `tid == 0`, load `float r0..r3 = vload_half(n_base + 0..3, residual)` and store `vstore_half(r_k + partial[k][0], n_base + k, residual)` for `k = 0..3`.

#### C. `fused_down_residual_m1_no4_img`
- **Signature:** `(__global const half* mlp_in, __read_only image2d_t Wd_img, __global half* residual, const int H)`
- **Threading:** 4 output rows per workgroup (`n_base = get_group_id(0) * 4`, `gws = (H / 4) * 64`, `lws = 64`).
- **Accumulation (`K = INTER = 1536 = 384` vec4 pixels = 6 full waves of 64, no tail):**
  - `#pragma unroll` `for (int j = 0; j < 6; ++j)`:
    - `x_off = j * (WG_SIZE * 4) + tid * 4`
    - `pix = j * WG_SIZE + tid`
    - `float4 xv = vload_half4(0, mlp_in + x_off);`
    - Read 4 weight pixels from `Wd_img` at `(int2)(pix, n_base + 0..3)` via `convert_float4(read_imageh(...))` and accumulate `acc0..3 += dot(xv, w0..3)`.
- **Reduction & Fused Residual Epilogue:** Same `__local float partial[4][WG_SIZE]` tree reduction and `residual` in-place update as `fused_oproj_residual_m1_no4_img`.

#### D. `fused_gate_up_silu_m1_v4_img` (and buffer fallback `fused_gate_up_silu_m1_v4`)
- **Why 1 output per WG:** Dual projections (`gate` + `up`) with 4 outputs per WG would require 8 fp32 accumulators and 8 `float4` weight reads per wave, causing register spilling on Adreno 6xx. Keep **1 output row per workgroup** (`c = (int)get_group_id(0)`, `tid = (int)get_local_id(0)`, return if `c >= INTER`, `gws = INTER * 64`, `lws = 64`).
- **Signature (`_v4_img`):** `(__global const half* x, __read_only image2d_t Wg_img, __read_only image2d_t Wu_img, __global half* out, const int INTER)`
- **Accumulation (`K = 576 = 144` vec4 pixels):**
  - `float gate_acc = 0.0f, up_acc = 0.0f;`
  - 2 full waves (`#pragma unroll` `for (int j = 0; j < 2; ++j)`):
    - `x_off = j * (WG_SIZE * 4) + tid * 4`, `pix = j * WG_SIZE + tid`
    - `float4 xv = vload_half4(0, x + x_off);`
    - `float4 wgv = convert_float4(read_imageh(Wg_img, kImgSampler, (int2)(pix, c)));`
    - `float4 wuv = convert_float4(read_imageh(Wu_img, kImgSampler, (int2)(pix, c)));`
    - `gate_acc += dot(xv, wgv); up_acc += dot(xv, wuv);`
  - Tail (`if (tid < 16)`):
    - `x_off = 2 * (WG_SIZE * 4) + tid * 4`, `pix = 2 * WG_SIZE + tid`
    - Same `xv`, `wgv`, `wuv` loads and `dot` accumulations.
- **Reduction & SwiGLU Write:**
  - Store into `__local float ls_gate[WG_SIZE]` and `__local float ls_up[WG_SIZE]`, `barrier(CLK_LOCAL_MEM_FENCE);`.
  - Tree reduce for `int s = WG_SIZE >> 1; s > 0; s >>= 1`: when `tid < s`, `ls_gate[tid] += ls_gate[tid + s]; ls_up[tid] += ls_up[tid + s];` with `barrier(CLK_LOCAL_MEM_FENCE);`.
  - When `tid == 0`: `float g = ls_gate[0]; vstore_half((g / (1.0f + exp(-g))) * ls_up[0], c, out);`.
- Also provide the buffer-backed vec4 variant `fused_gate_up_silu_m1_v4` (`vload_half4(0, w_gate + c * H + off)` and `vload_half4(0, w_up + c * H + off)`) as fallback.

---

### 3. Layer Wiring (`attention.*`, `mlp.*`, `lm_head.*`)

1. **Attention (`src/layers/attention.h`, `src/layers/attention.cpp`):**
   - In `Attention::initialize()` under `#ifdef NNOPT_USE_FP16`, create kernels `gemv_m1_k576_no4_img` and `fused_oproj_residual_m1_no4_img` from `block_fused_prog_`, and wrap `wq_` (`[576, 576]`), `wk_` (`[192, 576]`), `wv_` (`[192, 576]`), and `wo_` (`[576, 576]`) into 2D `CL_RGBA`/`CL_HALF_FLOAT` images (`width = 144`, `height = N`). Release them in `~Attention()`.
   - In `Attention::forward_decode_into_residual`:
     - **Step 1 (QKV):** Do **not** fuse Q, K, and V into a single 3-image branched kernel (branching over image arguments degrades texture cache flow). Instead, dispatch `gemv_m1_k576_no4_img` **three separate times** back-to-back:
       1. `(x, wq_img_, q, Q_DIM)` with `gws = (Q_DIM / 4) * 64, lws = 64`
       2. `(x, wk_img_, k, KV_DIM)` with `gws = (KV_DIM / 4) * 64, lws = 64`
       3. `(x, wv_img_, v, KV_DIM)` with `gws = (KV_DIM / 4) * 64, lws = 64`
     - **Steps 2 & 3:** Keep `fused_rope_kvwrite_m1` and `fused_decode_attn_m1` unchanged.
     - **Step 4 (O-proj + residual):** Dispatch `fused_oproj_residual_m1_no4_img(attn_out, wo_img_, residual, H)` with `gws = (H / 4) * 64, lws = 64`.

2. **MLP (`src/layers/mlp.h`, `src/layers/mlp.cpp`):**
   - In `Mlp::initialize()` under `#ifdef NNOPT_USE_FP16`, create `fused_gate_up_silu_m1_v4_img`, `fused_gate_up_silu_m1_v4`, and `fused_down_residual_m1_no4_img`. Wrap `w_gate_` and `w_up_` as 2D images of `(width = H / 4 = 144, height = INTER = 1536)` and `w_down_` as a 2D image of `(width = INTER / 4 = 384, height = H = 576)`. Release in `~Mlp()`.
   - In `Mlp::forward_decode_into_residual`:
     - **Step 1:** Dispatch `fused_gate_up_silu_m1_v4_img(x, w_gate_img_, w_up_img_, act, I)` with `gws = I * 64, lws = 64` (falling back to `_v4` then scalar if images are unavailable).
     - **Step 2:** Dispatch `fused_down_residual_m1_no4_img(act, w_down_img_, residual, H)` with `gws = (H / 4) * 64, lws = 64`.

3. **Row-Tiled `LmHead` (`src/layers/lm_head.h`, `src/layers/lm_head.cpp`):**
   - `w_` (`model.embed_tokens.weight`) has shape `[V = 49152, H = 576]`. On Adreno 6xx, `CL_DEVICE_IMAGE2D_MAX_HEIGHT` is `16384`, so a single 2D image of height `49152` will fail `clCreateImage`.
   - In `LmHead::initialize()` under `#ifdef NNOPT_USE_FP16`:
     - Query `CL_DEVICE_IMAGE2D_MAX_WIDTH` and `CL_DEVICE_IMAGE2D_MAX_HEIGHT` (`img_max_w`, `img_max_h`).
     - If `V <= img_max_h`, wrap `w_` in a single image. Otherwise, tile `w_` along the row dimension using `TILE_H = (int)img_max_h - ((int)img_max_h % 4)` (`16384` rows per tile, giving 3 tiles of `16384` rows for `V = 49152`).
     - Each row is `row_bytes = H * 2 = 1152` bytes (divisible by 128, satisfying Adreno sub-buffer alignment). For each tile (`row_offset`, `tile_n`), create a read-only sub-buffer of `w_` (`origin = row_offset * row_bytes`, `size = tile_n * row_bytes`) via `clCreateSubBuffer`, then create a 2D `CL_RGBA`/`CL_HALF_FLOAT` image of `width = H / 4 = 144`, `height = tile_n` over that sub-buffer.
   - In `LmHead::forward(queue, hidden, M)` when `M == 1` and the tiled image path is ready:
     - For each tile, create a sub-buffer `out_sub` over `out` (`origin = tile.row_offset * sizeof(nnopt_storage_t)`, `size = tile.row_count * sizeof(nnopt_storage_t)`), dispatch `gemv_m1_k576_no4_img(hidden, tile.image, out_sub, tile.row_count)` with `gws = (tile.row_count / 4) * 64, lws = 64`, and release `out_sub`.

---

### 4. On-GPU Argmax & Greedy Decode Fast Path (`kernels/argmax.cl`, `model.h`, `model.cpp`)

Eliminate the per-token 98 KB (`49152 * 2` bytes) blocking fp16 logits readback + CPU fp16-to-fp32 conversion during greedy decoding:

1. **New Kernel File `kernels/argmax.cl`:**
   - Support `#ifdef USE_FP16` (`half` storage with `vload_half(i, p)`) and fp32 fallback.
   - Default `WG_SIZE = 64`, `NUM_WG = 32`.
   - **`argmax_partial`** (`reqd_work_group_size(WG_SIZE, 1, 1)`):
     - Signature: `(__global const storage_t* logits, __global float* scratch_v, __global int* scratch_i, const int n, const int valid_n, const int row_off)`
     - `tid = get_local_id(0)`, `wg = get_group_id(0)`, `total_threads = WG_SIZE * NUM_WG` (`2048`), `gid = wg * WG_SIZE + tid`, `limit = (valid_n > 0 && valid_n < n) ? valid_n : n`.
     - Initialize `float best_v = -INFINITY; int best_i = 0;`.
     - Loop `for (int j = gid; j < limit; j += total_threads)`: load `float v = LOAD(logits, row_off + j); if (v > best_v) { best_v = v; best_i = j; }`.
     - Store `best_v` and `best_i` into `__local float ls_v[WG_SIZE]` and `__local int ls_i[WG_SIZE]`, barrier, then tree-reduce for `int s = WG_SIZE >> 1; s > 0; s >>= 1`:
       ```c
       if (tid < s) {
         if (ls_v[tid + s] > ls_v[tid]) {
           ls_v[tid] = ls_v[tid + s];
           ls_i[tid] = ls_i[tid + s];
         }
       }
       barrier(CLK_LOCAL_MEM_FENCE);
       ```
     - When `tid == 0`, write `scratch_v[wg] = ls_v[0]; scratch_i[wg] = ls_i[0];`.
   - **`argmax_final`** (`reqd_work_group_size(NUM_WG, 1, 1)`):
     - Signature: `(__global const float* scratch_v, __global const int* scratch_i, __global int* out_idx, __global int* cur_token, const int write_off)`
     - 1 workgroup of `NUM_WG = 32` threads loads `scratch_v[tid]` and `scratch_i[tid]` into `__local` arrays of size 32, performs the same tree reduction (`for (int s = NUM_WG >> 1; s > 0; s >>= 1)` with `ls_v[tid + s] > ls_v[tid]`), and when `tid == 0` writes `out_idx[write_off] = ls_i[0]; cur_token[0] = ls_i[0];`.

2. **`Model::forward_decode_greedy(int32_t token_id, int start_pos)` (`model.h`, `model.cpp`):**
   - Lazily build `kernels/argmax.cl` (with `-DNNOPT_USE_FP16=1 -DUSE_FP16=1 -DNUM_WG=32` under fp16) and allocate persistent buffers `argmax_scratch_v_` (`32 * sizeof(float)`), `argmax_scratch_i_` (`32 * sizeof(int)`), `argmax_out_idx_` (`sizeof(int)`), and `argmax_cur_token_` (`sizeof(int)`). Release them in `~Model()`.
   - Run token embedding (`seq_len = 1`), all 30 transformer layers (`forward_decode` / `forward_decode_into_residual`), `final_norm_->forward_decode`, and `lm_head_->forward(queue, final_hidden, 1)` to produce `logits_buf`.
   - Enqueue `argmax_partial` (`gws = 32 * 64, lws = 64` with `n = V, valid_n = V, row_off = 0`) followed by `argmax_final` (`gws = 32, lws = 32` with `write_off = 0`).
   - Read back the single `int32_t` winner from `argmax_out_idx_` via blocking `clEnqueueReadBuffer(queue, argmax_out_idx_, CL_TRUE, 0, sizeof(int32_t), &out, 0, nullptr, nullptr)`, release `logits_buf`, and return `out`.
   - In `Model::generate`, keep the prefill step (`i = 0`) unchanged, and in the decode loop (`for (int i = 1; i < max_new_tokens; i++)`), use `forward_decode_greedy(next_token, start_pos)` whenever `sampler_config.temperature <= 0.0f && sampler_config.repetition_penalty == 1.0f` (falling back to `forward` + `sampler.sample` only if `forward_decode_greedy` returns `< 0`).