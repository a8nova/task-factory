# Faster prompt processing for Bonsai-1.7B with a real 1-bit matrix-multiply tile

## Task

`src/models/bonsai` runs Bonsai, a Qwen3-style model with 1-bit weights (Q1_0: 1.125 bits per weight, sign bits
plus fp16 scales, never dequantized to a full matrix), on Adreno GPUs in OpenCL. This task uses Bonsai-1.7B
(bundle `bonsai1.7b.nnb`, 28 layers, hidden 2048, FFN 6144).

Today, prompt processing (prefill) runs the prompt through the same single-token GEMV kernels as decode, one
token at a time. The port's own benchmark log records that batching 8 tokens through the GEMV (M=8) was a
regression, because the 1-bit GEMV is limited by instruction issue rather than memory, and concludes:
**a true matrix-multiply tile would win, but it hits the Adreno register cliff.**

Build that tile. Write a 1-bit × fp32-activation GEMM kernel that processes many prompt tokens at once, with
weight reuse across tokens, sized to stay under the Adreno GPU's register limits. Use it for prefill so that
time-to-first-token on a long prompt drops. Decode may keep its existing GEMV path.

Deliverable: your change as a `git diff` against the starter commit, written to `/workspace/submission/patch.diff`,
plus a short `/workspace/submission/summary.md` explaining your tile shape and why it fits the register budget.

## Constraints

- Edit only `src/models/bonsai/kernels/` and `src/models/bonsai/src/`. You may add new kernel files there.
- Do not edit `CMakeLists.txt`, `scripts/`, `reference/`, `weights/`, or any other model's folder. Patches that
  touch them are rejected before building.
- Keep the default token-exact configuration (`BONSAI_STORAGE=fp32`). Don't change the weight format, the
  model's math, or the sampling.
- No network. You can't compile locally: your validation tool builds your patch with the Android NDK and runs it
  on the phone.

## Success criteria

Grading runs your patch with `BONSAI_NNB=bonsai1.7b.nnb` on a **Galaxy S26 (Snapdragon, latest Adreno)** from Google's
Device Run fleet:

1. **Correctness gate.** For every hidden prompt, the generated token IDs must exactly match the starter code's
   tokens, greedy, at every timed length. Anything else (a build failure, a crash, an off-limits edit, different
   tokens) scores 0. This is a token-exact port: reordering a reduction can change tokens, so keep the
   accumulation exact (fp32) and verify with your validation calls.
2. **Speed.** Prefill throughput is measured outside the binary: wall time against prompt length, fitted over
   several long prompts with only a few generated tokens, median across hidden prompts. Speedup = your prefill
   throughput ÷ the starter's on the same phone. The binary's own printed timings are ignored.
3. **Reward** = clip((speedup − 1) / (target − 1), 0, 1). The starter scores 0. The target is the best valid result
   found during calibration (2.0× until calibration sets it). Equal steps in reward cost equal gains in throughput.

Your tools:
- `run_validation` builds and runs your current patch on the phone. It returns build errors, prefill throughput, and
  whether your tokens match on the development prompts.
- `submit` grades your final patch on the hidden prompts.
- `time_left` reports your remaining budget.

A validation call takes several minutes, because a phone has to be provisioned for each one. Plan your calls.
