# SmolLM2-135M Greedy Decode Optimization on Adreno 620 (Android OpenCL)

This benchmark optimizes greedy decode throughput for Hugging Face's `SmolLM2-135M-Instruct` running in pure C++/OpenCL fp16 on a physical Motorola Razr 2020 (Adreno 620 GPU).

## Overview
- **Base Throughput**: 20.35 tokens/second
- **Reference Throughput**: 24.01 tokens/second (1.193x speedup)
- **Target Hardware**: Motorola Razr 2020 (Snapdragon 765G, Adreno 620 GPU, Android 11)
- **Key Bottlenecks**: Per-decode-token buffer reallocation and CPU trig computation in RoPE tables, OpenCL 1.2 default compiler scheduling, and software-emulated exponential math in hot kernel activations.

## Task Deliverables
- Patch: `/workspace/submission/patch.diff`
- Summary: `/workspace/submission/summary.md`
