# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

OSCAR is a 2-bit (INT2) KV-cache quantization method. The research contribution lives in a small,
self-contained pipeline under [rotation/](rotation/); the two `sglang-*` directories are forks of
SGLang that implement the dump and inference paths. The pipeline has three phases:

1. **Dump** — capture post-RoPE Q/K/V activations on a calibration set (runs an sglang server, sends prompts with `max_tokens=1`).
2. **Rotation** — eigendecompose attention-aware K/V covariances and compose per-layer orthogonal rotations.
3. **Eval** — serve the model with `--kv-cache-dtype int2` + the rotation, then run a benchmark (GPQA / LiveCodeBench).

`README.md` is the authoritative user-facing reference (results tables, env knobs, the spectral-covariance math). Read it before changing pipeline behavior.

## Repository layout

- [rotation/](rotation/) — **the OSCAR code you will most often edit.** Per-model subdirs (`qwen3-8B/`, `qwen3-32B/`, `qwen3-4B-thinking-2507/`, `MiniMax-M2.7/`, `GLM-4.7/`) each hold three thin wrapper scripts (`save_qkv_*.sh`, `compute_rotation.sh`, `eval_gpqa.sh`) that set per-model defaults and `exec` into the generic drivers.
  - [rotation/compute_kv_rotation.py](rotation/compute_kv_rotation.py) — phase 2 core: covariance estimation, eigendecomposition, rotation composition.
  - [rotation/eval_oscar_gpqa.sh](rotation/eval_oscar_gpqa.sh) — generic eval driver (the per-model `eval_gpqa.sh` scripts call this).
  - [rotation/_eval_runner/](rotation/_eval_runner/) — `dump_gpqa_prompts.py` (phase 1 client) and `run_simple_eval.py` (phase 3 client, drives `third_party/simple_evals`).
  - [rotation/_dump_compat/](rotation/_dump_compat/) and [rotation/_triton_per_rank/](rotation/_triton_per_rank/) — `sitecustomize.py` shims auto-loaded via `PYTHONPATH` (see "Two PYTHONPATH shims" below).
- `sglang-research/` — submodule; the **eval-side** SGLang fork that implements INT2 KV + OSCAR rotation loading. Installed editable (`pip install -e sglang-research/python`).
- `sglang-dump-qkv/` — vendored **older** SGLang fork used only for phase 1 dumping; loaded via `PYTHONPATH`, not installed.
- `third_party/simple_evals/` — submodule (openai/simple-evals), the benchmark harness used at eval time.
- `materials/` — README images only.

Clone with `--recursive` (there are git submodules). Branches `zhongzhu/*` carry model-/backend-specific variants (llama.cpp port, VL models, hybrid models, Lloyd-Max quant); `main` is the SGLang INT2 pipeline.

## Environment

One conda env (`oscar`, Python 3.12) serves all phases. Requires CUDA 12.8/12.9 with matching `nvcc` and a CUDA-capable GPU (H100-class). There is no test suite, linter, or build step — this is a research pipeline driven entirely by the shell scripts.

```bash
conda create -n oscar python=3.12 -y && conda activate oscar
pip install -e sglang-research/python   # + matching flashinfer / sgl_kernel wheels for your CUDA
```

## Running the pipeline (Qwen3-8B, single H100, ~20 min end to end)

```bash
bash rotation/qwen3-8B/save_qkv_8b.sh        # phase 1 → GPQA/seq<T>_prompt<N>_group<G>/qkv_dumps/
bash rotation/qwen3-8B/compute_rotation.sh   # phase 2 → .../rotations/{k,v}_rotation_qqt_r_h_pbr.pt
ROT_DIR=$(ls -1d rotation/qwen3-8B/GPQA/seq*_prompt*_group*/rotations | tail -1) \
  bash rotation/qwen3-8B/eval_gpqa.sh        # phase 3 → .../_eval_gpqa_oscar/
```

All three scripts are overridden via leading env vars (`bash rotation/<model>/save_qkv_<model>.sh DUMP_KVCACHE_TOKENS=10000`). Common knobs: `MODEL`, `TP_SIZE`/`GPU`/`CUDA_VISIBLE_DEVICES`, `DUMP_KVCACHE_TOKENS`, `GROUP_SIZE`, `HF_HOME` (defaults to `/shared/huggingface` — set to `$HOME/.cache/huggingface` on a fresh box), and for eval `ROT_DIR`, `RUN_DIR`, `K_CLIP`/`V_CLIP`, `MAX_NEW_TOKENS`, `N_REPEATS`. The full table is in `README.md`.

Phase 2 also takes `METHOD` (default `qqt_sst` = the calibrated recipe; `hadamard` = data-free, no dump needed) and `COMPOSITION` (default `r_h_pbr`).

## Key conventions that span multiple files

- **The calibration directory name encodes its config and is the join key across phases.** Phase 1 writes to `<model>/<DATASET>/latest/` then renames to `seq<TOKENS>_prompt<N>_group<G>/` after counting prompts. Phase 2 auto-discovers the newest such dir by mtime; phase 3 reads rotations from `<that dir>/rotations/`. Don't hand-rename these dirs.

- **The rotation checkpoint schema is a contract** between [compute_kv_rotation.py](rotation/compute_kv_rotation.py) (`torch.save`, naming `{target}_rotation_{hessian}_{composition}.pt`) and `load_oscar_rotations` in `sglang-research/python/sglang/srt/mem_cache/memory_pool.py` (`SGLANG_OSCAR_{K,V}_ROTATION_PATH` point at these files). Per-layer fp32 `(head_dim, head_dim)` orthogonal matrices. Changing the saved structure requires updating the loader too.

- **OSCAR inference is configured purely via `SGLANG_*` env vars**, not CLI flags — set in `eval_oscar_gpqa.sh` and consumed across `memory_pool.py`, `unified_kv_pool.py`, `model_runner_kv_cache_mixin.py`, `pool_configurator.py`. The mixed-precision scheme keeps a BF16 sink (`SGLANG_MIXED_KV_PREFIX_TOKENS`) + recent window (`SGLANG_MIXED_KV_RECENT_TOKENS`); everything else is INT2 in `--kv-cache-quant-group-size` groups along head-dim. `SGLANG_ENABLE_MIXED_KV_WINDOWS=1` turns the whole feature on.

- **Attention backends differ by phase:** dump uses `triton` prefill+decode (deterministic); eval uses `fa3` prefill + `triton` decode (the INT2 decode path is Triton-only).

- **Two PYTHONPATH shims, auto-activated by being on `PYTHONPATH`** (Python imports `sitecustomize` at startup):
  - `rotation/_dump_compat/` — stubs legacy `sgl_kernel` symbols the old dump fork imports but the new env dropped, and falls back to a pure-PyTorch argmax sampler (dump only needs one token).
  - `rotation/_triton_per_rank/` — routes each TP rank's Triton cache into `rank<N>/` subdirs so parallel workers don't race on shared compiled `.so` files.

## My research Goal 
- My research goal is to found out the inefficiency of inference system of rotation-based ultra low-bit KV cache. I assume that rotation operation might induce some inefficiencies compared to original token-wise group quantization due to the complexity of process. Since SOTA is OSCAR, I'm now trying to figure out inefficiency via profiling