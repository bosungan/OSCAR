# OSCAR server-backed Nsight sweep — multi-batch + chunked prefill

**Why this directory exists.** The other two profiling tiers drive
`sglang.bench_one_batch`, which calls the model's low-level `extend()` directly
and **bypasses the Scheduler**. Two consequences we hit in practice:

1. **`--chunked-prefill-size` is ignored** — chunking lives in the Scheduler, so
   `bench_one_batch` runs the *entire* prefill in one forward. The prefill
   activation then scales with `B*SEQ` and OOMs past **~256k total tokens** on a
   48 GB GPU no matter how `MEM_FRAC` is tuned (the binding limit is the MLP
   activation, not the KV pool). Proof: at 256k the OOM was a `[262144 × 12288]`
   SwiGLU buffer = the *full* sequence, not a 4096 chunk.
2. **No real batching scheduler** — it is single-process static.

This tier instead drives **`sglang.bench_one_batch_server`**, which **launches a
real server (HTTP + Scheduler)** and benchmarks one batch through it. With **zero
source changes** that gives us the two things the static path can't:

| capability | how |
|---|---|
| **multi-batch** | `--batch-size`/`--input-len` go through the real continuous-batching Scheduler |
| **chunked prefill** | `--chunked-prefill-size` is **honored** → prefill activation is bounded by the *chunk* size, not `B*SEQ` → **256k / 512k / 1M context becomes feasible** |

It is still a **wrapper**: nothing under `sglang-research/` is modified. The
script only builds the CLI per `(BATCH, SEQ)` combo and runs it under nsys.

---

## The three profiling tiers

| dir | driver | scheduler? | chunked prefill | best for |
|---|---|---|---|---|
| `profiling/single-batch/` | `bench_one_batch` | ❌ | ❌ (ignored) | one clean (B=1, S) shape, kernel-level attribution |
| `profiling/multi-batch/` | `bench_one_batch` | ❌ | ❌ (ignored) | `B×S` grid, static, **≤ ~256k total tokens** |
| **`profiling/server/`** (this) | **`bench_one_batch_server`** | ✅ | ✅ **honored** | **long context (256k–1M), chunked prefill, scheduler-realistic** |

---

## 1. Run one (batch, seq)

One invocation = one `(batch, seq)`. Positional args, like the single-batch script:

```bash
# usage: profile_oscar_server.sh BATCH SEQ [OUTPUT_LEN] [GPU]
bash profiling/server/profile_oscar_server.sh 1 16384            # smoke test
bash profiling/server/profile_oscar_server.sh 1 524288 8 3       # 512k on GPU 3
QUANT=bf16 bash profiling/server/profile_oscar_server.sh 1 262144
CHUNK=8192 MEM_FRAC=0.9 bash profiling/server/profile_oscar_server.sh 4 65536
```

It **launches its own server**, runs the one batch through it with sglang's
**built-in per-stage profiler** (`--profile-by-stage`), then tears the server
down. Outputs land in `profiling/server/traces/<QUANT>_<TAG>/<timestamp>/`:

| file | content |
|---|---|
| `...-TP-0-EXTEND.trace.json.gz` | **prefill** stage (chunked-prefill forwards) — Chrome/torch trace |
| `...-TP-0-DECODE.trace.json.gz` | **decode** stage (per-step decode kernels) — Chrome/torch trace |
| `run.log` | server stdout + bench latency + `max_total_num_tokens` |

Open a `.trace.json.gz` in **Perfetto** (https://ui.perfetto.dev, drag-drop).
The GPU row is the kernel timeline; search (`/`) for
`_fwd_grouped_kernel_stage1_quant_int2` to jump to the INT2 decode attention.

---

## 2. Knobs

Positional: `BATCH` `SEQ` `[OUTPUT_LEN]` `[GPU]`. The rest are env vars:

| var | default | meaning |
|---|---|---|
| `OUTPUT_LEN` (arg 3) | `8` | decode tokens to **generate** (the workload) |
| `GPU` (arg 4) | `0` | CUDA device |
| `PROFILE_STEPS` | `4` | decode steps to **record** into the DECODE trace (`<= OUTPUT_LEN`) |
| `START_STEP` | _(unset)_ | decode step to **start recording at**. Unset = from step 0 (includes the warmup/flush outlier). Set e.g. `8` for steady-state; needs `OUTPUT_LEN >= START_STEP + PROFILE_STEPS` |
| `CHUNK` | `8192` | **`--chunked-prefill-size`** — the whole point; caps activation |
| `QUANT` | `oscar` | `oscar` or `bf16` |
| `GRAPH` | `off` | `off` = eager (kernels visible); `on` = realistic latency |
| `MEM_FRAC` | `0.85` | **RAISE this** (opposite of single-batch) — chunked prefill removes the activation spike, so the KV pool is the only constraint |
| `MAX_RUNNING` | `=BATCH` | `--max-running-requests` |
| `PORT` | `31000` | server port |
| `TAG` | `b<B>_s<S>_chunk<CHUNK>` | output filename stem |

---

## 3. Memory: why 512k now fits (and why MEM_FRAC goes UP)

With chunked prefill the peak activation is bounded by **`CHUNK` tokens**, not
`B*SEQ`. So at 512k decode the budget is just:

```
weights 15.3 GB  +  INT2 KV pool ~11.4 GB (512k tok)  +  ~CHUNK-sized activation  ≈ 27 GB  → fits 48 GB
```

The **KV pool** is now the binding constraint, so **raise `MEM_FRAC`** to grow it
(0.85+). After boot, check `max_total_num_tokens=` in the log — it must exceed
`B*(SEQ+OUTPUT_LEN)`. Contrast with the single-batch script, where you had to
*lower* `MEM_FRAC` to free activation room — that tension is gone here.

Rough feasibility (B=1, A6000 48 GB), `CHUNK=8192`:

| context | KV pool (INT2) | total | fits? |
|---|---|---|---|
| 256k | ~5.6 GB | ~22 GB | ✅ |
| 512k | ~11.4 GB | ~27 GB | ✅ |
| 1M | ~22 GB | ~38 GB | ✅ (raise MEM_FRAC) |

---

## 4. Why the built-in profiler, not nsys

We tried nsys first and it **could not capture the GPU work**:
`bench_one_batch_server` launches the server (Scheduler / TP worker) in a
**child process** via `multiprocessing`, and the model forward runs there. nsys
profiled the parent (HTTP client) only; even with `--trace-fork-before-exec=true`
the resulting `.nsys-rep` came back `SKIPPED: does not contain CUDA kernel data`.

So this script uses sglang's **built-in `--profile --profile-by-stage`**, which
runs torch's profiler **inside the server process** via the `/start_profile`
endpoint — robust to the multiprocessing boundary. Verified: the EXTEND/DECODE
traces contain the full GPU kernel timeline (e.g. `_pretransformed_int2_set_kv_clip`
in prefill, the GEMM/flush/`_fwd_grouped_kernel_*` kernels in decode).

Trade-off: output is a **Chrome/torch trace** (Perfetto), not `.nsys-rep`. You
lose the nsys GUI / `nsys stats` CLI, but you gain reliable capture + automatic
prefill/decode split. (An nsys "attach mode" — launch `sglang.launch_server`
under nsys, drive it with `--base-url` — is possible but has the same child-process
risk, so we don't rely on it.)

---

## 5. What's the same vs the static tiers

- **Decode kernels are identical** — same model_runner, triton INT2 decode path,
  OSCAR rotation kernels. Match `B` and you get the same per-layer decode pattern
  documented in `profiling/single-batch/SINGLE_BATCH_GUIDE.md`.
- **What's new in the trace**: the **prefill now appears as multiple chunked
  forwards** (CHUNK tokens each) instead of one giant forward — which is exactly
  how production serving prefills, so it's *more* realistic, not less.
- **What the scheduler adds**: with `--max-running-requests=B` and one batch,
  behavior stays close to static; at larger loads you'd also see mixed
  prefill+decode steps (a regime the static tiers can't show).

Read the traces in **Perfetto** (https://ui.perfetto.dev → drag-drop the
`...-DECODE.trace.json.gz` / `...-EXTEND.trace.json.gz`). The single-batch
`report.sh` does NOT apply here (it's an `nsys stats` wrapper; these are Chrome
traces). For per-kernel sums, use Perfetto's query/“Slices” view, or load the
trace with `torch.profiler`/`tensorboard`. The kernel names match the
single-batch catalog in `SINGLE_BATCH_GUIDE.md`.
