# OSCAR multi-batch Nsight sweep

Profile OSCAR INT2 KV across **multiple (batch, seq-len) combinations** in one run.
Counterpart of the single-batch microbench (`profiling/single-batch/`): same OSCAR
config and kernels, but it sweeps a grid and emits **one self-contained trace + log
per combo**, so each `(batch, seq)` can be opened and read independently.

It is a **wrapper** around `sglang.bench_one_batch` — no project source is modified.

---

## 1. Run a sweep

```bash
# cartesian product of BATCHES x SEQS
BATCHES="1 4 8" SEQS="4096 16384 65536" \
  bash profiling/multi-batch/profile_oscar_multi.sh
```

Each `(B, S)` combo runs as its own process under nsys and writes, in
`profiling/multi-batch/traces/`:

| file | content |
|---|---|
| `oscar_b<B>_s<S>.nsys-rep` | whole-run trace (prefill + decode) for that combo |
| `oscar_b<B>_s<S>.log` | sglang stdout + `Prefill.` / `Decode. median` latency |

A failing combo (e.g. OOM) is logged and the sweep **continues**; a summary table
of OK / OOM / FAILED per combo is printed at the end.

Preview the grid without running anything:
```bash
BATCHES="1 4 8" SEQS="4096 16384" DRY_RUN=1 bash profiling/multi-batch/profile_oscar_multi.sh
```

---

## 2. Knobs (prepend as env)

| var | default | meaning |
|---|---|---|
| `BATCHES` | `1 4 8` | batch sizes to sweep (space-separated) |
| `SEQS` | `4096 16384 65536` | input/context lengths (space-separated) |
| `OUTPUT_LEN` | `2` | decode steps + 1 (`2` => prefill + a single decode step) |
| `GPU` | `3` | CUDA device index |
| `QUANT` | `oscar` | `oscar` (INT2 + rotation) or `bf16` baseline |
| `GRAPH` | `off` | `off` = eager (every kernel visible); `on` = realistic latency (graph replay; decode shows as one opaque graph) |
| `MEM_FRAC` | `0.8` | static-pool fraction — see Memory below |
| `SKIP_EXISTING` | `0` | `1` => skip combos whose `.nsys-rep` already exists (resume a sweep) |
| `DRY_RUN` | `0` | `1` => print the planned grid and exit |
| `MODEL` / `OUTDIR` / `CHUNKED_PREFILL` | … | as in the single-batch script |

---

## 3. Memory: the two competing budgets (read before large sweeps)

Multi-batch makes both KV and activation grow with **B**, so combos can OOM where
single batch did not. There are two separate budgets:

1. **KV pool** must hold `B*(SEQ+OUTPUT_LEN)` tokens. After boot each log prints
   `max_total_num_tokens=` — it must exceed `B*(SEQ+OUTPUT_LEN)`, else the combo
   errors. **Raise `MEM_FRAC`** to grow the pool.
2. **Prefill activation** scales with the total prefill tokens `B*SEQ` (one forward
   over the whole batch). Large `B*SEQ` can OOM in torch (e.g. the MLP SwiGLU
   buffer) even when the pool fits. **Lower `MEM_FRAC`** frees activation room but
   shrinks the pool.

These pull in opposite directions, so very large `B*SEQ` may not fit on one GPU.
Tune `MEM_FRAC` per sweep; for activation-OOM combos also try
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. (This is the same single-batch
lesson, amplified by batch.)

---

## 4. Read the traces

The single-batch reporter is generic and works on these traces too:

```bash
bash profiling/single-batch/report.sh profiling/multi-batch/traces/oscar_b4_s16384.nsys-rep
```

Compare a metric across the grid, e.g. decode median latency per combo:
```bash
grep -H "Decode.  median" profiling/multi-batch/traces/oscar_b*_s*.log
```

Per-instance decode attention kernel timeline for one combo:
```bash
nsys stats --report cuda_gpu_trace --format table \
  profiling/multi-batch/traces/oscar_b8_s16384.nsys-rep | grep _fwd_grouped_kernel_stage1_quant_int2
```

For the kernel-name → OSCAR-op map and the per-layer decode pattern, see
`profiling/single-batch/SINGLE_BATCH_GUIDE.md` (the kernels are identical; only
batch/seq scale).

---

## 5. Why batch matters for the OSCAR story

At batch=1 long context, decode is **weight-bandwidth bound** (weights ≫ KV), so KV
quantization and rotation overhead are hidden. Increasing **B** amortizes the weight
read across the batch while KV/attention traffic scales with `B*SEQ` — so the
attention path (INT2 dequant + HP/INT2 split + rotation) becomes a larger share of
decode. Sweeping `(B, SEQ)` is how you find the regime where the OSCAR machinery's
cost actually shows up in latency. Run `QUANT=oscar` and `QUANT=bf16` over the same
grid to isolate what INT2 buys vs. what the rotation/split gives back.
