#!/usr/bin/env python
# ============================================================================
# _run_bench_server_ncu.py — launcher for sglang.bench_one_batch_server tuned
# for NCU kernel capture of the REAL long-context decode. Two runtime patches,
# NO edits to sglang-research source:
#
#   1. DEFAULT_TIMEOUT -> OSCAR_BENCH_TIMEOUT (long chunked prefill is slow).
#   2. Neuter `/flush_cache` raise_for_status: the OSCAR int2 mixed-KV server
#      returns HTTP 400 on /flush_cache, which crashes run_one_case BEFORE the
#      real benchmark request ever runs. For a single-case profiling run, the
#      cache flush is unnecessary, so we make that one call non-fatal.
#
# Combined with `--skip-server-warmup` (no startup warmup decode) and
# `--skip-warmup` (no bench cache warmup), the FIRST `_fwd_grouped_kernel_
# stage1_quant_int2` launches are the real decode at full KV — so NCU's
# --launch-skip lands in genuine long-context decode, not an 8-token warmup.
# ============================================================================
import os
import runpy

import requests as _rq
import sglang.test.bench_one_batch_server_internal as _m

_to = int(os.environ.get("OSCAR_BENCH_TIMEOUT", "7200"))
_m.DEFAULT_TIMEOUT = _to

_orig_post = _rq.post
def _patched_post(url, *a, **k):
    r = _orig_post(url, *a, **k)
    if isinstance(url, str) and url.endswith("/flush_cache") and r.status_code >= 400:
        # neuter only the flush_cache failure; everything else behaves normally
        r.raise_for_status = lambda *x, **y: None
    return r
_rq.post = _patched_post
# the internal module imported `requests` as a name; patch its reference too
if getattr(_m, "requests", None) is _rq:
    _m.requests.post = _patched_post

print(f"[_run_bench_server_ncu] DEFAULT_TIMEOUT={_to}s, flush_cache 400 neutered")

runpy.run_module("sglang.bench_one_batch_server", run_name="__main__", alter_sys=True)
