#!/usr/bin/env python
# ============================================================================
# _run_bench_server.py — thin launcher for sglang.bench_one_batch_server that
# RAISES the hard-coded HTTP client read timeout WITHOUT editing sglang source.
#
# Why: bench_one_batch_server's client waits for the /generate response with
#   DEFAULT_TIMEOUT = 600 s  (sglang/test/bench_one_batch_server_internal.py).
# A long-context prefill (e.g. 512k tokens, O(N^2) attention) takes longer than
# 600 s, so the client raises ReadTimeoutError even though the server is healthy.
#
# This launcher monkey-patches that module-level constant in memory, then runs
# the real bench module unchanged (argv passes straight through). Nothing under
# sglang-research/ is modified.
#
# Timeout (seconds) is taken from $OSCAR_BENCH_TIMEOUT (default 7200 = 2 h).
# Usage (drop-in for `python -m sglang.bench_one_batch_server ...`):
#   python profiling/server/_run_bench_server.py <same args>
# ============================================================================
import os
import runpy

import sglang.test.bench_one_batch_server_internal as _m

_to = int(os.environ.get("OSCAR_BENCH_TIMEOUT", "7200"))
_m.DEFAULT_TIMEOUT = _to
print(f"[_run_bench_server] patched DEFAULT_TIMEOUT -> {_to}s (server-side prefill can be slow)")

# run the real entrypoint as __main__; sys.argv[1:] flows through to its argparse
runpy.run_module("sglang.bench_one_batch_server", run_name="__main__", alter_sys=True)
