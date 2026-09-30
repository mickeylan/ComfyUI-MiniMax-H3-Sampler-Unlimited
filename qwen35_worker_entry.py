"""Minimal Qwen3.5 worker bootstrap with pre-import diagnostics."""

import os
import sys
import time


started = time.monotonic()
print(
    f"[MINIMAX_H3_WORKER] bootstrap started pid={os.getpid()} python={sys.version.split()[0]} executable={sys.executable}",
    flush=True,
)
print("[MINIMAX_H3_WORKER] importing qwen35 module", flush=True)

import qwen35

print(f"[MINIMAX_H3_WORKER] qwen35 module imported elapsed={time.monotonic()-started:.1f}s", flush=True)
raise SystemExit(qwen35._worker_main())
