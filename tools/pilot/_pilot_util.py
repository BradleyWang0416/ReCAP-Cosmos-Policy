"""Shared helpers for the pilot diagnostics.

Two things the original serial scripts were losing on a 208-core box:
  * per-frame HDF5 gzip decompression ran on one core;
  * candidate distances ran as python loops of ``np.linalg.norm`` instead of one GEMM.

Everything here is float64 so the optimised scripts reproduce the serial numbers.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def default_workers(cap: int = 8) -> int:
    """Worker count for the image passes.

    Reading scales to 16+ workers (10.7x measured), but the block-mean descriptor is
    memory/cache bound and degrades past ~6-8 workers on this box.  The cap targets the
    slower of the two stages.  Override with PILOT_WORKERS.
    """
    env = os.environ.get("PILOT_WORKERS")
    if env:
        return max(1, int(env))
    return max(1, min(cap, (os.cpu_count() or 4) - 2))


def parallel_map(fn, items, workers: int | None = None, chunksize: int = 1):
    """Ordered parallel map. ``fn`` must be a module-level picklable callable."""
    items = list(items)
    workers = default_workers() if workers is None else workers
    if workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, items, chunksize=chunksize))


def pairwise_dist(Q: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Euclidean distance between every row of Q (nq,d) and C (nc,d) via one GEMM.

    Uses ||q||^2 + ||c||^2 - 2 q.c in float64.  Features here are either L2
    normalised or metric positions, so the cancellation term stays far above
    float64 noise; negatives from rounding are clamped before the sqrt.
    """
    Q = np.ascontiguousarray(Q, dtype=np.float64)
    C = np.ascontiguousarray(C, dtype=np.float64)
    if Q.ndim == 1:
        Q = Q[None, :]
    q2 = np.einsum("ij,ij->i", Q, Q)[:, None]
    c2 = np.einsum("ij,ij->i", C, C)[None, :]
    d2 = q2 + c2 - 2.0 * (Q @ C.T)
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2, out=d2)


def horizon_mask(starts, seg, n_rows: int, h_steps: int, k_steps: int) -> np.ndarray:
    """Vectorised replacement for the per-start ``seg_ok`` python loop.

    True where [start, start+H+K) fits inside the episode and stays in one segment.
    """
    starts = np.asarray(starts, dtype=np.int64)
    span = int(h_steps) + int(k_steps)
    out = np.zeros(len(starts), dtype=bool)
    fits = starts + span <= int(n_rows)
    if not fits.any():
        return out
    idx = starts[fits][:, None] + np.arange(span, dtype=np.int64)[None, :]
    window = np.asarray(seg, dtype=np.int64)[idx]
    out[np.flatnonzero(fits)] = window.min(1) == window.max(1)
    return out
