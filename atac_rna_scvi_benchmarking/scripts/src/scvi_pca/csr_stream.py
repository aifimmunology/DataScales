"""Stream row blocks of an anndata zarr CSR (X/data, X/indices, X/indptr) with read-ahead."""
import concurrent.futures as cf
import threading
import time
from dataclasses import dataclass

import numpy as np

BLOCK_NNZ = 200_000_000


@dataclass
class Block:
    r0: int
    r1: int
    indptr: np.ndarray   # (r1 - r0 + 1,) int64, rebased to 0
    data: np.ndarray
    indices: np.ndarray
    sel: np.ndarray      # local rows (sorted) of the requested subset inside [r0, r1)
    out0: int            # position of sel[0] within the requested subset


def set_zarr_io(concurrency=32, workers=12):
    import zarr
    zarr.config.set({"async.concurrency": concurrency, "threading.max_workers": workers})


def _ranges(indptr, rows, block_nnz):
    """Contiguous [r0, r1) ranges of ~block_nnz nnz that cover the requested rows."""
    n = len(indptr) - 1
    bounds = np.searchsorted(indptr, np.arange(0, indptr[-1], block_nnz), side="right") - 1
    bounds = np.unique(np.r_[np.clip(bounds, 0, n), n])
    out = []
    for r0, r1 in zip(bounds[:-1], bounds[1:]):
        lo, hi = np.searchsorted(rows, [r0, r1])
        if hi > lo:
            out.append((int(r0), int(r1), int(lo), int(hi)))
    return out


def blocks(X, rows=None, block_nnz=BLOCK_NNZ, prefetch=3, stats=None):
    """Yield Block per range; rows is a sorted int array (None = all rows).

    `prefetch` blocks are read concurrently. stats (dict) accumulates read_s (summed reader time), wait_s (consumer blocked on reads), nnz_read."""
    stats = {} if stats is None else stats
    for k in ("read_s", "wait_s", "nnz_read"):
        stats.setdefault(k, 0)
    indptr = np.asarray(X["indptr"][:], dtype=np.int64)
    n = len(indptr) - 1
    rows = np.arange(n, dtype=np.int64) if rows is None else np.asarray(rows, dtype=np.int64)
    data, indices = X["data"], X["indices"]
    lock = threading.Lock()

    def read(spec):
        r0, r1, lo, hi = spec
        a, b = indptr[r0], indptr[r1]
        t0 = time.perf_counter()
        blk = Block(r0, r1, indptr[r0:r1 + 1] - a, data[a:b], indices[a:b], rows[lo:hi] - r0, lo)
        with lock:
            stats["read_s"] += time.perf_counter() - t0
            stats["nnz_read"] += int(b - a)
        return blk

    specs = _ranges(indptr, rows, block_nnz)
    with cf.ThreadPoolExecutor(max_workers=prefetch) as pool:
        futs = [pool.submit(read, s) for s in specs[:prefetch]]
        for i in range(len(specs)):
            t0 = time.perf_counter()
            blk = futs[i].result()
            stats["wait_s"] += time.perf_counter() - t0
            futs[i] = None
            if i + prefetch < len(specs):
                futs.append(pool.submit(read, specs[i + prefetch]))
            yield blk


def from_scipy(m):
    """Adapter so an in-memory scipy CSR can be streamed like a zarr X group."""
    return {"indptr": m.indptr, "data": m.data, "indices": m.indices}
