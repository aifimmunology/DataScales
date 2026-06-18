"""Benchmark Zarr X-matrix query time for dense and sparse (CSR/CSC) stores.

The timed region is read-selection + convert-to-final-format, so a query that
asks for `dense` from a sparse store pays the densification cost (and vice
versa) — keeping comparisons across layouts fair. Chunk-fetch counts come from
a separate untimed pass through a counting store wrapper, so they never inflate
the wall-clock numbers.

Row selection has three modes: `sequential` (first N), `random` (N seeded
indices), and `celltype` (all obs rows whose `--obs-column` equals
`--cell-type`). The celltype mode forces `--axis row`, ignores `--count`, and
reads the obs column + builds the match mask *inside* the timed region, modeling
a real "filter by cell type, then fetch X" query.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import subprocess
import sys
from time import perf_counter

import numpy as np
import scipy.sparse as sp
import zarr
from anndata.io import sparse_dataset
from zarr.storage import LocalStore, WrapperStore


class CountingStore(WrapperStore):
    """Wraps a store and counts chunk fetches + bytes read. Reset before a read."""

    def __init__(self, store):
        super().__init__(store)
        self.gets = 0
        self.bytes = 0

    def reset(self):
        self.gets = 0
        self.bytes = 0

    async def get(self, key, prototype, byte_range=None):
        val = await self._store.get(key, prototype, byte_range=byte_range)
        self.gets += 1
        if val is not None:
            self.bytes += len(val)
        return val

    async def get_partial_values(self, prototype, key_ranges):
        vals = await self._store.get_partial_values(prototype, key_ranges)
        for v in vals:
            self.gets += 1
            if v is not None:
                self.bytes += len(v)
        return vals


def open_x(store):
    """Return (handle, is_sparse, src_format, shape, dtype). Handle is a zarr
    Array (dense) or an anndata backed sparse dataset, both slice-able."""
    x = zarr.open_group(store=store, mode="r")["X"]
    if isinstance(x, zarr.Group):
        enc = x.attrs.get("encoding-type", "")
        src_format = "csr" if enc == "csr_matrix" else "csc"
        return sparse_dataset(x), True, src_format, tuple(x.attrs["shape"]), x["data"].dtype
    return x, False, "dense", tuple(x.shape), x.dtype


def select(axis_len, count, mode, seed):
    if mode == "sequential":
        return slice(0, count)
    return np.sort(np.random.default_rng(seed).choice(axis_len, size=count, replace=False))


def _read_obs_column(group, obs_column):
    """Read a single obs column into a 1-D array (categorical or plain)."""
    from anndata.io import read_elem

    return np.asarray(read_elem(group["obs"][obs_column]))


def celltype_indices(group, obs_column, cell_type):
    """Sorted obs-row indices whose `obs[obs_column]` equals `cell_type`.

    Reads the obs column and builds the match mask live, so callers can put
    this inside the timed region (cell-type queries model 'filter then fetch').
    """
    return np.flatnonzero(_read_obs_column(group, obs_column) == cell_type)


def resolve_celltype_selection(group, obs_column, cell_type):
    """Validate obs column / cell-type and return matching indices, or exit.

    Exits non-zero with the available columns / values on a typo, since an
    empty selection almost always means a misspelled name rather than a real
    benchmark target.
    """
    obs = group["obs"]
    columns = list(obs.attrs.get("column-order", []))
    if obs_column not in obs:
        avail = ", ".join(columns) if columns else "(none)"
        sys.exit(f"ERROR: obs column '{obs_column}' not found. Available columns: {avail}")
    values = _read_obs_column(group, obs_column)
    sel = np.flatnonzero(values == cell_type)
    if sel.size == 0:
        uniq = np.unique(values)
        preview = ", ".join(map(str, uniq[:50]))
        more = "" if uniq.size <= 50 else f" (+{uniq.size - 50} more)"
        sys.exit(
            f"ERROR: no rows in obs['{obs_column}'] match cell type '{cell_type}'. "
            f"Available values: {preview}{more}"
        )
    return sel


def read_convert(handle, axis_dim, sel, final_format):
    sub = handle[sel, :] if axis_dim == 0 else handle[:, sel]
    if final_format == "dense":
        return sub.toarray() if sp.issparse(sub) else np.asarray(sub)
    return sub.tocsr() if sp.issparse(sub) else sp.csr_matrix(sub)


def _decompressed_bytes(result):
    """Working-set size of the materialized result (decompressed, in-memory).

    This is the footprint that drives RAM / host->device transfer, not the
    compressed `bytes_read` pulled from the store: a dense block holds
    rows*cols values, a CSR block only nnz (+ index overhead).
    """
    if sp.issparse(result):
        return int(result.data.nbytes + result.indices.nbytes + result.indptr.nbytes)
    return int(result.nbytes)


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def run_rss_probe(args):
    """Do exactly one open + read_convert and print peak RSS (ru_maxrss, raw).

    Runs in a fresh process so the high-water mark reflects this one read, not
    contamination from warmup/repeats. Unit normalization happens in the parent.
    """
    import resource

    store = LocalStore(args.store, read_only=True)
    handle, _is_sparse, _src_format, shape, _dtype = open_x(store)
    if args.mode == "celltype":
        group = zarr.open_group(store=store, mode="r")
        sel = celltype_indices(group, args.obs_column, args.cell_type)
        read_convert(handle, 0, sel, args.format)
    else:
        axis_dim = 0 if args.axis == "row" else 1
        sel = select(shape[axis_dim], args.count, args.mode, args.seed)
        read_convert(handle, axis_dim, sel, args.format)
    print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _measure_peak_rss(args):
    """Spawn an isolated probe process and return its peak RSS in bytes.

    ru_maxrss is bytes on macOS but KiB on Linux, so normalize to bytes.
    Returns None on any failure — the memory probe must never break the run.
    """
    cmd = [
        sys.executable, "-m", "zarr_query_benchmarking", "--_rss-probe",
        "--store", args.store, "--axis", args.axis,
        "--mode", args.mode, "--format", args.format, "--seed", str(args.seed),
    ]
    if args.mode == "celltype":
        cmd += ["--obs-column", args.obs_column, "--cell-type", args.cell_type]
    else:
        cmd += ["--count", str(args.count)]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
        rss = int(out)
        return rss if sys.platform == "darwin" else rss * 1024
    except Exception:
        return None


def run_inspect(path):
    x = zarr.open_group(store=LocalStore(path, read_only=True), mode="r")["X"]
    print(f"Store: {path}")
    if isinstance(x, zarr.Group):
        n_obs, n_vars = x.attrs["shape"]
        data = x["data"]
        print(f"Format: {x.attrs.get('encoding-type')}  shape=({n_obs}, {n_vars})")
        print(f"  nnz={data.shape[0]}  dtype={data.dtype}")
        for name in ("data", "indices", "indptr"):
            a = x[name]
            print(f"  {name}: chunks={a.chunks} codec={a.compressors}")
    else:
        print(f"Format: dense  shape={tuple(x.shape)} dtype={x.dtype}")
        print(f"  chunks={x.chunks} codec={x.compressors}")


def run_benchmark(args):
    if args.concurrency is not None:
        zarr.config.set({"async.concurrency": args.concurrency})
    concurrency = zarr.config.get("async.concurrency")

    axis_dim = 0 if args.axis == "row" else 1

    # Timing pass on a plain store.
    store = LocalStore(args.store, read_only=True)
    handle, is_sparse, src_format, shape, dtype = open_x(store)
    axis_len = shape[axis_dim]

    if args.mode == "celltype":
        # Validate up front (exits on a bad column / unmatched cell type), but
        # re-read obs inside the timed query so the obs read + mask build are
        # part of the measured "filter by cell type then fetch" cost.
        group = zarr.open_group(store=store, mode="r")
        resolve_celltype_selection(group, args.obs_column, args.cell_type)

        def do_query(h, g):
            sel = celltype_indices(g, args.obs_column, args.cell_type)
            return read_convert(h, 0, sel, args.format)
    else:
        if args.count > axis_len:
            sys.exit(f"ERROR: count={args.count} exceeds axis length {axis_len} for axis '{args.axis}'.")
        sel = select(axis_len, args.count, args.mode, args.seed)
        group = None

        def do_query(h, g):
            return read_convert(h, axis_dim, sel, args.format)

    for _ in range(args.warmup):
        do_query(handle, group)
    times = []
    result = None
    for _ in range(args.repeats):
        t0 = perf_counter()
        result = do_query(handle, group)
        times.append(perf_counter() - t0)

    # Untimed counting pass on a wrapped store (obs reads included for celltype).
    counter = CountingStore(LocalStore(args.store, read_only=True))
    chandle = open_x(counter)[0]
    cgroup = zarr.open_group(store=counter, mode="r") if args.mode == "celltype" else None
    counter.reset()
    do_query(chandle, cgroup)

    nnz_out = int(result.nnz) if sp.issparse(result) else int(np.count_nonzero(result))
    decompressed_bytes = _decompressed_bytes(result)
    peak_rss_bytes = _measure_peak_rss(args)
    summary = {
        "store": args.store,
        "source_format": src_format,
        "source_shape": list(shape),
        "dtype": str(dtype),
        "axis": args.axis,
        "count": args.count,
        "mode": args.mode,
        "obs_column": args.obs_column,
        "cell_type": args.cell_type,
        "selected": int(result.shape[axis_dim]),
        "final_format": args.format,
        "concurrency": concurrency,
        "result_shape": list(result.shape),
        "result_nnz": nnz_out,
        "chunks_fetched": counter.gets,
        "bytes_read": counter.bytes,
        "result_decompressed_bytes": decompressed_bytes,
        "peak_rss_bytes": peak_rss_bytes,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "median_s": float(np.median(times)),
        "min_s": float(np.min(times)),
        "p95_s": float(np.percentile(times, 95)),
        "timings_s": times,
        "versions": {p: md.version(p) for p in ("zarr", "anndata", "numpy", "scipy")},
        "git_commit": _git_commit(),
    }

    if args.json:
        print(json.dumps(summary))
        return
    print(f"Store: {summary['store']}")
    print(f"Source: {src_format}  shape={tuple(shape)}  dtype={dtype}")
    if args.mode == "celltype":
        print(
            f"Query: axis=row mode=celltype obs['{args.obs_column}']=='{args.cell_type}' "
            f"selected={summary['selected']} -> {args.format}"
        )
    else:
        print(f"Query: axis={args.axis} count={args.count} mode={args.mode} -> {args.format}")
    print(f"Concurrency: {concurrency}  (warm cache)")
    print(f"Result: shape={tuple(result.shape)}  nnz={nnz_out}")
    print(f"Chunks fetched: {counter.gets}   Bytes read: {counter.bytes / 1e6:.1f} MB")
    rss_str = "n/a" if peak_rss_bytes is None else f"{peak_rss_bytes / 1e6:.1f} MB"
    print(
        f"Decompressed result: {decompressed_bytes / 1e6:.1f} MB   "
        f"Peak RSS: {rss_str}"
    )
    print(
        f"Wall time (s): median={summary['median_s']:.4f} "
        f"min={summary['min_s']:.4f} p95={summary['p95_s']:.4f}  "
        f"(warmup={args.warmup}, repeats={args.repeats})"
    )


def main(argv=None):
    p = argparse.ArgumentParser(prog="zarr-bench", description=__doc__)
    p.add_argument("--store", required=True, help="Path to the .zarr store to query (reads its X).")
    p.add_argument("--inspect", action="store_true", help="Print X layout (format/shape/chunks/codec) and exit; no timing.")
    p.add_argument("--axis", choices=["row", "col"], help="Query rows (obs) or columns (var).")
    p.add_argument("--count", type=int, help="Number of rows/columns to select (ignored when --mode celltype).")
    p.add_argument("--mode", choices=["sequential", "random", "celltype"], default="sequential",
                   help="sequential = first N; random = N seeded random indices; "
                        "celltype = all obs rows whose --obs-column equals --cell-type "
                        "(forces --axis row, ignores --count). Default: sequential.")
    p.add_argument("--obs-column", dest="obs_column",
                   help="obs column to filter on (required for --mode celltype).")
    p.add_argument("--cell-type", dest="cell_type",
                   help="Value in --obs-column to select rows by (required for --mode celltype).")
    p.add_argument("--format", choices=["csr", "dense"], help="Final format the data is converted to (timed).")
    p.add_argument("--concurrency", type=int, help="zarr async.concurrency (parallel chunk fetches).")
    p.add_argument("--repeats", type=int, default=5, help="Timed repeats (default: 5).")
    p.add_argument("--warmup", type=int, default=1, help="Warmup runs discarded before timing (default: 1).")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for --mode random (default: 0).")
    p.add_argument("--json", action="store_true", help="Emit one JSON object (incl. raw timings + provenance).")
    p.add_argument("--_rss-probe", dest="rss_probe", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args(argv)

    if args.inspect:
        run_inspect(args.store)
        return
    if args.mode == "celltype":
        if args.axis == "col":
            p.error("--mode celltype selects obs rows; --axis col is not supported.")
        args.axis = "row"
        missing = [n for n in ("obs_column", "cell_type", "format") if getattr(args, n) is None]
        if missing:
            p.error("--mode celltype requires: "
                    + ", ".join("--" + m.replace("_", "-") for m in missing))
    else:
        missing = [n for n in ("axis", "count", "format") if getattr(args, n) is None]
        if missing:
            p.error("the following are required unless --inspect: " + ", ".join("--" + m for m in missing))
    if args.rss_probe:
        run_rss_probe(args)
        return
    run_benchmark(args)


if __name__ == "__main__":
    main()
