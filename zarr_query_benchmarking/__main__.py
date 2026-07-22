"""Benchmark Zarr X-matrix query time for dense and sparse (CSR/CSC) stores.

The timed region is read-selection + convert-to-final-format, so a query that
asks for `dense` from a sparse store pays the densification cost (and vice
versa) — keeping comparisons across layouts fair. Chunk-fetch counts come from
a separate untimed pass through a counting store wrapper, so they never inflate
the wall-clock numbers.

Row selection has three modes: `sequential` (first N), `random` (N seeded
indices), and `celltype` (all obs rows whose `--obs-column` equals
`--obs-value`). The celltype mode forces `--axis row`, ignores `--count`, and
reads the obs column + builds the match mask *inside* the timed region, modeling
a real "filter by obs value, then fetch X" query.

For celltype, ``--select-mode`` picks how matched rows are read (both first read obs
+ build the mask, so they differ *only* in the X read): ``slice`` (default) finds the
contiguous ``[start, end)`` run(s) of matched rows and reads ``X[start:end]`` slices —
anndata's contiguous fast path, which a *sorted* store turns into one cheap grab;
``fancy`` builds a ``flatnonzero`` integer index and gathers per row (anndata's
coordinate path). On an unsorted store the matched rows scatter into many length-1
runs, so slice degenerates — which is exactly why sorting helps. The run count is
reported either way as a locality metric.

Each timed read runs through a ``TimingStore`` that also records store-fetch intervals,
so every run is split into I/O (wall time with >=1 fetch in flight) vs CPU (decompress
+ gather + reassembly) to show where the time actually goes.
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
from zarr.storage import FsspecStore, LocalStore, WrapperStore


def open_store(path, read_only=True):
    """Open a zarr store from a local path or an fsspec URL.

    A URL with a scheme (``gs://``, ``s3://``, ``http://`` ...) goes through
    zarr's ``FsspecStore`` — the documented, stable remote path (``ObjectStore``
    is still flagged experimental). ``gs://`` requires ``gcsfs`` installed, which
    picks up gcloud Application Default Credentials automatically, so no token is
    threaded through here. Anything without a scheme is a local directory store.
    """
    if "://" in str(path):
        return FsspecStore.from_url(str(path), read_only=read_only)
    return LocalStore(path, read_only=read_only)


class TimingStore(WrapperStore):
    """Wraps a store; counts chunk fetches and records each fetch's wall interval.

    Recording the ``[start, end)`` interval of every ``store.get`` (relative to the
    last ``reset``) lets us take the *union* of those intervals — the wall time with
    at least one fetch in flight — rather than a naive sum of overlapping durations.
    Under ``async.concurrency`` > 1 fetches overlap, so the union is the honest,
    concurrency-correct split of wall time into I/O vs pure-CPU. Because the timed
    read itself runs through this store, fetch time and total time come from the
    *same* reads (no separate pass, so cloud warm-cache can't skew the split).
    """

    def __init__(self, store):
        super().__init__(store)
        self.gets = 0
        self._intervals = []
        self._epoch = 0.0

    def reset(self):
        self.gets = 0
        self._intervals = []
        self._epoch = perf_counter()

    async def get(self, key, prototype, byte_range=None):
        t0 = perf_counter()
        val = await self._store.get(key, prototype, byte_range=byte_range)
        self._intervals.append((t0 - self._epoch, perf_counter() - self._epoch))
        self.gets += 1
        return val

    async def get_partial_values(self, prototype, key_ranges):
        t0 = perf_counter()
        vals = await self._store.get_partial_values(prototype, key_ranges)
        self._intervals.append((t0 - self._epoch, perf_counter() - self._epoch))
        for _ in vals:
            self.gets += 1
        return vals

    def io_wall_s(self) -> float:
        """Union length of fetch intervals = wall time with >=1 fetch in flight."""
        if not self._intervals:
            return 0.0
        ivs = sorted(self._intervals)
        total = 0.0
        cur_s, cur_e = ivs[0]
        for s, e in ivs[1:]:
            if s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                total += cur_e - cur_s
                cur_s, cur_e = s, e
        return total + (cur_e - cur_s)


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


def celltype_indices(group, obs_column, obs_value):
    """Sorted obs-row indices whose `obs[obs_column]` equals `obs_value`.

    Reads the obs column and builds the match mask live, so callers can put
    this inside the timed region (obs-value queries model 'filter then fetch').
    """
    return np.flatnonzero(_read_obs_column(group, obs_column) == obs_value)


def resolve_celltype_selection(group, obs_column, obs_value):
    """Validate obs column / obs-value and return matching indices, or exit.

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
    sel = np.flatnonzero(values == obs_value)
    if sel.size == 0:
        uniq = np.unique(values)
        preview = ", ".join(map(str, uniq[:50]))
        more = "" if uniq.size <= 50 else f" (+{uniq.size - 50} more)"
        sys.exit(
            f"ERROR: no rows in obs['{obs_column}'] match obs value '{obs_value}'. "
            f"Available values: {preview}{more}"
        )
    return sel


def _read_sel(handle, axis_dim, sel):
    """Read the selection (unconverted): handle[sel, :] or handle[:, sel]."""
    return handle[sel, :] if axis_dim == 0 else handle[:, sel]


def _to_final(sub, final_format):
    """Convert a read result to the requested output format (timed separately).

    ``native`` returns ``sub`` unchanged (no conversion — measures the layout, not the
    format-conversion tax, and works for CSR *and* CSC); ``dense`` densifies; ``csr``
    compresses to CSR.
    """
    if final_format == "native":
        return sub
    if final_format == "dense":
        return sub.toarray() if sp.issparse(sub) else np.asarray(sub)
    return sub.tocsr() if sp.issparse(sub) else sp.csr_matrix(sub)


def read_convert(handle, axis_dim, sel, final_format):
    return _to_final(_read_sel(handle, axis_dim, sel), final_format)


def _runs_from_sorted_indices(idx):
    """Contiguous half-open [start, end) runs from a sorted 1-D integer index array.

    The number of runs is the selection's *locality*: a sorted store yields a few
    long runs (cheap contiguous slice reads); an unsorted store yields many length-1
    runs (slicing degenerates to ~per-row), which is exactly why sorting helps.
    """
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1) + 1
    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [idx.size]))
    return [(int(idx[s]), int(idx[e - 1]) + 1) for s, e in zip(starts, ends)]


_AUTO_MIN_AVG_RUN = 4  # avg rows/run >= this => contiguous (slice); below => scattered (fancy)


def _resolve_select_mode(select_mode, sel):
    """Resolve `auto` to `slice` (contiguous/sorted) or `fancy` (scattered).

    Returns ``(resolved_mode, n_runs, avg_run_len)``. `auto` picks `slice` when the matched
    rows form few long runs (a sorted store — slice visits each chunk once) and `fancy` when
    they scatter into many short runs (slice would re-fetch shared chunks once *per run*, so a
    single combined gather that visits each chunk once is faster). A forced mode is returned
    unchanged.
    """
    spans = _runs_from_sorted_indices(sel)
    n = len(spans)
    avg = sel.size / n if n else 0.0
    if select_mode == "auto":
        resolved = "slice" if avg >= _AUTO_MIN_AVG_RUN else "fancy"
    else:
        resolved = select_mode
    return resolved, n, avg


def _read_spans(handle, spans):
    """Read contiguous row spans via slices and concatenate (unconverted).

    Each ``handle[s:e]`` is a contiguous slice, so sparse takes anndata's fast path
    (`_get_contiguous_compressed_slice`: one `data[start:stop]`/`indices[start:stop]`
    read, no per-row gather, no coordinate indexer) and dense takes a plain
    `BasicIndexer` slice. Multiple runs are vstacked/concatenated.
    """
    if not spans:
        spans = [(0, 0)]
    parts = [handle[s:e] for s, e in spans]
    if len(parts) == 1:
        return parts[0]
    if sp.issparse(parts[0]):
        return sp.vstack(parts, format="csr")
    return np.concatenate(parts, axis=0)


def read_convert_spans(handle, spans, final_format):
    return _to_final(_read_spans(handle, spans), final_format)


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

    store = open_store(args.store)
    handle, _is_sparse, _src_format, shape, _dtype = open_x(store)
    final_format = "native" if args.native else args.format
    if args.mode == "celltype":
        group = zarr.open_group(store=store, mode="r")
        sel = celltype_indices(group, args.obs_column, args.obs_value)
        resolved, _, _ = _resolve_select_mode(args.select_mode, sel)
        if resolved == "slice":
            read_convert_spans(handle, _runs_from_sorted_indices(sel), final_format)
        else:
            read_convert(handle, 0, sel, final_format)
    else:
        axis_dim = 0 if args.axis == "row" else 1
        sel = select(shape[axis_dim], args.count, args.mode, args.seed)
        read_convert(handle, axis_dim, sel, final_format)
    print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _measure_peak_rss(args):
    """Spawn an isolated probe process and return its peak RSS in bytes.

    ru_maxrss is bytes on macOS but KiB on Linux, so normalize to bytes.
    Returns None on any failure — the memory probe must never break the run.
    """
    cmd = [
        sys.executable, "-m", "zarr_query_benchmarking", "--_rss-probe",
        "--store", args.store, "--axis", args.axis,
        "--mode", args.mode, "--seed", str(args.seed),
        "--select-mode", args.select_mode,
    ]
    cmd += ["--native"] if args.native else ["--format", args.format]
    if args.mode == "celltype":
        cmd += ["--obs-column", args.obs_column, "--obs-value", args.obs_value]
    else:
        cmd += ["--count", str(args.count)]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
        rss = int(out)
        return rss if sys.platform == "darwin" else rss * 1024
    except Exception:
        return None


def run_inspect(path):
    x = zarr.open_group(store=open_store(path), mode="r")["X"]
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
    # Two independent knobs, both needed to use multiple cores:
    #   async.concurrency  -- how many chunks are *dispatched* at once (a semaphore)
    #   threading.max_workers -- size of the pool that actually *decodes* them
    #                            (Blosc decode runs via asyncio.to_thread on this pool)
    # Raising concurrency alone leaves decode single-threaded -> ~1 core; raising
    # max_workers alone starves the pool if concurrency is low. Set both.
    cfg = {}
    if args.concurrency is not None:
        cfg["async.concurrency"] = args.concurrency
    if args.max_workers is not None:
        cfg["threading.max_workers"] = args.max_workers
    if cfg:
        zarr.config.set(cfg)
    concurrency = zarr.config.get("async.concurrency")
    max_workers = zarr.config.get("threading.max_workers")

    axis_dim = 0 if args.axis == "row" else 1

    # One pass through a TimingStore: it counts chunk fetches AND records each
    # store.get wall interval, so the fetch (I/O) vs CPU (decompress + gather +
    # reassembly) split comes from the *same* reads we time — no separate untimed
    # counting pass, and cloud warm-cache can't skew the split.
    store = TimingStore(open_store(args.store))
    handle, is_sparse, src_format, shape, dtype = open_x(store)
    axis_len = shape[axis_dim]

    final_format = "native" if args.native else args.format
    n_spans = resolved_mode = avg_run = None
    if args.mode == "celltype":
        # Validate up front (exits on a bad column / unmatched value) and resolve the read
        # mode from the run structure. 'auto' -> slice when the cells are contiguous (few long
        # runs, e.g. a sorted store), fancy when scattered (many short runs). The obs read +
        # mask build happen again *inside* the timed read so "filter then fetch" is measured —
        # the modes differ only in how X itself is read.
        group = zarr.open_group(store=store, mode="r")
        sel0 = resolve_celltype_selection(group, args.obs_column, args.obs_value)
        resolved_mode, n_spans, avg_run = _resolve_select_mode(args.select_mode, sel0)

        if args.select_mode == "slice" and avg_run < _AUTO_MIN_AVG_RUN:
            print(
                f"WARNING: matched cells span {n_spans} runs (avg {avg_run:.1f} rows/run) — "
                f"scattered for this cell type, so slice re-fetches shared chunks per run and "
                f"will be slow. Use --select-mode fancy (or auto) here.",
                file=sys.stderr,
            )

        if resolved_mode == "slice":
            def do_read(h, g):
                sel = celltype_indices(g, args.obs_column, args.obs_value)
                return _read_spans(h, _runs_from_sorted_indices(sel))
        else:
            def do_read(h, g):
                sel = celltype_indices(g, args.obs_column, args.obs_value)
                return _read_sel(h, 0, sel)
    else:
        if args.count > axis_len:
            sys.exit(f"ERROR: count={args.count} exceeds axis length {axis_len} for axis '{args.axis}'.")
        sel = select(axis_len, args.count, args.mode, args.seed)
        group = None

        def do_read(h, g):
            return _read_sel(h, axis_dim, sel)

    for _ in range(args.warmup):
        _to_final(do_read(handle, group), final_format)
    times, io_walls, cpu_walls, convert_walls = [], [], [], []
    result = None
    gets = 0
    for _ in range(args.repeats):
        store.reset()
        t0 = perf_counter()
        sub = do_read(handle, group)             # read + gather (all store fetches happen here)
        t_read = perf_counter()
        result = _to_final(sub, final_format)    # convert to --format (no fetches)
        t_end = perf_counter()
        io = store.io_wall_s()                   # wall with >=1 fetch in flight (concurrency-correct)
        total = t_end - t0
        convert = t_end - t_read
        times.append(total)
        io_walls.append(io)
        convert_walls.append(convert)
        # residual = decompress + gather (wall with no fetch in flight, not converting)
        cpu_walls.append(max(0.0, total - io - convert))
        gets = store.gets

    nnz_out = int(result.nnz) if sp.issparse(result) else int(np.count_nonzero(result))
    peak_rss_bytes = _measure_peak_rss(args)
    summary = {
        "store": args.store,
        "source_format": src_format,
        "source_shape": list(shape),
        "dtype": str(dtype),
        "axis": args.axis,
        "count": args.count,
        "mode": args.mode,
        "select_mode": resolved_mode,
        "select_mode_requested": args.select_mode if args.mode == "celltype" else None,
        "obs_column": args.obs_column,
        "obs_value": args.obs_value,
        "n_spans": n_spans,
        "avg_rows_per_run": avg_run,
        "selected": int(result.shape[axis_dim]),
        "final_format": final_format,
        "concurrency": concurrency,
        "max_workers": max_workers,
        "result_shape": list(result.shape),
        "result_nnz": nnz_out,
        "chunks_fetched": gets,
        "peak_rss_bytes": peak_rss_bytes,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "median_s": float(np.median(times)),
        "min_s": float(np.min(times)),
        "p95_s": float(np.percentile(times, 95)),
        "io_wall_median_s": float(np.median(io_walls)),
        "cpu_wall_median_s": float(np.median(cpu_walls)),
        "convert_median_s": float(np.median(convert_walls)),
        "timings_s": times,
        "io_wall_s": io_walls,
        "cpu_wall_s": cpu_walls,
        "convert_s": convert_walls,
        "versions": {p: md.version(p) for p in ("zarr", "anndata", "numpy", "scipy")},
        "git_commit": _git_commit(),
    }

    if args.json:
        print(json.dumps(summary))
        return
    print(f"Store: {summary['store']}")
    print(f"Source: {src_format}  shape={tuple(shape)}  dtype={dtype}")
    if args.mode == "celltype":
        sm_str = (args.select_mode if args.select_mode == resolved_mode
                  else f"{args.select_mode}->{resolved_mode}")
        print(
            f"Query: axis=row mode=celltype select={sm_str} "
            f"obs['{args.obs_column}']=='{args.obs_value}' "
            f"selected={summary['selected']} runs={n_spans} (avg {avg_run:.1f} rows/run) "
            f"-> {final_format}"
        )
    else:
        print(f"Query: axis={args.axis} count={args.count} mode={args.mode} -> {final_format}")
    print(f"Concurrency: {concurrency}  max_workers: {max_workers}  (warm cache)")
    print(f"Result: shape={tuple(result.shape)}  nnz={nnz_out}")
    print(f"Chunks fetched: {gets}")
    rss_str = "n/a" if peak_rss_bytes is None else f"{peak_rss_bytes / 1e6:.1f} MB"
    print(f"Peak RSS: {rss_str}")
    print(
        f"Wall time (s): median={summary['median_s']:.4f} "
        f"min={summary['min_s']:.4f} p95={summary['p95_s']:.4f}  "
        f"(warmup={args.warmup}, repeats={args.repeats})"
    )
    print(
        f"  split (median): I/O(fetch)={summary['io_wall_median_s']:.4f}s  "
        f"CPU-read(decompress+gather)={summary['cpu_wall_median_s']:.4f}s  "
        f"convert(->{final_format})={summary['convert_median_s']:.4f}s"
    )


def main(argv=None):
    p = argparse.ArgumentParser(prog="zarr-bench", description=__doc__)
    p.add_argument("--store", required=True,
                   help="Local path OR fsspec URL (gs://bucket/store.zarr, s3://...) of the "
                        ".zarr store to query (reads its X). gs:// needs gcsfs + gcloud ADC.")
    p.add_argument("--inspect", action="store_true", help="Print X layout (format/shape/chunks/codec) and exit; no timing.")
    p.add_argument("--axis", choices=["row", "col"], help="Query rows (obs) or columns (var).")
    p.add_argument("--count", type=int, help="Number of rows/columns to select (ignored when --mode celltype).")
    p.add_argument("--mode", choices=["sequential", "random", "celltype"], default="sequential",
                   help="sequential = first N; random = N seeded random indices; "
                        "celltype = all obs rows whose --obs-column equals --obs-value "
                        "(forces --axis row, ignores --count). Default: sequential.")
    p.add_argument("--select-mode", dest="select_mode", choices=["auto", "slice", "fancy"], default="auto",
                   help="For --mode celltype only (both first read obs + build the mask, so they "
                        "differ only in the X read). 'auto' (default) inspects the matched rows: slice "
                        "if they form few long contiguous runs (a sorted store — one cheap grab), fancy "
                        "if scattered into many short runs (one combined gather that visits each chunk "
                        "once, vs slice re-fetching shared chunks per run). 'slice'/'fancy' force it. "
                        "Ignored for sequential/random.")
    p.add_argument("--obs-column", dest="obs_column",
                   help="obs column to filter on (required for --mode celltype).")
    p.add_argument("--obs-value", dest="obs_value",
                   help="Value in --obs-column to select rows by (required for --mode celltype).")
    p.add_argument("--format", choices=["csr", "dense"],
                   help="Final format the result is converted to (timed). Required unless --native.")
    p.add_argument("--native", action="store_true",
                   help="Read at each store's NATIVE format (no conversion): skip .toarray()/.tocsr() "
                        "so you measure the layout, not the format-conversion tax (convert time ~0). "
                        "Overrides --format. Use for fair CSR-vs-dense / row-vs-col layout comparisons.")
    p.add_argument("--concurrency", type=int, help="zarr async.concurrency: how many chunks are dispatched at once (a semaphore). Pair with --max-workers.")
    p.add_argument("--max-workers", dest="max_workers", type=int,
                   help="zarr threading.max_workers: size of the decode thread pool that runs "
                        "Blosc/zstd decompression (the dominant stage of a dense read). Defaults "
                        "to min(32, cpu+4). Set ~= physical cores; raising --concurrency without "
                        "this leaves decode single-threaded (~1 core).")
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
        missing = [n for n in ("obs_column", "obs_value") if getattr(args, n) is None]
        if missing:
            p.error("--mode celltype requires: "
                    + ", ".join("--" + m.replace("_", "-") for m in missing))
    else:
        missing = [n for n in ("axis", "count") if getattr(args, n) is None]
        if missing:
            p.error("the following are required unless --inspect: " + ", ".join("--" + m for m in missing))
    if not args.native and args.format is None:
        p.error("one of --format {csr,dense} or --native is required.")
    if args.rss_probe:
        run_rss_probe(args)
        return
    run_benchmark(args)


if __name__ == "__main__":
    main()
