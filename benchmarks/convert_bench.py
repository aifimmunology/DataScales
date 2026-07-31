#!/usr/bin/env python
"""Simple h5ad -> zarr conversion benchmark (time + peak memory + output size).

Compares base implementations of a few ways to get an .h5ad into a zarr-family
store. Each tool is run at its **tuned best-effort**: it uses whatever
parallel / streaming / bounded-memory capability it actually has (not just the
out-of-box default). We do NOT force a common codec/chunking — the point is to
see what each tool produces and how fast, using its own machinery.

    anndata    ad.read_h5ad(eager) + adata.write_zarr(path)
               NOTE: write_zarr is SERIAL — anndata has no parallel-write knob in
               its public API and does not stream a sparse write. Input load is
               eager. This is anndata's ceiling; reported as-is.
    h5py       NAIVE baseline: a plain SERIAL recursive h5py -> zarr copy — the
               obvious hand-written way, with NO parallelism, NO chunk-aligned
               fan-out and NO streaming (each dataset is read whole and assigned
               in one shot). This is the row datascale's backed + process-parallel
               CSR copy is meant to beat on BOTH time and peak RAM. `workers` is
               ignored. zarr default codec/chunking applies.
    icechunk   ad.read_h5ad(eager) + write into an Icechunk store + one commit.
               NOTE: SERIAL single session — true parallel needs Session.fork()
               + icechunk.dask.store_dask (out of scope here). Reported as-is.
    datascale  datascale.convert_h5ad_to_zarr with --backed --cpus N: the backed
               CSR->CSR path flat-copies chunk-aligned nnz segments across
               PROCESSES (parallel + streaming + bounded RAM). This is the tool's
               intended optimized path.
    virtualizarr (optional, guarded) zero-copy byte-range reference over the raw
               HDF5 arrays. NOT installed here -> self-reports skipped. Needs
               `virtualizarr` + `imagecodecs` (imagecodecs supplies the lzf
               filter this file requires). It references raw arrays, not an
               anndata-readable store — a "don't rewrite at all" option.

    rechunker  EXCLUDED by design: rechunker rechunks an existing zarr/dask array
               to new chunks; it cannot read an .h5ad, and X here is sparse CSR
               (rechunker rechunks arrays, not CSR groups). It is a 2nd-stage
               re-layout tool, not an h5ad->zarr converter, so it is not a fair
               row in this comparison.

Each method runs in its OWN subprocess so peak RSS is isolated per method
(getrusage peak is a per-process high-water mark). The child reports wall time
and peak RSS as a JSON line; the parent adds output size and prints a table.

Designed for the big box (e.g. 400 GB RAM / 64 proc) where the 2M CSR (~27 GB in
memory) loads eagerly. On a small-RAM machine the eager methods (anndata,
icechunk) load the whole AnnData, and naive h5py spikes to the size of the
largest single array it reads whole; only datascale (always backed + streamed)
stays bounded. Use the 500k file to exercise the eager rows on a small box.

Usage
-----
    # tuned run across the core methods, 64-way parallel where supported
    pixi run python benchmarks/convert_bench.py \
        --input data/synthetic_2M_34k.h5ad --workers 64 \
        --outdir /scratch/convbench --json results_2M.json

    # single method / fewer workers
    pixi run python benchmarks/convert_bench.py \
        --input data/synthetic_2M_34k.h5ad --methods h5py datascale --workers 8

Notes
-----
* CACHE: on a big box the 17 GB input lands in page cache after method 1, so
  methods 2..N read *warm*. That is reported, not hidden. Drop caches between
  methods (Linux root: `echo 3 > /proc/sys/vm/drop_caches`) for cold numbers, or
  run one method per invocation.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Pin BLAS/OpenMP threads to 1 so N worker processes don't each spawn N BLAS
# threads (CLAUDE.md silent perf killer #1: N×N contention on a 64-core box).
# Deliberate + recorded in provenance. Must run before any numpy/scipy import —
# those are lazy (inside functions), so module top is early enough; child procs
# re-import this module and inherit these into their own worker pools.
# (Blosc's own internal thread count is left at library default — noted, not pinned.)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")


# ── measurement helpers ───────────────────────────────────────────────────────
def peak_rss_gb() -> float:
    """Peak resident set of THIS process, in GB. ru_maxrss is bytes on macOS,
    kilobytes on Linux — normalise so the number means the same on both."""
    m = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return m / (1024 ** 3) if sys.platform == "darwin" else m / (1024 ** 2)


def child_peak_rss_gb() -> float:
    """Peak RSS of any WORKER children this process spawned (RUSAGE_CHILDREN), GB.
    Same unit rules as above. Reported alongside self for the process-pool methods
    (h5py / datascale-backed) whose real memory lives in the workers."""
    m = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return m / (1024 ** 3) if sys.platform == "darwin" else m / (1024 ** 2)


def dir_size_gb(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 ** 3)


# ── conversion methods (base impls, tuned to each tool's parallel/stream knob) ─
def convert_anndata(inp: Path, out: Path, workers: int) -> None:
    # SERIAL by nature: no parallel-write knob in anndata's public API.
    import anndata as ad
    adata = ad.read_h5ad(inp)  # eager, full load
    adata.write_zarr(out)


def convert_h5py(inp: Path, out: Path, workers: int) -> None:
    """NAIVE baseline: plain SERIAL recursive h5py -> zarr copy. Walks the HDF5
    tree and copies every dataset with a single whole-dataset assignment — no
    process pool, no chunk-aligned fan-out, no block streaming. This is the
    "obvious" way someone hand-writes an h5ad->zarr copy, and the row datascale
    is meant to beat on time (serial) AND peak RAM (each big 1D CSR array is read
    whole into memory). `workers` is ignored. zarr default codec/chunking."""
    import h5py
    import numpy as np
    import zarr

    def attr(v):  # make an h5py attr value JSON-serialisable for zarr
        if isinstance(v, bytes):
            return v.decode()
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        return v

    def copy(h5obj, zgrp):
        for k, v in h5obj.attrs.items():
            zgrp.attrs[k] = attr(v)
        for key, item in h5obj.items():
            if isinstance(item, h5py.Group):
                copy(item, zgrp.require_group(key))
                continue
            # Variable-length / fixed string datasets (obs/_index, var/_index,
            # categorical `categories`, id columns) -> zarr variable-length str.
            if h5py.check_string_dtype(item.dtype) is not None:
                z = zgrp.require_array(key, shape=item.shape, dtype=str)
                for k, v in item.attrs.items():
                    z.attrs[k] = attr(v)
                z[...] = item.asstr()[()]
                continue

            z = zgrp.require_array(key, shape=item.shape, dtype=item.dtype)
            for k, v in item.attrs.items():
                z.attrs[k] = attr(v)
            z[...] = item[...]   # naive: read the whole dataset, then one assign

    root = zarr.open_group(str(out), mode="w")
    with h5py.File(inp, "r") as f:
        copy(f, root)


def convert_icechunk(inp: Path, out: Path, workers: int) -> None:
    # SERIAL single session + one commit (true parallel would need fork + dask).
    import anndata as ad
    import icechunk
    import zarr
    from anndata._io.specs import write_elem  # private API (see anndata skill)

    adata = ad.read_h5ad(inp)  # eager
    repo = icechunk.Repository.open_or_create(
        icechunk.local_filesystem_storage(str(out))
    )
    session = repo.writable_session("main")
    # NB: ad.write_zarr() would append zarr.consolidate_metadata(), which raises on
    # an IcechunkStore (supports_consolidated_metadata is False). Write the AnnData
    # element directly instead — same on-disk result, no consolidation tail.
    root = zarr.open_group(store=session.store, mode="w")
    write_elem(root, "/", adata)
    session.commit("convert_bench: h5ad -> icechunk")


def convert_datascale(inp: Path, out: Path, workers: int) -> None:
    # Tuned: backed streaming + process parallelism (bounded RAM per worker).
    from datascale.config import load_config, apply_cli_overrides
    from datascale.converter import convert_h5ad_to_zarr

    cfg = load_config(None)  # defaults: sparse-csr, zarr backend
    # Always backed so this row is ALWAYS the streaming/bounded CSR->CSR path
    # (workers just sets the process count). At workers=1 it is the serial
    # streaming copy, not eager — keeps the row's meaning stable.
    cfg = apply_cli_overrides(cfg, cpus=max(1, workers), backed=True)
    convert_h5ad_to_zarr(str(inp), str(out), cfg)


def convert_virtualizarr(inp: Path, out: Path, workers: int) -> None:
    """Zero-copy: byte-range reference manifest over the HDF5 arrays. NOT an
    anndata-readable store; lzf inputs need imagecodecs. ImportError if missing
    deps -> reported as skipped by the runner."""
    from virtualizarr import open_virtual_dataset

    vds = open_virtual_dataset(str(inp))
    vds.virtualize.to_kerchunk(str(out) + ".json", format="json")


METHODS = {
    "anndata": convert_anndata,
    "h5py": convert_h5py,
    "icechunk": convert_icechunk,
    "datascale": convert_datascale,
    "virtualizarr": convert_virtualizarr,
}
SERIAL_METHODS = {"anndata", "icechunk", "h5py"}  # no parallel path; workers is ignored


# ── child: run one method, report time + peak RSS ─────────────────────────────
def run_one(method: str, inp: Path, out: Path, workers: int) -> None:
    if out.exists():
        shutil.rmtree(out) if out.is_dir() else out.unlink()
    manifest = Path(str(out) + ".json")
    if manifest.exists():
        manifest.unlink()
    fn = METHODS[method]
    t0 = time.perf_counter()
    err = None
    try:
        fn(inp, out, workers)
    except Exception as e:  # report, don't crash the whole suite
        err = f"{type(e).__name__}: {e}"
    result = {
        "method": method,
        "seconds": round(time.perf_counter() - t0, 2),
        "peak_rss_gb": round(peak_rss_gb(), 2),
        "worker_peak_rss_gb": round(child_peak_rss_gb(), 2),
        "workers": 1 if method in SERIAL_METHODS else workers,
        "error": err,
    }
    print("RESULT_JSON:" + json.dumps(result), flush=True)


# ── parent: orchestrate subprocesses, collect + print ─────────────────────────
def provenance(workers: int) -> dict:
    import importlib.metadata as md
    vers = {}
    for pkg in ("anndata", "zarr", "numcodecs", "h5py", "icechunk", "dask",
                "scipy", "numpy", "virtualizarr", "imagecodecs", "rechunker"):
        try:
            vers[pkg] = md.version(pkg)
        except Exception:
            vers[pkg] = "MISSING"
    try:
        git = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        git = "unknown"
    for env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        vers[env] = os.environ.get(env, "unset")
    return {
        "host": platform.node(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "workers": workers,
        "git": git,
        "versions_and_env": vers,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", help="path to .h5ad")
    ap.add_argument("--outdir", default="/tmp/convbench", help="dir for output stores")
    ap.add_argument("--methods", nargs="+", default=["anndata", "h5py", "icechunk", "datascale"],
                    choices=list(METHODS))
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                    help="parallelism for tools that support it (datascale cpus); "
                         "ignored by the serial rows incl. the naive h5py baseline")
    ap.add_argument("--json", help="write raw results here")
    ap.add_argument("--run", help=argparse.SUPPRESS)  # internal child invocation
    args = ap.parse_args()

    if args.run:  # child path
        out = Path(args.outdir) / f"{args.run}.out"
        run_one(args.run, Path(args.input), out, args.workers)
        return

    if not args.input:
        ap.error("--input is required")
    inp = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    prov = provenance(args.workers)
    print("== provenance ==")
    print(json.dumps(prov, indent=2))
    print(f"\n== input: {inp}  ({inp.stat().st_size / 1024**3:.1f} GB on disk) ==")
    print(f"== workers={args.workers}  (serial methods ignore it: {sorted(SERIAL_METHODS)}) ==")
    print("NOTE: after method 1 the input is warm in page cache on a big box; "
          "results mix cold(1st)/warm unless you drop caches between runs.\n")

    rows = []
    for method in args.methods:
        for rep in range(args.repeats):
            out = outdir / f"{method}.out"
            cmd = [sys.executable, __file__, "--run", method, "--input", str(inp),
                   "--outdir", str(outdir), "--workers", str(args.workers)]
            print(f"-> {method} (rep {rep + 1}/{args.repeats}) ...", flush=True)
            proc = subprocess.run(cmd, capture_output=True, text=True)
            line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT_JSON:")), None)
            if line is None:
                print(f"   FAILED to get result. stderr tail:\n{proc.stderr[-2000:]}")
                rows.append({"method": method, "seconds": None, "peak_rss_gb": None,
                             "worker_peak_rss_gb": None, "out_size_gb": None,
                             "workers": None, "error": "no result line"})
                continue
            res = json.loads(line[len("RESULT_JSON:"):])
            manifest = Path(str(out) + ".json")
            if out.exists() and out.is_dir():
                res["out_size_gb"] = round(dir_size_gb(out), 3)
            elif manifest.exists():
                res["out_size_gb"] = round(manifest.stat().st_size / 1024**3, 6)
            else:
                res["out_size_gb"] = None
            rows.append(res)
            peak = max(res["peak_rss_gb"] or 0, res["worker_peak_rss_gb"] or 0)
            tag = f"   {res['seconds']}s  peak {peak} GB (self {res['peak_rss_gb']} / workers {res['worker_peak_rss_gb']})  out {res['out_size_gb']} GB"
            print(tag + (f"  ERROR {res['error']}" if res.get("error") else ""))

    # summary table
    print("\n== summary ==")
    hdr = f"{'method':14s} {'workers':>7s} {'time_s':>10s} {'peak_gb':>9s} {'wrk_gb':>8s} {'out_gb':>9s}  note"
    print(hdr)
    for r in rows:
        note = r.get("error") or ("serial" if r["method"] in SERIAL_METHODS else "")
        print(f"{r['method']:14s} {str(r.get('workers')):>7s} {str(r['seconds']):>10s} "
              f"{str(r['peak_rss_gb']):>9s} {str(r.get('worker_peak_rss_gb')):>8s} "
              f"{str(r['out_size_gb']):>9s}  {note}")
    print("\nNOTE: out_gb mixes codec + format + index dtype (real-world defaults, NOT "
          "held identical): datascale=Blosc-zstd-5+shuffle CSR; anndata=its own CSR "
          "(may down-cast indices); naive h5py=raw copy w/ zarr default zstd-0. "
          "peak_gb = max(self, workers); only datascale (backed+parallel) lives in "
          "workers — the serial rows (anndata/icechunk/h5py) spike in self.")

    if args.json:
        Path(args.json).write_text(json.dumps({"provenance": prov, "results": rows}, indent=2))
        print(f"\nraw results -> {args.json}")


if __name__ == "__main__":
    main()
