# CLAUDE.md — DataScale

Configurable **AnnData → Zarr v3** converter for non-spatial single-cell data, plus
benchmarking tools for Zarr/Icechunk query and conversion performance. The whole point
of this project is *correct, fast, parallel, memory-bounded* I/O — so the bar for any
change is: **it must not silently regress performance**. See the
[Silent performance killers](#silent-performance-killers) checklist before claiming
anything is "done".

---

## Ground-truth research protocol (READ THIS FIRST)

This project pins exact library versions and calls several **private/internal** APIs
(e.g. `anndata._io.specs.write_elem`, zarr v3 codec/sharding internals). API surfaces
change between versions and your training data is often stale. **Never guess an API —
verify it against the version actually in use.**

Authority order — consult in this order, stop when answered:

1. **Skill files** — `.claude/skills/<lib>/SKILL.md` for `zarr`, `icechunk`, `anndata`,
   `dask`, `rapids-singlecell`. These are the curated entry points; each one tells you
   *which* vendored files to grep for a given question.
2. **Vendored source + docs** — `.claude/vendor/<lib>/src/...` (source) and
   `.claude/vendor/<lib>/docs/...` (prose docs/guides) are the canonical ground truth.
   Grep the real source for signatures, defaults, and behavior; read the docs for intent
   and recommended usage. Prefer both over any remembered API.
3. **Installed package** — if not vendored, inspect what's installed:
   `pixi run python -c "import zarr, inspect; print(inspect.signature(zarr.create_array))"`
   and check `importlib.metadata.version(...)`.
4. **Web docs** — last resort, and only after confirming the version matches.

Vendored packages (read-only reference; do not edit): `zarr-python`, `icechunk`,
`anndata`, `dask`, `rapids-singlecell`.

> If `.claude/vendor/<lib>` or a skill file is missing, say so and offer to scaffold it — do
> **not** fall back to guessing the API from memory.

**Pinned versions (verify before relying on any API):**
`zarr 3.1.6` · `dask 2026.3.0` · `anndata 0.12.10` · `scipy 1.17.1` · `numpy 2.4.3` ·
`scanpy 1.12.1` · `numcodecs 0.16.5`. `icechunk` and `rapids-singlecell` are **not yet
installed** (see [Icechunk](#icechunk-storage-backend) / GPU notes) — adding them is a
real setup step, not an assumption.

---

## Environment & commands

Everything runs through **pixi** (conda-forge). There are two environments: `default`
(runtime) and `dev` (adds `pytest`). **`pytest` is only in `dev`.**

```bash
pixi install                                   # set up env
pixi run datascale convert-h5ad  --input in.h5ad --output out.zarr --config example_config.toml
pixi run datascale convert-10x-h5 --input fbm.h5 --output out.zarr
pixi run datascale concat-h5ads  --inputs a.h5ad b.h5ad --output out.zarr   # concat along obs
pixi run zarr-bench --store zarr_dbs/health_atlas_csr_1000.zarr --axis row --count 1000 --format csr
pixi run zarr-bench-inspect --store <store>    # show X layout (format/chunks/codec/shape), no timing
pixi run -e dev test-zarr-bench                # (currently stale: tests a removed API — see Repository map)
pixi run -e dev python -m pytest tests/ -q     # full suite (dev env)
```

Do **not** invoke bare `python`/`pytest` — they aren't on PATH. Always go through `pixi run`.

---

## Repository map

| Path | What |
|------|------|
| [datascale/converter.py](datascale/converter.py) | Core conversion: streaming dense/sparse writers, multi-h5ad concat |
| [datascale/config.py](datascale/config.py) | Frozen dataclass config (`AppConfig`/`IOConfig`/`ChunkConfig`/`ValidationConfig`) + TOML/YAML loader + CLI overrides |
| [datascale/cli.py](datascale/cli.py) | argparse CLI; maps subcommands → converter functions |
| [datascale/validation.py](datascale/validation.py) | Single-cell AnnData validation (rejects spatial, etc.) |
| `zarr_query_benchmarking/` | **CLI-only** query-time benchmark for a store's `X` (dense or CSR/CSC); `__main__.py` is the whole tool. Run via `python -m zarr_query_benchmarking` (pixi `zarr-bench` / `zarr-bench-inspect`). Args: `--store --axis {row,col} --count --mode {sequential,random} --format {csr,dense}` (+ `--concurrency/--repeats/--warmup/--seed/--json/--inspect`). Times read **+ convert-to-final-format** (fair across layouts); counts real `store.get()` chunk fetches + bytes via a `WrapperStore` in a separate untimed pass. No public library API by design. Dense reads use zarr orthogonal indexing; sparse reads go through `anndata.io.sparse_dataset` (the realistic downstream path). NOTE: the old [tests/test_zarr_query_benchmarking.py](tests/test_zarr_query_benchmarking.py) targets a since-removed library API and no longer matches this CLI tool — it fails to import until rewritten/removed. |
| [tests/](tests/) | `test_converter.py`, `test_config.py`, `test_validation.py`, `test_zarr_query_benchmarking.py` |
| `zarr_dbs/` | Pre-built fixture stores (csr/csc/dense variants) for benchmarking |
| `.claude/vendor/<lib>/{src,docs}` | Read-only vendored library source + docs (ground truth — see protocol above) |
| [.claude/skills/](.claude/skills/) | Per-library skills (`zarr`, `icechunk`, `anndata`, `dask`, `rapids-singlecell`) — invoked on demand; each points at the vendored source |

---

## AnnData ⇄ Zarr encoding conventions

Stores must stay **anndata-readable**. The repo writes Zarr v3 (`ad.settings.zarr_write_format = 3`)
and sets encoding attrs by hand. When you create arrays/groups, replicate these or the
store will not round-trip:

- Root group: `encoding-type="anndata"`, `encoding-version="0.1.0"`
- Dense array: `encoding-type="array"`, `encoding-version="0.2.0"`
- Sparse group: `encoding-type="csr_matrix"`/`"csc_matrix"`, `encoding-version="0.1.0"`,
  `shape=[n_obs, n_vars]`, with child arrays `data`, `indices`, `indptr`.

`write_elem`/`read_elem` live in `anndata._io.specs` — **private API**. Before using or
changing that call, check `.claude/vendor/anndata/src` for the signature in 0.12.x; it has moved
between versions.

---

## Zarr v3 performance & parallelism

This is zarr **v3** (`zarr.create_array` / `group.require_array`, `chunks=` for chunk
shape, `shards=` for sharding). Key levers — **verify exact defaults in `.claude/vendor/zarr-python/src`,
don't assume:**

- **Chunk shape is the master knob.** Reads/writes happen at chunk granularity. Chunks
  should match the dominant access pattern (row slices → wide-ish row chunks; column
  slices → tall col chunks). Write blocks **must align to the chunk grid** or you get
  read-modify-write (see the dense-tiling note below).
- **Sharding (`shards=`)** packs many inner chunks into one storage object. Use it when a
  store would otherwise have tens of thousands of tiny chunk files (filesystem inode
  pressure, slow `ls`, slow object-store listing) while keeping small read granularity.
  Always pick `shards` as an integer multiple of `chunks`.
- **Codecs / compression.** zarr v3 sets codecs via `compressors=`/`serializer=` (numcodecs
  v3 codecs, e.g. `BloscCodec`, `ZstdCodec`). Default is **not** "no compression" — inspect
  `array.metadata`/`array.compressors` to see what's actually applied. For float matrices,
  Blosc + shuffle usually helps; high zstd levels trade write throughput for size.
- **Concurrency.** zarr v3 parallelizes multi-chunk reads via its async layer; tune with
  `zarr.config.set({"async.concurrency": N})`. For writes this repo drives parallelism
  through dask (below), not zarr directly.
- **Consolidated metadata.** `zarr.consolidate_metadata(store)` collapses per-array metadata
  into one object — big win opening many-array stores, especially on S3/GCS. Note it emits a
  "not part of the Zarr v3 spec" warning (expected). Off by default; `--consolidate-metadata`
  or `io.consolidate_metadata=true` to enable.

### The dense-write tiling rule (don't reintroduce linear chunking)

Dense writes stream through `_build_tiled_dense_dask` in [converter.py](datascale/converter.py),
which builds a **2D-tiled** dask array whose blocks equal the `(x_row_chunk, x_col_chunk)`
zarr grid. This is deliberate: earlier code tiled only along rows (full-width column blocks),
which (a) misaligned with the chunk grid → read-modify-write, and (b) made per-block RAM scale
with total column count. **Any new dense writer must tile in 2D to the chunk grid** — never
slice full-width row bands. Peak RAM per worker should be bounded by one chunk, independent of
matrix width.

---

## Parallelism model (dask) & scheduler rules

In-memory input → `scheduler="threads"`, `num_workers=cfg.chunks.cpus` (releases the GIL via
numpy/blosc, so threads scale). Backed HDF5 input → **`scheduler="synchronous"`, 1 worker** —
`h5py is not thread-safe`; this is mandatory, not a tuning choice. Progress bars go to
`sys.stderr` via dask's `ProgressBar`.

---

## Silent performance killers

The explicit goal is that **nothing silently holds down performance**. Audit against this
list for any conversion/query/benchmark change:

1. **Thread oversubscription (the #1 killer).** dask threads × BLAS/OpenMP threads
   (numpy/scipy) × Blosc internal threads multiply. On an N-core box, `cpus=N` dask workers
   each spawning N BLAS threads = N² contention. Set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
   `MKL_NUM_THREADS`, and numcodecs Blosc thread count **deliberately**, and record them.
2. **Read-modify-write** from write blocks not aligned to the zarr chunk grid (see tiling rule).
3. **Accidental full materialization** — `.toarray()`/`.todense()` on a whole matrix,
   `.compute()` collecting everything, `np.asarray()` on a backed array. Stream instead.
4. **Wrong chunk size** — too many tiny chunks (per-chunk syscall/metadata overhead, slow open)
   or too few huge chunks (no parallelism, RAM spikes).
5. **Codec mismatch** — recompressing incompressible data, or a zstd level so high it bottlenecks
   writes; missing shuffle on float data.
6. **GIL-bound work under the threads scheduler** — pure-Python per-element loops don't
   parallelize; keep hot loops in numpy/scipy/blosc or use processes.
7. **Unconsolidated metadata on remote stores** — one network round-trip per array at open.
8. **Unnecessary synchronizer/locking** on a single-writer path.
9. **GPU↔CPU transfer thrash** (rapids-singlecell paths) — moving arrays across the PCIe bus
   per operation instead of keeping them resident on device.

---

## Icechunk (storage backend)

Icechunk is being **adopted as a storage backend** (transactional, versioned Zarr store). It
is **not yet installed** — adding it to `pyproject.toml`/pixi and vendoring `.claude/vendor/icechunk/src`
is required setup, not something to assume exists.

When working with it, verify the current API against `.claude/vendor/icechunk/src` (the Python binding
over the Rust core moves fast). Principles that hold regardless of version:

- Work happens in a **session**; changes are staged and made durable by a **commit**. Batch
  writes into few, large commits — a commit per chunk is pathological.
- Reads open a snapshot/ref; benchmark **open + first read** separately from steady-state reads.
- When benchmarking icechunk vs. plain zarr, hold chunk shape / codec / dataset identical so you
  measure the *store engine*, not a layout difference.

---

## GPU path (rapids-singlecell)

`rapids-singlecell` is GPU/CUDA-only and **not installed in the local (osx/CPU) env** — it runs
on an HPC/GPU environment. Don't import it in CPU code paths or tests. Benchmarks that exercise it
must record device, CUDA/driver version, and explicitly account for host↔device transfer time
(see killer #9).

---

## Benchmarking best practices

These tools are **general-purpose** — they're used to benchmark many things (query latency,
parallel write scaling, storage-layout tradeoffs, memory ceiling, task runtime). Hold to these
regardless of what's under test:

**Isolate the variable.** Change one knob per run (thread count, chunk shape, codec, store engine).
Hold everything else byte-identical. A comparison across two differences measures nothing.

**Control caches.** Distinguish **cold** (first touch, disk/network hit) from **warm** (OS page
cache hot). Report *which* you measured; ideally both. You usually can't drop the page cache
without root (`purge` on macOS, `echo 3 > /proc/sys/vm/drop_caches` on Linux) — if you can't,
say so and use a fresh/large-enough store so warm cache doesn't dominate.

**Warm up, then repeat.** Discard the first run(s) (import, JIT, lazy open, cache fill). Run
enough repetitions to characterize spread, not just center.

**Report distributions, not means.** Use **median + p95 (or min + IQR)**. A mean hides tail
latency and is wrecked by a single GC pause. Always report the spread; flag high variance instead
of averaging it away.

**Time the right clock.** `time.perf_counter()` for wall-clock (I/O-bound work); `time.process_time()`
for pure CPU. Record both when in doubt. Never time module import or one-off setup inside the hot loop.

**Measure memory deliberately.** Peak RSS via `resource.getrusage(RUSAGE_SELF).ru_maxrss` or
`psutil`; Python-level allocations via `tracemalloc`. For dask, watch worker memory. Confirm
streaming paths stay bounded (this is a core project guarantee).

**Pin and record parallelism.** Set and log dask `num_workers` *and* the BLAS/OMP/Blosc thread
env vars (killer #1). An unrecorded thread setting makes results irreproducible.

**Record full provenance** with every result: library versions (`zarr`, `dask`, `anndata`,
`numcodecs`, `icechunk`), git commit, hostname/CPU/GPU, dataset shape + nnz + dtype, and store
layout (chunks, shards, codec). Save **raw** per-run numbers (JSON/CSV), not just the summary.

**Fix randomness.** Seed RNGs; reuse fixed datasets/fixtures (`zarr_dbs/`, or tiny generated
stores like the test fixtures). Determinism is what makes a regression detectable.

**Use `zarr-bench-inspect`** (or `inspect_store`) to capture the layout you're benchmarking — a
latency number without its chunk/codec/shape context is not interpretable.

---

## Repo conventions

- **Lazy imports.** Heavy/optional deps (`numpy`, `dask`, `dask.array`, `ProgressBar`, scanpy,
  anndata internals) are imported **inside functions**, not at module top. Keep this pattern —
  it keeps CLI startup fast and GPU/optional deps out of CPU paths.
- **Config is immutable.** `AppConfig` and friends are frozen dataclasses; derive new config with
  `dataclasses.replace` / `apply_cli_overrides`, never mutate.
- **Stream, never fully materialize** dense matrices (tiling rule above).
- **Errors** wrap context in `ConversionError`/`QueryError`; progress/status prints go to
  `sys.stderr` (stdout stays clean for piping).
- **Tests are hermetic** — they build tiny stores in `tmp_path` matching the encoding
  conventions; don't make tests depend on the large `zarr_dbs/` fixtures.
