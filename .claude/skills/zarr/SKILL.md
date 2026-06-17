---
name: zarr
description: >-
  Use when working with Zarr v3 stores in this repo — creating/reading arrays or groups,
  choosing chunk shape, codecs/compression (Blosc/Zstd), sharding, consolidated metadata,
  fill values, the sync vs async API, or tuning parallel zarr I/O. Invoke before writing
  or changing any `zarr.*` call. Always verifies the API against the vendored zarr-python
  source rather than relying on memory, because this repo pins zarr 3.1.6 and uses v3-only
  features (`shards=`, `compressors=`, `zarr.config`).
---

# Zarr (zarr-python v3.1.6)

Ground truth for everything Zarr in DataScale. **Do not guess the v3 API from memory** —
v2↔v3 changed names, defaults, and the codec model. Verify against the vendored source.

## Where to look (ground truth)

- **Source:** `.claude/vendor/zarr-python/src/zarr/`
- **Docs:** `.claude/vendor/zarr-python/docs/`
- **Installed fallback:** `pixi run python -c "import zarr; print(zarr.__version__, zarr.__file__)"`

> Path assumes the vendored dir is named `zarr-python`. If yours differs, adjust the `<pkg>`
> segment everywhere below.

Key modules (verified against 3.1.6 layout):

| Question | File |
|----------|------|
| `create_array` / `Array` / indexing-set behavior | `src/zarr/core/array.py` |
| Group API (`require_array`, `require_group`, `create_group`) | `src/zarr/core/group.py` |
| Global config (threads, async concurrency, defaults) | `src/zarr/core/config.py` |
| Chunk grid math | `src/zarr/core/chunk_grids.py` |
| Codec pipeline (how codecs chain) | `src/zarr/core/codec_pipeline.py` |
| Codecs: Blosc / Zstd / Bytes / Transpose / **Sharding** | `src/zarr/codecs/{blosc,zstd,bytes,transpose,sharding}.py` |
| Sync vs async entry points | `src/zarr/api/{synchronous,asynchronous}.py` |
| Storage backends (local/memory/remote) | `src/zarr/storage/` |
| Fancy/orthogonal indexing | `src/zarr/core/indexing.py` |

### Grep recipes (symbol-based, layout-robust)

```bash
V=.claude/vendor/zarr-python/src/zarr
rg -n "def create_array" $V/core/array.py            # exact signature & param defaults
rg -n "class .*Codec" $V/codecs                      # available codecs
rg -n "default" $V/core/config.py                    # default compressor / thread settings
rg -n "consolidate_metadata" $V                      # consolidation behavior + the v3 warning
rg -n "shards|shard" $V/core/array.py $V/codecs/sharding.py
```

## Project-relevant essentials

`zarr.create_array` / `group.require_array` params that matter here (verify in source):
`shape, dtype, chunks, shards, filters, compressors, serializer, fill_value, order, config, overwrite`.

- **Chunking is the master perf knob.** I/O happens per-chunk; write blocks **must align to
  the chunk grid** or you trigger read-modify-write. Optimized chunking strategies will reflect the type of query being made. If this is in question for single cell analysis storage, see `rapids` and `dask` for designs on how chunks are loaded and processed. If no intructions are given for query use, choose a default chunking strategy that will work for both row and column queries.
- **Dense vs sparse storage drives query cost differently.** Dense arrays use a 2D chunk
  grid, so *both* row and column slices are chunk-aligned — a query materializes only the
  chunks it overlaps (selecting scattered gene columns uses orthogonal indexing,
  `core/indexing.py`), at the cost of storing/reading zeros. Sparse (CSR/CSC) stores only
  nonzeros, but anndata's lazy reader can **only chunk the major axis** (see the `anndata`
  skill), so for sparse the *format* (CSR = rows, CSC = cols) — not the chunk shape — is the
  dominant lever for which query is cheap.
- **Concurrency:** `zarr.config.set({"async.concurrency": N})` controls how many chunks the
  async layer fetches at once. DataScale drives *write* parallelism through dask, so check
  for double-counting threads (see "Silent performance killers" in CLAUDE.md).
- **`shards=`** packs many inner chunks into one object (fewer files / objects, same small
  read granularity). Pick `shards` as an integer multiple of `chunks`. Confirm semantics in
  `codecs/sharding.py` before using.
- **Codecs:** v3 uses `compressors=` / `serializer=` / `filters=` (numcodecs v3 codec
  objects), **not** the v2 `compressor=` kwarg. The default is *not* uncompressed — read
  `core/config.py` to see the active default, and inspect a real array with
  `arr.metadata` / `arr.compressors`.
- **Consolidated metadata:** `zarr.consolidate_metadata(store)` — speeds opening many-array
  stores (esp. S3/GCS). Emits a "not part of the Zarr v3 spec" warning; that's expected.

## Read/query access patterns (rows vs columns) — a benchmarking axis

DataScale stores get queried two main ways, which pull layout in **opposite** directions:

- **Row / cell-subset** (all cells of a cell-type or project-cohort): favors row-major
  locality — CSR sparse, or dense with wide-ish row chunks. Cheapest when the target cells are
  *contiguous*, so a query touches a small chunk range instead of scattering across the store.
- **Column / gene** (specific genes across all cells): favors column-major locality — CSC
  sparse, or dense with tall-ish column chunks.

One chunk shape / format can't be optimal for both. **Per the project's decision: don't bake in
a default — measure per store** with the benchmarking tools (record chunks-touched, bytes-read,
cold vs warm). When comparing engines (zarr vs icechunk) hold layout byte-identical.

**Research lever (not yet specified): row locality by metadata.** Reordering `obs` so a
cell-type / cohort is contiguous turns a scattered subset read into a contiguous chunk-range
read. Worth benchmarking; would need to persist the row permutation + per-group offsets and
keep the store anndata-readable. See the `anndata` skill for how reads slice the major axis.

**Downstream readers set the real access granularity.** The chunk shape + format you write here
*becomes* the `anndata` lazy block → the `dask` block → the `rapids` GPU transfer unit. Design
the layout with the read path in mind — see the `anndata`, `dask`, and `rapids` skills.

## DataScale encoding conventions (replicate)

Arrays/groups by default carry AnnData encoding attrs or the store won't round-trip. See the
`anndata` skill and the encoding table in CLAUDE.md. Quick reference: dense array →
`encoding-type="array"`/`"0.2.0"`; sparse group → `"csr_matrix"`/`"csc_matrix"` + `shape` +
child `data`/`indices`/`indptr`.

## Gotchas / version notes (3.1.6)

- `ad.settings.zarr_write_format = 3` should be set repo-wide; don't assume v2 on-disk layout.

## ✍️ Maintainer notes — ADD YOURS

<!-- TODO(you): Fill these in to sharpen the skill. Delete prompts you don't need. -->

- **Preferred default codec/level for this project's matrices** (e.g. Blosc+zstd level? shuffle?):
- **Chosen chunk-shape policy** (per storage mode / per access pattern):
- **Sharding policy** (when to enable, shard:chunk ratio):
- **Remote/object-store conventions** (S3/GCS bucket layout, credentials, consolidate-always?):
- **Specific vendored doc pages worth reading first** (paths under `docs/`):
- **Known sharp edges you've hit in 3.1.6:**
