# zarrsmith build spec

zarrsmith = convert-to-zarr (h5ad→zarr creation) merged with store-surgery ops on existing
AnnData zarr stores. One streaming/parallel engine, one config/storage/encoding layer, ops on top.

## Decisions (locked)

- Name: **zarrsmith**; CLI `zarrsmith convert|convert-10x|concat|sort|rechunk|append|add-expr`.
- Clean break from `convert-to-zarr` (no alias).
- Mutation policy: additive ops (append, add-expr) write in place on plain zarr (new
  arrays/groups only, re-consolidate metadata if present); rewriting ops (rechunk, sort)
  always write a new store via temp + atomic swap. `--icechunk` turns any op into one commit.
- Append input: zarr stores only (convert h5ad first).
- Build order: restructure → add-expr → rechunk → sort (standalone) → append.
- Code style: minimal, sparse comments (one-liners for real constraints only), no module headers.

## Status

- [x] Phase 0 — restructure (rename + converter.py split, tests green, zero behavior change)
- [x] Phase 1 — add-expr (ops/expr.py; csc/dense via disk-backed column-band buckets, csr streamed)
- [x] Phase 2 — rechunk (ops/rechunk.py; threaded tile copy, non-target elements copied as-is)
- [x] Phase 3 — sort (ops/sort.py sort_store; shares _stream_sorted_store with --backed convert)
- [x] Phase 4 — append (ops/append.py; in-place resize, --drop-obsp / --refresh-expr escapes)

Still open from the invariants list: peak-RSS assertions in tests, `--verify` flag,
parallel band finalize in add-expr.

## Module map

```
src/zarrsmith/
├── cli.py         subcommand → op (thin)
├── config.py      frozen dataclasses + TOML/YAML + CLI overrides + _resolve_backend_cfg
├── storage.py     plain-zarr / icechunk open_output_store + finalize
├── validation.py  input checks
├── errors.py      ConversionError
├── layout.py      codec + shard math (_x_compressors, _dense_shards)
├── engine.py      parallel band workers, _run_parallel, _stage progress
├── sources.py     input readers: h5ad eager/backed, 10x h5; zarr-store source lands in Phase 1+
├── writers.py     streaming dense/sparse/concat writers + anndata encoding attrs
└── ops/           convert, concat, sort (+ rechunk, append, expr as they land)
```

New ops read bands from a source and write through `writers.py`; each op should stay ~100–300
lines. The zarr-store source (open an existing store, expose X/parts/obs/var + band iteration)
is the shared piece the new ops need first.

## Phase 1 — add-expr

`zarrsmith add-expr STORE --format csc|dense|csr --chunk-elems N [--target-sum 1e4]`

- Reads CSR `X` (raw counts), writes log-normalized (`normalize_total(target_sum)` + `log1p`)
  values to `layers/gexp` with proper encoding attrs (`csc_matrix`/`csr_matrix`/`array`,
  shape attr, data/indices/indptr children). Create `layers` group with dict encoding if absent.
- csr output: per-row-band streaming (band sums → scale data → write). Trivially parallel.
- csc output: streamed transpose, two passes. Pass 1: histogram `indices` → CSC indptr.
  Pass 2: partition columns into K bands; scatter entries into per-band temp buckets; each
  band sorts + writes independently (parallel). Memory = one column band.
- dense output: column bands from the same pass; chunks `(n_obs, max(1, N // n_obs))`.
- `--chunk-elems N`: elements per chunk on the 1-D sparse arrays; dense derives from it.
- indptr/indices dtype: int64 when nnz > int32 max.
- This unblocks the datavis app's gene queries (browser reads the CSC column via
  indptr slice → data/indices range reads).

## Phase 2 — rechunk

`zarrsmith rechunk STORE --array X --chunks R C [--shards F] [--compressor ...] -o OUT`

- New store via temp dir + atomic rename; never in place.
- Dense: band size = common multiple of old/new row chunks → grid-aligned reads AND writes,
  disjoint bands in parallel, no locks (same pattern as _densify_band_segment).
- Sparse 1-D arrays: flat copy at new chunk size, parallel over aligned output ranges.
- dtype/codec byte-identical unless flagged (isolate the variable, per repo benchmarking rules).

## Phase 3 — sort (standalone)

`zarrsmith sort STORE --by COL... -o OUT`

- Zarr input; reuse the bucket-per-group + concat engine from ops/sort.py
  (`_write_sorted_backed` is the reference implementation).
- Refuse when obsp present (permutation must reorder both axes). Same policy as today.

## Phase 4 — append

`zarrsmith append STORE --cells NEW.zarr`

- In place: extend X data/indices (one boundary-chunk RMW), rewrite indptr, extend obs arrays.
- Hard errors: var mismatch (names + order), categorical category mismatch (no silent
  union/coercion), X dtype mismatch.
- Detect indices/indptr int32 → int64 promotion when nnz crosses 2^31.
- Loud refusal + explicit flags for invalidated elements: obsp graphs (`--drop-obsp`),
  sorted-store contiguity (`--resort`), stale gexp layer (`--refresh-expr`).

## Cross-cutting invariants (every op)

- Output stays anndata-readable: encoding attrs on every new array/group (see repo CLAUDE.md
  AnnData section); verify against vendored anndata source when touching write paths.
- Memory bound = workers × band; assert peak RSS in tests (hermetic, tmp_path, tiny stores).
- Pin + log BLAS/OMP/Blosc threads; zarr read concurrency needs both `async.concurrency`
  and `threading.max_workers`.
- Re-consolidate metadata after mutation if the store had it.
- `--verify` flag (worth adding with Phase 1): reopen output, sampled checksum vs source.
- Icechunk: one commit per op, commit message = op + params.

## Later candidates (not scheduled)

subset (obs-query → new store), recompress/reshard as rechunk knobs, drop (remove
layer/obsm entry). Keep out: pipelines/DAGs, distributed clusters, GPU paths.
