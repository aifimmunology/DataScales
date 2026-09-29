# Golden writer fixtures

Each `<case>.tar.gz` holds `input/*.h5ad` (tiny deterministic 100 x 50 fixtures), the
`expected/<case>.zarr` store, and a `case.json` with the `annizarr`-style parameters
(`x_storage`, `backed`, `cpus`, chunk/shard flags, `auto_shard`, `sort_by`, `inputs`). These are
**self-snapshots**: `tests/golden/generate.py` builds `expected/` by calling annizarr's own
`convert()` with the packed parameters, and `tests/test_golden_writers.py` extracts each
tarball, re-runs `annizarr.convert` with those same parameters, and asserts the produced store
is byte-identical to `expected/` — so a refactor that silently changes on-disk layout
(chunking, codec, encoding attrs, sharding math, …) fails loudly instead of drifting unnoticed.
Plain zarr only — Icechunk stores carry commit timestamps.

Cases: `csr_eager`, `csr_backed`, `csc_eager`, `dense_from_sparse_eager`,
`dense_from_sparse_backed`, `dense_from_dense_eager`, `dense_sharded`, `dense_small_chunks`,
`csr_small_flat_chunk`, `concat_csr`, `concat_dense`, `sorted_eager`, `sorted_backed`,
`csr_layers_raw_obsm`, `csr_int_counts`, `csr_cpus2` (in-memory thread path, tests that
parallel chunk writes are still deterministic) — all `auto_shard=False`. Plus six
`auto_shard=True` variants covering the same shapes: `csr_eager_autoshard`,
`csc_eager_autoshard`, `csr_layers_raw_obsm_autoshard`, `concat_csr_autoshard`,
`sorted_eager_autoshard` (a small `sparse_flat_chunk=100` on the sparse ones, so nnz spans more
than the 8 chunks zarr's `shards="auto"` heuristic requires before it actually shards), and
`dense_from_sparse_eager_autoshard` (dense X stays unsharded — only the anndata-written
elements are auto-sharded).

## Regenerating

Goldens are regenerated **only** on a deliberate, documented layout change (chunking, codec,
encoding attrs, sharding math, …) — never to make a failing comparison pass. Since these are
self-snapshots of annizarr's own output, regenerate with the package installed in this repo's
own env, no old converter and no git worktree needed:

```bash
cd tools/annizarr && pixi run -e default python tests/golden/generate.py
```

This overwrites every `tests/golden/*.tar.gz` in place; review the diff before committing.

**2026-09-29 regeneration** (autosharding): the 16 pre-existing cases were regenerated with the
current package and diffed byte-for-byte against the previous tarballs before being
overwritten — every `expected/*.zarr` store and `input/*.h5ad` fixture was unchanged; the only
difference was the new `"auto_shard": false` key added to each `case.json`. The six
`*_autoshard` cases above are new.
