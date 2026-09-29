# Golden writer fixtures

Each `<case>.tar.gz` holds `input/*.h5ad` (tiny deterministic 100 x 50 fixtures), the
`expected/<case>.zarr` store produced by the **old** `tools/convert-to-zarr` CLI, and a
`case.json` with the equivalent `annizarr`-style parameters (`x_storage`, `backed`, `cpus`,
chunk/shard flags, `sort_by`, `inputs`). `tests/test_golden_writers.py` extracts each tarball,
runs the **new** `annizarr.convert` with those parameters, and asserts the produced store is
byte-identical to `expected/`. Plain zarr only — Icechunk stores carry commit timestamps.

Cases: `csr_eager`, `csr_backed`, `csc_eager`, `dense_from_sparse_eager`,
`dense_from_sparse_backed`, `dense_from_dense_eager`, `dense_sharded`, `dense_small_chunks`,
`csr_small_flat_chunk`, `concat_csr`, `concat_dense`, `sorted_eager`, `sorted_backed`,
`csr_layers_raw_obsm`, `csr_int_counts`, `csr_cpus2` (in-memory thread path, tests that
parallel chunk writes are still deterministic).

## Regenerating

Goldens are regenerated **only** on a deliberate, documented layout change (chunking, codec,
encoding attrs, sharding math, …) — never to make a failing comparison pass. Regenerate with:

```bash
cd tools/convert-to-zarr && pixi run -e dev python \
    ../annizarr/tests/golden/generate.py
```

This overwrites every `tests/golden/*.tar.gz` in place; review the diff before committing.
