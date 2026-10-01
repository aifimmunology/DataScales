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

Nine cases, one per distinct writer code path:

| Case | Path exercised |
|---|---|
| `csr_eager` | in-memory CSR -> CSR; input is the layers/raw/obsm fixture, pinning `write_adata`'s layers/raw paths |
| `csr_backed_small_flat` | backed CSR -> CSR, `sparse_flat_chunk` far below nnz: process-pool flat copy, several chunks |
| `csc_eager` | in-memory CSR -> CSC (`write_transposed_sparse`) |
| `dense_from_sparse_backed_small_chunks` | backed CSR -> dense, small ragged row/col chunks: process-pool densify, several blocks per axis with a ragged tail |
| `dense_sharded_cpus2` | in-memory CSR -> sharded dense, `x_shard_factor=2`, `cpus=2`: threaded whole-shard blocks |
| `dense_from_dense_eager` | in-memory dense -> dense |
| `concat_csr` | concat of two eager CSR inputs |
| `sorted_backed` | backed input, `sort_by=["cell_type","batch"]`: streamed sorted bucketing |
| `csr_autoshard` | `auto_shard=True`, `sparse_flat_chunk` small enough that the sparse arrays actually shard |

## Regenerating

Goldens are regenerated **only** on a deliberate, documented layout change (chunking, codec,
encoding attrs, sharding math, …) — never to make a failing comparison pass. Since these are
self-snapshots of annizarr's own output, regenerate with the package installed in this repo's
own env, no old converter and no git worktree needed:

```bash
cd tools/annizarr && pixi run -e default python tests/golden/generate.py
```

This overwrites every `tests/golden/*.tar.gz` in place; review the diff before committing.

**2026-09-29 regeneration** (pruned to 9 cases): `csc_eager`, `dense_from_dense_eager`,
`concat_csr`, and `sorted_backed` kept the same parameters as before and were diffed
byte-for-byte against the previous tarballs before being overwritten — unchanged.
`csr_autoshard` (renamed from `csr_eager_autoshard`) also produced a byte-identical
`expected/` store under its new name. `csr_eager` (now built from the layers/raw/obsm
fixture instead of the plain one), `csr_backed_small_flat`,
`dense_from_sparse_backed_small_chunks`, and `dense_sharded_cpus2` are new combinations of
parameters and are new by definition. The other thirteen previous cases were folded into
these nine or dropped as redundant with a library-level test.
