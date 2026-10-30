# autoshard — annizarr `chunks.auto_shard` findings

annizarr's `ChunkConfig.auto_shard` (CLI `--auto-shard`) shards our own 1-D sparse `data`/
`indices` arrays with zarr v3 `shards="auto"`, and sets `ad.settings.auto_shard_zarr_v3` so
anndata's own writes (obs/var/obsm/…) auto-shard too. This measures the write-time /
object-count cost against the read-latency effect, to pick the default (dense X is never
auto-sharded — it keeps the explicit `x_shard_factor`, unaffected here).

**Instrument:** [`tools/annizarr`](../../tools/annizarr/) CLI (`convert --auto-shard` /
`--no-auto-shard`) to build the stores; [`tools/zarr-query-bench`](../../tools/zarr-query-bench/)
(`pixi run zarr-bench`) to read them. `results/` holds the raw per-run JSON.

**Machine:** Apple M4 Pro (arm64), 12 physical cores, 24 GB RAM, macOS 26.5.2 (Darwin
25.5.0), git `e79ee9c`. annizarr env: `zarr 3.4.0`, `anndata 0.12.19`, `numpy 2.5.3`,
`scipy 1.18.1`, `numcodecs 0.16.5`. zarr-query-bench env (separate pixi lockfile): `zarr
3.3.0`, `anndata 0.12.19`, `numpy 2.4.6` — a v3 sharded array's shard shape is resolved once
at write time and stored in its `zarr.json`, so the reader's zarr version doesn't change what
gets read; noted for full provenance.

**Dataset:** `benchmarking/.data/D1_sparse_100k.h5ad` — 100,000 × 2,000 CSR float32, 10%
density (20,000,000 nnz), 5-category `cell_type` obs column, fixed seed 42 (from
`writer_engine_bench.py`'s builder).

**Stores** (`annizarr convert D1_sparse_100k.h5ad -o <out> --x-storage <csr|dense>
[--x-shard-factor 1] [--auto-shard|--no-auto-shard]`, all other config left at defaults —
`sparse_flat_chunk=1,000,000`, `x_row_chunk=x_col_chunk=2048`): `csr_auto_off`/`csr_auto_on`
(the pair under test) and `dense_auto_off`/`dense_auto_on` (control — dense X unaffected by
`auto_shard`, only the anndata-written elements differ). `zarr-bench-inspect` on the CSR pair
confirms `data`/`indices` chunks stayed `(1_000_000,)` in both; `auto_on`'s shard shape is
`(2_000_000,)` (zarr's `shards="auto"` heuristic: with `array.target_shard_size_bytes` unset,
a 1-D array spanning more than 8 chunks gets 2 chunks/shard — see `_layout.write_grid` and
CLAUDE.md's "Zarr v3 performance & parallelism").

## Write time and object count

Warm cache (page cache primed with one `cat D1_sparse_100k.h5ad > /dev/null` before timing;
single run each, `pixi run -e default annizarr -q convert ...`, `time` around the CLI):

| Store | write wall s | output size | object count (`find -type f \| wc -l`) |
|---|--:|--:|--:|
| `csr_auto_off` | 1.22 | 76 MB | 62 |
| `csr_auto_on` | 1.16 | 76 MB | **42** (−32%) |
| `dense_auto_off` | 14.52 | 141 MB | 67 |
| `dense_auto_on` | 14.92 | 141 MB | 67 (unchanged) |

Write time is unaffected either way (±5%, within run-to-run noise for a single sample); output
size is identical (sharding changes file count, not the compressed byte total). The CSR
pair's object count drops from 62 → 42 (20 fewer chunk files: 20 `data` + 20 `indices` chunks
become 10 + 10 shards) — the object-count win `auto_shard` is meant to buy. The dense pair's
count is unchanged: dense X isn't sharded either way, and the sharded-but-single-chunk
`obs`/`var`/`obsm` arrays (100,000 rows fits in one chunk) don't add or remove files.

## Read latency: `csr_auto_off` vs `csr_auto_on`

`pixi run zarr-bench --store <store> --axis row --format csr --repeats 25 --warmup 3` (warm
cache; `celltype` at `--repeats 1 --warmup 0` — see caveat below) for `sequential`/`random`
`--count 500`, and `--mode celltype --obs-column cell_type --obs-value type_a` (D1's
`cell_type` is uniformly random per row, so the matched rows scatter into 15,967 runs of ~1.3
rows each on this **unsorted** store — a legitimately slow, scattered query, the same
pathology `zarr-query-bench`'s own docs describe).

| Mode | off median | on median | off p95 | on p95 | on vs off (median) | chunks fetched (off → on) |
|---|--:|--:|--:|--:|--:|--:|
| sequential (500 rows) | 6.87 ms | 9.36 ms | 7.22 ms | 10.08 ms | **+36% slower** | 2 → 2 |
| random (500 rows) | 28.27 ms | 35.26 ms | 38.49 ms | 46.36 ms | **+25% slower** | 40 → 20 |
| celltype (`type_a`, 19,981 rows, 1 rep) | 114.50 s | 132.00 s | — | — | **+15% slower** | 31,956 → 31,950 |

Every mode's `cpu_wall` (decompress + gather), not `io_wall`, dominates the wall time in both
stores (e.g. celltype: 103 s / 114.5 s off, 127 s / 132.0 s on) — this is a CPU/decode-path
cost, not a network- or disk-bound one.

## Decision

**`auto_shard` defaults to `False`.** The rule: auto-shard wins only if its median is within
5% of (or faster than) unsharded on *every* mode. Here it's **15–36% slower on all three** —
not close. `random` fetches half as many chunk objects (40 → 20, sharding's intended win) yet
is still slower, and `celltype`'s near-identical fetch count (31,956 vs 31,950) with a real
gap shows the cost isn't the object count at all: each individual chunk access through the
sharding codec pays a small extra indirection (locate the chunk inside its shard's index) on
top of the plain per-chunk read, and on this workload — an already-fast **local, warm-cache**
read, many small fetches — that per-access tax outweighs having fewer files. `--auto-shard`
stays available for anyone who wants the object-count win explicitly (e.g. a store expected to
sit on a remote/object store where per-object listing and inode pressure, not per-fetch CPU,
is the bottleneck), but that scenario is **not tested here** — this benchmark is local-disk
only, and CLAUDE.md's "don't assume a faster method based on what direction we want to go"
applies equally to assuming a *remote* win from a *local* loss. Someone benchmarking a `gs://`
or `s3://` target should re-run this before flipping the default.

## Caveats

- **`celltype` is 1 repeat, no warmup** (114–132 s each; 25×3 like the other modes would cost
  well over an hour for this pair alone) — no p95/spread for that row. The two numbers are
  still directly comparable (same store, same query, same process), just less precise than
  the 25-repeat sequential/random medians.
- **Warm cache, not dropped.** macOS page cache can't be dropped without root; all reads here
  are warm (first-touch/cold-start is not measured). Local disk only — no `gs://`/`s3://` run.
- **Single machine, single session** — no isolation from other processes, no repeated
  sessions to bound day-to-day variance.
- The multi-hundred-MB stores built for this benchmark were deleted after the runs above
  (`benchmarking/.data/autoshard_out/` is not committed).
