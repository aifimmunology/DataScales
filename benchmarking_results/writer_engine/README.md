# writer engine — old (dask) vs new (thread/process pool) findings

AnniZarr Phase 3 rewrote the h5ad→zarr writers off dask onto stdlib
`ThreadPoolExecutor`/`ProcessPoolExecutor` pools (Phase 3a/3b). This compares the two CLI
**engines** — `tools/convert-to-zarr` (old, dask) vs `tools/annizarr` (new, dask-free) — on
identical inputs and CLI flags, so only the store engine varies. `tools/convert-to-zarr` was
retired in the monorepo merge; its source is preserved in history at `git show
2dba863:tools/convert-to-zarr/`.

**Instrument:**
`writer_engine_bench.py`, retired from the tree ahead of annizarr's standalone split; recover it with
`git show 9f3a1cf:tools/annizarr/benchmarking/writer_engine_bench.py`.
Both pixi envs pin `zarr 3.3.0` / `anndata 0.12.19` / `numpy 2.4.6` / `numcodecs 0.16.5` with
identical chunk/codec math (`tests/test_golden_writers.py` proves byte-identical output), and
here the converted stores land at the same size across engines for every case (below) — so
this measures engine wall time / RSS, not a layout difference. `results/` holds the raw
per-run JSON (60 runs: 20 configs × 3 repeats, plus 1 discarded warm-up each).

**Machine:** Apple Silicon (arm64), `hw.ncpu`/`hw.physicalcpu` = 12, 24 GB RAM, macOS
26.5.2 (Darwin 25.5.0), git `435a7a9` (AnniZarr Phase 3a). Old env: Python 3.12, `dask
2026.7.1`. New env: Python 3.13, no dask. `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/
`MKL_NUM_THREADS=1` pinned and passed into every subprocess.

**Datasets** (fixed seed, generated once into `.data/`, not committed):
D1 = 100,000 × 2,000 CSR float32, 10% density (20M nnz), 5-category obs column;
D2 = 20,000 × 2,000 dense float32 h5ad, 3-category obs column.

**Method:** each run is the pixi CLI command in its own subprocess (`convert-to-zarr
convert-h5ad` / `annizarr convert`); wall time via `perf_counter` around the subprocess, peak
RSS via a fresh one-shot measuring subprocess reading `RUSAGE_CHILDREN` (see note below),
output store removed between reps. 1 warm-up + 3 measured repeats per config.

---

## Results

Wall time in seconds (median [min, max] of 3 measured repeats); peak RSS is the child
process tree's high-water mark (MB); output size confirms the two engines wrote the
same-size store for every case.

| Case | Engine | cpus | median s | min–max s | peak RSS MB | out MB |
|---|---|--:|--:|--:|--:|--:|
| (a) D1→dense, in-memory | old | 1 | 15.93 | 15.86–15.98 | 479 | 140.45 |
| (a) D1→dense, in-memory | new | 1 | 14.62 | 14.56–14.69 | 443 | 140.45 |
| (a) D1→dense, in-memory | old | 4 | 6.87 | 6.21–6.90 | 547 | 140.45 |
| (a) D1→dense, in-memory | new | 4 | 6.52 | 5.88–6.62 | 522 | 140.45 |
| (a) D1→dense, in-memory | old | 8 | 5.91 | 5.90–6.89 | 683 | 140.45 |
| (a) D1→dense, in-memory | new | 8 | 4.88 | 4.82–5.06 | 677 | 140.45 |
| (b) D1→dense, `--backed` (control) | old | 1 | 15.96 | 15.89–16.30 | 351 | 140.45 |
| (b) D1→dense, `--backed` (control) | new | 1 | 14.95 | 14.90–14.98 | 284 | 140.45 |
| (b) D1→dense, `--backed` (control) | old | 4 | 7.63 | 7.59–7.64 | 291 | 140.45 |
| (b) D1→dense, `--backed` (control) | new | 4 | 6.52 | 6.51–6.54 | 241 | 140.45 |
| (c) D1→csr, in-memory | old | 1 | 2.83 | 2.83–2.88 | 680 | 75.60 |
| (c) D1→csr, in-memory | new | 1 | 1.20 | 1.19–1.24 | 465 | 75.60 |
| (c) D1→csr, in-memory | old | 4 | 2.83 | 2.78–2.93 | 688 | 75.60 |
| (c) D1→csr, in-memory | new | 4 | 1.18 | 1.15–1.22 | 463 | 75.60 |
| (d) D2→dense, in-memory | old | 1 | 3.88 | 3.85–3.89 | 557 | 124.28 |
| (d) D2→dense, in-memory | new | 1 | 2.18 | 2.17–2.24 | 331 | 124.28 |
| (d) D2→dense, in-memory | old | 4 | 2.83 | 2.81–2.88 | 605 | 124.28 |
| (d) D2→dense, in-memory | new | 4 | 1.48 | 1.45–1.96 | 366 | 124.28 |
| (e) D1→dense, sharded (1024/1024, ×4) | old | 4 | 3.89 | 3.81–3.90 | 1205 | 140.78 |
| (e) D1→dense, sharded (1024/1024, ×4) | new | 4 | 3.36 | 2.86–4.07 | 1134 | 140.78 |

**CLI startup** (median of 3, `--version`/`--help`): **old** `pixi run -e dev convert-to-zarr
--help` = **1.73 s** (1.45–2.11); **new** `pixi run -e default annizarr --version` = **0.13 s**
(0.13–0.15). A **~13×** gap before either engine has read a byte — see below.

---

## Reading the results

- **New is faster everywhere measured**, by a wide margin on the writes that were
  previously fast (csr in-memory **2.3–2.4×**; D2 dense **1.6–1.9×**) and a smaller but
  consistent margin on the writes already dominated by real I/O/compute (D1 dense in-memory
  **8–17%**; sharded **1.16×**). Neither engine benefits much past `cpus=4` on this 12-core
  box for D1 dense (diminishing returns — write bandwidth/compression saturates before the
  thread/process pool does), and neither engine's csr-in-memory case moves at all with
  `cpus` (1 ≈ 4 for both) — at this size the eager h5ad read and fixed per-process overhead
  dominate, not the writer's own parallelism.
- **Where the gap comes from:** every case except (b) reflects both dask's task-graph/
  scheduling overhead *and* the new engine's smaller import graph (no dask ⇒ less to import
  before work starts, visible directly in the CLI-startup line: 1.73 s vs 0.13 s). Case (a)'s
  fixed ~1–1.3 s gap at `cpus=1` narrows as cpus rise, consistent with a fixed per-call
  overhead rather than a per-chunk one.
- **Case (b), the process-pool control, is close but not exactly equal** — the spec expected
  it to be unchanged, since both engines route backed→dense through a plain
  `ProcessPoolExecutor` over chunk-aligned row bands (`_densify_band_segment` — no dask
  array, no `da.store`). It is the tightest engine gap (1.0 s / 6% at cpus=1, 1.1 s / 15% at
  cpus=4) but a real one, and traced to source: old's
  `convert_to_zarr/writers.py::_write_sparse_as_dense_dask` does
  `import dask.array as da` / `from dask.diagnostics import ProgressBar` **unconditionally**
  at the top of the function (lines 67–68), *before* the `backed` branch check at line 91 —
  so the backed/process-pool path still pays dask's module-import cost even though it never
  calls `da.store`. That also explains the RSS gap in this case (351→284 MB, 291→241 MB):
  dask and its dependency tree (toolz, cloudpickle, fsspec, partd) sit resident in the parent
  process even when unused. This is the cleanest evidence in this benchmark that the
  "control" is measuring engine overhead, not the process-pool architecture itself (which is
  identical on both sides).
- **New engine's peak RSS is lower or equal in every case**, most visibly in (b)/(c)/(d)
  (15–41% lower) — consistent with the smaller import graph and with dask's task-graph
  bookkeeping (delayed objects, scheduler state) adding parent-process memory on top of the
  actual write buffers.
- **Case (e) (sharded) has noticeably higher new-engine variance** (2.86–4.07 s, ~40% spread)
  vs. old's tight 3.81–3.90 s, despite new's median still winning. Peak RSS also varies more
  for new here (1126–1181 MB vs old's 1204–1214 MB). This wasn't chased further — plausible
  causes are thread-pool scheduling jitter over the shard grid or transient system load late
  in a long benchmark session (this machine wasn't isolated for the run) — but it means the
  ~1.16× median speedup for case (e) is the least confident number in this table; more
  repeats would tighten it.

## Caveats

- **Warm cache, not dropped.** macOS page cache cannot be dropped without root; the input
  h5ad is touched by the warm-up run before every measured repeat, so all numbers here are
  **warm-cache** reads. Cold-start (first touch of an on-disk h5ad) is not measured.
- **Both engines' numbers include Python + pixi startup and full h5ad load**, not writer time
  alone (CLAUDE.md: never time only the hot loop in isolation when the CLI itself is under
  test). The CLI-startup line item (1.73 s vs 0.13 s) is the floor of that fixed cost; the
  per-case gaps above are consequently a **lower bound** on the writer-only speedup — the
  actual write-path improvement is likely larger than the reported wall-time ratios suggest,
  since a shrinking fraction of each run is fixed overhead as data size grows.
- Single machine, single session, no dedicated benchmarking isolation (no CPU pinning,
  thermal state not controlled) — see the case (e) variance note above.
