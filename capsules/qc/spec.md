# `qc` — what the build changed from the main spec, and why

The main spec (`capsules/spec.md` §4) is the design this capsule was built from. This file lists
where the built capsule differs and why; everything not listed here is as §4 describes. Usage is in
`code/README.md`.

## Inputs
| Input | Supported | Why |
|---|---|---|
| AnnData icechunk repo (local, data asset, `s3://`) | yes | as §4 |
| AnnData `.zarr` (local or data asset) | yes | as §4 |
| SpatialData `.zarr` (local or data asset) | yes, new | Xenium tables are reviewed with the same loop (see Spatial) |
| SpatialData icechunk repo | refused | out of scope; stock `spatialdata` can't read from icechunk yet |
| plain `.zarr` on `s3://` | refused | out of scope until annizarr reads plain zarr through obstore |

## App panel (§4.2)
- `apply = true/false` became **`mode`**: a list, `report` (default) or `apply`.
- Parameters are grouped into three categories: *1. Data (start here)*, *2. Save the passing cells*,
  *3. Advanced*. A first-time user fills in the input and runs.
- The plain-zarr output of `apply` is a **local** path only (§4.2 allowed S3), matching the input rule.
- `THREADS` is gone: annizarr sizes its own workers (see Writing).

## Profiles (new: `profiles.py`)
Each assay's metrics, cell rules and default thresholds live in one profile, so spatial support
didn't touch the scRNA rules. `qc_decisions.json` records its `profile` and refuses the other kind.

| | `scrna` (AnnData) | `spatial` (SpatialData table) |
|---|---|---|
| Rules (fail when) | as §4.3: doublet, mito ≥ 10, genes ≤ 200, genes ≥ 5000 | `transcript_counts < 20`, genes `< 5`, `nucleus_count < 1` |
| Missing metrics | stop, unless `compute missing metrics` | computed from X (only the absent columns) |
| Ambient indicator, mito | yes | no |
| Embedding cells per sample | 1,000 | 20,000 (a slide is often one sample) |
| Samples | `sample_id` | `sample_id`, else the table's region column, with a warning |

The spatial defaults come from 10x Xenium lung (11,898 cells × 289 genes): its 1% / 5% quantiles
were 11 / 20 transcripts and 8 / 13 genes. `transcript_counts` is used rather than Xenium's
`total_counts`, which also counts control probes. A tissue plot per sample is added.

## Apply: where it goes (§4.10)
- **icechunk:** the curated cells are written on a session based on the fingerprinted `main`
  snapshot and saved with `session.flush()`, which makes a snapshot without moving any branch.
  Only then is `qc` created at it or reset to it (`reset_branch` checks that nobody moved `qc`
  meanwhile).
  - **Why:** §4.10 pointed `qc` at `main` first and wrote after. Tested on icechunk 2.3: a failed or
    killed write then leaves `qc` holding the raw cells. With flush, a failed write leaves no
    trace, so no staging branch is needed.
  - **Lineage:** `qc`'s snapshot has the reviewed `main` snapshot as its parent.
- **No garbage collection.** Reclaiming old curations is left to the user, and annizarr 0.2.4 has
  no `garbage_collect`.
- **SpatialData `.zarr`:** the output is a new local zarr holding the root attributes and the curated
  `tables/<name>`. The images and labels stay in the input; the table keeps its region links.

## Apply: writing (§4.10)
§4.10 assumed an `annizarr.subset` op. annizarr 0.2.4 has none, so the capsule adds one small piece
and leaves the rest to annizarr:
- **`rows.py`:** the passing rows of a CSR matrix as an annizarr reader (about 50 lines).
- **X and CSR layers:** written by annizarr's `write_adata` / `write_matrix` from that reader.
- **Layers annizarr's `add_layer` made** (they record their source matrix, e.g. a CSC copy of X):
  re-derived from the curated matrix in their own layout, instead of being cut. A CSC layer would
  otherwise have to be read whole.
- **Other layers, obs, obsm:** filtered to the passing rows. var, varm, varp, uns unchanged; obsp
  dropped, as §4.10.
- **Workers and compression are annizarr's defaults:** its CPU count honours the container's CPU
  limit (new in 0.2.4), and matrices are written at Blosc-zstd level 3. `indptr` is int32 or int64
  as annizarr picks (§4.10 said int64).

Measured on this capsule's container (1 CPU allowed by its quota, though 16 are visible), keeping
80% of cells; the output's X is identical in every case:

| Store | Hand-written banded copy | annizarr 0.2.3 | annizarr 0.2.4 |
|---|---|---|---|
| 400k cells, 600M nonzeros | 64 s, 3.6 GB | 105 s, 6.4 GB | 51 s, 1.1 GB |

annizarr 0.2.3 planned 16 parallel bands on one usable core, which is where its memory went.

## Code layout (§2.3)
`config`, `source`, `profiles`, `metrics`, `samples`, `decisions`, `embed`, `ambient`,
`celltypes`, `report`, `plots`, `rows`, `write`, `main`. Tests: `pytest` in `code/qc_check`
(93 tests: unit tests, plus report and apply runs on synthetic stores of every supported kind).

## Known gaps
- **Private annizarr pieces:** `rows.py` subclasses annizarr's internal CSR reader, and `write.py`
  calls `annizarr._writers` and `annizarr.ops._layer._write_layer`. The public `add_layer` takes a
  path or a `Repo`, but apply writes into an open icechunk session or a `tables/<name>` group.
  A public subset op, or ops that take a zarr group, would remove these.
- **A layer derived from `raw/X`** is cut, not re-derived, because `raw` is dropped.
- **`spatialdata.read_zarr`** reads the SpatialData output but warns that the annotated element
  (e.g. `cell_labels`) isn't in it.
- **App panel keys** `"type": "file"` (decisions picker) and `"required": true` (input) are
  unverified in the App Builder.
