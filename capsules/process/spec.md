# `process` capsule: changes and why

The shared design is in [`../spec.md`](../spec.md) §5. This sheet records what was built or changed in
this capsule, and why. Newest decisions win where they disagree with older text.

## 1. What the capsule is for
- **The RAPIDS environment first.** The capsule exists to hand out a working GPU single-cell environment
  (RAPIDS, rapids-singlecell, Dask-CUDA, scanpy, scvi-tools/torch, squidpy, spatialdata), plus guide
  notebooks for out-of-core and spatial analysis.
  - *Why:* other capsules and people need the environment more than they need one fixed pipeline.
- **No App Panel.** The pipeline runs from a terminal, `./run --input <store> [--flag value]`. Every setting
  is a flag (`./run --help`) and an upper-case environment variable.
  - *Why:* the analysis is exploratory; a form would freeze choices that belong in a notebook. Plain flags
    keep `run` small.
- **Users bring their own data asset.** Nothing is bundled; notebooks point at `/data/<asset>/...`.

## 2. Layout
```
code/
├── README.md       using the environment and notebooks, and `run`
├── environment/    requirements.txt and pixi.toml
├── notebooks/      out-of-core and spatial guides
├── run             ./run --input <store>
└── process/        the pipeline (config, cluster, steps, store, main) and tests
```
- **Environment files moved into `code/`** (from `capsules/process/environment/`).
  - *Why:* `code/` is what people copy or download from a capsule, so the package list should travel with
    it. `environment/` at the capsule root keeps only `postInstall`, like the other capsules.

## 3. Environment
- **Code Ocean builds from the pip3 list on the environment page.** `requirements.txt` is a copy of that
  list (32 pins) for pasting into another capsule's **pip3 → Edit Bulk**, or for `uv pip install -r`.
  - Base image `codeocean/cuda12-runtime-uv:main-5692e50`, Python 3.13, CUDA 13 wheels.
  - Checked 2026-10-29: every installed version matches its pin. `jupyter==1.1.1` was on the page but
    missing from the files, so it was added.
- **`pixi.toml` is for use outside Code Ocean only** (`CONDA_OVERRIDE_CUDA=13 pixi install`). Code Ocean
  never reads it. **It has not been solved yet**: installing pixi from inside the capsule was blocked.
- **annizarr stays pinned at 0.2.3**, matching the environment page. The code also works on 0.2.4 (§7).
- **Why exact pins:** this is an environment, not a library. The CUDA 13 RAPIDS wheels, torch and
  `cuda-toolkit` have to match each other, and zarr/anndata/spatialdata change fast. annizarr 0.2.4 is an
  example: it renamed `add_expr`. Without pins, a rebuild quietly installs whatever is newest that day.
- **`MPLBACKEND=Agg` is set by the image.** Both notebooks start with `%matplotlib inline`.
  - *Why:* without it, JupyterLab shows no plots.

## 4. Pipeline (`run`)
HVG → scale → PCA → optional harmony → neighbors → UMAP → leiden, plus `layers/lognorm`, on a
Dask-CUDA cluster with managed memory. Defaults follow `10-Rapids_GPU_Analysis.ipynb`.

- **rapids-singlecell 0.18 takes `rng=`**, not `random_state=`, for PCA/neighbors/UMAP/leiden.
- **lognorm is computed on the GPU and streamed to disk** block by block (anndata writes a dask CSR one
  chunk at a time from the client, so it's safe with an icechunk session). The layer gets a `target_sum`
  attr.
  - *Why:* one writer for every target (icechunk, in place, copies, SpatialData tables). annizarr's
    normalizing `add_expr` only worked on an AnnData at the store root, and is gone in 0.2.4.
- **`obsm/X_scVI` is no longer deleted** by a run.
  - *Why:* this pipeline doesn't make it; another step will.
- **Batch keys with NA values are refused**, and batch keys are only required when integrating.

## 5. Where results go, and the safety fixes
| Input | Results |
|---|---|
| icechunk repo | one commit on `--branch` (default `qc`); a read-only mount needs `--origin` |
| writable `.zarr` | in place |
| read-only `.zarr`, or `--output` given, or `--output-mode results-only` | a copy, default `/results/<stem>.processed.zarr` |
| remote plain `.zarr` | refused; use icechunk on S3 |

Fixes from two independent reviews, each with a test:
- **The store changing during a run.** icechunk: the branch tip is checked before the commit. Plain zarr: a
  fingerprint (X attrs and nnz, obs index; not `column-order`, which this pipeline rewrites) is checked
  before writing.
  - *Why:* results written in another cell order land on the wrong cells, silently.
- **A run that dies midway on a consolidated store.** Consolidated metadata is removed from the root and
  every group down to the table before the first change, written atomically, then the root is always
  consolidated again at the end.
  - *Why:* stale consolidated metadata made deleted elements read back as zeros, and made the "complete"
    marker (`uns/process`, written last) look present.
- **`--output` / `results-only` with a writable input** now always copy.
  - *Why:* they were ignored, so the input was modified.
- **`results-only` keeps everything but the counts** (`X`, `layers`, `raw`), not just obs and var.
- **Copies are built in `<dest>.partial-<pid>` and renamed into place.** A failure removes only its own
  partial copy.
  - *Why:* a failed copy could delete an output another run had just finished.
- **Cluster or NVML shutdown errors are logged, not raised.** `cluster_summary.csv` is cleared at the start
  and written before `run.json`.
  - *Why:* a run could be reported as failed after its results had landed.
- **A failed icechunk write discards its session**, and the branch tip stays where it was.

## 6. SpatialData and icechunk
- **A SpatialData store is processed through its table.** When the root has no `X`, `run` and both
  notebooks use `tables/<--table>` (default `table`). Only that group is written; images, labels, points
  and shapes are never changed. A read-only full copy copies the whole store; `results-only` gives just
  the table as an AnnData `.zarr`.
- **To edit just the table with history, use an icechunk repo laid out like a SpatialData store.** Tested:
  a `run` commit and a notebook save each touch only `/tables/table` in icechunk's diff. On a plain
  SpatialData `.zarr`, image and shape files were byte-identical after a run.
  - *Why icechunk:* plain zarr also allows table-only edits, but with no history and no way to undo.
- **Why only the table, not the whole object, in icechunk** (spatialdata 0.8.0, checked against earthmover's
  `2026-scverse/icechunk-SpatialData`):
  - `sdata.write()` and `sd.read_zarr()` don't accept an icechunk store; that needs the open spatialdata
    PR #1107 plus earthmover's patches, and their own advice is not to build on it yet.
  - spatialdata's element writers work on an icechunk session for images, labels, tables and zarr-encoded
    shapes. Points and GeoParquet shapes fail.
  - So: write the table with `anndata.io.write_elem`, read it with `read_elem(root["tables/<name>"])`, and
    keep a plain SpatialData `.zarr` for tools that need the whole object (napari, `sd.read_zarr`).
  - annizarr's ops assume the AnnData at the repo root, which is another reason `run` writes lognorm
    itself.

## 7. annizarr 0.2.4 review (2026-10-30)
- **Breaking:** `add_expr` was renamed `add_layer`, and now only copies a matrix in another layout, with no
  normalization or `target_sum`. This capsule no longer calls it, so nothing breaks. The layer attr was
  renamed from `annizarr_target_sum` to `target_sum`, since annizarr no longer writes or reads it.
- **Other changes:** the matrix copy moved into `_writers/_copy.py` (used by `rechunk`), and the CPU count
  respects cgroup quotas (`_core/_system.py`). That suits Code Ocean containers.
- **Not in 0.2.4:** `subset` and `Repo.garbage_collect`, which `../spec.md` §8 planned for qc apply.
- **Checked:** the 38 tests pass against both 0.2.3 and 0.2.4. The `Repo` API this capsule uses (`exists`,
  `create`, `open_zarr`, `log`, `commit`, `discard`, `checkout`, `dirty`) is unchanged.

## 8. Notebooks
- **`rapids_scrna_out_of_core_dask.ipynb`** is rebuilt as 9 short steps matching `run`. It reads an AnnData
  or SpatialData `.zarr`, or an icechunk repo holding either. It only reads.
- **`rapids_spatial_atera_breast_cancer.ipynb`:**
  - It reads the table from a SpatialData `.zarr` or an icechunk repo. The shapes are no longer required,
    and the `nucleus_count` filter is skipped when that column is missing.
  - Integer counts are cast to float32 before the move to the GPU.
    - *Why:* `anndata_to_GPU` refused int32.
  - **New optional §7 save (`SAVE = True`).** It writes the cell types, clusters, niches and embeddings
    into the table under their own names (`spatial_cell_type`, `spatial_leiden`, `spatial_niche`,
    `X_pca_spatial`, `X_umap_spatial`). It writes only those columns, refuses when the icechunk branch
    moved since loading, and checks that the cell IDs are unique and match.
    - *Why:* so it never replaces `run`'s results or puts values on the wrong cells.

## 9. Tested, and not
| What | Result |
|---|---|
| 38 tests (11 full GPU runs): AnnData, SpatialData, icechunk, failure paths | pass on the T4, with annizarr 0.2.3 and 0.2.4 |
| Out-of-core notebook, headless, on a SpatialData `.zarr` and an icechunk repo | every cell runs |
| Spatial notebook, headless, 25,600 synthetic cells, as `.zarr` and icechunk, with saving on | every cell runs, ~56 s, 0.6 GB GPU memory |
| 1M cells × 10k genes with harmony | ~3.5 min, 3.3 GB GPU / 7.3 GB RAM at peak (before the lognorm change) |

**Not tested yet:**
- real data;
- more than one GPU;
- the pixi solve;
- a 1M-cell rerun with the GPU lognorm writer.

## 10. Open
- **scVI / scANVI integration**, in a subprocess (`../spec.md` §5.2 step 5).
- **Code Ocean versioned data assets** (AllenNeuralDynamics/aind-code-ocean-info#11; private, not read yet).
  The question is whether a new asset version copies everything, or can point at an icechunk repo on S3,
  where commits cost very little.
- **ingest and qc still pin annizarr 0.2.3.** Neither calls `add_expr`; bumping them is up to their own
  testing.
