# Single-cell curation capsules — spec

Four Code Ocean capsules that take HISE scRNA samples from raw files to a curated, processed store:

| # | Capsule | Hardware | Writes | Job |
|---|---|---|---|---|
| 1 | `ingest/` | CPU, sized to the largest **sample** | `main` | Per-sample QC metrics, CellTypist, Scrublet; append to the raw store. No whole-store analysis |
| 2 | `qc_check/` | CPU | nothing | QC report for proposed thresholds, with a downsampled embedding; emits the decisions + cell mask |
| 3 | `qc_apply/` | CPU, I/O-bound | the `qc` branch, or a new zarr | Writes the passing cells as the curated object, exactly as the reviewed `qc_check` run decided |
| 4 | `process/` | **GPU, CUDA 12** (rapids-singlecell + Dask, scvi-tools) | a commit on the `qc` branch, or a new zarr | HVG, PCA, optional Harmony or scVI, neighbors, UMAP and leiden over the curated cells |

```
samples ─► ingest ─► main (raw, all cells)
                       │
                       ├─► qc_check ─► report + decisions ◄── person reviews, reruns qc_check
                       │                     │
                       └──────────► qc_apply ┘
                                       │
                                       ▼
                          qc (curated cells only; latest curation)
                                       │
                                    process ─► + embeddings, clusters
```

Cell labeling and cluster-level doublet cleanup (standardized notebooks 06–08) come after `process`
and are a separate stage. Existing stores built by the old pipeline aren't migrated; stores are
rebuilt from samples with the new ingest.

The capsules are meant to be shared and run by anyone on their own data: every location (data
asset, local path, S3) comes in through the app panel, nothing is hard-coded, and there are no
per-user permissions.

### What QC reviews, at three levels
| Level | What the report shows | What the reviewer can do |
|---|---|---|
| **Dataset** (all uncurated cells) | distributions with the thresholds drawn; batch and chip views; a downsampled UMAP split by batch and chip | change the **global** thresholds (one set for every cell) |
| **Sample** | one row per sample; only **flagged** samples need a look | keep (default), **exclude**, or **override** a threshold for that sample |
| **Cell** | nothing to review | — the thresholds decide each cell |

### Relation to the standardized notebooks
| Notebook | Capsule | Differences |
|---|---|---|
| 01 pipeline catch-up | `ingest` | CellTypist and Scrublet run per sample at ingest |
| 03 dataset QC | `qc_check` | Same idea: downsample, then HVG/PCA/UMAP/leiden on CPU, plus grouped plots |
| 05 apply QC filters | `qc_check` + `qc_apply` | Same rules and order; see the table below |
| 10 rapids GPU analysis | `process` | Same steps and defaults, as a capsule |
| 02 ambient RNA, 06–09 | — | Not covered (§4.8) |

Deliberate differences from notebook 05:

| Topic | Notebook 05 | Capsules |
|---|---|---|
| Column names | `pct_counts_mito`, `predicted_doublet` | the store's `pct_counts_mt`, `predicted_doublets` |
| Gene count | prefers `n_genes` if present | always `n_genes_by_counts`; the store's `n_genes` is a different HISE field |
| Scrublet NA | dropped (`== False` is false for NA) | **kept**, and counted per sample as "doublet status unknown" |
| Per-cell-type thresholds | none | optional overrides (§4.5), off by default |

The default thresholds (`min_genes 200`, `max_genes 5000`, `max_mito 10`) are **PBMC defaults**.

---

## 1. Repo layout and deployment

```
capsules/
├── spec.md                 this file
├── environments.json       every capsule's packages (§7.2)
├── ingest/                 ┐
├── qc_check/               │ each folder is one complete capsule root
├── qc_apply/               │
└── process/                ┘
    ├── code/               run (entrypoint), <package>/, tests/
    ├── environment/        postInstall (and the Dockerfile Code Ocean generates)
    ├── .codeocean/         environment.json (the environment page), resources.json
    ├── metadata/
    └── .gitignore
```

- **Development:** develop in a full DataScales clone (branch `codeocean_capsules`), where every capsule is visible side by side.
- **Deployment:** each folder goes to its own capsule with `git subtree push --prefix=capsules/<name> co-<name> master`. Edits made inside Code Ocean come back with `git subtree pull`.
- **Shared code:** there is no shared package. Each capsule depends on `annizarr` for store I/O and keeps its own copy of the few column names it needs (§2.2).
- **Clutter:** inside the deployed capsules, git-ignore any reference clones (`code/DataScales/`, `code/standardized_notebooks/`).

---

## 2. Shared conventions

### 2.1 Inputs
| Input | Opened with | Used by |
|---|---|---|
| icechunk repo (data-asset mount, local path, `s3://bucket/prefix`) | `annizarr.Repo(path, origin=...)`, then `open_zarr("r")` at a branch tip | all four |
| plain AnnData `.zarr` (local path, data-asset mount or `s3://`), not in icechunk | `zarr.open_group(path, mode="r")` | all three QC/process capsules |

- **Detecting the kind:** with `annizarr.Repo.exists(path)`.
- **Supported locations:** local zarr, data-asset zarr, icechunk on S3, and icechunk on a data asset.
- **Where results go** (`qc_apply` and `process`):

  | Input | `qc_apply` | `process` |
  |---|---|---|
  | icechunk | the curation branch (default `qc`) in the repo | a commit on that branch |
  | plain zarr, writable (local or S3) | a new zarr (cells are removed, so the input can't be reused) | **in place**: the results are added to the input |
  | plain zarr, read-only (e.g. a data asset) | a new zarr | a new zarr (§6.4) |

  - **icechunk must be writable.** That means S3, a writable local repo, or a data-asset mount given its `origin`. Icechunk is never copied out to a zarr; a read-only repo without an `origin` is refused.
  - **Plain zarr writability is probed,** not guessed: write and delete a small `.write_probe` key at the store root. Data-asset mounts fail the probe.
- **`.h5ad`:** out of scope. Convert it first with `annizarr convert in.h5ad out.zarr`.
- **Same code after opening:** both kinds give a `zarr.Group` root, and everything after that is shared.

Reading rules:
- **Read obs column by column.** Use `anndata.io.read_elem(root["obs"][col])` for only the columns needed.
  - Never `read_elem(root["obs"])`: per-cell string columns (`cell_name`, `cell_uuid`, `original_barcodes`) make full obs about 1 KB per cell.
  - Convert sample, group and label columns to `category` on read.
- **X is CSR only.** Dense X is an error.
  - CPU reads slice `indptr`, `data` and `indices` directly, in bands of whole storage chunks.
  - Outputs always use an int64 `indptr`; int32 overflows at about 2.1B nonzeros, roughly 1M cells.

### 2.2 Column contract
Written per sample by the ingest:

| Purpose | Column | Type |
|---|---|---|
| Sample key | `sample_id` | categorical |
| Group keys | `batch_id`, `pool_id`, `chip_id` | categorical |
| Genes detected | `n_genes_by_counts` | Int64 |
| Total UMIs | `total_counts` | float64 |
| Mito % | `pct_counts_mt` | float64 |
| Doublet score / call | `scrublet_score`, `predicted_doublets` | float64 / Int64. NA = Scrublet failed or found no threshold |
| Cell-type label | `celltypist_label`, `celltypist_prob`, `celltypist_model` | categorical / float64 / categorical |
| Hashing / ADT QC (report only) | `hto_category` (multiplexed pools), `adt_qc_flag` (CITE-seq) | categorical. Empty for samples without hashing or CITE-seq |

Written by `process`: `obs/leiden`, `obsm/X_pca`, `obsm/X_pca_harmony` or `obsm/X_scVI` (if integrated), `obsm/X_umap`, `var/highly_variable`, `layers/lognorm`, `uns/process`.

### 2.3 Entrypoint and code style
- **Entrypoint:**
  - `code/run` passes app-panel parameters positionally. An empty value falls back to an env var, then a default.
  - One `main.py`, with a `<Capsule>Error(RuntimeError)` for user-facing failures (logged as one line, exit code 1).
  - Every run writes `run.json` under `/results`: parameters, input fingerprint, versions, timestamps, peak memory.
- **Modular within each capsule:** small modules with one job each.
  - `ingest`: `config`, `hise`, `annotate`, `contract`, `store`, `main`.
  - `qc_check`: `config`, `source`, `metrics`, `decisions`, `embed`, `report`, `main`.
  - `qc_apply`: `config`, `source`, `verify`, `write`, `main`.
  - `process`: `config`, `cluster` (Dask-CUDA setup), `steps`, `store`, `main`.
- **No comments.** Names and short docstrings on public functions carry the meaning.
- **Not overengineered:**
  - no plugin layers or abstract base classes;
  - no options nobody asked for;
  - branch on input kind only where icechunk and plain zarr actually differ.
- **Tests sit beside the code** and run against small synthetic stores. `process` tests need a GPU and are skipped without one.

---

## 3. `ingest` — append samples only

### 3.1 What changes from today
- **Removed:**
  - all whole-store analysis: `derive.py`, `run_derived`, `working_copy`, `write_derived`;
  - `leiden` from the schema;
  - `n_clusters` from the summary;
  - creating the `lognorm` layer (`ensure_layer`).
- **Kept:**
  - per sample: read → QC metrics → CellTypist → Scrublet → project to the contract → append;
  - all appends staged on one `Repo` with `commit=False` and committed once per run;
  - the bootstrap seed commit on a new store.
- **Effect:** memory is set by the largest single sample, since annizarr streams the append. `main` holds only X (raw counts), obs and var, with no derived elements, so `append` never needs `drop_derived` or `extend_layers`.
- **Scrublet:** when it finds no threshold, `predicted_doublets` is NA and the scores are kept. This is done, and tested.

### 3.2 Outputs
- **The commit on `main`:** its metadata records `pipeline`, `annizarr` version, `appended_samples`, `n_cells` and `n_samples`.
- **`/results`:**
  - `ingest_summary.csv`, per sample: cells, mean genes, mean mito %, doublets, whether Scrublet scored it, number of cell types, model;
  - `ingest_run.json`.

---

## 4. `qc_check` — the report (read-only)

### 4.1 App panel
| # | Parameter | Default | Notes |
|---|---|---|---|
| 1 | input | — | icechunk repo (reads `main`) or plain zarr |
| 2 | decisions | *(none)* | A `qc_decisions.json` (attached data asset). None → defaults, plus the decisions on the curation branch (§4.6) |
| 3 | branch | `qc` | icechunk only: the curation branch whose decisions are carried forward (§4.6) |
| 4 | group columns | `batch_id,pool_id,chip_id` | A column that's absent or has a single value is skipped |
| 5 | per-cell-type QC | `false` | Optional in v1. Turns on §4.5 and the `cell_types` overrides |
| 6 | label column | `celltypist_label` | Used only when per-cell-type QC is on |
| 7 | compute missing metrics | `false` | Allow the X-based metrics fallback (§4.3) |

The sample column is fixed to `sample_id`, the store's own column.

### 4.2 Decisions file (`qc_decisions.json`)
This file is the contract between `qc_check` and `qc_apply`.

```json
{
  "version": 1,
  "input": {"kind": "icechunk", "location": "s3://…", "fingerprint": "<main snapshot id>"},
  "label_column": null,
  "defaults": {"min_genes": 200, "max_genes": 5000, "max_mito": 10, "filter_doublets": true},
  "cell_types": {},
  "samples": {
    "SP0001": {"exclude": true, "note": "low viability"},
    "SP0002": {"max_genes": 7000, "note": "deeper sequencing"},
    "SP0003": {}
  },
  "pending": ["SP0010"],
  "note": ""
}
```

- **Precedence:** `defaults` → `cell_types[label]` → `samples[id]`. The most specific entry wins, key by key.
  - `exclude: true` removes all of a sample's cells.
  - `{}` means reviewed, defaults are fine.
  - `cell_types` applies only when per-cell-type QC is on and the label column exists; otherwise it's ignored, with a warning.
- **`pending`:** new samples (in the input, in no curation yet) that are **flagged** (§4.3).
  - New samples without flags are accepted with the defaults automatically and written into `samples` as `{}`.
  - A reviewer only looks at new samples that look off.
- **Cell rules.** Each rule is evaluated independently, with 05's strict inequalities:

  | Rule | A cell fails when |
  |---|---|
  | `sample_excluded` | its sample has `exclude: true` |
  | `doublet` | `filter_doublets` and `predicted_doublets == 1` |
  | `high_mito` | `pct_counts_mt >= max_mito` |
  | `low_genes` | `n_genes_by_counts <= min_genes` |
  | `high_genes` | `n_genes_by_counts >= max_genes` |

  - A cell passes when no rule fails.
  - `qc_fail_reason` is the first failing rule in this order, which is 05's order.
  - Per-rule counts in the report count every failing rule.
  - Doublet NA passes.
- **Fingerprint:**
  - icechunk: the `main` snapshot id.
  - plain zarr: SHA-256 over:
    - the root, `obs` and `X` attrs;
    - the X shape and `indptr[-1]`;
    - the obs index, streamed chunk by chunk.

### 4.3 Per-sample section
- **Missing metrics.** If `n_genes_by_counts`, `total_counts` or `pct_counts_mt` are absent (typical for a converted external zarr):
  - by default, stop: *"Store lacks QC metrics; rerun with 'compute missing metrics' (reads all of X, about 16 KB per cell)"*;
  - when allowed, compute them from X in bands of storage chunks:
    - with `np.add.reduceat` per band;
    - with an `MT-` mask over `var` names;
    - with a small thread pool;
    - using the same definitions as `scanpy.pp.calculate_qc_metrics`.
  - If X isn't integer counts, stop.
- **Mito sanity check:** if `var` has no `MT-` genes, warn at the top of the summary.
- **Per-sample metrics** (`sample_qc.csv`), from a pandas groupby on numeric columns:
  - cell count;
  - median and MAD of genes, UMIs and mito %;
  - doublet rate and doublet-NA count;
  - per-rule and overall fail %;
  - cell-type composition when labels are on;
  - `hto_category` and `adt_qc_flag` counts where present;
  - status (`reviewed` / `pending` / `excluded`);
  - flags.
- **Flags.** Advisory only; they never change a decision.
  - `few_cells`: fewer than 500 cells.
  - `doublet_rate`: zero, or above 15%.
  - `doublet_unknown`: any doublet-NA cells.
  - `outlier:<metric>`: the median genes, UMIs or mito % lies more than 3 MADs from the cohort median.
    - Needs at least 5 samples.
    - The MAD has a floor of 5% of the median.
    - Fail % is never a flag input.
- **Plots** (PNG, Agg backend, dpi 120, figures closed after saving):
  - Box plots of genes, UMIs and mito % per sample, from quantiles of all cells, with threshold lines.
    - At most 50 samples per figure, as pages.
    - Flagged samples first.
  - Cells per sample before and after filtering, and fail % per sample stacked by reason.
  - The same box plots and fail-% bars for each group column, plus cells per sample by group.
  - Mito % against genes for each flagged sample (at most 20k random cells each).

### 4.4 Downsampled embedding (dataset level)
This follows notebook 03: a small, fixed-size embedding that shows batch structure whatever the store's size.

- **Sampling:**
  - up to 1,000 cells per sample, capped at 200k cells in total;
  - fixed seed;
  - QC-failed cells included, so they can be seen.
  - Cells of a sample are stored contiguously, so the sampling is **chunk-aware**: take the needed cells from one or two randomly chosen storage chunks per sample, instead of scattering reads across every chunk.
- **Steps (scanpy, CPU):** `normalize_total(1e4)` → `log1p` → HVG (2,000, `seurat`) → `scale(max_value=10)` → PCA (50) → neighbors (15, 30 PCs) → UMAP → leiden (1.0).
- **Plots:**
  - UMAP coloured by sample, by each group column, by fail reason, and by `celltypist_label`;
  - UMAP split by each group column, plus density maps (raw and mean-subtracted), as in notebook 03;
  - leiden composition per cluster, by sample and by each group column (batch mixing). A cluster made almost entirely of one sample or chip points to a batch effect, or to a sample-specific population.
- **No batch correction, on purpose.** The embedding is meant to show batch effects as they are; Harmony or scVI here would hide exactly what the reviewer is looking for. Correction happens in `process`, on curated cells.
- **Use:** report only. It helps decide whether `process` needs integration and on which column.

### 4.5 Per-cell-type section (optional in v1)
- **Off (default):** skipped silently.
- **On, but the label column is absent:**
  - the section is skipped;
  - a warning goes to the log and the top of `report_summary.txt`: *"No `celltypist_label` column; per-cell-type QC skipped. Re-ingest with CellTypist on, or set a label column."*;
  - the run still succeeds.
- **On, with labels** (`celltype_qc.csv`, plus plots):
  - per label: cell count, medians and MADs, per-rule fail %;
  - a heatmap of fail % per label × sample;
  - box plots of genes and mito % per label, with that label's effective thresholds;
  - `celltypist_prob` per label.
- **Caveat:** labels are provisional and least reliable in low-quality cells. QC never changes labels.

### 4.6 Incremental runs (icechunk)
- **The curation branch (parameter 3, default `qc`) is the starting point,** if it exists. Its `uns/qc/decisions` (JSON string) supplies the decisions, and reviewed samples keep their entries.
- **New samples on `main`** are flagged or accepted as described in §4.2.
- **If a decisions file is passed:** it overrides key by key.
- **The summary's first line says whether `qc` is current:** *"qc covers 3 of 5 samples on main (curated 2026-10-20 from 1CECHNKR); 2 new samples (1 flagged, pending: SP0010)."*

### 4.7 Outputs (`/results`)
```
qc_decisions.json      the effective decisions (edit and rerun, or hand to qc_apply)
cell_qc.parquet        one row per cell in store order: sample (dict), qc_pass (bool),
                       qc_fail_reason (dict); plus the three metrics (float32) only if the
                       fallback computed them. zstd, ~1M-row groups
sample_qc.csv          §4.3
celltype_qc.csv        §4.5 (absent when skipped)
embedding.parquet      the downsampled cells: row position, UMAP, leiden
plots/samples/  plots/groups/  plots/embedding/  plots/celltypes/
report_summary.txt     curation status, headline counts, warnings, flags, pending samples
run.json
```
- **`cell_qc.parquet`'s file metadata** holds the input fingerprint and the SHA-256 of the canonical `qc_decisions.json`.
- **To feed `qc_apply`:** capture the reviewed run's results as a data asset.

### 4.8 Not in v1
- **Ambient RNA (notebook 02, SoupX):** it needs each sample's raw droplet matrix, which the store doesn't hold. If wanted, it belongs before or inside the ingest.
- **Ribosomal and hemoglobin %:** later. Adding them means a new ingest column and a backfill of existing cells.
- **Adaptive per-sample thresholds:** cutoffs computed automatically from each sample's own distribution (e.g. median ± 3 MADs). No review is needed, but cutoffs differ per sample in ways that are harder to explain. v1 uses global thresholds plus overrides for flagged samples.
- **Integration:** belongs to `process`, not QC.

---

## 5. `qc_apply` — write the curated object

### 5.1 App panel
| # | Parameter | Default | Notes |
|---|---|---|---|
| 1 | input | — | The same store `qc_check` read |
| 2 | qc_check results | — | Data asset holding `qc_decisions.json` + `cell_qc.parquet` |
| 3 | origin | *(none)* | icechunk only: the writable location behind a read-only mount |
| 4 | output | `qc` / `/results/<input stem>.qc.zarr` | icechunk: the curation branch name. Plain zarr: the output zarr path (local or S3) |
| 5 | reviewer | — | Required. Recorded in `uns/qc` and the commit metadata |

### 5.2 Refuse to run when
- **The input changed since the report** (the fingerprint differs). *"Input changed since this report; rerun qc_check."*
- **The decisions were edited after the report** (the hash differs). *"Decisions edited after the report; rerun qc_check to preview them."*
- **There are pending samples**, i.e. `pending` is non-empty.
- **The row count differs:** `cell_qc.parquet` must have one row per cell.
- **icechunk is read-only** and no `origin` was given.

`qc_apply` never evaluates thresholds. It keeps exactly the rows `qc_check` marked `qc_pass`.

### 5.3 Output: the passing cells only
Both input kinds go through one writer, `write.copy_rows(src_root, dst_root, keep)`.
- **Copied filtered, in bands of whole storage chunks:** X (and any layers), keeping passing rows via the mask over each band's `indptr` slice. They're written into pre-sized arrays with the source's chunking and compression, with an int64 `indptr`.
- **Copied filtered row-wise:** `obs` (all columns) and any `obsm`.
- **Copied unchanged:** `var`, `varm` and `uns`.
- **Dropped:** `obsp`.
- **Added:** `uns/qc`: `decisions` (JSON string), `cell_qc_sha256`, `source` (snapshot or plain-zarr fingerprint), `samples`, `reviewer`, per-sample before/after counts.

Where it goes:
- **icechunk:** one curation branch (default **`qc`**, or the name given), which always holds **only the latest curation** made under that name. Different names give independent curations, e.g. a strict and a lenient one.
  - **Point `qc` at the reviewed snapshot:**
    - create it on first use (`checkout(name, create=True)` while on `main`);
    - otherwise reset it with `cherrypick(fingerprint)`.

    Either way, `qc` becomes the fingerprinted `main` snapshot.
  - **Write it:** `open_zarr("w", truncate=True)`, then `copy_rows` from the `main` snapshot, then one commit: `commit(msg, metadata={pipeline: "qc_apply", created, source_snapshot, samples, decisions_sha256, reviewer, n_pass, n_cells})`. The message reads like `QC curation: 3 samples, 61,204 / 65,766 cells pass (source 1CECHNKR)`.
  - **The previous curation** (and any `process` commits on it) is then no longer on any branch.
  - **Reclaiming its storage:** icechunk's `garbage_collect(delete_object_older_than=…)` deletes objects that no branch or tag can reach.
    - It runs after the commit, with a 24-hour margin, so that an ingest writing at the same time can't lose uncommitted chunks.
    - `main` and its whole history stay reachable, so nothing raw is touched.
    - annizarr's `Repo` has no GC method (§10).
  - **`main` is untouched,** so ingest appends keep working.
- **plain zarr:** a new zarr at the output path. The input is never modified, writable or not.
- **Cost:** the curated copy rereads and rewrites X, about 16 KB per cell (160 GB at 10M cells), mostly I/O. Only the latest curation is kept, once GC has run.
- **Also written:** `apply_summary.csv` (per sample: before, after, removed by reason) and `run.json`. The removed cells are the `qc_pass == false` rows of `cell_qc.parquet`.

---

## 6. `process` — embeddings and clusters on GPU

This follows `standardized_notebooks/scRNA/rapids-sc/10-Rapids_GPU_Analysis.ipynb`, plus an optional scVI path.

### 6.1 App panel
| # | Parameter | Default | Notes |
|---|---|---|---|
| 1 | input | — | Any location in §2.1: an icechunk repo, or a curated plain zarr |
| 2 | branch | `qc` | icechunk only: the curation branch to process |
| 3 | origin | *(none)* | icechunk only: the writable location behind a read-only mount |
| 4 | integration | `none` | `none`, `harmony` or `scvi` |
| 5 | batch keys | *(none)* | Comma-separated obs columns (e.g. `chip_id`). Required unless integration is `none` |
| 6 | output mode | `results-only` | Read-only plain zarr only (§6.4): `results-only` or `full` |
| 7 | leiden resolution | `1.1` | ≈ scanpy 1.0, per the notebook |

Fixed in v1, with the notebook's values:

| Setting | Value |
|---|---|
| Rows per Dask chunk | 24,000, rounded to a multiple of the storage row chunk |
| HVG | 2,000, `seurat` flavor (the only one that works with Dask) |
| Scale | `zero_center=False`, `max_value=10`, float64 for this step |
| PCA | 50 components |
| Neighbors | 20 neighbors, 30 PCs (or the full scVI latent); `ivfflat` (`n_probes=20`) on one GPU |
| UMAP | `min_dist=0.45`, spectral init |
| Leiden | 100 iterations |
| scVI | `n_latent=30`, `n_layers=2`, `gene_likelihood="nb"`, `batch_size=1024`, scvi-tools' default epoch count with early stopping |
| Seed | 4242 |

### 6.2 Steps
1. **Dask-CUDA cluster:**
   - `LocalCUDACluster` over all visible GPUs;
   - `rmm_managed_memory=True` and `enable_cudf_spill=True`, so GPU memory spills to host RAM;
   - `dask.config.set({"distributed.scheduler.worker-ttl": None})`;
   - RMM managed memory with the cupy allocator.
2. **Open the curated cells read-only:**
   - X: `anndata.experimental.read_elem_lazy(root["X"], (rows_per_chunk, n_vars))`, cast to float32;
   - obs: only `sample_id` and the batch keys;
   - var as is.
3. **Normalize:** `rsc.get.anndata_to_GPU`, `rsc.pp.normalize_total`, `rsc.pp.log1p`.
4. **HVG and PCA:** HVG, subset to the HVGs, rechunk across workers and persist; scale; PCA, persisted and computed.
5. **Integration:**
   - **`harmony`:** `rsc.pp.harmony_integrate(key=batch_keys)` on `X_pca` → `X_pca_harmony`.
   - **`scvi`**, in a **separate Python subprocess**:
     - PyTorch and RMM/cupy in one process compete for GPU memory: torch's caching allocator and RMM each hold their own reservations, and OOM is likely. A fresh process gets a clean CUDA context and frees everything on exit.
     - **Before:** close the Dask-CUDA cluster (its workers exit and release their GPUs), free the client's cupy pool, then write the HVG raw counts (CSR, streamed in row bands) and the combined `_batch` label to a temporary zarr under `/scratch`. scVI trains on counts, not lognorm.
     - **The subprocess** (`python -m process.scvi_train`, which never imports RMM):
       - `scvi.model.SCVI.setup_anndata(adata, batch_key="_batch")`;
       - train on the GPU;
       - write `get_latent_representation()` to the temporary zarr;
       - save the model to `/results/scvi_model/` for reuse (e.g. scANVI at the labeling stage).
     - **After:** the main process reads `X_scVI` back.
6. **Graph and clusters** (single GPU, no Dask needed): neighbors on `X_pca`, `X_pca_harmony` or `X_scVI`, then UMAP, then leiden. `rsc.pp.harmony_integrate` accepts the batch keys as a list directly.
7. **lognorm layer:** `annizarr.add_expr(layout="csr", layer="lognorm", target_sum=1e4)`, streaming on CPU. It's there for the labeling stage (markers, dotplots).
8. **Write** (where: §6.4):
   - `obsm/X_pca`, `obsm/X_pca_harmony` or `obsm/X_scVI`, `obsm/X_umap`;
   - `obs/leiden` (categorical, added to `column-order`);
   - `var/highly_variable`;
   - `layers/lognorm`;
   - `uns/process`: all parameters, the number of GPUs, package versions, the PCA variance ratio, and the source (branch and snapshot, or plain-zarr fingerprint);
   - on icechunk, commit metadata `{pipeline: "process", curation_branch, source_snapshot, integration, batch_keys, n_cells, n_clusters}`.

   The neighbor graph (`obsp`) isn't written in v1, because it's large and recomputable.
9. **`/results`:**
   - UMAP PNGs coloured by `leiden`, `sample_id`, the batch keys and `celltypist_label`;
   - `cluster_summary.csv` (cells per cluster, by sample);
   - `run.json` with per-step timings and peak GPU and host memory.

### 6.3 Reruns (icechunk)
Rerunning `process` on the same branch, for example with a different integration, adds another
commit. The derived elements are replaced, and the previous result stays in the branch's history.

### 6.4 Where results go
- **icechunk:** one commit on the given branch, through `Repo(input, origin=origin)` with `open_zarr("w")` and `add_expr(..., commit=False)`. A read-only repo without an `origin` is refused.
- **Writable plain zarr:** in place. The new elements are written into the input zarr, replacing earlier `process` results; nothing else is touched.
- **Read-only plain zarr:** a new zarr at `/results/<input stem>.processed.zarr`.
  - **`results-only` (default):**
    - holds `obs` (all columns), `var`, the new `obsm`, `obs/leiden`, `var/highly_variable`, `layers/lognorm` and `uns/process`;
    - has **no X**, which anndata reads back as `X=None`;
    - is small apart from `lognorm`;
    - `uns/process/source` records the input it pairs with.
  - **`full`:** first a full streamed copy of the input with `annizarr.rechunk(input, output=…)`, then the results written onto it. It's a self-contained object, at the cost of a full copy of X.
    - `rechunk` rewrites X with the configured chunking and stream-copies everything else.
    - It's verified to carry X, obs, var, layers, obsm, varm, obsp and uns, from both a plain zarr and an icechunk branch.
    - `convert` refuses a store as input.

---|---|
  | ingest | scanpy, scrublet, annoy (apt `g++` to compile it), celltypist |
  | qc_check | scanpy, leidenalg, igraph, umap-learn (downsampled embedding); pyarrow, matplotlib, seaborn |
  | qc_apply | pyarrow |
  | process | rapids-singlecell-cu12, cudf/cuml/cugraph/cuvs/rmm-cu12, dask-cuda, cupy-cuda12x, scanpy, matplotlib (page); torch + scvi-tools stack (`postInstall`) |

- **numpy is pinned to 2.4.6 everywhere.**
  - cudf 26.10 requires `numba < 0.65`, and numba 0.64 supports only `numpy < 2.5`. That's why notebook 10 pins `numpy < 2.5`.
  - The CPU capsules resolve with 2.4.6 too, so `core` stays identical.
- **The `process` stack is verified to resolve** (186 packages) with a `pip install --dry-run`:
  - on CUDA 12: RAPIDS 26.10 cu12 + torch 2.13.0+cu129 + scvi-tools 1.5.1, with the `core` pins;
  - it pulls CUDA toolkit pip packages 12.9.1 throughout, and dask/distributed 2026.7.1.
  - **It hasn't been installed or run on a GPU yet.**
- **Why torch is in `postInstall`:**
  - PyPI's torch builds after 2.10 are CUDA 13.
  - PyPI's torch 2.10.0 (CUDA 12.8) pins `nvidia-nvjitlink-cu12==12.8.93`, but RAPIDS 26.10 needs `≥ 12.9`.
  - So torch comes from PyTorch's CUDA 12.9 index, in `postInstall`:

    ```bash
    pip3 install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu129 \
        "torch==2.13.0+cu129" "scvi-tools==1.5.1" "lightning==2.6.6" "torchmetrics==1.9.0" "scikit-misc==0.5.3"
    ```

    scvi-tools and its stack go in the same command, so the page can't pull PyPI's CUDA 13 torch first.

### 7.3 Base images
- **CPU capsules:** `codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04`.
- **`process`:** one of your Code Ocean CUDA 12 GPU environments with Python 3.12.
  - The pip wheels bring their own CUDA 12.9 runtime libraries, so the image's own CUDA toolkit version doesn't matter much.
  - The host **NVIDIA driver** must be at least 525.60.13 (the CUDA 12 minimum), and 550 or newer is preferred.
  - CUDA minor-version compatibility lets the 12.9 runtime run on older 12.x drivers, but JIT-compiled (PTX) kernels won't, so a newer driver avoids surprises.
- **annizarr:** 0.2.3 from PyPI. It's published, imports on Python 3.12 (the `TypeIs` fix is in), and the ingest's 20 tests pass against it.

### 7.4 Dev tooling (every image)
- **On the page:** `pytest` (in `core`) and the VS Code extensions already listed.
- **In `postInstall`:** only Claude Code, pinned: `curl -fsSL https://claude.ai/install.sh | bash -s <version>`.

### 7.5 Resource classes
| Capsule | Class |
|---|---|
| ingest | `small`; it depends on the largest sample, not the store |
| qc_check | `small` up to about 1M cells; 16–32 GB around 10M. The embedding is fixed-size |
| qc_apply | I/O-bound; give it cores rather than memory. Roughly 12 minutes per 1M cells from local disk, slower from S3 |
| process | One GPU with ≥ 15 GB VRAM handled 5M cells in about 15 minutes in the notebook's benchmark; above about 13M cells, ≥ 30 GB VRAM. Host RAM must absorb the spill |

---

## 8. Tests (per capsule, `code/tests`)

**`ingest`**
- appends without any derived elements on `main`;
- one commit per run, plus the bootstrap seed;
- the schema has no `leiden`;
- Scrublet with no threshold gives NA calls and keeps scores (done);
- the existing per-input tests (HISE, 10x, h5ad, variable-length strings, ADT features).

**`qc_check`**
- precedence, `exclude`, and `{}`;
- the strict-inequality boundaries;
- reason order versus per-rule counts;
- NA doublet passes;
- each flag, including no outlier flags below 5 samples and the MAD floor;
- unflagged new samples are accepted as `{}`, and flagged ones go to `pending`;
- the mito warning;
- the metrics fallback: refused unless allowed, and matches stored expected values when run;
- per-cell-type off and on, including the missing-label warning;
- only the listed obs columns are read;
- the plain-zarr fingerprint is stable, and changes when cells are added or obs is rewritten;
- chunk-aware sampling reads at most two storage chunks per sample;
- the embedding is deterministic for a seed;
- the input is never written.

**`qc_apply`**
- refuses on each condition in §5.2;
- `qc`'s parent is the fingerprinted snapshot, and its metadata is complete;
- the output holds exactly the passing rows, with X, layers, obs and obsm equal to the input's rows, an int64 `indptr`, and no `obsp`;
- `main` is untouched, and a later ingest append works;
- a second curation after new samples resets `qc` onto the new snapshot, and GC removes the first curation's objects but nothing reachable from `main`;
- plain zarr writes the same content to the output path, and never touches the input;
- a custom branch name creates an independent curation, and `qc` is left alone;

**`process`** (GPU only)
- on a small curated branch, writes every element in §6.2 in one commit;
- `harmony` and `scvi` each write their own embedding, and `none` writes neither;
- integration without batch keys is refused;
- plain-zarr input: `results-only` has no X and reads back with `anndata.read_zarr`; `full` equals the input plus the results;
- read-only icechunk without an origin is refused;
- a writable plain zarr gets the results in place; a read-only one gets a new zarr;
- `leiden` is in `column-order`, and the result is readable with `anndata.read_zarr`;
- a rerun adds a commit and replaces the derived elements.

---

## 9. Build order
1. annizarr 0.2.3 on PyPI, with the `TypeIs` fix. **Done.**
2. `ingest`: remove the whole-store analysis (§3), move its environment to the page (§7), and rebuild a store from samples.
3. `qc_check`: icechunk first, then plain zarr. Review the report on the rebuilt store.
4. `qc_apply`: icechunk first, then plain zarr.
5. `process` on a GPU capsule, against the first `qc` branch.
6. Subtree-push each capsule to its own Code Ocean capsule.

## 10. Open questions
- **GPU driver:** run `nvidia-smi` in your CUDA 12 GPU environment and note the driver version. ≥ 550 is preferred; ≥ 525.60.13 is the minimum.
- **Garbage collection in annizarr:** add a public `Repo.garbage_collect(older_than)` (wrapping icechunk's)? Otherwise `qc_apply` must reach the icechunk repository through annizarr's private `_writer_repo()`.
- **First GPU run:** the stack resolves but hasn't run. The first `process` build needs a real test of torch and RAPIDS loading in the same environment, and of the scVI subprocess handoff.
