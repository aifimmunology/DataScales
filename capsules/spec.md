# Single-cell curation capsules — spec

Three Code Ocean capsules that take HISE scRNA samples from raw files to a curated, processed store:

| # | Capsule | Hardware | Writes | Job |
|---|---|---|---|---|
| 1 | `ingest/` | CPU, sized to the largest **sample** | `main` | Per-sample QC metrics, CellTypist, Scrublet; append to the raw store. No whole-store analysis |
| 2 | `qc/` | CPU; I/O-bound when applying | nothing by default; with `apply`, the `qc` branch or a new zarr | QC report for proposed thresholds, with a downsampled embedding. With `apply` and reviewed decisions, writes the passing cells as the curated object |
| 3 | `process/` | **GPU, CUDA 13** (rapids-singlecell + Dask, scvi-tools) | a commit on the `qc` branch, or a new zarr | HVG, PCA, optional Harmony or scVI, neighbors, UMAP and leiden over the curated cells |

```
samples ─► ingest ─► main (raw, all cells)
                       │
                       ▼
                  qc (report) ─► report + qc_decisions.json ◄── person reviews, edits, reruns
                       │                     │
                       ▼                     │ the reviewed file
                  qc (apply) ◄───────────────┘
                       │
                       ▼
          qc branch (curated cells only; latest curation)
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
| 03 dataset QC | `qc` | Same idea: downsample, then HVG/PCA/UMAP/leiden on CPU, plus grouped plots |
| 05 apply QC filters | `qc` | Same rules and order; see the table below. Applied only with `apply` |
| 10 rapids GPU analysis | `process` | Same steps and defaults, as a capsule |
| 02 ambient RNA | `qc` | Detection only (§4.6); correction needs raw droplets, not in v1 |
| 06–09 | — | Not covered (§4.12) |

Deliberate differences from notebook 05:

| Topic | Notebook 05 | Capsules |
|---|---|---|
| Column names | `pct_counts_mito`, `predicted_doublet` | the store's `pct_counts_mt`, `predicted_doublets` |
| Gene count | prefers `n_genes` if present | always `n_genes_by_counts`; the store's `n_genes` is a different HISE field |
| Scrublet NA | dropped (`== False` is false for NA) | **kept**, and counted per sample as "doublet status unknown" |
| Per-cell-type thresholds | none | optional overrides (§4.7), off by default |

The default thresholds (`min_genes 200`, `max_genes 5000`, `max_mito 10`) are **PBMC defaults**.

---

## 1. Repo layout and deployment

```
capsules/
├── spec.md                 this file
├── ingest/                 ┐
├── qc/                     │ each folder is one complete capsule root
└── process/                ┘
    ├── code/               run (entrypoint), <package>/, tests/
    ├── environment/        postInstall; Code Ocean generates its own Dockerfile
    ├── .codeocean/         environment.json (the environment page), resources.json
    ├── metadata/
    └── .gitignore
```

- **Development:** develop in a full DataScales clone (branch `codeocean_capsules`), where every capsule is visible side by side.
- **Updating a capsule checkout during development:** copy the capsule's `code/` entries, its `.codeocean/environment.json` and `environment/postInstall`, and `.codeocean/resources.json` over their namesakes in the checkout (e.g. `/root/capsule`). Data assets, secrets, reference clones and the Dockerfile are left alone. No reset is needed; an environment change applies on the next restart (rebuild).
- **Deployment:** each folder goes to its own capsule with `git subtree push --prefix=capsules/<name> co-<name> master`. Edits made inside Code Ocean come back with `git subtree pull`.
- **Shared code:** there is no shared package. Each capsule depends on `annizarr` for store I/O and keeps its own copy of the few column names it needs (§2.2).
- **Clutter:** inside the deployed capsules, git-ignore any reference clones (`code/DataScales/`, `code/standardized_notebooks/`).

---

## 2. Shared conventions

### 2.1 Inputs
| Input | Opened with | Used by |
|---|---|---|
| icechunk repo (data-asset mount, local path, `s3://bucket/prefix`) | `annizarr.Repo(path, origin=...)`, then `open_zarr("r")` at a branch tip | all three |
| plain AnnData `.zarr` (local path, data-asset mount or `s3://`), not in icechunk | `zarr.open_group(path, mode="r")` | `qc` and `process` |

- **Detecting the kind:** with `annizarr.Repo.exists(path)`.
- **Supported locations:** local zarr, data-asset zarr, icechunk on S3, and icechunk on a data asset.
- **Where results go** (`qc` with `apply`, and `process`):

  | Input | `qc` with `apply` | `process` |
  |---|---|---|
  | icechunk | the curation branch (default `qc`) in the repo | a commit on that branch |
  | plain zarr, writable (local or S3) | a new zarr (cells are removed, so the input can't be reused) | **in place**: the results are added to the input |
  | plain zarr, read-only (e.g. a data asset) | a new zarr | a new zarr (§5.4) |

  - **icechunk must be writable.** That means S3, a writable local repo, or a data-asset mount given its `origin`. Icechunk is never copied out to a zarr; a read-only repo without an `origin` is refused.
  - **Plain zarr writability is probed,** not guessed: write and delete a small `.write_probe` key at the store root. Data-asset mounts fail the probe.
  - **A `qc` report run never needs write access.** Writability is checked only when `apply` is set.
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
| Cell-type labels | `<model>_label`, `<model>_prob` per CellTypist model; by default `pbmc_clean_AIFI_L1_2024`, `_L2_2024`, `_L3_2024` | categorical / float64 |
| Hashing / ADT QC (report only) | `hto_category` (multiplexed pools), `adt_qc_flag` (CITE-seq) | categorical. Empty for samples without hashing or CITE-seq |

Written by `process`: `obs/leiden`, `obsm/X_pca`, `obsm/X_pca_harmony` or `obsm/X_scVI` (if integrated), `obsm/X_umap`, `var/highly_variable`, `layers/lognorm`, `uns/process`.

### 2.3 Entrypoint and code style
- **Entrypoint:**
  - `code/run` passes app-panel parameters positionally. An empty value falls back to an env var, then a default.
  - One `main.py`, with a `<Capsule>Error(RuntimeError)` for user-facing failures (logged as one line, exit code 1).
  - Every run writes `run.json` under `/results`: parameters, input fingerprint, versions, timestamps, peak memory.
- **Modular within each capsule:** small modules with one job each.
  - `ingest`: `config`, `hise`, `annotate`, `contract`, `store`, `main`.
  - `qc`: `config`, `source`, `metrics`, `decisions`, `embed`, `report`, `write`, `main`.
  - `process`: `config`, `cluster` (Dask-CUDA setup), `steps`, `store`, `main`.
- **No comments.** Names and short docstrings on public functions carry the meaning.
- **Not overengineered:**
  - no plugin layers or abstract base classes;
  - no options nobody asked for;
  - branch on input kind only where icechunk and plain zarr actually differ.
- **Tests sit beside the code** and run against small synthetic stores. `process` tests need a GPU and are skipped without one.

---

## 3. `ingest` — append samples only (built)

The source is `capsules/ingest/`. Its `code/` is copied into the capsule checkout (`/root/capsule/code`, §1), so the capsule keeps working once this repo is gone. The README there describes it in full.

### 3.1 What it does
- **Store:** an icechunk repo (one commit per run, plus a seed commit when a run creates the store), or a path ending in `.zarr` for a plain AnnData zarr appended in place.
- **Per sample, one in memory at a time:** read (HISE / 10x / h5ad), QC metrics, CellTypist, Scrublet, project onto the contract, stage as a zarr. Samples with no cells are skipped with a warning.
- **The store is touched only after every sample is staged,** so a failing sample leaves either kind unchanged.
  - On icechunk, a failed append discards the run.
  - A plain zarr can't roll back, which the README states.
- **`main` holds only X (raw counts), obs and var.** There is no whole-store analysis, so memory depends on the largest sample.
- **Scrublet:** NA calls when it fails or finds no threshold; scores are kept when it only finds no threshold.
- **Create:** explicit.
  - It refuses a path that already holds a store, and a non-empty folder that isn't one.
  - An empty repo left by a failed create run can be reused.
- **Read-only locations are refused before any work:** an icechunk repo without an `origin`, or an unwritable zarr.

### 3.2 App panel
`samples`, `store`, `origin`, `celltypist models`, `create`. Every other setting is an environment variable named after its `Settings` field, in upper case.

### 3.3 Outputs
- **The commit on `main`:** its metadata records `pipeline`, `annizarr` version, `appended_samples`, `n_cells` and `n_samples`.
- **`/results`:**
  - `ingest_summary.csv`, per appended sample: cells, mean genes, mean mito %, doublets, whether Scrublet scored it, number of cell types, model;
  - `ingest_run.json`: store, kind, snapshot ids, appended samples, totals, annizarr version, peak memory, settings.

### 3.4 Verified
- 63 tests: unit tests, plus end-to-end runs on both store kinds covering mixed inputs, CellTypist (a locally trained model) and Scrublet, failure and rerun semantics, read-only and create guards, and config parsing. They pass against annizarr 0.2.3; annizarr 0.2.1 fails the icechunk runs.
- The five real B092 samples (127,800 cells) on both kinds, with Scrublet. Stored counts, barcodes and HISE obs match each source file exactly, and the icechunk history is one commit per run.
- The bundled AIFI L1/L2/L3 models on real samples: labels and probabilities identical to running CellTypist directly, as `06-label_h5s.py` does.
- **CellTypist:** several models per store (default AIFI L1, L2, L3, bundled in `code/models` with a `SOURCE.json` of commit and checksums), columns `<model>_label` / `<model>_prob`. The model set is part of the schema.
- **HISE layouts:** tenx-scrna-pipeline's observation names (`Batch`, `Pool`, `Chip`, `uuid`, …) are mapped onto the older HISE ones; labels already in a file are recomputed.

---

## 4. `qc` — review, then apply

One capsule, two modes. Every run computes the decisions, the cell mask and the full report. Only a
run with `apply` set writes anything, and only with decisions that a previous report run produced
and that haven't changed since.

### 4.1 The two modes
| | Report (default) | Apply (`apply = true`) |
|---|---|---|
| Decisions | defaults → carried forward from the curation branch (§4.8) → the file, if given | **exactly the given file**, nothing merged |
| Input access | read-only | read, plus write to the curation branch or output zarr |
| Report | yes | yes, the same report, so `/results` documents exactly what was written |
| Writes | nothing | the passing cells (§4.10) |
| Preconditions | none | §4.9, all checked **before** any report work |

The review loop:
1. Run without `apply`. Read the report and `qc_decisions.json`.
2. To change anything, edit `qc_decisions.json`, attach it (as a data asset), and run **without** `apply` again. That run previews the edit and writes a fresh `qc_decisions.json`.
3. When the report is right, run with `apply` and that run's `qc_decisions.json`.

If an ingest lands between steps 1 and 3, `main`'s snapshot changes, so the fingerprint no longer matches and `apply` refuses (§4.9). Run step 1 again; reviewed samples keep their entries.

### 4.2 App panel
| # | Parameter | Default | Notes |
|---|---|---|---|
| 1 | input | — | icechunk repo (reads `main`) or plain zarr |
| 2 | decisions | *(none)* | A `qc_decisions.json` (attached data asset). Report: none → defaults plus the curation branch's decisions (§4.8). Apply: required |
| 3 | apply | `false` | `true` writes the curated object, after the checks in §4.9 |
| 4 | branch / output | `qc` / `/results/<input stem>.qc.zarr` | icechunk: the curation branch, read for carried-forward decisions and written by `apply`. Plain zarr: the output zarr path (local or S3), used only by `apply` |
| 5 | origin | *(none)* | icechunk only: the writable location behind a read-only mount. Needed only for `apply` |
| 6 | reviewer | *(none)* | Optional. Recorded in `uns/qc` and the commit metadata when given |
| 7 | group columns | `batch_id,pool_id,chip_id` | A column that's absent or has a single value is skipped |
| 8 | label column | `pbmc_clean_AIFI_L1_2024_label` | Groups cells for the ambient indicator (§4.6) and, when on, per-cell-type QC (§4.7); any `<model>_label` column works |
| 9 | per-cell-type QC | `false` | Optional in v1. Turns on §4.7 and the `cell_types` overrides |
| 10 | compute missing metrics | `false` | Allow the X-based metrics fallback (§4.4) |

The sample column is fixed to `sample_id`, the store's own column.

### 4.3 Decisions file (`qc_decisions.json`)
This file is what a reviewer edits and what `apply` writes from.

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
  "note": "",
  "previewed_sha256": "<SHA-256 of this file without this key, canonical JSON>"
}
```

- **Precedence:** `defaults` → `cell_types[label]` → `samples[id]`. The most specific entry wins, key by key.
  - `exclude: true` removes all of a sample's cells.
  - `{}` means reviewed, defaults are fine.
  - `cell_types` applies only when per-cell-type QC is on and the label column exists; otherwise it's ignored, with a warning.
- **`pending`:** new samples (in the input, in no curation yet) that are **flagged** (§4.4).
  - New samples without flags are accepted with the defaults automatically and written into `samples` as `{}`.
  - A reviewer only looks at new samples that look off, and clears them from `pending` by giving them an entry in `samples`.
- **`previewed_sha256`:** set by every report run, over the canonical JSON (sorted keys, no whitespace) of the file minus this key. A hand edit changes the content but not this value, which is how `apply` knows the edit hasn't been previewed.
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

### 4.4 Per-sample section
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
  - `ambient:<set>`: a marker set at ≥ 1.5× the cohort median (§4.6).
- **Plots** (PNG, Agg backend, dpi 120, figures closed after saving):
  - Box plots of genes, UMIs and mito % per sample, from quantiles of all cells, with threshold lines.
    - At most 50 samples per figure, as pages.
    - Flagged samples first.
  - Cells per sample before and after filtering, and fail % per sample stacked by reason.
  - The same box plots and fail-% bars for each group column, plus cells per sample by group.
  - Mito % against genes for each flagged sample (at most 20k random cells each).

### 4.5 Downsampled embedding (dataset level)
This follows notebook 03: a small, fixed-size embedding that shows batch structure whatever the store's size.

- **Sampling:**
  - up to 1,000 cells per sample, capped at 200k cells in total;
  - fixed seed;
  - QC-failed cells included, so they can be seen.
  - Cells of a sample are stored contiguously, so the sampling is **chunk-aware**: take the needed cells from one or two randomly chosen storage chunks per sample, instead of scattering reads across every chunk.
- **Steps (scanpy, CPU):** `normalize_total(1e4)` → `log1p` → HVG (2,000, `seurat`) → `scale(max_value=10)` → PCA (50) → neighbors (15, 30 PCs) → UMAP → leiden (1.0).
- **Plots:**
  - UMAP coloured by sample, by each group column, by fail reason, and by `pbmc_clean_AIFI_L1_2024_label`;
  - UMAP split by each group column, plus density maps (raw and mean-subtracted), as in notebook 03;
  - leiden composition per cluster, by sample and by each group column (batch mixing). A cluster made almost entirely of one sample or chip points to a batch effect, or to a sample-specific population.
- **No batch correction, on purpose.** The embedding is meant to show batch effects as they are; Harmony or scVI here would hide exactly what the reviewer is looking for. Correction happens in `process`, on curated cells.
- **Use:** report only. It helps decide whether `process` needs integration and on which column. It is never written to the store.

### 4.6 Ambient RNA indicator (on by default)
This is **detection, not correction**: no counts are changed. Correction (SoupX, notebook 02) needs each sample's raw droplet matrix, which the store doesn't hold (§4.12). This step flags samples whose cells carry an unusual amount of another lineage's highly expressed transcripts, the typical signature of ambient RNA, so the reviewer can exclude them or give them stricter thresholds.

- **Cells used:**
  - the downsampled cells of §4.5, so no extra X reads;
  - predicted doublets are left out, because a doublet shows another lineage's genes for its own reason;
  - cells are grouped by the label column (parameter 8, default `pbmc_clean_AIFI_L1_2024_label`).
- **Marker sets** (fixed in v1, PBMC). Each is measured in cells that shouldn't express it:

  | Set | Genes | Measured in |
  |---|---|---|
  | `monocyte` | `LYZ`, `S100A8`, `S100A9` | T cell, B cell, NK cell |
  | `hemoglobin` | `HBB`, `HBA1`, `HBA2` | every cell not labelled Erythrocyte |
  | `immunoglobulin` | `IGKC`, `IGHM`, `IGLC2` | T cell, NK cell, Monocyte |

- **Per sample, per set** (columns in `sample_qc.csv`):
  - `ambient_<set>_pct`: the median, over those cells, of the set's share of each cell's UMIs, in %;
  - `ambient_<set>_p90`: the 90th percentile;
  - `ambient_<set>_ratio`: the median divided by the cohort's median.
- **Flag** `ambient:<set>` when the ratio is **≥ 1.5**. It needs at least 5 samples, and is advisory like every other flag. A relative rule, because absolute levels depend on chemistry and tissue.
- **Skipped,** with a warning in the summary, when the label column is absent or a set has none of its genes in `var`. A set missing only some genes uses the rest, and the report names the missing ones.
- **Plots:** per-sample bars of each set's median, flagged samples first, with the 1.5× line; and a heatmap of the ratios (sets × samples).
- **Measured on the five B092 samples:**
  - monocyte-gene medians were 0.02–0.03% for four samples and 0.045% for SP02423 (1.8×);
  - its hemoglobin and immunoglobulin ratios were about 1.5×;
  - SP02423 also has the highest mito % and the fewest genes per cell, so all three sets would flag it and nothing else.

### 4.7 Per-cell-type section (optional in v1)
- **Off (default):** skipped silently.
- **On, but the label column is absent:**
  - the section is skipped;
  - a warning goes to the log and the top of `report_summary.txt`: *"No `pbmc_clean_AIFI_L1_2024_label` column; per-cell-type QC skipped. Re-ingest with CellTypist on, or set a label column."*;
  - the run still succeeds.
- **On, with labels** (`celltype_qc.csv`, plus plots):
  - per label: cell count, medians and MADs, per-rule fail %;
  - a heatmap of fail % per label × sample;
  - box plots of genes and mito % per label, with that label's effective thresholds;
  - the label probability (`<model>_prob`) per label.
- **Caveat:** labels are provisional and least reliable in low-quality cells. QC never changes labels.

### 4.8 Incremental runs (icechunk)
- **The curation branch (parameter 4, default `qc`) is the starting point** of a report run, if it exists. Its `uns/qc/decisions` (JSON string) supplies the decisions, and reviewed samples keep their entries.
- **New samples on `main`** are flagged or accepted as described in §4.3.
- **If a decisions file is passed:** it overrides key by key.
- **The summary's first line says whether `qc` is current:** *"qc covers 3 of 5 samples on main (curated 2026-10-20 from 1CECHNKR); 2 new samples (1 flagged, pending: SP0010)."*
- **An apply run doesn't merge anything:** the file it's given already holds the carried-forward entries from its report run, and the fingerprint check guarantees no samples were added since.

### 4.9 Apply: refuse to run when
Checked first, before any report work, so a refused run costs seconds:

| Condition | Message |
|---|---|
| No decisions file was given | *"apply needs the qc_decisions.json of a report run; run without apply first."* |
| `previewed_sha256` is missing or doesn't match the file's content | *"Decisions edited after their report; rerun without apply to preview them."* |
| The input's fingerprint differs from `input.fingerprint` | *"Input changed since this report (e.g. new samples were ingested); rerun without apply."* |
| `pending` is non-empty | *"Pending samples need a decision: SP0010, …"* |
| icechunk is read-only and no `origin` was given, or the plain-zarr output isn't writable | the location, and what to pass |

There's no separate mask to verify: the mask is a deterministic function of the decisions and the
input, both of which are checked, so the apply run recomputes it and writes exactly the cells its
report shows as passing.

### 4.10 Apply: write the passing cells
The copy is **out of core**, done by annizarr, so apply fits the same machine as the report whatever the store's size: memory is bounded by one band of storage chunks, never by the number of cells.

Both input kinds go through one call, `annizarr.subset(store, output=…, keep=mask, branch=…, commit=False)` (new in annizarr 0.2.4, §9). `write` in the capsule only builds the mask, the `uns/qc` entry and the commit around it.
- **Copied filtered, in bands of whole storage chunks:** X (and any layers), keeping passing rows via the mask over each band's `indptr` slice. They're streamed into pre-sized arrays with the source's chunking and compression, with an int64 `indptr`.
- **Copied filtered row-wise:** `obs` (all columns) and any `obsm`.
- **Copied unchanged:** `var`, `varm` and `uns`.
- **Dropped:** `obsp`.
- **Added:** `uns/qc`: `decisions` (JSON string), `cell_qc_sha256`, `source` (snapshot or plain-zarr fingerprint), `samples`, `reviewer` (if given), per-sample before/after counts.

Where it goes:
- **icechunk:** one curation branch (default **`qc`**, or the name given), which always holds **only the latest curation** made under that name. Different names give independent curations, e.g. a strict and a lenient one.
  - **Point `qc` at the reviewed snapshot:**
    - create it on first use (`checkout(name, create=True)` while on `main`);
    - otherwise reset it with `cherrypick(fingerprint)`.

    Either way, `qc` becomes the fingerprinted `main` snapshot.
  - **Write it:** `open_zarr("w", truncate=True)`, then `annizarr.subset` from the `main` snapshot, then one commit: `commit(msg, metadata={pipeline: "qc", created, source_snapshot, samples, decisions_sha256, reviewer, n_pass, n_cells})`. The message reads like `QC curation: 3 samples, 61,204 / 65,766 cells pass (source 1CECHNKR)`.
  - **The previous curation** (and any `process` commits on it) is then no longer on any branch.
  - **Reclaiming its storage:** icechunk's `garbage_collect(delete_object_older_than=…)` deletes objects that no branch or tag can reach.
    - It runs after the commit, with a 24-hour margin, so that an ingest writing at the same time can't lose uncommitted chunks.
    - `main` and its whole history stay reachable, so nothing raw is touched.
    - Through `Repo.garbage_collect(older_than)`, new in annizarr 0.2.4 (§9).
  - **`main` is untouched,** so ingest appends keep working.
- **plain zarr:** a new zarr at the output path. The input is never modified, writable or not.
- **Cost:** the curated copy rereads and rewrites X, about 16 KB per cell (160 GB at 10M cells), mostly I/O; time grows with the store, memory doesn't. Only the latest curation is kept, once GC has run.

### 4.11 Outputs (`/results`)
Every run:
```
qc_decisions.json      the effective decisions, with previewed_sha256 (edit and rerun, or pass to apply)
cell_qc.parquet        one row per cell in store order: sample (dict), qc_pass (bool),
                       qc_fail_reason (dict); plus the three metrics (float32) only if the
                       fallback computed them. zstd, ~1M-row groups
sample_qc.csv          §4.4, plus the ambient columns of §4.6
celltype_qc.csv        §4.7 (absent when skipped)
embedding.parquet      the downsampled cells: row position, UMAP, leiden
plots/samples/  plots/groups/  plots/embedding/  plots/ambient/  plots/celltypes/
report_summary.txt     mode, curation status, headline counts, warnings, flags, pending samples
run.json
```
Apply runs also write `apply_summary.csv` (per sample: before, after, removed by reason), and `run.json` records the commit id or output path.

- **`cell_qc.parquet`'s file metadata** holds the input fingerprint and `previewed_sha256`.
- **`report_summary.txt`'s first lines** say which mode ran and, for a report run, whether `apply` would currently be accepted (no pending samples) or what's blocking it.

### 4.12 Not in v1
- **Ambient RNA correction (notebook 02, SoupX):** it needs each sample's raw droplet matrix, which the store doesn't hold. If wanted, it belongs before or inside the ingest. If it's added, estimate the contamination fraction with Scrublet's doublets left out (they inflate it), then re-score doublets on the corrected counts. §4.6 detects ambient RNA in the meantime.
- **Ribosomal and hemoglobin %:** later. Adding them means a new ingest column and a backfill of existing cells.
- **Adaptive per-sample thresholds:** cutoffs computed automatically from each sample's own distribution (e.g. median ± 3 MADs). No review is needed, but cutoffs differ per sample in ways that are harder to explain. v1 uses global thresholds plus overrides for flagged samples.
- **Integration:** belongs to `process`, not QC.

---

## 5. `process` — embeddings and clusters on GPU

This follows `standardized_notebooks/scRNA/rapids-sc/10-Rapids_GPU_Analysis.ipynb`, plus an optional scVI path.

### 5.1 App panel
| # | Parameter | Default | Notes |
|---|---|---|---|
| 1 | input | — | Any location in §2.1: an icechunk repo, or a curated plain zarr |
| 2 | branch | `qc` | icechunk only: the curation branch to process |
| 3 | origin | *(none)* | icechunk only: the writable location behind a read-only mount |
| 4 | integration | `none` | `none`, `harmony` or `scvi` |
| 5 | batch keys | *(none)* | Comma-separated obs columns (e.g. `chip_id`). Required unless integration is `none` |
| 6 | output mode | `results-only` | Read-only plain zarr only (§5.4): `results-only` or `full` |
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

### 5.2 Steps
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
8. **Write** (where: §5.4):
   - `obsm/X_pca`, `obsm/X_pca_harmony` or `obsm/X_scVI`, `obsm/X_umap`;
   - `obs/leiden` (categorical, added to `column-order`);
   - `var/highly_variable`;
   - `layers/lognorm`;
   - `uns/process`: all parameters, the number of GPUs, package versions, the PCA variance ratio, and the source (branch and snapshot, or plain-zarr fingerprint);
   - on icechunk, commit metadata `{pipeline: "process", curation_branch, source_snapshot, integration, batch_keys, n_cells, n_clusters}`.

   The neighbor graph (`obsp`) isn't written in v1, because it's large and recomputable.
9. **`/results`:**
   - UMAP PNGs coloured by `leiden`, `sample_id`, the batch keys and `pbmc_clean_AIFI_L1_2024_label`;
   - `cluster_summary.csv` (cells per cluster, by sample);
   - `run.json` with per-step timings and peak GPU and host memory.

### 5.3 Reruns (icechunk)
Rerunning `process` on the same branch, for example with a different integration, adds another
commit. The derived elements are replaced, and the previous result stays in the branch's history.

### 5.4 Where results go
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

---

## 6. Environments

### 6.1 How a capsule's environment is built
- **Files, edited by hand.** Each capsule's environment is its own `.codeocean/environment.json` (the environment page; the two are the same data) and `environment/postInstall`. There is no generator.
- **Code Ocean generates `environment/Dockerfile` from them.** It has one `RUN pip3 install …` per page installer, then `COPY postInstall` + `RUN /postInstall`. Its `# hash:` line is Code Ocean's own, so the Dockerfile is never edited by hand.
- **Only `postInstall` is copied into the build.** Nothing else from the capsule exists there, so anything it needs (e.g. a pinned package list) is written inside the script.
- **Everything installs into the Docker image at build time.** A change takes effect only after a rebuild (a restart). The page's pip packages install before `postInstall` runs; anything that needs a system package to build (annoy needs `g++`) goes in `postInstall`.
- **Same path, different storage.** A build installs into the image's Python (`/opt/conda` on the CPU image, `/opt/venv` on the GPU image) as read-only image layers. A `pip install` in a running capsule goes to the same path but lands in the container's writable layer: about 5 GB, reset every run, and absent from Reproducible Runs. So image size is not bounded by that 5 GB, and anything a capsule needs goes in its environment, never a runtime install or `/scratch`.
- **Hand edits:** a changed `environment.json` (copied into the checkout or subtree-pushed) should show up on the page. Verify this on the first rebuild.
- **Getting them into a capsule:** copy them into the checkout as in §1, or subtree-push the capsule folder.

### 6.2 Packages
- **Shared pins.** The packages every capsule uses to read and write the store carry the same versions everywhere: `annizarr 0.2.3`, `icechunk 2.3.0`, `boto3`, `zarr 3.4.1`, `anndata 0.13.4`, `numpy 2.4.6`, `pandas 3.0.6`, `scipy 1.18.1`, `h5py 3.16.0`, `pytest`. A change to one is made in all three capsules.
- **What each capsule adds:**

  | Capsule | Installed by | Adds |
  |---|---|---|
  | ingest | page; `postInstall` | scanpy, celltypist (+ leidenalg, igraph, umap-learn, pinned); `g++`, annoy, scrublet in `postInstall` |
  | qc | page | scanpy, leidenalg, igraph, umap-learn (downsampled embedding); pyarrow, matplotlib, seaborn |
  | process | `postInstall` only | the full pinned list below |

- **CPU capsules:** transitive dependencies aren't locked. A released image is built once and frozen; if a rebuild breaks, pin the offending package on the page.
- **`process`: CUDA 13, fully pinned, from PyPI only.**
  - `process/environment/requirements.in` lists the top-level packages: the shared pins, RAPIDS 26.10 cu13 (`cudf`, `cuml`, `cugraph`, `cuvs`, `rmm`, `dask-cuda`), `rapids-singlecell-cu13 0.18.1`, `cupy-cuda13x`, scanpy, matplotlib, torch, `scvi-tools 1.5.1`, scikit-misc, ipykernel.
  - Its first lines hold the `uv pip compile` command that pins it for Python 3.13 on x86-64 Linux, wheels only (except `docrep`, a pure-Python scvi-tools dependency with no wheel). The result, 205 packages (torch 2.14.1, CUDA toolkit wheels 13.0, numpy 2.4.6, dask 2026.7.1), is pasted into `postInstall`, which installs it with `uv pip install --python /opt/venv/bin/python --no-cache`.
  - The target is explicit because the build shell may not have the terminal's `VIRTUAL_ENV`, and a bare `pip` would then install into the image's other Python (`/usr/local`).
  - Nothing goes on the page, so one resolver run decides every version.
  - **Not yet installed or run on a GPU.** After the first build, check `torch.cuda.get_arch_list()` includes `sm_75` (the T4) and `torch.cuda.is_available()`.
- **numpy 2.4.6 everywhere:** cudf 26.10 needs `numba < 0.65`, and numba 0.64 supports only `numpy < 2.5`. The CPU capsules resolve with it too.

### 6.3 Base images
- **CPU capsules:** `codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04` (conda Python 3.12 in `/opt/conda`).
- **`process`:** `codeocean/cuda12-runtime-uv:main-5692e50` (uv, Python 3.13 in the venv `/opt/venv`).
  - The wheels bring their own CUDA 13 libraries, so the image's CUDA 12 toolkit doesn't matter; the host driver does.
  - Measured: driver 610.57.04 (CUDA up to 13.3) on a Tesla T4 (15 GB, compute capability 7.5, the oldest generation CUDA 13 supports).
- **annizarr:** 0.2.3 from PyPI. It imports on Python 3.12 (the `TypeIs` fix is in), and the ingest's 63 tests pass against it with numpy 2.4.6.

### 6.4 Dev tooling (every image)
- **On the page:** `pytest` and the VS Code extensions (CPU capsules); `process` gets `pytest` from its pinned list.
- **In `postInstall`, last in every capsule:** Claude Code.
  - `postInstall` runs at image build time, before `/root/capsule` is mounted, so anything it writes under `/root/capsule` is hidden at runtime. The binary therefore installs to the image's `~/.local/bin`.
  - `~` is part of the image layer and is reset every run, so Claude Code's state (settings, history, sessions) goes to `CLAUDE_CONFIG_DIR=/root/capsule/code/.claude`, exported in `~/.bashrc`, on the persistent capsule volume. Git-ignore `code/.claude/` in a deployed capsule.

### 6.5 Resource classes
| Capsule | Class |
|---|---|
| ingest | `medium`; it depends on the largest sample, not the store. About 1 minute per 30k-cell sample with CellTypist and Scrublet |
| qc | `medium`, for both modes. The report reads only a few obs columns plus a fixed-size embedding sample; apply streams X out of core (§4.10). Apply is I/O-bound, roughly 12 minutes per 1M cells from local disk, slower from S3 |
| process | A GPU machine (the T4 has 15 GB). One GPU with ≥ 15 GB VRAM handled 5M cells in about 15 minutes in the notebook's benchmark, on a faster GPU than a T4; above about 13M cells, ≥ 30 GB VRAM. Host RAM must absorb the spill |

---

## 7. Tests (per capsule, `code/tests`)

**`ingest`**
- appends without any derived elements on `main`;
- one commit per run, plus the bootstrap seed;
- the schema has no `leiden`;
- Scrublet with no threshold gives NA calls and keeps scores (done);
- the existing per-input tests (HISE, 10x, h5ad, variable-length strings, ADT features).

**`qc`, report mode**
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
- `previewed_sha256` is stable across key order and whitespace, and changes on any content edit;
- chunk-aware sampling reads at most two storage chunks per sample;
- the ambient indicator: per-set medians on a synthetic store with known marker shares, doublets excluded, the 1.5× flag, the 5-sample minimum, and skipping with a warning when labels or genes are missing;
- the embedding is deterministic for a seed;
- the input is never written, and a read-only icechunk without an `origin` is accepted.

**`qc`, apply mode**
- refuses on each condition in §4.9, before any report output is written;
- a report run's `qc_decisions.json` passed straight to `apply` is accepted, and its `cell_qc.parquet` equals the apply run's;
- `qc`'s parent is the fingerprinted snapshot, and its metadata is complete;
- the output holds exactly the passing rows, with X, layers, obs and obsm equal to the input's rows, an int64 `indptr`, and no `obsp`;
- `main` is untouched, and a later ingest append works, after which `apply` with the old file is refused;
- a second curation after new samples resets `qc` onto the new snapshot, and GC removes the first curation's objects but nothing reachable from `main`;
- plain zarr writes the same content to the output path, and never touches the input;
- a custom branch name creates an independent curation, and `qc` is left alone.

**`process`** (GPU only)
- on a small curated branch, writes every element in §5.2 in one commit;
- `harmony` and `scvi` each write their own embedding, and `none` writes neither;
- integration without batch keys is refused;
- plain-zarr input: `results-only` has no X and reads back with `anndata.read_zarr`; `full` equals the input plus the results;
- read-only icechunk without an origin is refused;
- a writable plain zarr gets the results in place; a read-only one gets a new zarr;
- `leiden` is in `column-order`, and the result is readable with `anndata.read_zarr`;
- a rerun adds a commit and replaces the derived elements.

---

## 8. Build order
1. annizarr 0.2.3 on PyPI, with the `TypeIs` fix. **Done.**
2. `ingest`: remove the whole-store analysis (§3) **(done)**, move its environment to the page (§6) **(copied into the checkout; rebuild pending)**, and rebuild a store from samples.
3. `qc`, report mode: icechunk first, then plain zarr. Review the report on the rebuilt store.
4. annizarr 0.2.4: `subset` (streaming row filter, both store kinds, int64 `indptr`, drops `obsp`) and `Repo.garbage_collect(older_than)`.
5. `qc`, apply mode: icechunk first, then plain zarr.
6. `process` on a GPU capsule, against the first `qc` branch.
7. Subtree-push each capsule to its own Code Ocean capsule.

## 9. Open questions
- **annizarr 0.2.4:** `subset(store, *, output, keep, branch, commit)` mirrors `rechunk`'s streaming copy with a row mask; `Repo.garbage_collect(older_than)` wraps icechunk's. Both keep the capsule free of store internals (no private `_writer_repo()`).
- **First GPU run:** the CUDA 13 list resolves but hasn't been installed. The first `process` build needs a real test of torch and RAPIDS loading in the same environment on the T4, and of the scVI subprocess handoff.
- **Using RAPIDS in other capsules:** the same `postInstall` could serve a capsule with its own environment by installing into a separate venv (`/opt/rapids`) with a Jupyter kernel. Untested whether Code Ocean exposes the GPU to a non-GPU base image.
