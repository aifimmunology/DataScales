# `qc` — spec

Two parts:
- **Part 1, the design** (§4, moved in full from the former `capsules/spec.md` on 2026-11-15, section
  numbers kept), with the overview tables that were at the top of that file.
- **Part 2, what the build changed from §4, and why.** Where they disagree, Part 2 wins.

Shared rules (inputs, column contract, environments, git, testing) are in [`../CLAUDE.md`](../CLAUDE.md);
usage is in `code/README.md`.

# Part 1: design

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

## 4. `qc` — review, then apply

One capsule, two modes. Every run computes the decisions, the cell mask and the full report. Only a
run with `apply` set writes anything, and only with decisions that a previous report run produced
and that haven't changed since.

### 4.1 The two modes
| | Report (default) | Apply (`apply = true`) |
|---|---|---|
| Decisions | defaults → carried forward from `main`'s `uns/qc` (§4.8) → the file, if given | **exactly the given file**, nothing merged |
| Input access | read-only | read, plus a commit on `main` or a new output zarr |
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
| 2 | decisions | *(none)* | A `qc_decisions.json` (attached data asset). Report: none → defaults plus `main`'s carried-forward decisions (§4.8). Apply: required |
| 3 | apply | `false` | `true` writes the curated object, after the checks in §4.9 |
| 4 | output | `/results/<input stem>.qc.zarr` | Plain zarr only: the output zarr path (local or S3), used only by `apply`. icechunk always commits to `main` |
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
**Revised 2026-11-02:** curation is a commit on `main` (§4.10), so `main` holds the curated cells of reviewed samples plus any raw samples ingested since. A report run reads `main`; samples listed in `uns/qc/decisions` are already curated and aren't filtered again, and only the others are reviewed.
- **`main`'s `uns/qc/decisions`** (JSON string) supplies the decisions, and reviewed samples keep their entries.
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
**Revised 2026-11-02.** annizarr 0.2.4 shipped without `subset` and `Repo.garbage_collect`, so the capsule streams the subset itself; it swaps in `annizarr.subset` when that ships.

- **Out of core:** X (and any layers) is read with `read_elem_lazy` in whole-chunk bands, passing rows are kept by the mask, and the dask CSR is written with `write_elem` (int64 `indptr`). Memory is bounded by one band, never by the number of cells.
- **Copied filtered row-wise:** `obs` (all columns) and any `obsm`. **Unchanged:** `var`, `varm`, `uns`. **Dropped:** `obsp`.
- **Added:** `uns/qc`: `decisions` (JSON string), `cell_qc_sha256`, `source` (snapshot or plain-zarr fingerprint), `samples`, `reviewer` (if given), per-sample before/after counts.

Where it goes:
- **icechunk: a new commit on `main`** holding only the curated cells, written through the origin when the store is a read-only mount.
  - **Tag first:** `raw-<snapshot>` on the reviewed `main` snapshot, so the raw cells are one checkout away. No garbage collection; raw history is kept.
  - **One commit:** `commit(msg, metadata={pipeline: "qc", created, source_snapshot, samples, decisions_sha256, reviewer, n_pass, n_cells})`, message like `QC curation: 3 samples, 61,204 / 65,766 cells pass (source 1CECHNKR)`.
  - **Later ingest runs append raw samples onto curated `main`;** the next report reviews only those (§4.8).
- **plain zarr:** a new zarr at the output path. The input is never modified, writable or not.
- **Cost:** rereads and rewrites X, about 16 KB per cell (160 GB at 10M cells), mostly I/O; each curation adds a full copy to the history.

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

### 4.13 Tests (design)
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
- icechunk: a `raw-<snapshot>` tag points at the fingerprinted `main` snapshot, and the new `main` commit's parent is that snapshot, with complete metadata;
- the output holds exactly the passing rows, with X, layers, obs and obsm equal to the input's rows, an int64 `indptr`, and no `obsp`;
- a later ingest append onto curated `main` works, after which `apply` with the old file is refused, and the next report reviews only the new samples;
- the raw cells are readable at the `raw-<snapshot>` tag;
- plain zarr writes the same content to the output path, and never touches the input.

### 4.14 Environment and resources
- **Adds** to the shared pins (`../CLAUDE.md` §6.2): scanpy, leidenalg, igraph, umap-learn (downsampled
  embedding); pyarrow, matplotlib, seaborn.
- **Resource class** `medium`, for both modes. The report reads only a few obs columns plus a fixed-size
  embedding sample; apply streams X out of core (§4.10). Apply is I/O-bound, roughly 12 minutes per 1M
  cells from local disk, slower from S3.
- **Code layout (design):** `config`, `source`, `metrics`, `decisions`, `embed`, `report`, `write`,
  `main` (as built: Part 2, Code layout).

### 4.15 Decisions from the working notes (user, 2026-11-02)
- **qc apply:** on zarr, write the subset to the output location. On icechunk, make a new commit on
  **main** holding only the curated cells, with a raw tag first and no GC. annizarr 0.2.4 has no subset
  or garbage_collect. (Also `../process/rework_spec.md` §7.1.)
- **qc on spatial stores** was first skipped (PBMC assumptions); the build added a `spatial` profile (Part 2).

---

# Part 2: what the build changed from §4, and why

Everything not listed here is as §4 describes.

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

## Code layout (`../CLAUDE.md` §2.3)
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
