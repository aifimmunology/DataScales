# Single-cell curation capsules — spec

Code Ocean capsules that take HISE scRNA samples from raw files to a curated, processed store, plus a
spatial ingest that builds one SpatialData object from Xenium and Atera bundles (§3S):

| # | Capsule | Hardware | Writes | Job |
|---|---|---|---|---|
| 1 | `anndata_ingest/` | CPU, sized to the largest **sample** | `main` | Per-sample QC metrics, CellTypist, Scrublet; append to the raw store. Takes .h5, .h5ad and SpatialData tables. No whole-store analysis |
| 1S | `spatial_ingest/` (design tested, not built) | CPU, ~16 GB per Xenium sample | a local joint SpatialData `.zarr` | Append Xenium/Atera bundles: per-sample elements and table, plus one joint AnnData (§3S) |
| 2 | `qc/` | CPU; I/O-bound when applying | nothing by default; with `apply`, a commit on `main` (icechunk) or a new zarr | QC report for proposed thresholds, with a downsampled embedding. With `apply` and reviewed decisions, writes the passing cells as the curated object |
| 3 | `process/` | **GPU, CUDA 13** (rapids-singlecell + Dask, scvi-tools) | a new zarr, or a commit (icechunk) | First the shared RAPIDS environment with guide notebooks (out-of-core scRNA, spatial); optionally an App Panel / Reproducible Run pipeline: PCA (± harmony) or scVI, then neighbors, UMAP, leiden (§5) |

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
        main, new commit (curated cells only; raw tagged and kept in history)
                       │
              process notebooks ─► + embeddings, clusters (new zarr, or a commit)
```

Cell labeling and cluster-level doublet cleanup (standardized notebooks 06–08) come after `process`
and are a separate stage. Existing stores built by the old pipeline aren't migrated; stores are
rebuilt from samples with the new ingest.

The capsules are meant to be shared and run by anyone on their own data: every location (data
asset, local path, S3) comes in through the app panel (ingest, qc) or a notebook's settings cell
(process), nothing is hard-coded, and there are no per-user permissions.

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
├── anndata_ingest/         ┐ (was ingest/)
├── spatial_ingest/         │ (planned)
├── qc/                     │ each folder is one complete capsule root
└── process/                ┘
    ├── code/               run (entrypoint), <package>/, tests/
    ├── environment/        requirements.txt, environment.yml (if conda is needed), pixi.toml, postInstall
    ├── .codeocean/         resources.json
    ├── metadata/
    └── .gitignore
```

- **Development:** develop in a full DataScales clone (branch `codeocean_capsules`), where every capsule is visible side by side.
- **Updating a capsule checkout during development:** copy the capsule's `code/` entries, `environment/postInstall` and `.codeocean/resources.json` over their namesakes in the checkout (e.g. `/root/capsule`), and load `requirements.txt` / `environment.yml` onto the page with **Edit Bulk**. Data assets, secrets, reference clones and the Dockerfile are left alone. No reset is needed; an environment change applies on the next restart (rebuild).
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
- **Where results go** (`qc` with `apply`, and the `process` notebooks):

  | Input | `qc` with `apply` | `process` notebooks |
  |---|---|---|
  | icechunk | a new commit on `main` with only the curated cells (§4.10) | a commit on the branch read |
  | plain zarr (writable or not) | a new zarr at the output path | a new full zarr in `/results` (§5) |

  - **icechunk writes need write access:** S3, a writable local repo, or a data-asset mount given its `origin`.
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

Written by `process`: `obs/leiden`, `obsm/X_pca`, `obsm/X_pca_harmony` (if integrated; `X_scVI` will come from the scVI step), `obsm/X_umap`, `var/highly_variable`, `layers/lognorm`, `uns/process`.

### 2.3 Entrypoint and code style
- **Entrypoint** (ingest and qc; `process` has notebooks only, §5):
  - `code/run` passes app-panel parameters positionally. An empty value falls back to an env var, then a default.
  - One `main.py`, with a `<Capsule>Error(RuntimeError)` for user-facing failures (logged as one line, exit code 1).
  - Every run writes `run.json` under `/results`: parameters, input fingerprint, versions, timestamps, peak memory.
- **Modular within each capsule:** small modules with one job each.
  - `ingest`: `config`, `hise`, `annotate`, `contract`, `store`, `main`.
  - `qc`: `config`, `source`, `metrics`, `decisions`, `embed`, `report`, `write`, `main`.
  - `process`: no package; two notebooks sharing one identical helper cell (open, save).
- **No comments.** Names and short docstrings on public functions carry the meaning.
- **Not overengineered:**
  - no plugin layers or abstract base classes;
  - no options nobody asked for;
  - branch on input kind only where icechunk and plain zarr actually differ.
- **Tests sit beside the code** and run against small synthetic stores. `process` runs its notebooks headless with `code/notebooks/test_notebooks.sh`, on the GPU capsule.

---

## 3. `anndata_ingest` — append samples only (built)

The source is `capsules/anndata_ingest/` (renamed from `capsules/ingest/` on 2026-10-30; the Code Ocean capsule and its `code/ingest/` package keep the name `ingest`). Its `code/` is copied into the capsule checkout (`/root/capsule/code`, §1), so the capsule keeps working once this repo is gone. The README there describes it in full.

### 3.1 What it does
- **Store:** an icechunk repo (one commit per run, plus a seed commit when a run creates the store), or a path ending in `.zarr` for a plain AnnData zarr appended in place.
- **Per sample, one in memory at a time:** read (HISE / 10x / Xenium `cell_feature_matrix.h5` / h5ad / a SpatialData `.zarr`'s `tables/table`, read with zarr alone), QC metrics, CellTypist, Scrublet, project onto the contract, stage as a zarr. Samples with no cells are skipped with a warning. Only Gene Expression features are kept (Xenium control probes and codewords are dropped).
- **Sample IDs:** parsed from the file name; a generic 10x name (`cell_feature_matrix.h5`, `filtered_feature_bc_matrix.h5`) takes its bundle folder's name (skipping `outs`, `extract`), a SpatialData `.zarr` its folder name.
- **Gene panels:** every new sample's genes are compared with the store's (or the first sample's) from var alone, before any sample is read. A difference is a warning (n shared / only in store / only in sample, with examples), and the sample is fitted to the store's genes: missing genes zero-filled, extra genes dropped. annizarr 0.2.3 appends only identical var (names and order); growing the store's genes ("append new genes") is planned in annizarr.
- **Storage warning on every run** for a store in `/scratch` (cleared) or `/results` (it can be saved as a data asset, but data assets are read-only, so later runs can't append), recommending S3.
- **The store is touched only after every sample is staged,** so a failing sample leaves either kind unchanged.
  - On icechunk, a failed append discards the run.
  - A plain zarr can't roll back, which the README states.
- **`main` holds only X (raw counts), obs and var.** There is no whole-store analysis, so memory depends on the largest sample.
- **Scrublet:** NA calls when it fails or finds no threshold; scores are kept when it only finds no threshold.
- **Create:** automatic for a missing store at an `s3://` prefix (the write is the permission check) or inside `/results` or `/scratch`; any other local path only with `CREATE=true`.
  - It refuses a non-empty folder that isn't a store.
  - An empty repo left by a failed create run can be reused.
- **Appending drops results built for the old cells** (2026-11-02): `obsm`, `obsp`, `layers` and `uns/process`, with a warning, through annizarr `append(drop_derived=True)` (without the flag annizarr raises). The commit records them as `dropped`; on icechunk they stay in history. Earlier, such stores were refused.
- **Read-only locations are refused before any work:** an icechunk repo without an `origin`, or an unwritable zarr.

### 3.2 App panel
`store`, `origin`, `samples`, `Celltypist Model` (a dropdown: AIFI L1, L2, L3 · AIFI L1 only · Immune_All High and Low · None), `Custom Celltypist model` (optional `.pkl` file(s) that replace the dropdown, e.g. a model trained for a spatial panel), passed positionally in that order; when Code Ocean drops an empty optional value, values are placed by shape. Defaults make a Reproducible Run ingest `/data/samples` into a new store at `/results/ingest_icechunk`. Every other setting is an environment variable named after its `Settings` field, in upper case.

### 3.3 Outputs
- **The commit on `main`:** the message lists one `- <sample>: <source path>` line per sample (with the fitting, if any); the metadata records `pipeline`, `annizarr` version, `appended_samples`, `sources` (sample → path), `fitted` (sample → how it was fitted), `n_cells` and `n_samples`.
- **`/results`:**
  - `ingest_summary.csv`, per appended sample: cells, mean genes, mean mito %, doublets, whether Scrublet scored it, number of cell types, model;
  - `ingest_run.json`: store, kind, snapshot ids, appended samples, totals, annizarr version, peak memory, settings.

### 3.4 Verified
- 85 tests (2026-10-30): unit tests, SpatialData-table inputs, gene-panel fitting and the custom `.pkl` field, plus plus end-to-end runs on both store kinds covering mixed inputs, every app-panel input shape and dropdown option, automatic creation, a mount that lags its origin, CellTypist (a locally trained model) and Scrublet, failure and rerun semantics, read-only and create guards, and config parsing. They pass against annizarr 0.2.3; annizarr 0.2.1 fails the icechunk runs.
- The five real B092 samples (127,800 cells) on both kinds, with Scrublet. Stored counts, barcodes and HISE obs match each source file exactly, and the icechunk history is one commit per run.
- The bundled AIFI L1/L2/L3 models on real samples: labels and probabilities identical to running CellTypist directly, as `06-label_h5s.py` does.
- **CellTypist:** several models per store (default AIFI L1, L2, L3, bundled in `code/models` with a `SOURCE.json` of commit and checksums), columns `<model>_label` / `<model>_prob`. The model set is part of the schema.
- **Xenium (2026-10-30):** two public bundles' `cell_feature_matrix.h5` (Breast 280 genes, Lung 289) through `./run`: named after their bundle folders, Lung fitted with a warning (67 shared, 213 zero-filled, 222 dropped), commit lines name both sources, and a rerun skips both. The PBMC CellTypist models are meaningless on such panels (AIFI L1 shares 94 of 1,109 genes; 75% of breast cells labelled "T cell"), so spatial stores use None or a custom `.pkl`.
- **HISE layouts:** tenx-scrna-pipeline's observation names (`Batch`, `Pool`, `Chip`, `uuid`, …) are mapped onto the older HISE ones; labels already in a file are recomputed.

---

## 3S. `spatial_ingest` — Xenium and Atera into one SpatialData (design tested, not built)

The source will be `capsules/spatial_ingest/`. Findings below were tested on 2026-10-29 with spatialdata 0.8.0, spatialdata-io 0.7.1 and zarr 3.4.2 on public 10x bundles; the test scripts lived in `/scratch/spatial_tests/` (not kept).

### 3S.1 Decisions
- **Output:** one joint SpatialData `.zarr`, **local only for now** (a path under `/results`, or a workstation path). Each run appends new bundles; samples already present are skipped by sample ID.
- **Per sample:** its images, labels, cell/nucleus shapes and transcripts (points), and its own table, all prefixed by the sample ID and in its own coordinate system. The per-sample table keeps every gene of its panel and is the source of truth.
- **Joint AnnData:** one table over all cells, appended with annizarr (no in-memory rewrite). Genes follow the same rule as `anndata_ingest`: the first sample's panel; later panels are fitted with a warning, until annizarr can append new genes. Gene key: Ensembl ID.
- **CellTypist:** default None; an optional custom `.pkl` (same panel field as `anndata_ingest`).
- **Crash safety (plain zarr):** each run writes its new samples into a temporary `.zarr` beside the store and renames their folders in; the joint table is appended last; refreshing the consolidated metadata is the commit step. At startup, leftovers are deleted, and a joint table that doesn't match the per-sample tables is rebuilt from them. Icechunk stores (`anndata_ingest`) already roll back, so they don't need this.
- **Atera:** vendor the reader from spatialdata-io PR #426 (`10XGenomics/spatialdata-io@738204de`, `atera.py` + `_atera_common.py`, ~1.2k lines), pinned, until it is released.

### 3S.2 Why not icechunk (yet)
- Released spatialdata can't write to or read from icechunk: `write()`/`read_zarr()` take paths, and the store-backed writes in scverse/spatialdata PR #1107 are unmerged.
- Points and shapes are Parquet files inside the `.zarr` folder. Icechunk holds only zarr keys and rejects them ("invalid zarr key format"). Packing each Parquet file into a uint8 zarr array works (prototyped) but needs our own read/write layer, loads whole files, and no other tool can open the result.
- The earthmover `2026-scverse/icechunk-SpatialData` material is talk code on a patched spatialdata fork plus the non-public `zarr-files` package.

### 3S.3 Tested findings
| Area | Finding |
|---|---|
| Incremental append | 4 public Xenium bundles in 3 runs into one `.zarr` with `write_element`; stock `read_zarr` reads it back; bbox query and table↔shapes join work; a verifier catches corrupted transforms, obs and X. Rerun of a present sample: 1.5 s, all files byte-identical |
| Gene key | All Xenium bundles and Atera carry Ensembl IDs (`gene_ids` / `feature_id`); accept `ENSG` and `ENST` (CD45RA/RO use `ENST`). Symbols collide (F3 human vs mouse) |
| S3 | spatialdata 0.8.0 can't write or read S3: `UPath` is rejected, and a plain `"s3://…"` string silently creates a local `./s3:/` folder. Workaround tested: build locally, sync with root `zarr.json` last |
| Crash | `write_element(overwrite=True)` refuses elements of its own store; delete-then-write loses the table on a crash and the store no longer opens. Staged write + `os.rename` + startup recovery tested OK. An element killed mid-write is invisible to consolidated metadata but blocks the rerun until deleted |
| Scale (Xenium Prime 5K skin, 112k cells, 74M transcripts, 7.6 GB zip) | ~2–2.5 min and ~14 GB peak per sample append; 4.3 GB per sample on disk (images 2.9, transcripts 1.3); keep native Parquet row groups (2–3M rows). Opening the store: +0.85 s per sample (shapes load eagerly). Container memory cap is 61.7 GiB |
| Joint table | In-memory rewrite: 1M cells 19 s / 7.5 GB, 5M 52 s / 27 GB, ~21M OOM. In-place append: ~3–4 s at any size, hence annizarr for the joint table |
| Atera (tiny bundle) | PR #426 reader works on released spatialdata 0.8.0; 0.8.0 reads the joint Xenium+Atera `.zarr`. `morphology_3d_images=True` fails (chunks bug; workaround `chunks=(1,1,4096,4096)`). Full public Atera sets are ~50–65 GB |
| Standard workflow | None published for incremental multi-sample SpatialData: `spatialdata.concatenate` is in-memory (and intersects genes by default), sopa keeps one `.zarr` per sample |

### 3S.4 annizarr into the joint table (tested 2026-10-30)
- `annizarr.append("<store>.zarr/tables/table", cells=<staged sample .zarr>)` appends in place: 7,275 cells onto a 19,828 × 5,753 int32 table in 1.4 s. `uns["spatialdata_attrs"]` is kept, and after `zarr.consolidate_metadata` stock `spatialdata.read_zarr` reads the store with the new cells.
- The capsule must also add the new sample's region (e.g. `<sample>_cell_boundaries`) to `uns["spatialdata_attrs"]["region"]` and keep `region`/`instance_key` in obs, so the table still annotates every sample's cell shapes.

### 3S.5 Still to test
- A full-size Atera bundle (~50–65 GB).

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

---

## 5. `process` — the RAPIDS environment and notebooks

**Reworked 2026-11-02** from a command-line pipeline into an environment capsule; the design is
[`process/rework_spec.md`](process/rework_spec.md) and the change log [`process/spec.md`](process/spec.md).

- **What it is, first:** the pinned RAPIDS GPU environment (`code/environment/`, copyable to other capsules) and two notebooks: out-of-core scRNA (normalize, HVG, scale, PCA, optional harmony, neighbors, UMAP, leiden, following `10-Rapids_GPU_Analysis.ipynb`) and a spatial walkthrough.
- **Optional pipeline (2026-11-03):** `code/run`, from the App Panel (named parameters, five categories) or a terminal. Method PCA (± harmony) or scVI (train), then neighbors, UMAP and leiden at one or more resolutions, optionally on an obs subset. One `Config` dataclass modelled on `dev_tools/rapids-benchmark/rapids_benchmark.py`: the panel sets the common fields, everything else (cluster preset, GPUs, scVI training knobs, …) is a flag or environment variable. A Reproducible Run with no input exits 0 with usage. Details and the GPU test plan: `process/spec.md` §12.
- **Inputs:** an AnnData `.zarr` or a SpatialData `.zarr` (its `tables/<name>`) by default; icechunk through commented-out cells (branch, optional S3 origin).
- **Saving (notebooks and pipeline):** zarr input → a new full object at `/results/<stem>.processed.zarr` (SpatialData: the whole store copied, results in its table); icechunk → one commit back to the branch. Results: `obsm/X_pca` (`X_pca_harmony`), `obsm/X_umap`, `obs/leiden`, `var/highly_variable`, `layers/lognorm` (with `target_sum`), `uns/process`.
- **scVI:** in the pipeline (train a new model; `obsm/X_scVI`, model saved to `/results/scvi_model/`); HVGs are picked on the Dask-CUDA cluster, which is closed before training.

## 6. Environments

### 6.1 How a capsule's environment is built
- **Files, edited by hand.** Each capsule's packages are `environment/requirements.txt` (pip) and, when conda packages are needed, `environment/environment.yml`. They are loaded onto the environment page with the page's bulk editor (gear icon → **Edit Bulk**, paste or drop the file). `environment/postInstall` installs only Claude Code. The base image and VS Code extensions are set on the page. There is no generator.
- **Code Ocean generates `environment/Dockerfile` from them.** It has one `RUN` per page installer, in this order: `conda install -y …` (conda-forge on the mambaforge image), `pip3 install -U …`, the VS Code extensions, then `COPY postInstall` + `RUN /postInstall`. Its `# hash:` line is Code Ocean's own, so the Dockerfile is never edited by hand.
- **Only `postInstall` is copied into the build.** Nothing else from the capsule exists there, so anything it needs (e.g. a pinned package list) is written inside the script.
- **Everything installs into the Docker image at build time.** A change takes effect only after a rebuild (a restart). The page's packages install before `postInstall` runs. Prefer a prebuilt conda package over a pip source build: `annoy` has no Linux wheel, so it comes from conda (`python-annoy`) before pip installs scrublet, and no `g++` is needed.
- **Same path, different storage.** A build installs into the image's Python (`/opt/conda` on the CPU image, `/opt/venv` on the GPU image) as read-only image layers. A `pip install` in a running capsule goes to the same path but lands in the container's writable layer: about 5 GB, reset every run, and absent from Reproducible Runs. So image size is not bounded by that 5 GB, and anything a capsule needs goes in its environment, never a runtime install or `/scratch`.
- **Hand edits:** copying a changed `environment.json` into a running capsule's checkout does **not** change the page (verified 2026-10-22: the rebuild used the page, and the checkout file was overwritten). Make changes on the page, then update the capsule's `requirements.txt` / `environment.yml` to match (see `README.md`). The repo keeps no copy of `environment.json`.
- **Getting them into a capsule:** copy them into the checkout as in §1, or subtree-push the capsule folder.

### 6.2 Packages
- **Shared pins.** The packages every capsule uses to read and write the store carry the same versions everywhere: `annizarr 0.2.3`, `icechunk 2.3.0`, `boto3`, `zarr 3.4.1`, `anndata 0.13.4`, `numpy 2.4.6`, `pandas 3.0.6`, `scipy 1.18.1`, `h5py 3.16.0`, `pytest`. A change to one is made in all three capsules.
- **What each capsule adds:**

  | Capsule | Installed by | Adds |
  |---|---|---|
  | ingest | page (`requirements.txt`, `environment.yml`) | scanpy, celltypist, scrublet (pip); `python-annoy`, `gh` (conda) |
  | qc | page (`requirements.txt`) | scanpy, leidenalg, igraph, umap-learn (downsampled embedding); pyarrow, matplotlib, seaborn |
  | process | page (`requirements.txt`) | RAPIDS, torch and scvi-tools, pinned as below; squidpy, spatialdata, omnipath for the tutorial notebooks |

- **CPU capsules:** transitive dependencies aren't locked. A released image is built once and frozen; if a rebuild breaks, pin the offending package on the page.
- **`process`: CUDA 13, from PyPI only, on the page.**
  - `process/code/environment/requirements.txt` (with `pixi.toml` beside it, kept in `code/` so people can take them from the capsule) pins exact versions of the shared pins, RAPIDS 26.10 cu13 (`cudf`, `cuml`, `cugraph`, `cuvs`, `rmm`, `dask-cuda`), `rapids-singlecell-cu13 0.18.1`, scanpy, matplotlib, torch, `scvi-tools 1.5.1`, scikit-misc and ipykernel. It also pins the CUDA-stack packages that must agree: `numba`, `numba-cuda`, `cupy-cuda13x`, `cuda-toolkit`, `dask` and `distributed`.
  - Everything else is resolved at build time. On 2026-10-23 the list resolved (`uv pip compile`, Python 3.13, x86-64 Linux) to the same 205 packages as the earlier full lock.
  - The page's `pip3` on this image is `/opt/venv/bin/pip3` (verified 2026-10-23), so page packages land in the venv the code runs in.
  - **Built and working on the GPU capsule** (user-tested 2026-10-23, before the spatial packages were added). squidpy 1.8.3, spatialdata 0.8.0 and omnipath 1.0.12 resolve with the pins (246 packages, re-checked 2026-10-27); to be added on the page.
  - **Tutorial notebooks** in `process/code/notebooks/`: `rapids_scrna_out_of_core_dask` and `rapids_spatial_atera_breast_cancer`, from rapids-singlecell's tutorials `06_out-of-core` and `08_atera_demo`, adapted for one T4. Interactive only. Each reads its input from a data asset and states the format it needs.
- **numpy 2.4.6 everywhere:** cudf 26.10 needs `numba < 0.65`, and numba 0.64 supports only `numpy < 2.5`. The CPU capsules resolve with it too.

### 6.3 Base images
- **CPU capsules:** `codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04` (conda Python 3.12 in `/opt/conda`).
- **`process`:** `codeocean/cuda12-runtime-uv:main-5692e50` (uv, Python 3.13 in the venv `/opt/venv`).
  - The wheels bring their own CUDA 13 libraries, so the image's CUDA 12 toolkit doesn't matter; the host driver does.
  - Measured: driver 610.57.04 (CUDA up to 13.3) on a Tesla T4 (15 GB, compute capability 7.5, the oldest generation CUDA 13 supports).
- **annizarr:** 0.2.3 from PyPI. It imports on Python 3.12 (the `TypeIs` fix is in), and the ingest's 63 tests pass against it with numpy 2.4.6.

### 6.4 Dev tooling (every image)
- **On the page:** `pytest` (every capsule's `requirements.txt`) and the VS Code extensions.
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
- icechunk: a `raw-<snapshot>` tag points at the fingerprinted `main` snapshot, and the new `main` commit's parent is that snapshot, with complete metadata;
- the output holds exactly the passing rows, with X, layers, obs and obsm equal to the input's rows, an int64 `indptr`, and no `obsp`;
- a later ingest append onto curated `main` works, after which `apply` with the old file is refused, and the next report reviews only the new samples;
- the raw cells are readable at the `raw-<snapshot>` tag;
- plain zarr writes the same content to the output path, and never touches the input.

**`process`** (`code/notebooks/test_notebooks.sh`, on the GPU capsule)
- each notebook runs headless on tiny AnnData `.zarr`, SpatialData `.zarr`, ingest-style (no `layers`, no `sample_id`) and icechunk stores;
- a zarr save writes a new full copy that reads back (`anndata.read_zarr`, `sd.read_zarr`), never touches the input, and never overwrites an earlier output;
- an icechunk save makes one commit, also twice in a row through an `ORIGIN`, and refuses when the branch moved;
- `lognorm` matches scanpy's normalize + log1p; the shared helper cell is identical in both notebooks;
- the pipeline (`code/test_gpu.sh`): PCA, harmony, scVI, several resolutions, a subset, SpatialData, an ingest store, icechunk through an origin twice, and the no-input Reproducible Run; the full checklist is `process/spec.md` §12.6.

---

## 8. Build order
1. annizarr 0.2.3 on PyPI, with the `TypeIs` fix. **Done.**
2. `ingest`: remove the whole-store analysis (§3) **(done)**, move its environment to the page (§6) **(copied into the checkout; rebuild pending)**, and rebuild a store from samples.
3. `qc`, report mode: icechunk first, then plain zarr. Review the report on the rebuilt store.
4. annizarr `subset` and `Repo.garbage_collect`: **not in 0.2.4**; qc apply streams its own subset meanwhile (§4.10).
5. `qc`, apply mode: icechunk (a commit on `main`) first, then plain zarr.
6. `process`: built as a pipeline on 2026-10-30, **reworked into an environment + notebooks capsule on 2026-11-02** (§5).
7. Subtree-push each capsule to its own Code Ocean capsule.
8. `anndata_ingest`: SpatialData-table inputs, gene-panel fitting, source paths in commits, custom `.pkl` **(done 2026-10-30)**; S3 creation without `CREATE`, dropping results on append **(done 2026-11-02)**.
9. `spatial_ingest` (§3S): the annizarr joint-table test, then the capsule.

## 9. Open questions
- **annizarr 0.2.4 shipped without `subset` and `Repo.garbage_collect`** (checked 2026-10-30). It renamed `add_expr` to `add_layer`, which now only copies a matrix in another layout (no normalization), split the matrix copy into `_writers/_copy.py`, and caps its CPU count by the cgroup quota. `process` works on it unchanged; qc apply still needs the two below, now for a later release.
- **Planned for annizarr:** `subset(store, *, output, keep, branch, commit)` mirrors `rechunk`'s streaming copy with a row mask; `Repo.garbage_collect(older_than)` wraps icechunk's. Both keep the capsule free of store internals (no private `_writer_repo()`).
- **First GPU run:** the environment builds and works on the T4. The reworked notebooks still need a headless run there (`code/notebooks/test_notebooks.sh`).
- **Spatial `obsm["spatial"]`:** annizarr `append` drops `obsm`, so a joint table appended by `spatial_ingest` loses cell centers; keep them as obs columns, or have annizarr keep named `obsm` keys.
- **Using RAPIDS in other capsules:** a capsule with its own environment could install `process/code/environment/requirements.txt` into a separate venv (`/opt/rapids`) from `postInstall`, with a Jupyter kernel. Untested whether Code Ocean exposes the GPU to a non-GPU base image.
