# `anndata_ingest` — spec

The design of this capsule, its decisions and findings. Moved from the former `capsules/spec.md` §3
(2026-11-15), section numbers kept. Shared rules (layout, column contract, environments, git, testing)
are in [`../CLAUDE.md`](../CLAUDE.md); usage is in `code/README.md`.

## 3. `anndata_ingest` — append samples only (built)

The source is `capsules/anndata_ingest/` (renamed from `capsules/ingest/` on 2026-10-30; the Code Ocean capsule and its `code/ingest/` package keep the name `ingest`). Its `code/` is copied into the capsule checkout (`/root/capsule/code`, `../CLAUDE.md` §1), so the capsule keeps working once this repo is gone. The README there describes it in full.

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
- **Appending drops results built for the old cells** (2026-11-02; `raw` added 2026-11-05): `obsm`, `obsp`, `layers`, `raw` and `uns/process`, with a warning, through annizarr `append(drop_derived=True)` (without the flag annizarr raises). The commit records them as `dropped`; on icechunk they stay in history. Earlier, such stores were refused.
- **Read-only locations are refused before any work:** an icechunk repo without an `origin`, or an unwritable zarr.

- **Input type** (2026-11-04), a panel dropdown:
  - **Raw samples:** as above.
  - **Existing AnnData or SpatialData table (as is):** `.h5ad`, AnnData `.zarr` (with or without the suffix), or a SpatialData `.zarr` table, read whole with nothing recomputed. Identified by `sample_id` (or the obs column in `SAMPLE_COLUMN`, copied into `sample_id`; the file name without one). A rerun skips an object whose samples are all present; a partial overlap is refused. The first object of a new store keeps everything (obs, layers, obsm, uns); later ones add X, obs (only the store's columns) and are gene-fitted like raw samples; the store's embeddings and layers are dropped on append. Cell names are prefixed `<sample_id>:` like raw samples. X is cast to the store's dtype (counts must fit it; a store whose X holds counts takes only counts), obs columns are converted to the store's column types (one that can't convert is left empty, with a warning). Several objects into a new store in one run: the first object's embeddings and layers are dropped when the rest append. Objects are streamed by annizarr (`convert` for `.h5ad`, `rechunk` for AnnData zarr v3 and SpatialData tables; zarr v2 inputs, which annizarr can't rechunk, get a v3 staging copy in row blocks) into staging; only obs/var are edited in memory and X is rewritten in row blocks only for a gene or dtype change. A new store is seeded from its staging zarr with `rechunk`, so no input is read whole. Peak memory is about 3× X, from annizarr's append (300k cells, 0.48 GB X: 0.69 GB staging, 0.91 GB seeding, 1.59 GB appending). `raw/` is never kept (annizarr can't append it); empty objects are skipped before staging. `.h5` inputs are refused in this mode.
  - Both types can share a store. Raw samples into a store seeded from an existing AnnData keep only the store's columns, with a warning; a store built from raw samples keeps its strict schema check.
  - **MuData** is refused by name (`.h5mu`, or a zarr with `mod`/`encoding-type: MuData`), with how to write a modality as AnnData.

### 3.2 App panel
`Input type` (Raw samples · Existing AnnData or SpatialData table), `samples`, `store`, `origin`, `New store format` (2026-11-16; Auto · Icechunk · Zarr, the kind of a store that doesn't exist yet; Auto keeps the old rule, a `.zarr` path is a plain zarr), `Celltypist Model` (raw samples only; a dropdown: AIFI L1, L2, L3 · AIFI L1 only · Immune_All High and Low · None), `Custom Celltypist model` (optional `.pkl` file(s) that replace the dropdown, e.g. a model trained for a spatial panel), passed positionally in that order (Sample Information first since 2026-11-16); when Code Ocean drops an empty optional value, dropdown labels and `.pkl` values are placed by shape and the remaining paths, in order, are samples, store, origin. The default store is the placeholder `s3://bucket/prefix` (2026-11-16, replacing `/results/ingest_icechunk`), so a Reproducible Run needs the Storage Location replaced first. Every other setting is an environment variable named after its `Settings` field, in upper case.

### 3.3 Outputs
- **The commit on `main`** (the seed commit too, since 2026-11-05): the message lists one `- <sample>: <source path>` line per sample (with the fitting, if any); the metadata records `pipeline`, `annizarr` version, `appended_samples`, `sources` (sample → path), `fitted` (sample → how it was fitted), `n_cells` and `n_samples`.
- **`/results`:**
  - `ingest_summary.csv`, per appended sample: cells, mean genes, mean mito %, doublets, whether Scrublet scored it, number of cell types, model;
  - `ingest_run.json`: store, kind, snapshot ids, appended samples, totals, annizarr version, peak memory, settings.

### 3.4 Verified
- 112 tests (2026-11-05): unit tests, SpatialData-table and existing-AnnData inputs (mixed stores, sample identity, overlap refusal, dtype casting, MuData refusal), gene-panel fitting and the custom `.pkl` field, plus end-to-end runs on both store kinds covering mixed inputs, every app-panel input shape and dropdown option, automatic creation, a mount that lags its origin, CellTypist (a locally trained model) and Scrublet, failure and rerun semantics, read-only and create guards, and config parsing. They pass against annizarr 0.2.3; annizarr 0.2.1 fails the icechunk runs.
- The five real B092 samples (127,800 cells) on both kinds, with Scrublet. Stored counts, barcodes and HISE obs match each source file exactly, and the icechunk history is one commit per run.
- The bundled AIFI L1/L2/L3 models on real samples: labels and probabilities identical to running CellTypist directly, as `06-label_h5s.py` does.
- **CellTypist:** several models per store (default AIFI L1, L2, L3, bundled in `code/models` with a `SOURCE.json` of commit and checksums), columns `<model>_label` / `<model>_prob`. The model set is part of the schema.
- **Xenium (2026-10-30):** two public bundles' `cell_feature_matrix.h5` (Breast 280 genes, Lung 289) through `./run`: named after their bundle folders, Lung fitted with a warning (67 shared, 213 zero-filled, 222 dropped), commit lines name both sources, and a rerun skips both. The PBMC CellTypist models are meaningless on such panels (AIFI L1 shares 94 of 1,109 genes; 75% of breast cells labelled "T cell"), so spatial stores use None or a custom `.pkl`.
- **HISE layouts:** tenx-scrna-pipeline's observation names (`Batch`, `Pool`, `Chip`, `uuid`, …) are mapped onto the older HISE ones; labels already in a file are recomputed.

### 3.5 Code layout
Consolidated 2026-11-15 into five modules by function (README code map): `config` (settings, panel args,
logging) ← `inputs` (reading, precheck of every input/model problem before any write, picking new
inputs; RawInput, ObjectInput, IngestError) ← `prepare` (QC, CellTypist, Scrublet, the one gene policy,
the obs/X contract, staging; Staged, ContractError); `store`; `main` runs them top to bottom. Annizarr's
planned "append new genes" changes only `prepare`'s gene functions. `store` records each sample's source (icechunk commit
metadata, `attrs["ingest"]["sources"]` on plain zarr) so a re-sequenced file with the same ID is warned
about. Plain zarr on S3 is refused: annizarr 0.2.4 writes remote stores as icechunk only (tested
2026-11-15). 147 tests pass on annizarr 0.2.4 and 0.2.3 (2026-11-15).

### 3.6 Tests
- appends without any derived elements on `main`;
- one commit per run, plus the bootstrap seed;
- the schema has no `leiden`;
- Scrublet with no threshold gives NA calls and keeps scores (done);
- the existing per-input tests (HISE, 10x, h5ad, variable-length strings, ADT features).

### 3.7 Environment and resources
- **Adds** to the shared pins (`../CLAUDE.md` §6.2): scanpy, celltypist, scrublet (pip); `python-annoy`,
  `gh` (conda), loaded on the page from `requirements.txt` and `environment.yml`.
- **Resource class** `medium`; it depends on the largest sample, not the store. About 1 minute per
  30k-cell sample with CellTypist and Scrublet.

### 3.8 Decisions and history from the working notes (folded in 2026-11-15)
- **Checkout:** the Code Ocean checkout is `/root/capsule`, package `code/ingest`. The README's source of
  truth is `/root/capsule/code/README.md`; the copy in `capsules/anndata_ingest/code/` is overwritten from it.
- **Gene mismatch** (user, 2026-10-29/30): warn with the diff, then fit the sample to the store's var
  (zero-fill missing genes, drop extras, record `fitted` in the commit). Done 2026-10-30. The user will
  add "append new genes" to annizarr later.
- **CellTypist:** a dropdown plus an optional custom `.pkl` field (`CELLTYPIST_PKL`, panel value 7) that
  overrides it. Spatial defaults to None.
- **Rerun:** samples already present are skipped. Commits list each sample and its source path.
- **Input type dropdown (2026-11-04),** in this capsule (no separate import capsule): Raw samples
  (annotated) vs Existing AnnData/SpatialData table (as is). Identity by `sample_id` or `SAMPLE_COLUMN`;
  both kinds can share one store. MuData is dropped and refused by name. 107 tests passed then.
- **S3 and drop-derived (user, 2026-11-02):** create at `s3://` whenever it is writable (no
  `CREATE=true`). With the default `/results`, warn that the store becomes a read-only data asset.
  Appending onto a store with obsm/layers drops them with a warning (`drop_derived=True`) instead of
  refusing. The user believed annizarr drops by default; it doesn't, it raises (verified on 0.2.3).
  Built 2026-11-03 (89 tests, S3 create tested) and mirrored to `capsules/anndata_ingest`.
- **Xenium `.h5` findings (2026-10-29):** the sample ID used to parse as `cell_feature_matrix`, so
  duplicates were silently dropped; samples are named after the bundle folder instead (§3.1). The PBMC
  CellTypist models are meaningless on a ~300-gene panel. annizarr append requires identical var (names
  and order).
- **Commit 7d4ba4e** (2026-11-05, `codeocean_capsules`): the input-type and streaming work (112 tests)
  plus the spatial design (now `../spatial_ingest/spec.md` §3S).
