# `spatial_ingest` — spec

The design of this capsule, its decisions and findings. Moved from the former `capsules/spec.md` §3S
(2026-11-15), section numbers kept. Shared rules (layout, column contract, environments, git, testing)
are in [`../CLAUDE.md`](../CLAUDE.md); usage is in `code/README.md`. This capsule is the reference for
the structural patterns in `../CLAUDE.md` ("Modularity and future-proofing").

## 3S. `spatial_ingest` — Xenium and Atera into one SpatialData (built 2026-11-11)

The source will be `capsules/spatial_ingest/`, laid out like `anndata_ingest` (`code/run`, one package, tests beside it, `../CLAUDE.md` §2.3). Findings below were tested from 2026-10-29 to 2026-11-05 with spatialdata 0.8.0, spatialdata-io 0.7.1 and zarr 3.4.2 on public 10x bundles. The prototype and test scripts are in `/scratch` (§3S.12): not saved to the repo, and gone when that machine's `/scratch` is cleared. Everything a build needs is in this section.

### 3S.0 As built (2026-11-11)
The capsule is in `capsules/spatial_ingest/`; its `code/README.md` documents it in full. It follows this section, with these changes:
- **Multi-sample SpatialData inputs (user, 2026-11-10)** replace the v1 refusal in §3S.7. An input `.zarr` may hold several samples; each becomes its own sample, and every input merges into the one store:
  - **Grouping:** regions are grouped by the table's `sample_id` (or `SAMPLE_COLUMN`) when present. Otherwise:
    - regions linked by a shared coordinate system are one sample;
    - separate systems are separate samples, named `<object>_<tag>` (`cell_labels-a` in `pair.zarr` → `pair_a`);
    - a sample annotated by several tables keeps one (`table` first);
    - samples sharing one system need the column.
  - **Elements:** a sample gets the elements in coordinate systems only it uses, else those whose name carries its tag. Others are left out with a warning.
  - **Tables:** tables that annotate nothing (this capsule's joint table) are left out, so a store from this capsule can be merged into another. Only a sample's rows are loaded (`anndata.experimental.read_lazy`).
  - **Naming:** tags already in element or coordinate-system names are stripped before prefixing (`global-a` → `<sid>`), so names never double up.
- **Atera** is read with `transcripts=True`. The vendored commit defaults to `False`, and §3S.1 wants transcripts.
- **Joint table `indptr`:** annizarr writes int32 and widens to int64 itself when the nonzeros outgrow int32, on write and on append. The capsule doesn't force int64.
- **Joint table attrs:** it carries `spatialdata-encoding-type: ngff:regions_table` and `version: 0.2` (with no region), without which `spatialdata.read_zarr` refuses it. The first sample's rows are written to staging and renamed into place, then later samples are appended in place.
- **`uns/spatial_ingest`** also holds the species, the obs schema and the samples in table order; the repair rebuilds the joint table in that order.
- **Repair** compares the joint table with the per-sample tables from their obs index alone. anndata 0.13 with pandas 3 writes string indexes as `nullable-string-array` groups.
- **Verification runs before the S3 sync,** so a broken store is never pushed.
- **annizarr 0.2.4,** like the other capsules' `requirements.txt`.
- **After the code review and black-box validation (2026-11-11):**
  - **Stores:** only stores this capsule created are grown; it marks them with `spatial_ingest` in the root attributes, because its repair deletes unlisted element folders. Another tool's SpatialData is refused as the target and goes in as an input instead. A new store is created only when its first sample is ready, so a refused run leaves nothing.
  - **Sample IDs:** tag-derived IDs are `<object>_<tag>`. A skipped present sample from another path is warned about. Each per-sample table keeps `uns/spatial_ingest` (CellTypist model, source, fingerprint), and the same cells under a new ID are refused.
  - **CellTypist:** the joint schema takes its model from that record, never from obs columns; the `.pkl` is loaded before any sample is read.
  - **Failures:** a failing sample triggers repair, verification and, on S3, a sync of the samples finished before it.
  - **Repair** always ends with a consolidate.
  - **Atera:** `morphology_3d` is read only when the bundle has it.
  - **S3 sync** trusts files unchanged since the download (same size and mtime) without hashing them.
  - **Names:** a sample of several regions whose cell IDs repeat gets joint names `<sid>:<region>:<cell_id>`. Element-name collisions are checked across kinds.
- **Second review pass:**
  - joint rows are built before a sample is written, so the joint table can't reject a committed sample;
  - S3 stores are judged from their root `zarr.json` before any download, folder-marker keys are skipped, and the working copy goes in `SCRATCH_DIR/store/`;
  - the fingerprint hashes the exact CSR counts.
- **Tested:** 53 tests, about 3 min.
- **Tested** (45 tests at first build):
  - synthetic single- and multi-sample objects;
  - SIGKILL at three points (an element write, the joint staging, the per-sample table rename), each repaired by the next run;
  - an S3 round trip under moto, with the root `zarr.json` uploaded last and a no-op rerun uploading nothing;
  - the real tiny bundles (Xenium Breast/Ovary/Mouse Ileum, Atera) and their read-back;
  - the joint table into `anndata_ingest`'s existing mode.
- **Robustness rework (user decisions 2026-11-13, after 260 test cases on 83 public datasets):**
  - **No Input type:** the panel is Storage Location, List of paths to samples, Custom CellTypist `.pkl`. Each sample's kind (Xenium, Atera, SpatialData) is detected from its files; kinds mix in one run. Older 4-value calls and `INPUT_TYPE` are accepted, the input type ignored (INFO). `~` expanded, `.pkl` any case.
  - **Discovery** (`discover.py`): archives (`.zip`, `.tar(.gz)`, `.tgz`) listed, not extracted; every bundle root at any depth is a sample; per-file prefixes (10x downloads, GEO `GSM…_`, `MB263-`) grouped one sample per prefix; GEO per-file `.gz` decompressed to scratch on demand (`stage.py`); old SpatialData layouts and `.zarr.zip`; junk ignored silently; anything else left out with one warning.
  - **Precheck** (`precheck.py`) before any write: every bundle's inventory (essential: count matrix; cells table with IDs + x/y matching the matrix), every SpatialData split, species, duplicate IDs, CellTypist overlap (refuse below max(10, 5% of the model's genes), warn below 50%). Any problem stops the run with one message listing every sample's problem; optional parts missing are one warning line per sample.
  - **Joint table = union** of genes and obs columns (supersedes §3S.1 "first sample's genes"): new genes extend var at the end; `var["n_samples_measured"]` and `uns/spatial_ingest/measured_genes/<sid>` tell measured zeros from filled ones; differing panels get one strong warning in the precheck.
  - **Out-of-core:** count matrices streamed to CSR zarr on scratch and processed in row chunks (QC, labels, fingerprint, joint append). Resource class: 64 GB ("large").
  - **Logging** (`logs.py`): library noise silenced at its source; a clean run logs no WARNING line. celltypist is imported inside `warnings.catch_warnings()` (it disables UserWarning/FutureWarning/RuntimeWarning globally on import).

### 3S.1 Decisions (user, 2026-11-05)
- **Store:** always a plain SpatialData `.zarr`, never icechunk (§3S.2). The Storage Location is a local path (written or grown in place) or an `s3://` prefix (copied to `/scratch`, grown there, synced back; §3S.6).
- **Inputs:** Xenium bundles, Atera bundles (folder or `.zip`), or an existing single-sample SpatialData `.zarr`.
- **Per sample:** its images, labels, cell/nucleus shapes, transcripts (points) and its own table, all prefixed by the sample ID and in its own coordinate system. The per-sample table keeps every Gene Expression feature of its panel, `obsm["spatial"]` and its region link, and is the source of truth.
- **Joint table: single-cell data only.** One AnnData over all cells at `tables/table`: X, obs, var. No `obsm`, no `spatialdata_attrs`, no region link: samples don't share a coordinate space, so stacked `spatial` coordinates would be meaningless. Appended with annizarr in place.
- **Genes:** every table is indexed by gene symbol (§3S.4). The joint table takes the first sample's genes; later samples are fitted to them with a warning (zero-fill missing, drop extra), as in `anndata_ingest`.
- **Per-sample QC metrics** with scanpy; no Scrublet. **CellTypist:** default None; an optional custom `.pkl`.
- **Atera is in v1:** the reader from spatialdata-io PR #426 vendored, pinned, with the `morphology_3d` fix (§3S.7).

### 3S.2 Why not icechunk
- Released spatialdata can't write to or read from icechunk: `write()`/`read_zarr()` take paths, and the store-backed writes in scverse/spatialdata PR #1107 are unmerged.
- Points and shapes are Parquet files inside the `.zarr` folder. Icechunk holds only zarr keys and rejects them ("invalid zarr key format"). Packing each Parquet file into a uint8 zarr array works (prototyped) but needs our own read/write layer, loads whole files, and no other tool can open the result.
- The earthmover `2026-scverse/icechunk-SpatialData` material is talk code on a patched spatialdata fork plus the non-public `zarr-files` package.

### 3S.3 Tested findings
| Area | Finding |
|---|---|
| Incremental append | 4 public Xenium bundles in 3 runs into one `.zarr` with `write_element`; stock `read_zarr` reads it back; bbox query and table↔shapes join work; a verifier catches corrupted transforms, obs and X. Rerun of a present sample: 1.5 s, all files byte-identical |
| Gene index | Verified 2026-11-05, §3S.4 |
| S3 | spatialdata 0.8.0 can't write or read S3: `UPath` is rejected, and a plain `"s3://…"` string silently creates a local `./s3:/` folder. Build locally, then sync with the root `zarr.json` last: tested |
| Crash | `write_element(overwrite=True)` refuses elements of its own store; delete-then-write loses the table on a crash and the store no longer opens. Staged write + `os.rename` + startup repair tested OK. An element killed mid-write is invisible to consolidated metadata but blocks the rerun until deleted |
| Scale (Xenium Prime 5K skin, 112k cells, 74M transcripts, 7.6 GB zip) | ~2–2.5 min and ~14 GB peak per sample append; 4.3 GB per sample on disk (images 2.9, transcripts 1.3); keep native Parquet row groups (2–3M rows). Opening the store: +0.85 s per sample (shapes load eagerly). Container memory cap is 61.7 GiB |
| Joint table | In-memory rewrite: 1M cells 19 s / 7.5 GB, 5M 52 s / 27 GB, ~21M OOM. annizarr in-place append: ~3–4 s at any size (7,275 cells onto 19,828 × 5,753 in 1.4 s); stock `read_zarr` reads it after `zarr.consolidate_metadata` |
| Atera (tiny bundle) | PR #426 reader works on released spatialdata 0.8.0; 0.8.0 reads the joint Xenium+Atera `.zarr`. Full public Atera sets are ~50–65 GB |
| Standard workflow | None published for incremental multi-sample SpatialData: `spatialdata.concatenate` is in-memory (and intersects genes by default), sopa keeps one `.zarr` per sample |
| annizarr `append` | Raises on `obsm` unless `drop_derived=True` (since 0.2.3 / 2026-11-02). The joint table has no `obsm`, so a plain `append` works; the capsule never passes the flag |

### 3S.4 Gene index: symbols everywhere (verified 2026-11-05)
What each reader gives on its own:

| Source | var index | Symbol / ID columns | Controls in the table |
|---|---|---|---|
| Xenium, spatialdata-io 0.7.1 (Breast 2fov 280 genes, Prime Mouse Ileum 5,006) | **symbol** (`ABCC11`, `A1cf`), unnamed, no duplicates | `gene_ids` (Ensembl, 100%), `feature_types`, `genome` | none (reader keeps Gene Expression only) |
| Xenium `cell_feature_matrix.h5` | — | `features/name` = symbol, `features/id` = Ensembl | in the file (Breast: 261 control features beside 280 genes; Ileum: 8,774 beside 5,006) |
| Atera, PR #426 reader (tiny bundle, 132 features) | **`feature_id`**: Ensembl for genes, the control's name for controls (`NegControlCodeword_0035`) | `feature_name` (symbol), `feature_type`, `genome`, `filtered`, `highly_variable` | 79 of 132 (Negative Control Codeword 62, Unassigned Codeword 11, Negative Control Probe 6) |
| Atera bundle's own `cell_feature_matrix.zarr.zip` / `csc_…` (no `.h5` in the bundle) | **`feature_id`** (`var/.zattrs` `_index: feature_id`) | `feature_name` = symbol | in the file |
| 10x scRNA `.h5` in `anndata_ingest` (`ingest/hise.py` `load`, `_gene_names`) | **symbol**: `matrix/features/name`, Gene Expression only, then `anndata.utils.make_index_unique` | none kept | dropped |

So 10x itself makes Ensembl the primary key for Atera; the capsule re-indexes. For every table (per-sample and joint):
1. Keep Gene Expression only: `feature_types` or `feature_type` == `"Gene Expression"` (`anndata_ingest`'s `gene_mask`; no such column = all genes). Control probes and codewords leave the tables; the transcripts points keep them as the reader wrote them.
2. `var_names` = symbol: Xenium as read; Atera `feature_name`; an existing SpatialData's index if it isn't Ensembl, else its first of `feature_name`, `gene_symbols`, `gene_name`, else refused ("table is indexed by Ensembl IDs and has no symbol column").
3. Duplicates made unique after filtering with `anndata.utils.make_index_unique` (`-` join: the second `X` becomes `X-1`), as `anndata_ingest`. Applied per sample before fitting, so the same panel always yields the same names.
4. Ensembl kept as var column `gene_ids` (Atera `feature_id` renamed; Xenium already `gene_ids`). var keeps only `gene_ids` and `feature_types`; the joint table's var is the first sample's.
- **Caveat, mixed species:** symbols collide across species (`F3` is a human and a mouse gene). The joint table holds one species: each sample's species is taken from `gene_ids` (`ENSG`/`ENST` human, `ENSMUSG`/`ENSMUST` mouse; `anndata_ingest` accepts `ENST` for CD45RA/RO), and a sample whose species differs from the store's is refused before any write. Mouse panels are mixed-case (`A1cf`), so fitting a mouse sample onto a human panel would anyway share almost nothing.

### 3S.5 App panel
Fields, passed positionally in this order (as `anndata_ingest` §3.2 in `../anndata_ingest/spec.md`; when Code Ocean drops an empty optional value, values are placed by shape):
1. **Storage Location** (text, required, default `/results/spatialdata.zarr`): a local path or `s3://bucket/prefix` to a SpatialData `.zarr`, created when missing at an `s3://` prefix or inside `/results` or `/scratch` (any other local path only with `CREATE=true`). Help text: a store you keep adding to belongs on S3; `/results` is a one-off build (saved as a data asset it is read-only); `/scratch` is cleared.
2. **Input type** (dropdown, default the first): `Xenium bundle (folder or .zip)` · `Atera bundle (folder or .zip)` · `Existing SpatialData (.zarr)`. One type per run; run again for another type into the same store.
3. **Samples** (text, default `/data/samples`): bundle folders, bundle `.zip` files, SpatialData `.zarr` folders, folders holding them, or globs, comma/semicolon/newline separated (`anndata_ingest`'s `split_list`). A folder that is itself a bundle (has `experiment.xenium` for Xenium, `experiment.spatial` for Atera) is one sample; otherwise its direct children are the samples (not recursive).
4. **Custom CellTypist model** (optional `.pkl` file): labels added to every new sample; empty = no labels.
- Every other setting is an environment variable named after its `Settings` field in upper case (`CREATE`, `SCRATCH_DIR`, `N_JOBS`).
- No `origin` field: there is no icechunk, and a data-asset mount of a plain zarr is read-only; S3 is given directly.

### 3S.6 Store location
- **Local path:** written in place. Writability is probed as in `../CLAUDE.md` §2.1 (`.write_probe`); a read-only mount is refused before any work. A path in `/results` gets the same warning as `anndata_ingest` (one-off build; can't be appended to after it becomes a data asset).
- **`s3://` prefix**, per run:
  1. If the prefix holds a root `zarr.json`, copy the whole store to `/scratch/spatial_ingest/<store name>` (boto3, parallel); none = a new store, created there.
  2. Run every step below on that local copy.
  3. Sync back as `s3sync.py` does: upload new and changed files (compared by S3 ETag, computing the multipart ETag with boto3's 8 MiB part size), delete remote keys no longer present locally, the root `zarr.json` last, so a reader never sees new consolidated metadata before its files. A run that fails before the sync leaves S3 unchanged.
  - `/scratch` needs room for the whole store plus the new samples (and extracted zips). Copying only the `zarr.json` files and `tables/` is a possible later saving; not in v1.
  - No locking: one run per store at a time (stated in the README and panel help).
- **Growing a store across Code Ocean runs:** use S3. `/results` is cleared each run and a saved data asset is read-only, so a `/results` store is a single build; to continue it, copy it to S3 once and point later runs there.
- **Refused:** an icechunk repo (`annizarr.Repo.exists`), an AnnData zarr, or a non-empty folder that isn't a SpatialData store.

### 3S.7 Reading inputs
- **Xenium:** `spatialdata_io.xenium(path, n_jobs=N_JOBS)` with reader defaults.
- **Atera:** the vendored reader `atera(path, morphology_3d_images=True)`.
  - Vendored as `code/spatial_ingest/vendor/atera.py` + `_atera_common.py` from `10XGenomics/spatialdata-io@738204de` (PR #426, unreviewed), with a `SOURCE.json` of the commit and file checksums. Imports rewritten to the package; nothing else changed except the fix below.
  - **`morphology_3d` fix:** the reader passes the same `image_models_kwargs` to 2D and 3D images, and the 3D parse fails (chunks of the wrong length, then scale factor 2 along `z`). The 3D call gets its own kwargs `{"chunks": (1, 1, 4096, 4096), "scale_factors": [{"z": 1, "y": 2, "x": 2}] * 2}` (verified on the tiny bundle: 3-level pyramid, `z` kept at 5).
  - X comes back as a dask array (int32); it is computed and converted to CSR.
- **Bundle `.zip`:** extracted to `/scratch/spatial_ingest/extract/<sid>/` first (10x zips hold the bundle files at the root; a single top-level folder inside is descended into), then read like a folder, and deleted after the sample is written.
- **Existing SpatialData:** `sd.read_zarr(path)` (lazy). It must hold exactly one table annotating one region; several tables, several regions, or a table whose `sample_id` has more than one value is refused in v1: "<path> holds several samples; write one SpatialData per sample". A store written by this capsule is refused the same way. **Superseded (2026-11-10):** multi-sample inputs are split per sample (§3S.0).
- **Sample IDs:** the bundle folder's name, or the `.zip` stem without a trailing `_outs` (10x download names: `Xenium_V1_human_Breast_2fov_outs.zip` → `Xenium_V1_human_Breast_2fov`); generic folders (`outs`, `extract`) are skipped up to the first meaningful parent (`anndata_ingest`'s rule); an existing SpatialData's folder name without `.zarr`. An ID must match `[A-Za-z0-9_.-]+` (SpatialData element names); anything else, or two inputs with the same ID in one run, is refused before any work.

### 3S.8 Layout inside the store
- **Elements:** each reader element `<name>` is written as `<sid>_<name>` (`<sid>_morphology_focus`, `<sid>_cell_boundaries`, `<sid>_nucleus_labels`, `<sid>_transcripts`, …).
- **Coordinate systems:** `global` → `<sid>`; any other system `c` → `<sid>_<c>`.
- **Per-sample table** `tables/<sid>_table`: annotates `<sid>_cell_boundaries`, `region_key="region"` (every row `<sid>_cell_boundaries`), `instance_key="cell_id"` (both readers index `cell_boundaries` by `cell_id`; the Xenium/Atera label link columns `cell_labels` / `label_id` stay as plain obs columns). Keeps the reader's obs, `obsm` (`spatial`; Atera also `X_pca`, `X_umap`) and the symbol-indexed var (§3S.4), plus `sample_id`, the QC columns and the CellTypist columns. scanpy's `total_counts` (Gene Expression only) replaces the reader's column of that name.
- **Joint table** `tables/table` (read by `anndata_ingest`'s SpatialData mode and `process`'s default `TABLE`):
  - cell names `<sid>:<cell_id>`;
  - obs: `sample_id` (categorical), `cell_id` (string), `n_genes_by_counts`, `total_counts`, `pct_counts_mt`, `<model>_label` / `<model>_prob` when a `.pkl` is given (model = the `.pkl` stem), then the first sample's remaining reader obs columns (`cell_area`, `nucleus_area`, `transcript_counts`, `control_*_counts`, …; Atera `azimuth_*`), except `region`, `cell_labels`, `label_id`, `barcode` and `leiden_*` (per-sample clusters). Later samples: missing columns empty, extra columns dropped with one warning, types converted to the store's as in `anndata_ingest`;
  - X: raw counts, CSR, int64 `indptr`, the first sample's dtype (Xenium float32, Atera int32); later samples cast (counts must fit);
  - var: the first sample's symbols, `gene_ids`, `feature_types`;
  - `uns["spatial_ingest"]`: the gene panel's source sample and the CellTypist model, so a later run with a different model is refused, as in `anndata_ingest`.
- **QC:** `scanpy.pp.calculate_qc_metrics(qc_vars=["mt"], percent_top=None, log1p=False)` per sample, mito = `MT-` prefix (`anndata_ingest`'s `contract.is_mito`). Spatial panels usually carry no mitochondrial genes, so `pct_counts_mt` is mostly 0; mouse `mt-` isn't matched (same as `anndata_ingest`).

### 3S.9 Presence, crash safety and repair
- **Present:** a sample is present when `tables/<sid>_table` is in the root consolidated metadata. Present samples are skipped (logged), even with a different input type.
- **Commit points:** per-sample elements become present at a consolidate; the joint table is appended after that, then consolidated again. A crash between the two leaves a present sample missing from the joint table, which the next run appends.
- **Repair on start**, before reading any input (logged, and listed in `spatial_ingest_run.json`):
  1. delete `<store>/.spatial_ingest_staging/` if left over;
  2. delete every element folder on disk that isn't in the consolidated metadata (a killed write; it would block `write_element`);
  3. check the joint table: `obs` length, X shape and `indptr` length agree, and each `sample_id` in it has as many cells as its `<sid>_table`. If not (killed mid-append), rebuild it from the per-sample tables into staging (first table seeds it, the rest append, as in a normal run) and swap it in with `os.rename`;
  4. append present samples missing from the joint table (step 7 below);
  5. consolidate.
- On S3, repair runs on the local copy, and the sync carries its deletions back.

### 3S.10 Steps per run
1. Parse the panel and environment; resolve the store (local, or copy from S3 to `/scratch`); refuse a read-only, icechunk, AnnData or foreign location; create when allowed.
2. Repair (§3S.9).
3. List the inputs, derive sample IDs, refuse bad or duplicate IDs; skip present samples. Nothing new: write the outputs and exit 0.
4. Check panels before reading any sample: read each new sample's var alone (Xenium `cell_feature_matrix.h5`, Atera `cell_feature_matrix.zarr.zip` `var`, an existing table's `var`), apply §3S.4, refuse a species mismatch, and warn per sample about the fit to the joint table's genes (n shared / only in store / only in sample, with examples).
5. Per new sample, one at a time: extract a zip; read; prefix element names and coordinate systems; build the per-sample table (§3S.4, §3S.8, QC, CellTypist); write every element with `write_element` into `<store>/.spatial_ingest_staging/<sid>.zarr`; `os.rename` each element folder into the store; consolidate (the sample is now present); delete the extract.
6. Stage this sample's joint rows (genes fitted, obs projected, names `<sid>:<cell_id>`) as an AnnData zarr in staging.
7. Append the staged rows to `tables/table` with `annizarr.append` (the first sample of a new store seeds `tables/table` with `annizarr.rechunk` from its staging zarr instead); consolidate.
8. Delete staging; on S3, sync back (§3S.6).
9. Verify: a fresh `sd.read_zarr(store)` lists every element, `tables/table` has the expected cells; write the outputs.
- Each sample commits on its own (steps 5–7), so a failure keeps the samples finished before it and the next run continues; the error names the failed sample. This differs from `anndata_ingest`, which touches the store only after every sample is staged: a spatial sample is several GB, so staging all of them first would double the disk need.

### 3S.11 Outputs, environment, resources
- **`/results`:**
  - `spatial_ingest_summary.csv`, one row per sample handled this run: `sample_id`, input type, source path, status (appended / skipped / repaired), cells, genes in panel, genes shared with the joint table, mean genes, mean `total_counts`, number of CellTypist labels, seconds;
  - `spatial_ingest_run.json`: parameters, store (and S3 prefix), repair actions, appended and skipped samples, fitting per sample, totals (samples, cells, store size), versions (spatialdata, spatialdata-io, vendored Atera commit, annizarr, zarr), timestamps, peak memory.
- **Environment** (CPU image, `../CLAUDE.md` §6.3): `spatialdata 0.8.0`, `spatialdata-io 0.7.1` plus the vendored Atera reader, the shared pins of `../CLAUDE.md` §6.2 (zarr 3.4.x: §6.2 pins 3.4.1, the tests ran on 3.4.2; check 3.4.1 in the first build), `annizarr 0.2.4`, scanpy, celltypist, geopandas, pyarrow, s3fs, boto3, pytest.
- **Resources:** ~16 GB peak per Xenium Prime 5K sample (14 GB measured), so a class with ≥ 32 GB RAM for Prime 5K or Atera; `/scratch` sized for the store copy plus the largest unzipped bundle.

### 3S.12 Prototype pointers (ephemeral: `/scratch` only, not in the repo)
- `/scratch/spatial_tests/xenium_e2e/scripts/`: `ingest.py` (the prototype append; it tracked presence in `uns['samples']`, superseded by §3S.9), **`verify.py`** (fresh-process read-back against the source bundles: a template for the read-back test), **`killtest.sh`** (SIGKILL when a path appears: a template for the crash test), **`s3sync.py`** (ETag sync, root `zarr.json` last: copy into the capsule), `snap.py` (file snapshot diff, for "rerun leaves files byte-identical"), `corrupt.py`, `open_check.py`, `probe.py`, `s3patch.py` (a failed S3 read patch, evidence only).
- `/scratch/spatial_tests/atera/`: **`pr_src/atera.py`, `pr_src/_atera_common.py`** (the PR #426 files to vendor), `scripts/t1_read_atera.py`, `t2_readback.py`, `t3_mix.py` (Atera + Xenium in one store), `logs/` (3D failure, PR metadata); `venv/` holds the PR reader.
- `/scratch/spatial_tests/scale/scripts/` (Prime 5K timing and memory, joint-table rewrite vs append), `/scratch/spatial_tests/annizarr_joint/` (`t_append.py`, `t_read.py`), `/scratch/spatial_proto/scripts/` (first prototypes, incl. the icechunk probes behind §3S.2).
- Test data: `/scratch/spatial_proto/data/` — `Xenium_V1_human_Breast_2fov`, `Xenium_V1_human_Lung_2fov`, `Xenium_V1_Human_Ovary_tiny`, `Xenium_Prime_Mouse_Ileum_tiny`, `tiny_atera_dataset`, each as `zip/` and `extract/`. All are public 10x downloads.

### 3S.13 Relation to `anndata_ingest`
- Its "Existing AnnData or SpatialData table" mode ingests a `spatial_ingest` store's `tables/table` into an AnnData store as is: symbol-indexed, `sample_id` per cell, no `obsm` to drop. Both capsules now key genes by symbol, so spatial and scRNA samples fit to each other's panels by name.
- Its raw mode's Xenium `cell_feature_matrix.h5` path stays for a table-only ingest without images.

### 3S.14 Still to test
- A full-size Atera bundle (~50–65 GB): time, memory, `/scratch` need.

### 3S.15 Tests (original plan; synthetic bundles and the tiny public ones in §3S.12)
- Xenium and Atera bundles, as folders and as `.zip`, into a new store; Atera + Xenium in one store;
- every table (per-sample and joint) is symbol-indexed with `gene_ids`, Gene Expression only; a mouse sample into a human store is refused;
- a mixed panel is fitted with a warning; a rerun skips present samples and leaves every file byte-identical;
- read-back in a fresh process with `sd.read_zarr`: elements, coordinate systems `<sid>`, table↔`<sid>_cell_boundaries` join, joint cell names and obs (template `verify.py`);
- SIGKILL during an element write and during the joint append, then repair on the next run (template `killtest.sh`); a present sample missing from the joint table is appended;
- an S3 round trip: create, append, rerun; the root `zarr.json` uploaded last;
- an existing single-sample SpatialData goes in; a multi-sample one is refused (superseded 2026-11-10: split per sample, §3S.0); the joint table goes into `anndata_ingest`'s existing-SpatialData mode.

### 3S.16 From the working notes (folded in 2026-11-15)
- **Test venv (2026-10-29):** spatialdata 0.8.0, spatialdata-io 0.7.1, zarr 3.4.2 in `/scratch/spatial_proto/venv`; tests in `/scratch/spatial_tests/` (cleared with `/scratch`).
- **Container memory cap** is 61.7 GiB, not the 124 GB shown. Appending the joint table in place stays ~3 s.
- **Atera `morphology_3d`** workaround first found: `chunks=(1,1,4096,4096)` (full fix in §3S.7). The PR #426 reader is ~1.2k lines.
- **Gene keys:** all Xenium bundles and Atera carry Ensembl IDs. Accept ENSG and ENST (CD45RA/RO use ENST).
- **Prototype scripts:** the user chose not to save them into the repo.
- **History:** planned 2026-10-29/30; decisions 2026-11-05 (§3S.1); built 2026-11-11; on `origin/codeocean_capsules` since; the user considers it the well-set-up reference (2026-11-15).
