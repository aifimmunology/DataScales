# `process` rework: an environment and notebooks capsule — spec

> **Decision changed (2026-11-03):** the user wants an **optional** App Panel / Reproducible Run pipeline
> back, besides the notebooks: `code/run` and `code/process/`, with named panel parameters, PCA (+harmony)
> or scVI training, several leiden resolutions and an obs subset, saving by the notebooks' rules (§4.3). The
> capsule stays first an environment and notebook holder. §2's "Remove `code/run`, `code/process/`" no
> longer holds for them; see `spec.md` §12.

Status: **built 2026-11-02** (see `spec.md` §11). Changed while building: a `RESULTS` setting (where `umap.png`, `cluster_summary.csv` and the default `OUTPUT` go, for tests); the spatial notebook also writes `uns/spatial_process`, so it doesn't replace `uns/process`; an existing output gives `<stem>-<n>.zarr`, skipping names whose `.partial` exists; icechunk is one commented block in the settings cell that sets `repo`, and with `ORIGIN` the store is read through the origin (2026-11-03, `spec.md` §11.1). Tested on CPU only. It replaces the `run` pipeline described in `spec.md` §4–§5
here and in `../spec.md` §5. The decisions it rests on are the user's (2026-11-02). Related decisions for
ingest and qc, made at the same time, are in §7.

## 1. What changes, in one paragraph
`process` stops being a pipeline. It is the RAPIDS GPU environment plus two notebooks. The notebooks
work first on plain zarr: an AnnData `.zarr`, or a SpatialData `.zarr` and its table. Icechunk is an
optional path in each notebook, as commented-out cells. Saving never edits a zarr input: on zarr, a
notebook writes a **new, self-contained object** with the results; on icechunk, it **commits the results
back** to the repo. Nothing is refused for being read-only, being on `main`, or being remote. A
notebook just reads what it's given and writes where it's told.

- *Why:* the capsule's value is the environment and worked examples. The CLI froze choices that belong
  in a notebook, and its guards (refusing read-only inputs, the `qc` branch default) made it a dead end
  on a fresh ingest store.

## 2. Remove
| Item | Why |
|---|---|
| `code/run`, `code/process/` (config, cluster, steps, store, main) and its 38 tests | The notebooks hold the same steps; one copy of the logic |
| The CLI flags, environment variables, `run.json`, `cluster_summary.csv` | Notebook outputs replace them (§4.4) |
| Every refusal: read-only input, `--branch qc` default, remote plain zarr, NA-batch-key exit | The user decides in the notebook; a cell that can't work fails with the library's own error |
| `../spec.md` §5 as a pipeline (App Panel/flags table, "Where results go", reruns) | Replaced by §4 here |

Keep: `code/environment/` (requirements.txt, pixi.toml), `environment/postInstall`, `.codeocean/resources.json`.

The review findings that only applied to `run` become moot:
- crash leaves an unreadable in-place store;
- `--output` ignored on icechunk;
- relative paths resolved from `code/process`;
- the fingerprint gaps;
- a refused rerun wiping the run records;
- the no-lock interleaving.

The user accepted that a write which dies midway may leave a broken output (2026-11-02).

## 3. Layout
```
code/
├── README.md                  the environment first, then the notebooks
├── environment/               requirements.txt, pixi.toml
└── notebooks/
    ├── rapids_scrna_out_of_core_dask.ipynb      AnnData / SpatialData table, out of core
    └── rapids_spatial_atera_breast_cancer.ipynb  SpatialData, in memory
```
The notebooks are self-contained: no local package to import. The few helpers they need (open, save)
are short cells at the top of each notebook, written the same way in both.

## 4. Notebook pattern (both notebooks)

### 4.1 Settings cell (top)
```python
STORE = "/data/<data asset>/<name>.zarr"   # AnnData .zarr, or SpatialData .zarr
TABLE = "table"                             # SpatialData only: tables/<TABLE>
OUTPUT = None                               # None: /results/<name>.processed.zarr
SAVE = False                                # write results at the end

# --- icechunk (optional): uncomment to read a repo instead of a zarr ---
# from annizarr import Repo
# BRANCH = "main"
# ORIGIN = None          # s3://bucket/prefix behind a read-only data asset, to commit back
```

### 4.2 Open
- **Zarr (default):** `zarr.open_group(STORE, mode="r")`. If the root has no `X`, it's a SpatialData store,
  and the notebook uses `root["tables/<TABLE>"]`. A read-only data asset is fine, because nothing is written to it.
- **Icechunk (commented):** `Repo(STORE, branch=BRANCH).open_zarr("r")`, with the same `tables/<TABLE>` rule.
  With `ORIGIN`, reads go through the mount and the later commit goes to the origin (as `anndata_ingest`
  does).
- The out-of-core notebook keeps X lazy (`read_elem_lazy`, 24k-row blocks); the spatial one loads the
  table.

### 4.3 Save (last section, only when `SAVE = True`)
Results: `obsm/X_pca` (and `X_pca_harmony` if run), `obsm/X_umap`, `obs/leiden`, `var/highly_variable`,
`layers/lognorm` (with a `target_sum` attr), and `uns/process`. `uns/process` records the settings, the
input path, and the input's snapshot id or "zarr". The spatial notebook keeps its own names
(`spatial_cell_type`, `spatial_leiden`, `spatial_niche`, `X_pca_spatial`, `X_umap_spatial`).

- **Zarr input → a new full object** at `OUTPUT` (default `/results/<input stem>.processed.zarr`):
  - **AnnData:** `annizarr.rechunk(STORE, output=OUTPUT)` makes a streamed full copy (X, obs, var, layers,
    obsm, …). Then the results are written onto the copy: `write_elem` for obs/obsm/var/uns, and the
    lognorm layer streamed block by block from the GPU (moved from `process/store.py:write_lognorm`).
  - **SpatialData:** copy the whole store directory (`shutil.copytree`: images, labels, shapes, points,
    tables), then write the results into `tables/<TABLE>` of the copy, and re-consolidate the metadata.
    The copy must open with `sd.read_zarr`; the existing test only checked a table-only edit.
  - Build the copy at `<OUTPUT>.partial` and rename it into place when it's complete. If `OUTPUT` already
    exists, write `<name>.processed-<n>.zarr`, never over it.
  - *Why a full copy:* the result is standalone (it can become its own data asset), and the input is never
    changed. That makes read-only inputs a non-issue.
- **Icechunk input (commented cells) → one commit on `BRANCH`:** `Repo(STORE, branch=BRANCH,
  origin=ORIGIN)`, `open_zarr("w")`, write the same elements (into `tables/<TABLE>` for a SpatialData
  layout), then `commit(message, metadata=...)`. If the branch tip moved since loading, the commit fails
  rather than landing results on the wrong cells. That check exists today and is kept as a few lines.
  - Writing results to `main` is allowed. §7.3 is how ingest then handles those results on its next
    append.
- **Spatial notebook, cells removed by QC:** the results are written for the kept cells only, with NA
  elsewhere (as today). On a zarr input, that's in the new copy's table.

### 4.4 Outputs besides the store
A UMAP PNG and `cluster_summary.csv` (cells per cluster by `sample_id`) go to `/results`, written by a
cell. No `run.json`.

## 5. The two notebooks
| Notebook | Default input | Changes |
|---|---|---|
| `rapids_scrna_out_of_core_dask` | AnnData `.zarr` (or a SpatialData table) | Gains §4.3 save, so it's what `run` did. Optional harmony cell (commented; `BATCH_KEYS`). Treats `""` as missing in batch keys, warning, not refusing (ingest writes `""`, not NA) |
| `rapids_spatial_atera_breast_cancer` | SpatialData `.zarr` | Save changes from "into the table in place" to §4.3: a new SpatialData copy on zarr, a commit on icechunk. The icechunk loading moves to commented cells |

Both keep their current step order and defaults:
- **scRNA:** HVG seurat 2000, scale (max 10, no centering), PCA 50, neighbors 20/30, UMAP 0.45, leiden 1.1.
- **Spatial:** unchanged.

## 6. Docs and tests
- **README:** the environment and how to copy it; the two notebooks, the data each needs, and the save
  rules (§4.3). A short "icechunk instead of zarr" section points to the commented cells, and the
  SpatialData-on-icechunk limits stay (spatialdata 0.8.0: no `sdata.write()`/`read_zarr` on icechunk;
  table only).
- **`../spec.md` §5:** a short page pointing to this file. §2.3's entrypoint rule and §8's build order change to match.
- **Tests:** on the GPU capsule, run each notebook headless (`jupyter nbconvert --execute`) on tiny
  synthetic stores:
  - scRNA: AnnData `.zarr`, SpatialData `.zarr`, and icechunk with the cells uncommented;
  - spatial: SpatialData `.zarr`, and icechunk.

  Each test checks the new object or commit holds the results and reads back (`ad.read_zarr`,
  `sd.read_zarr`). It's a script `code/notebooks/test_notebooks.sh`, not a pytest package. The save
  helper's CPU logic (copy, write onto the copy, rename) can also be tested with scanpy standing in for
  rapids, as the review did (`/scratch/review_process/stubs`, now gone with `/scratch`).
- **Pins:** annizarr stays 0.2.3 on the page; both versions work (tested 2026-10-30/11-02).

## 7. Related decisions (2026-11-02), for the other capsules

### 7.1 qc apply: where curated cells go
- **Plain zarr:** write the subset (passing cells, all elements filtered row-wise) to the output location,
  as a new zarr. The input is never modified.
- **Icechunk:** a **new commit on `main`** whose X, obs and obsm hold only the curated cells, written through
  the origin when the asset is a read-only mount. The raw cells stay in `main`'s history.
  - **Tag before writing** (`raw-<snapshot>`), so the raw store is one checkout away. No garbage collection:
    raw history is kept.
- **How, without annizarr `subset`** (not in 0.2.4): stream X in whole-chunk bands with `read_elem_lazy`,
  mask rows, and write the dask CSR with `write_elem` into the new zarr, or into the working session. This is
  the same streaming the notebooks use for lognorm. Swap in `annizarr.subset` when it ships.
- **Consequences to settle in the qc spec:**
  - after qc, `main` mixes curated samples and later raw appends;
  - the next qc report must know which samples are already curated. `uns/qc/decisions` carries that, and
    curated samples are not re-filtered;
  - `../spec.md` §4.10 (the `qc` branch, GC) and §4.8 change accordingly.

### 7.2 Ingest: S3 and the default location
- **A new store at an `s3://` location is created when it's writable**, without `CREATE=true`. Today any
  missing location outside `/results` and `/scratch` needs it (`config.py` `may_create`). The create
  attempt is the write check; a permission error is reported plainly.
- **No storage location given (the default `/results/ingest_icechunk`):** warn that the store can be saved as
  a data asset but is then read-only, so later runs can't append to it, and that S3 is where a growing
  store belongs. A shorter warning already exists for `/results`; this adds the data-asset wording.

### 7.3 Ingest onto a store with results (embeddings, layers)
- **Appending new cells drops the elements built for the old cell set:** `obsm/*`, `obsp/*`, `layers/*`
  and `uns/process`, with a warning naming them, recorded in the commit metadata (`dropped`). Today ingest
  refuses such stores (`store.py` `derived_elements`).
- **annizarr does not drop them by default.** 0.2.3 and 0.2.4 raise ("will drop derived elements … Pass
  `drop_derived=True`"; tested 2026-11-02), so ingest passes `drop_derived=True` after warning.
  - Tested on 0.2.3 (2026-11-02): without the flag, `ConversionError … obsm/X_pca, layers/lognorm`. With it,
    80 cells and no `obsm`, but `list(adata.layers)` reads back as `[None]`; check whether a stray
    `layers` entry is left behind.
  - Re-check on 0.2.4, whose `append` now extends layers both stores share (staged samples have none).
- On icechunk the dropped results stay in history. On plain zarr they're gone, which is why the warning
  names them.

## 8. Open
- **Spatial `obsm["spatial"]`:** annizarr `append` drops `obsm`, so a joint table appended by
  `spatial_ingest` loses the cell centers that the spatial notebook needs. Options:
  - store the centers as obs columns (`x_centroid`, `y_centroid`), and have the notebook build `obsm["spatial"]`
    from them;
  - or annizarr keeps named `obsm` keys on append.
- **qc on spatial stores:** the qc design assumes PBMC (`MT-` genes, markers), so spatial stores skip qc
  for now.
- **The Code Ocean result-asset flow:** a notebook's new `.zarr` in `/results` becomes a read-only data
  asset, which is fine for the next notebook, since saving always makes a new object.
