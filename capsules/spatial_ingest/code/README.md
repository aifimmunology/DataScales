# spatial ingest

A Code Ocean capsule that grows one [SpatialData](https://spatialdata.scverse.org) `.zarr`, run by run,
from three kinds of input, picked with **Input type** on the App Panel:

- **Xenium bundles** (a Xenium Onboard Analysis output folder, or its `.zip`), read with
  `spatialdata_io.xenium`;
- **Atera bundles** (folder or `.zip`), read with the Atera reader of
  [spatialdata-io PR #426](https://github.com/scverse/spatialdata-io/pull/426), vendored in
  `spatial_ingest/vendor/`;
- **Existing SpatialData objects** (`.zarr`), single- or multi-sample, for example a store this
  capsule wrote, or the output of `spatialdata.concatenate`. Each is split into its samples.

Every sample keeps everything it came with (images, labels, cell and nucleus shapes, transcripts and
its own table), renamed `<sample>_<element>` in its own coordinate system `<sample>`. Its cells also
go into one **joint single-cell table**, `tables/table`, which holds every sample's counts on one gene
panel. That table is what scRNA tools, the `anndata_ingest` capsule and `process` read. All tables
are indexed by gene symbol, so spatial and scRNA samples line up by gene name. The joint table is
appended in place with [annizarr](https://pypi.org/project/annizarr/), so it is never loaded whole.

**Design spec:** this capsule is `capsules/spatial_ingest` in the DataScales repo. The repo's
`capsules/spec.md` §3S describes it, and the other capsules too. In a workstation session the checkout
may be at `/scratch/DataScales`.

## Quick start

Attach your bundles (folders or the `.zip` files 10x gives you) or SpatialData `.zarr` folders to this
capsule as a data asset, so they appear under `/data`.

1. Open the **App Panel** and fill in the fields:

   | Field | What to enter | Example |
   |---|---|---|
   | **Storage Location** | where the store is, or should be created | `s3://my-bucket/spatial/lung_atlas.zarr` |
   | **Input type** | Xenium bundle · Atera bundle · Existing SpatialData | Xenium bundle (default) |
   | **List of paths to samples** | bundles, `.zip` files, `.zarr` folders, or folders holding them, separated by commas | `/data/xenium_runs, /data/extra/run7_outs.zip` |
   | **Custom Celltypist model** (optional) | a CellTypist `.pkl` trained for your gene panel | `/data/models/lung_panel.pkl` |

2. Click **Run**. A small Xenium bundle takes 15–20 s; a Xenium Prime 5K sample about 2–2.5 minutes
   and ~14 GB of memory.
3. Look in `/results`:
   - `spatial_ingest_summary.csv`: one row per sample, with its `status` (`appended`,
     `skipped (present)` or `skipped (no cells)`), cells, panel size, genes shared with the joint
     table, mean genes and counts, and seconds taken;
   - `spatial_ingest_run.json`: what the run did, any repairs, the store's totals and size, the
     package versions and the settings;
   - the store itself, if you created it in `/results`.

A **Reproducible Run** (the panel's defaults) adds every Xenium bundle in `/data/samples` to a new
store at `/results/spatialdata.zarr`.

### Common tasks

| I want to… | Storage Location | Input type |
|---|---|---|
| try it, or build a one-off store I'll download | `/results/<name>.zarr` (new) | the type of my inputs |
| start a store I'll keep adding to | `s3://<bucket>/<prefix>/<name>.zarr` (new; needs write access) | the type of my inputs |
| add Xenium runs, then Atera runs, to the same store | the same location, twice | Xenium, then Atera |
| merge several SpatialData objects (single- or multi-sample) into one | a new location | Existing SpatialData |
| continue a store I built in `/results` | copy it to S3 once, then use the S3 prefix | — |

### When a run stops with an error

Errors that come from the inputs or the store are one line in the log, and the run exits with code 1:

| The message says… | What to do |
|---|---|
| `… is Atera bundle; choose the input type 'Atera bundle'` (or the reverse) | pick the input type that matches the bundle |
| `Sample IDs must use only letters, digits, '_', '.' and '-'` | rename the bundle folder or `.zip` (sample IDs become element names) |
| `Two inputs have the sample ID …` | two bundles share a name: rename one |
| `… is a mouse panel, but the joint table holds human genes` | start a separate store per species (symbols such as `F3` exist in both) |
| `The store's joint table was built with CellTypist model …` | give the same `.pkl` as the store's first run (or none), or start a new store |
| `No store at …` | check the path of an existing store, or create the new one in `/results`, `/scratch` or S3 |
| `… is a SpatialData store this capsule didn't create` | give that object as an input (Existing SpatialData) into a new store |
| `… holds the same cells as …, which is already in the store` | the same sample under another name: leave it out |
| `… isn't a CellTypist model` | check the `.pkl` (it is loaded before any sample is read) |
| `… is read-only` | a data asset can't be written: give its S3 prefix instead |
| `… holds cells of several samples` / `… samples can't be told apart` | see "Existing SpatialData" below |

A run that fails partway keeps every sample finished before the failure:
- it repairs what the failing sample left and verifies the store;
- on S3 it syncs those samples back before stopping;
- the next run continues from there.

A run refused before any sample is read leaves nothing behind: a new store is created only when its
first sample is ready to go in. A killed run is repaired by the next one (see "Crash safety").

## How a run works

1. **Store:** a local path is written in place. An `s3://` store is copied to
   `SCRATCH_DIR/<name>`, grown there, and synced back at the end.
2. **Repair** what an earlier, killed run left (see below).
3. **Inputs:** list them, derive sample IDs, refuse bad or duplicate IDs, and skip samples already in
   the store. A sample is present when its `<sample>_table` is in the store.
   - A skipped sample that was added from a different path gets a warning, in case it is a
     different sample with the same name.
   - Cells already in the store under another sample ID are refused. Each sample's fingerprint (cell
     IDs, shape, total counts) is kept in its table.
4. **Panels**, from each new sample's `var` alone, before any sample is read:
   - refuse a species different from the store's;
   - warn about each sample whose genes differ from the joint table's.
5. **Per sample, one at a time:**
   1. Extract a `.zip`, and read the sample.
   2. Rename its elements and coordinate systems.
   3. Build its table: Gene Expression features on gene symbols, QC metrics (scanpy, mito = `MT-`),
      and CellTypist labels if a model is given.
   4. Write the sample to a staging folder inside the store, and move each element into place with
      `os.rename`. Consolidate the metadata: the sample is now present.
   5. Fit its cells to the joint table's genes and append them; consolidate again.
6. **Verify:** a fresh `spatialdata.read_zarr` lists every element, and the joint table holds exactly
   each per-sample table's cells.
7. On S3, sync back. Only changed files are uploaded, and the root `zarr.json` goes last, so a reader
   never sees the new samples before their files are there. A run that fails before the sync leaves
   S3 unchanged.

### Gene panels

- **Every table is indexed by gene symbol, with Gene Expression features only.**
  - Control probes and codewords leave the tables; the transcripts keep them.
  - Atera tables, which 10x indexes by Ensembl ID, are re-indexed by `feature_name`.
  - The Ensembl ID is kept in `var["gene_ids"]`.
- **Each sample's own table keeps its full panel.**
- **The joint table takes the first sample's genes.** Later samples are fitted to them, with a
  warning: missing genes are zero-filled and extra genes dropped (recorded under `fitted` in the run
  JSON). Put the sample with the panel you want first, or start the store with it.
- **One species per store,** read from the Ensembl prefixes (`ENSG` human, `ENSMUSG` mouse).

### Existing SpatialData

An input `.zarr` may hold one sample or many:

- **One annotated region:** the whole object is one sample, named after the folder (without `.zarr`),
  unless its table has a `sample_id` column.
- **Several samples:**
  - When the table has a `sample_id` column (`SAMPLE_COLUMN` to use another), the regions are grouped
    by it, and the sample IDs are its values. It is needed when several samples share one coordinate
    system.
  - Without the column, samples are told apart by coordinate system:
    - regions linked by a shared coordinate system are one sample (cell and nucleus boundaries of one
      section, or several bin sizes);
    - regions in separate systems are separate samples, named after the object and what sets the
      region's name apart: in `pair.zarr`, `cell_labels-a` (system `global-a`) and `cell_labels-b`
      (system `global-b`) become samples `pair_a` and `pair_b`.
    - Several samples that share one coordinate system (all in `global`) can't be told apart this
      way and go in as one sample, so give them a `sample_id` column.
  - A sample annotated by several tables (one per bin size, say) keeps one: `table` if present,
    else the first by name. The others are left out, with a warning.
- **Each sample's other elements:**
  - those in coordinate systems only that sample uses;
  - otherwise those whose name carries its tag (`morphology-a`, `a_transcripts`).
  - An element that matches no single sample is left out, with a warning.
- **Tables:**
  - A table that annotates nothing, such as this capsule's own joint table, is left out. The joint
    table is rebuilt from the per-sample tables.
  - Only the sample's rows of its table are loaded.
  - X must hold raw counts; when it doesn't, `layers["counts"]` is used if present.

So a store from this capsule can be merged into another, keeping its sample IDs. The output of
`spatialdata.concatenate({"a": sdata_a, "b": sdata_b})`, saved as `both.zarr`, goes in as samples
`both_a` and `both_b`; add a `sample_id` column to its table to choose the names yourself.

### Crash safety

Commit points are metadata consolidations: an element exists for readers once it is in the root
consolidated metadata. On start, every run:

1. deletes `<store>/.spatial_ingest_staging/` if a run left it;
2. deletes element folders that aren't in the consolidated metadata (a killed write; they would
   block the next write);
3. checks the joint table (`obs`, X shape and `indptr` agree, and each sample has as many cells as
   its own table). If not, it rebuilds the joint table from the per-sample tables;
4. appends present samples that are missing from the joint table.

This is tested by killing runs (SIGKILL) while an element is written and while the joint table is
appended.

## Running: the details

### Code Ocean app

`code/run` is the entrypoint. It takes four positional app-panel values, in panel order. An empty
value keeps the environment variable or default.

| # | category | parameter | meaning | default |
|---|---|---|---|---|
| 1 | Central Storage | Storage Location | a local path or `s3://bucket/prefix` to a SpatialData `.zarr` | `/results/spatialdata.zarr` |
| 2 | Sample Information | Input type | Xenium bundle (folder or .zip) · Atera bundle (folder or .zip) · Existing SpatialData (.zarr) | Xenium bundle |
| 3 | Sample Information | List of paths to samples | bundles, `.zip` files, SpatialData `.zarr` folders, folders holding them, or globs | `/data/samples` |
| 4 | Run params | Custom Celltypist model (optional) | a CellTypist `.pkl`; its labels go on every new sample | none |

How the samples list is read:
- **Separators:** commas, semicolons or newlines. Quotes, brackets and spaces around each item are
  ignored; spaces inside a path are kept.
- **What counts as one sample:**
  - A folder holding `experiment.xenium` (Xenium) or `experiment.spatial` (Atera) is one sample.
  - A `.zip` is one sample, found by the same marker files at its root or in its single top folder.
  - Any other folder's direct children are the samples; it isn't searched recursively. Children of
    another kind are ignored, and the log names them.
- **Relative paths** that don't exist are looked up under `/data`.

**Sample IDs:**
- **Bundle folder:** its name. Generic folders such as `outs` are skipped up to a meaningful parent.
- **`.zip`:** its name without `_outs` (`Xenium_V1_human_Breast_2fov_outs.zip` →
  `Xenium_V1_human_Breast_2fov`).
- **SpatialData:** see "Existing SpatialData" above.
- **Allowed characters:** letters, digits, `_`, `.` and `-`.

If Code Ocean leaves out an empty optional value, the values after the store are placed by their
shape: an input-type option, a `.pkl` the model, anything else the samples.

### Where the store lives

- **The Storage Location** is an absolute path or an `s3://` prefix.
- **Only stores this capsule built are grown.**
  - It marks them in their root attributes.
  - Its repair deletes element folders that the consolidated metadata doesn't list. In another
    tool's store those can be real data.
  - To add another SpatialData object, give it as an input into a store of this capsule.
- **S3 (recommended for a store you keep adding to):**
  - A missing store is created at the prefix; writing it is the permission check.
  - `/scratch` needs room for the whole store plus the largest unzipped bundle.
  - There is no locking: run one job per store at a time.
- **`/results`:** a one-off build. It can be saved as a data asset, but data assets are read-only,
  so later runs can't add to it; to continue it, copy it to S3 once.
- **`/scratch`:** cleared when the run ends; for trying things out.
- **Starting a store:**
  - **Where:** a missing local store is created only inside `CREATE_ROOTS` (`/results`, `/scratch`),
    not as either folder itself. Set `CREATE=true` to create one elsewhere.
  - **Refused:** an AnnData `.zarr`, an icechunk repo, another tool's SpatialData store, or a folder
    holding other files.
  - **Why:** a mistyped path to an existing store can't silently start a new one.

### Other settings

Every setting is an environment variable named after its field in `config.Settings`.
`spatial_ingest/.env.example` lists them with their defaults:

| Variable | Default | What it sets |
|---|---|---|
| `SAMPLE_COLUMN` | `sample_id` | the table column that names samples in existing SpatialData objects |
| `CREATE` / `CREATE_ROOTS` | `false` / `/results,/scratch` | where a missing local store may be created |
| `SCRATCH_DIR` | `/scratch/spatial_ingest` | extracted `.zip` bundles and the working copy of an S3 store |
| `N_JOBS` | 4 | Xenium reader workers |
| `WRITE_CPUS` | 8 | annizarr writers for the joint table |
| `S3_WORKERS` | 16 | parallel S3 transfers |

### Locally

```bash
cd code/spatial_ingest
python main.py /scratch/atlas.zarr "Xenium bundle (folder or .zip)" /data/xenium_runs ""   # panel order
python main.py --store s3://bucket/atlas.zarr --input-type atera --samples /data/atera     # or flags
python -m pytest                                                                         # ~3 min
```

The tests use synthetic SpatialData objects. They also run on the tiny public 10x bundles (Xenium
Breast 2fov, Ovary tiny, Mouse Ileum tiny, the Atera tiny dataset) when those are in
`SPATIAL_TEST_DATA/zip` and `SPATIAL_TEST_DATA/extract` (default `/scratch/spatial_dev/data`), and
are skipped otherwise. The download URLs are in spatialdata-io's `prepare_test_data.yaml`.

## The stored object

```
<store>.zarr
├── images/<sid>_morphology_focus, <sid>_morphology (Atera), <sid>_morphology_3d (Atera) …
├── labels/<sid>_cell_labels, <sid>_nucleus_labels
├── shapes/<sid>_cell_boundaries, <sid>_nucleus_boundaries
├── points/<sid>_transcripts
└── tables/
    ├── <sid>_table   one sample: its full panel (symbols, gene_ids), the reader's obs and obsm["spatial"],
    │                 sample_id, QC metrics, CellTypist labels, uns["spatial_ingest"] (model, source,
    │                 fingerprint); a bundle's annotates <sid>_cell_boundaries by cell_id, an existing
    │                 object's keeps its own region (renamed) and instance key
    └── table         the joint single-cell table: every sample's cells, no spatial link
```

- **Coordinate systems:** each sample's are `<sid>` (from the reader's `global`) and `<sid>_<other>`.
- **Joint table** (`tables/table`):
  - cell names `<sid>:<cell_id>` (`<sid>:<region>:<cell_id>` for a sample of several regions whose
    cell IDs repeat);
  - obs: `sample_id`, `cell_id`, `n_genes_by_counts`, `total_counts`, `pct_counts_mt`, the CellTypist
    `<model>_label` / `<model>_prob`, then the first sample's other reader columns (cell area,
    transcript counts, …). It leaves out the region links and per-sample clusters (`leiden*`);
  - X: raw counts, CSR, in the first sample's dtype (Xenium float32, Atera int32). annizarr widens
    `indptr` to int64 when the nonzeros outgrow int32;
  - `uns["spatial_ingest"]`: the panel's source sample, the CellTypist model, the species, the obs
    schema, and the samples in table order.

Open the store with `spatialdata.read_zarr(path)`. For the joint table alone, without the images, use
`anndata.read_zarr(f"{path}/tables/table")`, or `anndata.experimental.read_lazy` for big stores.
`anndata_ingest`'s "Existing AnnData or SpatialData table" mode ingests the joint table into an
AnnData store (tested).

## Code map

```
code/run                  Code Ocean entrypoint: passes the app-panel values on to main.py
code/spatial_ingest/
├── main.py               the run: store, repair, panels, one sample at a time, verify, sync, report
├── config.py             Settings from environment variables and app-panel values
├── inputs.py             inputs: bundle folders and .zip files, SpatialData folders; sample IDs; panels from var alone
├── existing.py           existing SpatialData objects split into samples (tables, regions, elements)
├── sample.py             one sample read and built: renamed elements, its table (counts, symbols, QC, labels)
├── genes.py              symbol indexing, species, panel differences and fitting
├── annotate.py           QC metrics and CellTypist labels
├── layout.py             element and coordinate-system names inside the store
├── joint.py              the joint table: schema, rows, seed/append with annizarr, consistency check
├── store.py              the local store: kinds, writability, presence, staged installs, repair
├── s3.py                 copy an S3 store down and sync it back (ETags, root zarr.json last)
├── vendor/               the Atera reader from spatialdata-io PR #426 (SOURCE.json: commit, checksums, changes)
└── tests/                unit tests; end-to-end runs (synthetic objects, S3 via moto, SIGKILL repair); real bundles
```

## Environment

- **`environment/requirements.txt`** pins the packages, with `pixi.toml` beside it for use outside
  Code Ocean:
  - shared with the other capsules: annizarr, zarr, anndata, numpy, pandas, boto3;
  - this capsule's own: spatialdata 0.8.0, spatialdata-io 0.7.1, scanpy, CellTypist, geopandas,
    pyarrow, s3fs, and moto for the S3 tests.
  - Load it on the environment page with **Edit Bulk**.
- **`environment/postInstall`** installs Claude Code only.
- **Resources:** about 16 GB peak per Xenium Prime 5K sample, one sample in memory at a time. Use a
  class with ≥ 32 GB RAM for Prime 5K or Atera.
