# spatial ingest

A Code Ocean capsule that grows one [SpatialData](https://spatialdata.scverse.org) `.zarr`, run by run,
from three kinds of samples. You don't say which kind you give: each sample's kind is detected from its
files, and the kinds can be mixed in one run.

- **Xenium bundles**: a Xenium Onboard Analysis (or Xenium Ranger) output folder, its `.zip` / `.tar.gz`,
  or its files downloaded one by one (10x per-file downloads, GEO supplementary files). Read with
  `spatialdata_io.xenium` when the bundle is complete, else from its files directly.
- **Atera bundles** (folder or archive), read with the Atera reader of
  [spatialdata-io PR #426](https://github.com/scverse/spatialdata-io/pull/426), vendored in
  `spatial_ingest/vendor/`.
- **Existing SpatialData objects** (`.zarr` folder or `.zarr.zip`), single- or multi-sample, old or new
  format: for example a store this capsule wrote, or the output of `spatialdata.concatenate`. Each is
  split into its samples.

Every sample keeps everything it came with (images, labels, cell and nucleus shapes, transcripts and
its own table), renamed `<sample>_<element>` in its own coordinate system `<sample>`. Its cells also
go into one **joint single-cell table**, `tables/table`, which holds every sample's counts on the
**union** of the samples' genes. That table is what scRNA tools, the `anndata_ingest` capsule and
`process` read. All tables are indexed by gene symbol, so spatial and scRNA samples line up by gene
name. Count matrices are streamed (read and written in row chunks), so no sample's matrix is ever
held in memory whole, and the joint table is appended in place with
[annizarr](https://pypi.org/project/annizarr/).

**Design spec:** this capsule is `capsules/spatial_ingest` in the DataScales repo. The repo's
`capsules/spec.md` §3S describes it, and the other capsules too. In a workstation session the checkout
may be at `/scratch/DataScales`.

## Quick start

Attach your bundles (folders, or the `.zip` / `.tar.gz` files you downloaded) or SpatialData `.zarr`
folders to this capsule as a data asset. A data asset appears under `/data/<mount name>`.

1. Open the **App Panel** and fill in the fields:

   | Field | What to enter | Example |
   |---|---|---|
   | **Storage Location** | where the store is, or should be created (a `.zarr` path) | `s3://my-bucket/spatial/lung_atlas.zarr` |
   | **List of paths to samples** | bundles, archives, `.zarr` folders, folders holding any of them, or globs; separated by commas, semicolons or new lines | `/data/xenium_runs, /data/extra/run7_outs.zip` |
   | **Custom CellTypist model** (optional) | a CellTypist `.pkl` trained on your gene panel, same species and gene symbols | `/data/models/lung_panel.pkl` |

2. Click **Run**. The run first checks every sample (the **precheck**, below) and stops before
   writing anything if a sample can't be added. A small Xenium bundle takes about 30 s; a Xenium
   Prime 5K sample a few minutes.
3. Look in `/results`:
   - `spatial_ingest_summary.csv`: one row per sample, with its `kind`, `source`, `status`
     (`appended`, `skipped (present)` or `skipped (no cells)`), cells, panel size, genes shared with
     the other panels and new to the joint table, genes shared with the CellTypist model, mean genes
     and counts, and seconds taken;
   - `spatial_ingest_run.json`: what the run did: repairs, left-out items, the joint table's gene
     count, the store's totals and size, package versions, peak memory and the settings;
   - the store itself, if you created it in `/results`.

A **Reproducible Run** (the panel's defaults) adds every sample in `/data/samples` to a new store at
`/results/spatialdata.zarr`; it needs a data asset mounted at exactly `/data/samples`. Without one it
stops and lists what `/data` holds.

### Common tasks

| I want to… | Storage Location | List of paths to samples |
|---|---|---|
| try it, or build a one-off store I'll download | `/results/<name>.zarr` (new) | my samples |
| start a store I'll keep adding to | `s3://<bucket>/<prefix>/<name>.zarr` (new; needs write access) | my samples |
| add Xenium and Atera runs to the same store | the same location | both, in one run or in several |
| merge several SpatialData objects (single- or multi-sample) into one | a new location | the `.zarr` folders |
| continue a store I built in `/results` | copy it to S3 once, then use the S3 prefix | the new samples |

## What counts as a sample

Every path in the list is searched (folders and archives, down to 8 levels), and everything found is
one of: a **Xenium bundle**, an **Atera bundle**, a **SpatialData object**, or **left out**.

- **A bundle** is a folder (or archive folder) with `experiment.xenium` (Xenium) or
  `experiment.spatial` (Atera). Without the marker file, a folder with a count matrix and cell
  positions is a bundle too; its kind comes from its files (Atera's zarr-only tables and
  `morphology_2d/` → Atera, else Xenium).
- **Archives** (`.zip`, `.tar`, `.tar.gz`, `.tgz`) are listed, not unpacked: every bundle inside is its
  own sample (a zip holding two runs gives two samples), at any depth, e.g. Xenium Ranger's
  `<run>/outs/`. Only the files a step needs are extracted to scratch.
- **Files downloaded one by one** are grouped by the prefix in front of the 10x name:
  `Xenium_FFPE_Human_Breast_Rep1_cells.parquet`, `GSM7990532_Lung_cell_feature_matrix.h5`,
  `MB263-cells.parquet`. One prefix is one sample; several prefixes in one folder are several samples.
  Per-file gzip from GEO (`.parquet.gz`, `.h5.gz`, `.ome.tif.gz`, `experiment.xenium.gz`) is
  decompressed to scratch; an MTX triplet (`matrix.mtx.gz`, `features.tsv.gz`, `barcodes.tsv.gz`) is a
  count matrix.
- **A SpatialData object** is a zarr group with the SpatialData root attributes, or the element groups
  of an older store (`images`, `labels`, `shapes`, `points`, `tables`, or `table`), zarr v2 or v3, as a
  folder or a `.zarr.zip`. A `cells.zarr.zip` or `analysis.zarr.zip` beside bundle files belongs to the
  bundle.
- **A path to a file inside a bundle** means that bundle. A bundle's `outs/` folder and its parent
  both work.
- **Left out, with one warning listing each item and why:** other files, folders with no sample, plain
  zarr groups that aren't SpatialData, archives with no sample. Junk is ignored silently: `__MACOSX/`,
  `._*`, `.DS_Store`, Xenium Ranger pipestance internals. A run stops only when no sample is found at
  all, and the message says why each path was left out.
- **Relative paths** that don't exist are looked up under `/data`; `~` is expanded; globs work.

**Sample IDs:**
- **Bundle:** its folder or archive name, without `_outs` / `-outs`. Generic names (`outs`, `output`,
  `data`, `spatialdata`, …) are skipped up to a meaningful parent. A prefix group: the prefix
  (`GSM7990532_Lung`).
- **SpatialData:** see "Existing SpatialData" below.
- **Characters:** letters, digits, `_`, `.` and `-`; other characters become `_` (noted in the log).
- **Two samples with the same ID** stop the run, naming both sources.

### Incomplete bundles

A bundle needs two **essential** parts:
- a **count matrix**: `cell_feature_matrix.h5`, `cell_feature_matrix.zarr.zip` (Atera, Xenium Explorer
  exports), or an MTX triplet;
- **cell positions**: a cells table (`cells.parquet`, `cells.csv(.gz)`, Atera `cells.zarr.zip`) with
  cell IDs and x/y centroids that match the matrix's barcodes.

Everything else is **optional**: morphology images, transcripts, cell and nucleus boundaries, labels,
`experiment.xenium`, `cells.zarr.zip`. A bundle missing optional parts is added without them, with one
warning line naming what is missing. With no boundaries, the cells are drawn as circles at their
centroids (`<sid>_cell_circles`), which the table annotates. When a bundle holds several transcripts or
boundaries files (e.g. `reseg_transcripts.parquet` next to `transcripts.parquet`), the one whose cell
IDs match the count matrix is used, and the log says which.

Protein features (Xenium Protein) aren't RNA counts and are left out of the tables (one INFO line);
their images are kept.

## The precheck

Before anything is written, every sample is found and checked, and the log shows one report:

```
INFO    Precheck: 3 sample(s) to add, 1 already in the store (Breast_A).
  - Lung_1: Xenium bundle from /data/runs/Lung_1, 162000 cells, 377 genes
  - Lung_2: Xenium bundle from /data/runs/lung2.zip, 150000 cells, 377 genes
  - atlas_s1: SpatialData object from /data/atlas.zarr, 5001 genes
WARNING Lung_2: no morphology images, transcripts in the bundle; it is added without them.
WARNING GENE PANELS DIFFER: … after this run it holds the union of 5101 genes …
  - atlas_s1: 5001 genes = 277 also in other panels (6%) + 4724 only in this panel; 100 of the 5101 joint genes not measured
```

**The run stops**, writing nothing, with one message listing every problem, when:
- a sample misses an essential part (each sample and what it misses);
- a SpatialData object can't be split into samples, or has no table (a table of cells × genes counts
  is required);
- samples are of different species, or of another species than the store (one store holds one
  species; a single mouse bundle in a folder of human ones stops the run — add it to another store);
- two samples have the same ID;
- the CellTypist model can't be used (below), or the store was built with another model.

When some samples miss essentials and others are fine, the message also lists how to give only the
fine ones (their folders, or a glob such as `/data/GSE…/GSM9366444_*` for files downloaded one by one),
and names 0-byte files (a broken upload: download again, or ask the authors).

**The run goes on**, with one warning each, when:
- items in the inputs aren't samples (one warning listing them);
- a sample misses optional parts (one line per sample);
- gene panels differ (one block: per sample, its genes, how many it shares with the other panels and
  as % of its panel, how many it adds to the joint table, and how many joint genes it didn't measure);
- the CellTypist model shares fewer than half of its genes with a panel;
- a bundle and a SpatialData object are the only samples in one folder (e.g. a SEA-AD section's raw
  Xenium output beside its `combined_spatialdata_zarr`): they may be the same tissue; both are added,
  so give only one of them if they are.

**Same cells under another ID** (e.g. a bundle's zip and a `.zarr` made from it, given in one run, or a
sample already in the store under another name) can only be told once the sample is read: the later
one is skipped with one warning (`skipped (same cells as …)` in the summary, `same_cells_skipped` in
the run JSON), and the run goes on.

A clean run logs no WARNING line, with or without a CellTypist model: library chatter is silenced at
its source (`logs.py`). The only warning by design on a clean run is the notice that a store under
`/scratch` will be deleted.

## When a run stops with an error

Errors that come from the inputs, the settings or the store are one message in the log (no
traceback), and the run exits with code 1:

| The message says… | What to do |
|---|---|
| `The precheck found problems, so nothing was written` | read the list below it: each line names a sample and what's wrong |
| `… missing count matrix …` / `… missing cell positions …` | the bundle lacks an essential file: add it, or leave the sample out |
| `No samples given, and /data/samples doesn't exist` | fill in List of paths to samples, or mount your data asset at `/data/samples` |
| `Input not found: …` | check the path (relative paths are also looked up under `/data`) |
| `No Xenium bundle, Atera bundle or SpatialData object found in the inputs` | the message lists why each path was left out |
| `Nothing matches …` | a glob matched nothing: check it |
| `Two samples have the ID …` / `Two inputs have the sample ID …` | two inputs share a name: rename one, or give only one of them |
| `Samples of different species …` | start a separate store per species (symbols such as `F3` exist in both) |
| `… has no table` | a SpatialData object needs a table of cells × genes (counts) |
| `CellTypist model … shares N of its M genes with …'s panel` | give a model trained on this panel and species (gene symbols, not Ensembl IDs); if the store labels every sample with that model, add the sample to a separate store |
| `… isn't a CellTypist model` | check the `.pkl` (it is loaded before anything else) |
| `This store labels every sample with CellTypist model …` / `This store was built without CellTypist labels …` | give the same `.pkl` as the store's first run (or none, for a store without labels), or add the samples to a separate store |
| `… is a folder for stores, not a store` | give a `.zarr` path inside it, e.g. `/results/spatialdata.zarr` |
| `No store at …` | check the path of an existing store, or create the new one in `/results/…`, `/scratch/…` or at an `s3://` prefix |
| `… is under /data, where data assets are mounted read-only` | create the store in `/results`, `/scratch` or S3 |
| `… is a SpatialData store this capsule didn't create` | give that object as a sample into a new store |
| `… is read-only` | a data asset can't be written: give its S3 prefix instead |
| `… holds cells of several samples` / `… samples can't be told apart` | see "Existing SpatialData" below |

A run that fails partway keeps every sample finished before the failure:
- it repairs what the failing sample left and verifies the store;
- on S3 it syncs those samples back before stopping;
- the next run continues from there.

A run refused before any sample is read leaves nothing behind: a new store is created only when the
precheck has passed. A killed run is repaired by the next one (see "Crash safety").

## How a run works

1. **Store:** a local path is written in place. An `s3://` store is copied to
   `SCRATCH_DIR/store/<name>`, grown there, and synced back at the end.
2. **Repair** what an earlier, killed run left (see below).
3. **Precheck** (above). A sample is already present when its `<sample>_table` is in the store; it is
   skipped, with a warning if it was added from a different path.
4. **Per sample, one at a time:**
   1. Read it: images, transcripts and shapes stay lazy; the count matrix (Gene Expression only) is
      streamed into a compressed sparse table on scratch.
   2. Rename its elements and coordinate systems.
   3. Build its table chunk by chunk: gene symbols, QC metrics (scanpy, mito = `MT-`), CellTypist
      labels if a model is given, and a fingerprint of its cells.
   4. Refuse cells already in the store under another sample ID (same fingerprint).
   5. Move its elements and table into the store; consolidate the metadata: the sample is now present.
   6. Append its cells to the joint table (new genes extend it at the end); consolidate again.
5. **Verify:** a fresh `spatialdata.read_zarr` lists every element, every table opens, and the joint
   table holds exactly each per-sample table's cells.
6. On S3, sync back. Only changed files are uploaded, and the root `zarr.json` goes last, so a reader
   never sees the new samples before their files are there. A run that fails before the sync leaves
   S3 unchanged.
7. Scratch files of the run are deleted.

### Gene panels

- **Every table is indexed by gene symbol, with Gene Expression features only.**
  - Control probes and codewords leave the tables; the transcripts keep them.
  - Atera tables, which 10x indexes by Ensembl ID, are re-indexed by `feature_name`.
  - The Ensembl ID is kept in `var["gene_ids"]`.
- **Each sample's own table keeps its full panel.**
- **The joint table holds the union of all samples' genes.** A sample with new genes extends it (at
  the end, so earlier rows stay valid). A cell reads 0 for genes its sample didn't measure; to tell
  those zeros from measured ones, `var["n_samples_measured"]` counts the samples that measured each
  gene, and `uns["spatial_ingest"]["measured_genes"][<sample>]` lists each sample's genes. Panels
  usually match; when they don't, the precheck warns before anything is processed.
- **One species per store,** read from the Ensembl prefixes (`ENSG` human, `ENSMUSG` mouse).

### CellTypist labels

With a model, every new sample gets `<model>_label` and `<model>_prob` (the label's probability)
columns. Counts are normalized to 10,000 per cell and log1p-transformed, and each cell is labelled on
its own, so labels don't depend on the other cells. The precheck compares the model's genes with each
new sample's panel:
- **refused** when the model shares fewer than max(10, 5% of its genes) with a panel: labels from a
  handful of genes are meaningless (another species, or Ensembl IDs instead of symbols);
- **warned** when it shares fewer than half of its genes;
- the genes shared are in the summary CSV (`celltypist_genes_shared`).

A store keeps the model it was built with: use the same `.pkl` (same file name) for every run into it.

### Existing SpatialData

An input `.zarr` may hold one sample or many:

- **One annotated region:** the whole object is one sample, named after the folder (without `.zarr`;
  a generic name such as `spatialdata.zarr` takes its parent folder's name), unless its table has a
  `sample_id` column.
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
  - An element that matches no single sample, or an image with no coordinate system, is left out,
    with a warning.
- **Tables:**
  - A table that annotates nothing, such as this capsule's own joint table, is left out. The joint
    table is rebuilt from the per-sample tables.
  - Only the sample's rows of its table are read, in chunks.
  - Counts come from X, `layers["counts"]` or `.raw`. A table without counts (protein intensities,
    metabolites, normalized values) is refused, with the reason.
  - Obs columns whose names clash with the capsule's own (also when only the case differs, such as
    `cell_ID` and `cell_id`) are renamed `source_<name>`.
- **Old stores** (before SpatialData 0.1 tables, zarr v2, no root attributes) are read as well.

So a store from this capsule can be merged into another, keeping its sample IDs. The output of
`spatialdata.concatenate({"a": sdata_a, "b": sdata_b})`, saved as `both.zarr`, goes in as samples
`both_a` and `both_b`; add a `sample_id` column to its table to choose the names yourself.

### Crash safety

Commit points are metadata consolidations: an element exists for readers once it is in the root
consolidated metadata. On start, every run:

1. deletes `<store>/.spatial_ingest_staging/` if a run left it;
2. deletes element folders that aren't in the consolidated metadata (a killed write; they would
   block the next write);
3. checks the joint table (`obs`, X shape, `indptr` and var agree, its record is whole, and each
   sample has as many cells as its own table). If not, it rebuilds the joint table from the
   per-sample tables;
4. appends present samples that are missing from the joint table.

This is tested by killing runs (SIGKILL) while an element is written and while the joint table is
appended.

## Running: the details

### Code Ocean app

`code/run` is the entrypoint. It takes three positional app-panel values, in panel order. An empty
value keeps the environment variable or default.

| # | category | parameter | meaning | default |
|---|---|---|---|---|
| 1 | Central Storage | Storage Location | a local path or `s3://bucket/prefix` to a SpatialData `.zarr` | `/results/spatialdata.zarr` |
| 2 | Sample Information | List of paths to samples | bundles, archives, SpatialData `.zarr` / `.zarr.zip`, folders holding them, or globs | `/data/samples` |
| 3 | Run params | Custom CellTypist model (optional) | a CellTypist `.pkl` (any case); its labels go on every new sample | none |

- **Separators:** commas, semicolons or newlines. Spaces around each item, and quotes or brackets that
  wrap an item or the whole list (`["/data/a", "/data/b"]`), are ignored; spaces and brackets inside a
  path are kept (`/data/tiny_atera (1)`).
- **Older panels** sent a fourth value, the Input type (store, input type, samples, `.pkl`). Such
  calls, the `INPUT_TYPE` variable and `--input-type` still work: the input type is ignored, with one
  INFO line.
- If Code Ocean leaves out an empty optional value, the values after the store are placed by their
  shape: a `.pkl` is the model, anything else the samples.

### Where the store lives

- **The Storage Location** is an absolute path or an `s3://` prefix.
- **Only stores this capsule built are grown.**
  - It marks them in their root attributes.
  - Its repair deletes element folders that the consolidated metadata doesn't list. In another
    tool's store those can be real data.
  - To add another SpatialData object, give it as a sample into a store of this capsule.
- **S3 (recommended for a store you keep adding to):**
  - A missing store is created at the prefix; writing it is the permission check.
  - `/scratch` needs room for the whole store plus the largest decompressed sample files (images
    from GEO `.ome.tif.gz` can be tens of GB).
  - There is no locking: run one job per store at a time.
- **`/results`:** a one-off build. It can be saved as a data asset, but data assets are read-only,
  so later runs can't add to it; to continue it, copy it to S3 once.
- **`/scratch`:** cleared when the run ends; for trying things out (the run warns about it).
- **`/data`:** read-only data assets; a store can't be created there.
- **Starting a store:**
  - **Where:** a missing local store is created only at a path inside `CREATE_ROOTS` (`/results`,
    `/scratch`), not as either folder itself. Outside the App Panel, `CREATE=true` allows other
    folders.
  - **Refused:** an AnnData `.zarr`, an icechunk repo, another tool's SpatialData store, a file, or a
    folder holding other files.
  - **Why:** a mistyped path to an existing store can't silently start a new one.

### Memory and resources

- Count matrices are streamed in row chunks (from `.h5`, MTX or zarr) into a sparse table on
  scratch; QC, CellTypist labels, the fingerprint and the joint-table append all work chunk by
  chunk. Images, transcripts and shapes are read lazily and written element by element.
- Memory is therefore set by the largest element written at once, not by the number of cells or
  samples. Use the **64 GB ("large")** resource class for Xenium Prime 5K, whole-transcriptome and
  Atera samples; 16–32 GB suffices for small panels.
- Scratch disk: the store (for S3) plus the decompressed/extracted files of the sample in progress;
  each sample's scratch files are deleted once it is in.

### Other settings

Every setting is an environment variable named after its field in `config.Settings`.
`spatial_ingest/.env.example` lists them with their defaults:

| Variable | Default | What it sets |
|---|---|---|
| `SAMPLE_COLUMN` | `sample_id` | the table column that names samples in existing SpatialData objects |
| `CREATE` / `CREATE_ROOTS` | `false` / `/results,/scratch` | where a missing local store may be created |
| `SCRATCH_DIR` | `/scratch/spatial_ingest` | staged sample files, tables in progress, the working copy of an S3 store |
| `N_JOBS` | 4 | Xenium reader workers |
| `WRITE_CPUS` | 8 | annizarr writers for the joint table |
| `S3_WORKERS` | 16 | parallel S3 transfers |

### Locally

```bash
cd code/spatial_ingest
python main.py /scratch/atlas.zarr /data/xenium_runs ""                   # panel order: store, samples, .pkl
python main.py --store s3://bucket/atlas.zarr --samples /data/atera       # or flags
python -m pytest                                                          # unit and end-to-end tests
```

The tests build tiny synthetic bundles, archives and SpatialData objects at test time. They also run
on the tiny public 10x bundles (Xenium Breast 2fov, Ovary tiny, Mouse Ileum tiny, the Atera tiny
dataset) when those are in `SPATIAL_TEST_DATA/zip` and `SPATIAL_TEST_DATA/extract` (default
`/scratch/spatial_dev/data`), and are skipped otherwise. The download URLs are in spatialdata-io's
`prepare_test_data.yaml`. The real-S3 test runs only when `SPATIAL_S3_TEST_PREFIX` is set.

## The stored object

```
<store>.zarr
├── images/<sid>_morphology_focus, <sid>_morphology (Xenium 3D z-stack; Atera 2D), <sid>_morphology_3d (Atera) …
├── labels/<sid>_cell_labels, <sid>_nucleus_labels
├── shapes/<sid>_cell_boundaries (or <sid>_cell_circles), <sid>_nucleus_boundaries
├── points/<sid>_transcripts
└── tables/
    ├── <sid>_table   one sample: its full panel (symbols, gene_ids), its source obs, sample_id, QC
    │                 metrics, CellTypist labels, uns["spatial_ingest"] (model, source, fingerprint,
    │                 genes in panel); a bundle's annotates <sid>_cell_boundaries (or _cell_circles)
    │                 by cell_id, an existing object's keeps its own region (renamed) and instance key
    └── table         the joint single-cell table: every sample's cells, no spatial link
```

- **Coordinate systems:** each sample's are `<sid>` (from the reader's `global`) and `<sid>_<other>`.
- **Joint table** (`tables/table`):
  - cell names `<sid>:<cell_id>` (`<sid>:<region>:<cell_id>` for a sample of several regions whose
    cell IDs repeat);
  - obs: `sample_id`, `cell_id`, `n_genes_by_counts`, `total_counts`, `pct_counts_mt`, the CellTypist
    `<model>_label` / `<model>_prob`, then the union of the samples' other columns (cell area,
    transcript counts, …; missing for samples without them). It leaves out the region links and
    per-sample clusters (`leiden*`). QC columns describe each sample's own panel;
  - var: the union of genes, with `gene_ids`, `feature_types` and `n_samples_measured`;
  - X: raw counts, CSR, in the first sample's dtype (bundles float32, exact for counts; a SpatialData
    keeps its own), `indptr` int64 so the table never outgrows it;
  - `uns["spatial_ingest"]`: the panel's first sample, the CellTypist model, the species, the obs
    schema, the samples in table order, and the genes each sample measured (`measured_genes`).

Open the store with `spatialdata.read_zarr(path)`. For the joint table alone, without the images, use
`anndata.read_zarr(f"{path}/tables/table")`, or `anndata.experimental.read_lazy` for big stores.
`spatialdata.read_zarr("s3://…")` fails with spatialdata 0.8 / zarr 3 (`isinstance() arg 2 must be a
type`): copy an S3 store to local disk first (`aws s3 sync s3://bucket/prefix/store.zarr store.zarr`),
or read one table straight from S3 with `anndata.read_zarr(zarr.open_group("s3://…/tables/table"))`.
Cell IDs keep the bundle's type in a sample's table and its shapes: integers for XOA 1.0, text from
XOA 1.3 on (as spatialdata-io stores them). `spatialdata.join_spatialelement_table` matches text IDs
slowly (numpy's `isin` on text; about 30 s for 40k cells, minutes past 150k); for big samples join on
the table's `cell_id` with pandas (`table.obs.join(shapes, on="cell_id")`) instead.
`anndata_ingest`'s "Existing AnnData or SpatialData table" mode ingests the joint table into an
AnnData store (tested).

## Code map

```
code/run                  Code Ocean entrypoint: passes the app-panel values on to main.py
code/spatial_ingest/
├── main.py               the run: store, repair, precheck, one sample at a time, verify, sync, report
├── config.py             Settings from environment variables and app-panel values
├── logs.py               log format; library noise silenced at its source
├── contracts.py          the types passed between stages, and the data flow
├── discover.py           inputs -> samples: folders, archives, prefixed/GEO files, SpatialData; sample IDs; left-out items
├── stage.py              a sample's files materialised on scratch (symlink, extract, gunzip) on demand
├── inventory.py          what a bundle holds: essential and optional parts, file choice by cell IDs, raw var
├── precheck.py           every sample checked before any write; the report; panel and model checks
├── readers/              a sample read: elements (lazy) and its count matrix streamed to scratch
├── matrix.py             CSR tables on scratch: chunked writes and row-chunk reads
├── sample.py             one sample prepared: renamed elements, its table (QC, labels, fingerprint) chunk-wise
├── existing.py           existing SpatialData objects split into samples (tables, regions, elements)
├── genes.py              symbol indexing, species, panel differences
├── annotate.py           QC metrics, CellTypist labels and the model-to-panel overlap check
├── layout.py             element and coordinate-system names inside the store
├── joint.py              the joint table: gene and obs union, append with annizarr, consistency check
├── store.py              the local store: kinds, writability, presence, installs, repair, table checks
├── s3.py                 copy an S3 store down and sync it back (ETags, root zarr.json last)
├── vendor/               the Atera reader from spatialdata-io PR #426 (SOURCE.json: commit, checksums, changes)
└── tests/                unit tests per module; end-to-end runs (synthetic bundles, archives, SpatialData,
                          S3 via moto, SIGKILL repair, no-warning check); real bundles when present
```

## Environment

- **`environment/requirements.txt`** pins the packages, with `pixi.toml` beside it for use outside
  Code Ocean:
  - shared with the other capsules: annizarr, zarr, anndata, numpy, pandas, boto3;
  - this capsule's own: spatialdata 0.8.0, spatialdata-io 0.7.1, scanpy, CellTypist, geopandas,
    pyarrow, s3fs, and moto for the S3 tests.
  - Load it on the environment page with **Edit Bulk**.
- **`environment/postInstall`** installs Claude Code only.
- **Resources:** see "Memory and resources" above: the 64 GB ("large") class for Prime 5K,
  whole-transcriptome and Atera samples.
