# spatial ingest

Adds Xenium bundles, Atera bundles and existing SpatialData objects to one
[SpatialData](https://spatialdata.scverse.org) `.zarr` (the **store**), run by run. Each sample keeps its
images, shapes, transcripts and table; all cells also go into one joint single-cell table for scRNA tools.

## Quick start

1. Attach your data to the capsule as a data asset: bundle folders, the `.zip` / `.tar.gz` files you
   downloaded, or SpatialData `.zarr` folders. A data asset appears under `/data/<mount name>`.
2. Open the **App Panel** and fill in:

   | Field | What to enter | Example |
   |---|---|---|
   | **Storage Location** | the store: a new or existing `.zarr` path, or an `s3://` prefix | `/results/spatialdata.zarr` (one-off) or `s3://my-bucket/spatial/lung_atlas.zarr` (keep adding to it) |
   | **List of paths to samples** | bundles, archives, `.zarr` folders, folders holding any of them, or globs; separated by commas, semicolons or new lines | `/data/xenium_runs, /data/extra/run7_outs.zip` |
   | **Custom CellTypist model** (optional) | a CellTypist `.pkl` trained on your gene panel (same species, gene symbols) | `/data/models/lung_panel.pkl` |

3. Click **Run**. Every sample is checked first; if one can't be added, the run stops before writing
   anything and says why. Then open the store (see "How to open the result").

**S3 needs an AWS secret with write access to the bucket.** In Code Ocean, attach one to the capsule
(capsule › Environment › Secrets / cloud credentials) before the run; without it the run stops at once
with "access denied" and nothing is written.

**Reproducible Run:** it adds every sample in a data asset mounted at `/data/samples` to a new store at
`/results/spatialdata.zarr`. With nothing at `/data/samples` it does nothing: it logs how to use the
capsule and succeeds. A path you type that doesn't exist is an error.

| I want to… | Storage Location | List of paths to samples |
|---|---|---|
| try it, or build a store once and download it | `/results/<name>.zarr` | my samples |
| start a store I'll keep adding to | `s3://<bucket>/<prefix>/<name>.zarr` | my samples |
| add more samples later (Xenium, Atera, SpatialData, in any mix) | the same S3 location | the new samples |
| merge several SpatialData objects into one | a new location | the `.zarr` folders |
| continue a store I built in `/results` | copy it to S3 once, then use the S3 prefix | the new samples |

`/results` is a one-off build: saved as a data asset it is read-only, so later runs can't add to it.
`/scratch` is cleared when the run ends (for quick tests only). `/data` is read-only. One run per store at
a time.

## What lands in /results

- **the store**, when the Storage Location is in `/results`;
- **`spatial_ingest_summary.csv`**, one row per sample:

  | Column | Meaning |
  |---|---|
  | `sample_id`, `kind`, `source` | the sample, its kind (`xenium`, `atera`, `spatialdata`) and the path it was added from |
  | `status` | `appended`, `skipped (present)`, `skipped (no cells)`, `skipped (same cells as …)` or `failed` |
  | `n_cells`, `genes_in_panel` | cells, and genes in the sample's own panel |
  | `genes_shared`, `genes_added` | genes it shares with the store or the other samples of this run (0 for the only sample of a new store), and genes it adds to the joint table |
  | `celltypist_genes_shared`, `n_celltypist_cell_types` | the model's genes in this panel, and the distinct labels given |
  | `mean_genes`, `mean_total_counts` | per cell, 1 decimal |
  | `n_cells_zero_counts`, `n_cells_without_position` | cells with no counts; cells the cells table gives no position (`has_position` False) |
  | `seconds`, `note`, `error` | time taken; for a skipped present sample, the path given this run; why a sample failed |

- **`spatial_ingest_run.json`**, also written when the run fails: `status` (`succeeded` / `failed`) and
  `error`, `store`, repairs, appended / skipped / failed samples, left-out inputs, the joint table's gene
  count and `gene_notes` per sample, the store's totals and size, the S3 sync counts, package versions,
  peak memory of the main process, and the settings.

The log (the run's output) has the precheck report and one line per problem.

## How to open the result

```python
import spatialdata as sd

sdata = sd.read_zarr("/results/spatialdata.zarr")       # an S3 store: aws s3 sync it to local disk first
joint = sdata.tables["table"]                           # every sample's cells, raw counts, union of genes
sid = "Xenium_V1_human_Breast_2fov"
one = sdata.tables[f"{sid}_table"]                      # one sample: its panel, obsm["spatial"], region link

# clusters or labels computed on the joint table, back onto one sample's cells and coordinates
mine = joint.obs[joint.obs["sample_id"] == sid].set_index("cell_id")
one.obs["leiden"] = mine["leiden"].reindex(one.obs["cell_id"].astype(str)).to_numpy()
placed = one[one.obs["has_position"].to_numpy()]        # cells with a position, before spatial analysis

crop = sdata.query.bounding_box(axes=("x", "y"), min_coordinate=[0, 0], max_coordinate=[2000, 2000],
                                target_coordinate_system=sid, filter_table=False)
section = sdata.filter_by_coordinate_system(sid)        # one sample's elements and table
```

- **Tables.** `table` is the joint single-cell table: cell names `<sid>:<cell_id>`, obs `sample_id`,
  `cell_id`, `n_genes_by_counts`, `total_counts`, `pct_counts_mt`, the CellTypist `<model>_label` /
  `<model>_prob`, then the samples' other columns (`has_position`, cell area, transcript counts, …). It
  has no coordinates and no link to shapes: map results back to a sample's table by `cell_id`, as above.
  `<sid>_table` is one sample's table, indexed by its `cell_id`, annotating `<sid>_cell_boundaries` (or
  `_cell_circles`).
- **Spatial queries** (`bounding_box`, `polygon_query`) on the whole store need `filter_table=False`,
  because the joint table annotates no element; or work per sample with
  `sdata.filter_by_coordinate_system(sid)`.
- **Genes.** Every table is indexed by gene symbol, Gene Expression only, with the Ensembl ID in
  `var["gene_ids"]`. The joint table holds the union of all panels: a cell reads 0 for genes its sample
  didn't measure. `var["n_samples_measured"]` and `uns["spatial_ingest"]["measured_genes"][sid]` tell
  measured zeros from filled ones; compare samples on genes measured in all of them. Genes are matched
  across samples by Ensembl ID (version suffixes such as `.7` ignored); a symbol whose ID differs between
  samples stays two genes (`SYM` and `SYM-<gene_id>`), with an INFO line in the precheck.
- **QC.** `pct_counts_mt` is NaN when a panel has no mitochondrial genes (`MT-`, mouse `mt-`), which is
  usual for spatial panels.
- **Elements** are named `<sid>_<element>`: `images/<sid>_morphology_focus` (Atera: `<sid>_morphology`),
  `labels/<sid>_cell_labels`, `shapes/<sid>_cell_boundaries`, `<sid>_nucleus_boundaries`,
  `points/<sid>_transcripts`.
- **The joint table alone**, without images: `anndata.read_zarr(f"{store}/tables/table")`, or
  `anndata.experimental.read_lazy` for a big store. `spatialdata.read_zarr("s3://…")` doesn't work with
  the pinned spatialdata; copy the store to local disk first.
- `read_zarr` prints `no parent found for <ome_zarr.reader.Label …>` for each labels element; that comes
  from ome-zarr and is harmless.

### Units and coordinate frames

- `obsm["spatial"]`, `cell_area` / `nucleus_area` and the transcripts' `x`, `y`, `z` are in **µm**.
- Each sample has its own coordinate system, **named after the sample** (`sid`). It is in **pixels of
  `morphology_focus` at full resolution**: the image maps to it unchanged, and shapes and transcripts
  (stored in µm) carry `Scale(1 / pixel_size)`, e.g. 4.706 for Xenium's 0.2125 µm pixels. To overlay
  `obsm["spatial"]` on the image, divide it by the pixel size. A SpatialData input keeps its own
  transformations, with its systems renamed for the sample.
- There is **no `global` system**: pass the sample's system, e.g.
  `sd.get_extent(el, coordinate_system=sid)`, `sd.aggregate(..., target_coordinate_system=sid)`.

## Machine size and run time

Memory depends on the largest sample, not on the store. The capsule's default class is `large`. Change it
in the capsule's resource settings (the machine selector before a run; saved in `.codeocean/resources.json`).

| Your samples | Class | Observed |
|---|---|---|
| demo bundles (2-FOV, "tiny"), small Xenium v1 sections | small | peaks ≤ 1.3 GB |
| Xenium v1 whole slides | medium | |
| Xenium Prime 5K, Atera, whole-transcriptome panels | large (64 GB) | 8–14 GB per sample |

- **3D z-stacks are not stored.** Xenium's `morphology.ome.tif` and Atera's `morphology_3d/` are skipped
  when the bundle has a 2D image; a z-stack that is the only image is stored as its max projection
  (`<sid>_morphology_mip`).
- A 2-FOV Xenium bundle (7k cells) takes about 160 s;
  whole slides take tens of minutes each. A rerun with every sample already present takes seconds
  (about 7 s), plus the download of an S3 store.
- **Disk (`/scratch`):** a zipped sample is extracted while it is read: plan for about 2.3× the archive's
  size at the peak. An S3 store is downloaded whole on every run, so add the store's size.

## What counts as a sample

Every path in the list is searched (folders and archives, 8 levels deep). Each thing found is a Xenium
bundle, an Atera bundle, a SpatialData object, or left out (one warning listing every item and why).

- **A bundle** is a folder or archive folder with `experiment.xenium` (Xenium) or `experiment.spatial`
  (Atera), or without one, a count matrix and cell positions. Archives (`.zip`, `.tar`, `.tar.gz`,
  `.tgz`) are read without unpacking them yourself; a zip holding two runs gives two samples.
- **Files downloaded one by one** (10x per-file downloads, GEO supplementary files) are grouped by the
  prefix in front of the 10x name (`GSM7990532_Lung_cells.parquet.gz`): one prefix, one sample.
- **A SpatialData object** (`.zarr` folder or `.zarr.zip`, any format version) is split into its samples
  (below).
- **Relative paths** are also looked up under `/data`; `~` and globs work. A path to a file inside a
  bundle means that bundle.

**Sample IDs** come from the bundle's folder or archive name without `_outs` (generic names such as
`outs` take the parent's name), a prefix group's prefix, or a SpatialData's `sample_id` column. Other
characters than letters, digits, `_`, `.` and `-` become `_`. A sample whose ID is already in the store is
skipped (with a warning when it came from another path: rename it if it is a different sample).

**Essential parts of a bundle:** a count matrix (`cell_feature_matrix.h5`, `cell_feature_matrix.zarr.zip`
or an MTX triplet) and cell positions (`cells.parquet`, `cells.csv(.gz)` or `cells.zarr.zip`, with IDs
matching the matrix). Everything else (images, transcripts, boundaries) is optional; a bundle missing
some is added without them, with one warning. Cells without a position are kept with
`has_position = False` and NaN coordinates. Protein features are left out of the tables (their images are
kept).

**Existing SpatialData objects:**
- one annotated region is one sample, named after the `.zarr` (or its parent, for a generic name);
- with a `sample_id` column in the table (`SAMPLE_COLUMN` for another name), regions are grouped by it;
- without it, regions in separate coordinate systems are separate samples, named
  `<object>_<tag>` (`cell_labels-a` in `pair.zarr` → `pair_a`); samples sharing one system need the column;
- counts come from X, `layers["counts"]` or `.raw`; a table without counts is refused;
- a sample ID already taken by another object becomes `<object>_<sid>`; identical cells already in the
  store are skipped (`skipped (same cells as …)`);
- a store from this capsule can be merged into another (its joint table is rebuilt, not copied).

## Checks and warnings

Before anything is written, the **precheck** finds and checks every sample and logs one report. **The
run stops**, writing nothing, with one message listing every problem, when a bundle misses an essential
part, a SpatialData object can't be split or has no counts table, samples are of different species (one
store holds one species), two samples have the same ID, or the CellTypist model can't be used. When some
samples are fine, the message says how to give only those.

**The run goes on, with one warning each,** when items in the inputs aren't samples, a bundle misses
optional parts, the CellTypist model shares fewer than half of its genes with a panel, a bundle and a
SpatialData object may be the same tissue, or **gene panels differ**. That last one is expected when
you mix panels: the joint table holds the union of genes, and nothing needs fixing.

A clean run logs no WARNING line, except the gene-panel notice when panels differ and the notice that a
store under `/scratch` will be deleted.

**CellTypist:** with a model, every new sample gets `<model>_label` and `<model>_prob`, each cell labelled
on its own (normalized to 10,000 counts, log1p). The model is refused when it shares fewer than max(10, 5%
of its genes) with a panel. A store keeps the model it was built with: give the same `.pkl` (same file
name) on every run into it.

## When a run stops with an error

Errors from the inputs, the settings or the store are one message in the log, and the run exits with
code 1.

| The message says… | What to do |
|---|---|
| `The precheck found problems, so nothing was written` | read the list below it: each line names a sample and what's wrong |
| `Input not found: …` / `N inputs not found` | check each path; the message suggests close names from the folder |
| `No Xenium bundle, Atera bundle or SpatialData object found in the inputs` | the message says why each path was left out |
| `… missing count matrix …` / `… missing cell positions …` | the bundle lacks an essential file: add it, or leave the sample out |
| `Two samples have the ID …` / `Two inputs have the sample ID …` | rename one, or give only one of them |
| `Samples of different species …` | one store per species: give the odd sample to another store |
| `… has no table` | a SpatialData object needs a table of cells × genes (counts) |
| `CellTypist model … shares N of its M genes with …'s panel` | give a model trained on this panel and species (gene symbols) |
| `… isn't a CellTypist model` | check the `.pkl` |
| `This store labels every sample with CellTypist model …` / `… built without CellTypist labels …` | give the same `.pkl` as the store's first run (or none), or use a separate store |
| `Can't … s3://…: access denied …` / `… no AWS credentials` | attach an AWS secret with write access to the bucket, or use `/results` |
| `… is a whole bucket; give a prefix inside it …` | an S3 store needs a prefix: `s3://bucket/<prefix>/<name>.zarr`, not `s3://bucket` |
| `Another run is adding to …` | wait for it; a lock left by a killed run (e.g. out of memory) is taken over 10 min after its last heartbeat |
| `… is a folder for stores, not a store` | give a `.zarr` path inside it, e.g. `/results/spatialdata.zarr` |
| `No store at …` | check the path of an existing store, or create one in `/results/…`, `/scratch/…` or at an `s3://` prefix |
| `… is under /data, where data assets are mounted read-only` | create the store in `/results`, `/scratch` or S3 |
| `… is a SpatialData store this capsule didn't create` | give that object as a sample into a new store |
| `region … holds cells of several samples` | the SpatialData object mixes samples in one region: write one region per sample |

**A sample that fails while it is added** is taken back out of the store with one ERROR line; the other
samples go in, and the run ends with exit code 1 and the list of failed samples (`failed` in the summary,
`failed_samples` in the run JSON). Fix them and run again: samples already in are skipped.

**A run killed without an error** (usually out of memory) leaves the store as it was before the sample it
was adding: the next run repairs it, and an S3 store's lock is taken over 10 min after the killed run's last
heartbeat. Run again on the large machine. The precheck warns when a sample has over 100k cells and the
machine has under 24 GB.

## Settings

The App Panel sets `STORE`, `SAMPLES` and `CELLTYPIST_PKL`. Every setting is an environment variable named
after its field in `config.Settings` (`spatial_ingest/.env.example`):

| Variable | Default | What it sets |
|---|---|---|
| `SAMPLE_COLUMN` | `sample_id` | the table column naming samples in SpatialData inputs |
| `CREATE` / `CREATE_ROOTS` | `false` / `/results,/scratch` | where a missing local store may be created |
| `SCRATCH_DIR` | `/scratch` | the run works in `<SCRATCH_DIR>/spatial_ingest/` (staged files, tables in progress, an S3 store's working copy) and clears only that folder |
| `RESULTS_DIR` | `/results` | where the summary CSV and run JSON go |
| `WRITE_CPUS` | 8 | annizarr writers for the joint table |
| `S3_WORKERS` | 16 | parallel S3 transfers |
| `S3_LOCK_STALE_MINUTES` | 10 | after how long an S3 lock with no heartbeat is taken over |

## For developers

Design and decisions: `capsules/spatial_ingest/spec.md`; rules shared by all capsules: `capsules/CLAUDE.md`.

**A run:** open the store (an S3 store is copied to `<SCRATCH_DIR>/spatial_ingest/store/`); repair what a
killed run left; precheck; create a new store only now; then one sample at a time: read (images,
transcripts and shapes lazy; counts streamed to a CSR table on scratch), rename its elements, build its
table (QC, CellTypist, fingerprint) chunk by chunk, check the joint rows, move its elements and table in
(consolidate: the sample is present), append its rows to the joint table with annizarr (consolidate
again; rolled back if it fails); verify (only what this run wrote, plus every table's metadata); sync to
S3; write the outputs.

**Crash safety:** commit points are consolidations of the root metadata. Every run starts by deleting
`<store>/.spatial_ingest_staging/` and element folders the consolidated metadata doesn't list, rebuilding
the joint table from the per-sample tables when they disagree, and appending present samples missing from
it. A present sample whose rows can't go back into the joint table (a data error) is taken out of the
store with one warning; an I/O error during the repair stops the run with one line and removes nothing.
Tested by SIGKILL at several points.

**Locks:** one run per store. A local store has a lock file beside it (`.<name>.zarr.spatial_ingest_lock`),
taken over when its process is gone on the same host, or after 12 h from another host. An S3 store has a
lock object `<prefix>/.spatial_ingest_lock` with a heartbeat, taken over after `S3_LOCK_STALE_MINUTES`;
before uploading, the run also checks that the root `zarr.json` is still the one it downloaded.

**S3 sync order:** new files, changed data, other metadata (deepest first), the root `zarr.json`, then
deletions; readers see new samples only when the root changes. Details in `store.py` (`sync`).

**Running locally and tests:**

```bash
cd code/spatial_ingest
python main.py /scratch/atlas.zarr /data/xenium_runs ""                   # panel order: store, samples, .pkl
python main.py --store s3://bucket/atlas.zarr --samples /data/atera       # or flags
python -m pytest                                                          # unit and end-to-end tests
```

About 40 unit tests, under a minute. They build small synthetic bundles, archives, SpatialData objects
and per-sample tables; the end-to-end tests run the tiny public 10x bundles (Xenium Ovary tiny, Atera tiny)
from `SPATIAL_TEST_DATA/zip` and `/extract` when present, and skip otherwise. S3 is tested under moto; the
real-S3 test runs only when `SPATIAL_S3_TEST_PREFIX` is set.

**Code map:**

```
code/run                  Code Ocean entrypoint: the app-panel values, positionally, to main.py
code/spatial_ingest/
├── main.py               the run: store, repair, precheck, one sample at a time, verify, sync, outputs
├── config.py             settings (env and app panel), logging, the records passed between stages, UserError
├── inputs.py             inputs -> samples (folders, archives, per-file downloads, SpatialData), staging,
│                         what a bundle holds; count matrices and cells tables streamed to CSR, gene policy
├── prepare.py            the precheck; reading a sample (SpatialData objects split into samples, bundle
│                         elements built lazily); its table with QC, CellTypist labels and fingerprint
├── store.py              the local store (installs, repair, locks, element names), the joint table, and the
│                         S3 copy down, sync back and lock
├── vendor/               the Atera reader of spatialdata-io PR #426 (SOURCE.json: commit, checksums)
└── tests/                conftest.py (synthetic inputs), test_inputs.py (discovery, inventory),
                          test_prepare.py (precheck, reading, per-sample table), test_store.py (joint
                          table, repair, locks, S3 sync), test_run.py (end to end, settings, the
                          private-API canary)
```

**Environment:** `environment/requirements.txt` pins the packages (with `pixi.toml` beside it): spatialdata
0.8.0, spatialdata-io 0.7.1, annizarr, zarr, anndata, scanpy, CellTypist, geopandas, pyarrow, boto3, moto
for the S3 tests. Private APIs of these pins that the code uses are listed in `tests/test_run.py`, which
fails when a pin bump moves one.
