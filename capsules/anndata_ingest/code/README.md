# ingest

A Code Ocean capsule that grows one single-cell store, sample by sample.

Each run reads new samples (HISE `.h5`, plain 10x or Xenium `.h5`, raw-count `.h5ad`, or the table
of a SpatialData `.zarr`), annotates
each one on its own (QC metrics, CellTypist labels, Scrublet doublet scores), and appends its
cells to an [AnnData](https://anndata.readthedocs.io) store. The store is either an
[Icechunk](https://icechunk.io) repository, where every run lands as one commit, or a plain
AnnData `.zarr`. Nothing is filtered and nothing is computed across the whole store. QC
thresholds, embeddings and clusters come later, from the `qc` and `process`
capsules, on curated cells. All store operations go through
[annizarr](https://pypi.org/project/annizarr/).

**Design spec:** this capsule is `capsules/anndata_ingest` in the DataScales repo, and the repo's
`capsules/spec.md` describes it and the other capsules (including `spatial_ingest`). In a
workstation session the checkout may be at `/scratch/DataScales`.

## Quick start

You need: samples as `.h5` (HISE, 10x or Xenium `cell_feature_matrix.h5`), raw-count `.h5ad`
files, or SpatialData `.zarr` folders, attached to this capsule as a data asset so they appear
under `/data`.

1. Open the **App Panel** and fill in the fields:

   | Field | What to enter | Example |
   |---|---|---|
   | **Storage Location** | where the store is, or should be created | `/results/my_store` |
   | **Origin of s3** (optional) | only for a store on a data asset linked to S3: that bucket prefix | `s3://my-bucket/my_store` |
   | **List of paths to samples** | files and/or folders, separated by commas | `/data/samples, /data/more/x.h5ad` |
   | **Celltypist Model** | which cell-type labels to add | AIFI L1, L2, L3 (default) |
| **Custom Celltypist model** (optional) | a `.pkl` used instead of the dropdown | `/data/models/xenium_panel.pkl` |

2. Click **Run**. Each 25k-cell sample takes about 1.5 minutes.
3. Look in `/results`:
   - `ingest_summary.csv`: one row per sample added (cells, genes, mito %, doublets, cell types);
   - `ingest_run.json`: what the run did and with which settings;
   - the store itself, if you created it in `/results`.
4. Explore the store with **`Icechunk_annizarr_notebook.ipynb`**: history, cell counts per sample
   and cell type, loading cells into scanpy, reading older versions.

A **Reproducible Run** (no changes to the panel) ingests every sample in `/data/samples` into a new
store at `/results/ingest_icechunk`.

### Common tasks

| I want to… | Storage Location | Origin of s3 |
|---|---|---|
| try it, or build a one-off store I'll download | `/results/<name>` (new) | empty |
| start a shared store on S3 | `s3://<bucket>/<prefix>` (new; needs write access) | empty |
| add samples to a shared store on S3 | `/data/<linked data asset>` | the S3 prefix that asset is linked to |
| test without keeping anything | `/scratch/<name>` (cleared after the run) | empty |

- **Running again is safe.** Samples already in the store are skipped, so you can rerun with the
  same folder after adding files to it.
- **Keep the same Celltypist choice** for every run into one store; a different choice is refused.
- **Spatial samples:** pick **None** for the Celltypist model, or give a custom `.pkl` trained for
  your gene panel. The bundled models are PBMC whole-transcriptome models and give meaningless
  labels on a 300–5,000-gene panel.
- **A store you'll keep adding to belongs on S3.** A store in `/scratch` or `/results` is warned
  about on every run: `/scratch` is cleared when the run ends, and a store in `/results` can be saved
  as a data asset, but data assets are read-only, so later runs can't append to it.
- **Adding samples to a processed store** (one with embeddings, clusters or layers from `process`)
  works, but those results were built for the old cells, so they're dropped with a warning and
  listed in the commit (`dropped`). Re-run the analysis afterwards.

### When a run stops with an error

Nothing is written when a run fails before the append, and on icechunk a failed append is rolled
back. The log's `ERROR` line says what to change:

| The error says | What to do |
|---|---|
| `No store at … created only inside /results,/scratch` | the path to your existing store is wrong, or start a new store under `/results` or on `s3://` |
| `… is read-only; give its writable origin` | fill in **Origin of s3** with the bucket prefix of that data asset |
| `obs schema differs from the store's` | pick the Celltypist option the store was created with (the message names it) |
| `holds no .h5/.h5ad samples` / `not a .h5/.h5ad file` / `not found` | check the sample paths; folders are not searched into subfolders |
| `var mismatch` | the sample's genes couldn't be fitted to the store's; report it, this shouldn't happen |
| `a .zarr sample must be a SpatialData object` | a `.zarr` in the sample list isn't SpatialData (no `tables/table`) |

A **gene panel that differs** from the store's doesn't stop the run. The log warns with how it
differs (for example `67 shared, 213 only in the store, 222 only in the sample`), and the sample
is fitted to the store's genes: genes it lacks are zero-filled and its other genes are dropped. The
commit records each fitted sample. QC metrics are computed before fitting, on the sample's own genes,
so for a fitted sample `total_counts` and `n_genes_by_counts` can differ from its stored counts.

## How a run works

```
new samples ──► read ──► QC metrics ──► CellTypist ──► Scrublet ──► project to the contract ──► stage as zarr
                                                                                                     │
                                  new store: the first sample seeds it ◄─────────────────────────────┤
                                                the rest are appended ◄──────────────────────────────┘
                                                          │
                                       icechunk: one commit ──► ingest_summary.csv + ingest_run.json
```

- **One sample in memory at a time.** Memory depends on the largest sample, not on the store.
- **The store is untouched until every sample is staged.** A sample that fails to read or
  annotate stops the run before anything is written, for both store kinds.
- **Writes:**
  - On icechunk, a failure while appending discards the whole run. A new store's seed is its
    own commit; rerun to finish a bootstrap that failed after it.
  - A plain zarr has no history and no rollback. A failure in the middle of an append can leave
    it inconsistent, so prefer icechunk for anything shared.
- **Reruns are safe.** Samples already in the store are skipped by sample ID, parsed from the
  file name (`B092-P1_SP02425-01.h5 → SP02425`); a run with nothing new changes nothing. A generic
  10x name (`cell_feature_matrix.h5`, `filtered_feature_bc_matrix.h5`) takes its bundle folder's
  name instead, skipping folders such as `outs`; a SpatialData `.zarr` is named after its folder.
- **Every commit names its samples and where each came from**: one `- <sample>: <path>` line per
  sample in the message, and `sources` (plus `fitted`, for fitted samples) in its metadata.

## The store contract

`contract.project` reshapes every sample into exactly what the store accepts:
- CSR `uint32` raw counts. Fractional or negative values are refused.
- An obs table with a fixed set of columns in a fixed order:
  - `sample_id`;
  - 16 HISE columns, empty for non-HISE inputs;
  - QC metrics;
  - Scrublet: NA where Scrublet failed or found no threshold;
  - `<model>_label` and `<model>_prob` for each CellTypist model, as tenx-scrna-pipeline names them.
- A gene index identical to the store's. A sample with a different gene set or order is fitted to
  it (zero-filled, extra genes dropped), because an append can't change the store's genes.
- Gene Expression features only: Xenium control probes and unassigned codewords are dropped.

Scrublet and the set of CellTypist models are part of the schema. A store is created with them,
and a later run with a different setting or model set is refused before anything is written.

**HISE layouts.** Both HISE observation layouts land in the same columns. tenx-scrna-pipeline's
names (`Batch`, `Pool`, `Chip`, `uuid`, `sample_name`, `hashtag`, `barcodes`,
`n_mitochondrial_counts`) are renamed to the older ones (`batch_id`, `pool_id`, `chip_id`,
`cell_uuid`, `pbmc_sample_id`, `hto_barcode`, `original_barcodes`, `n_mito_umis`). Labels or
Scrublet scores already in an input file are recomputed, so every sample is labelled the same way.

## CellTypist models

`code/models/` bundles the five models from
[tenx-scrna-pipeline](https://github.com/aifimmunology/tenx-scrna-pipeline/tree/main/models)
(a private repo, so they ship with the capsule). `SOURCE.json` records the commit and each file's
SHA-256, and a test checks them.
- **Default:** the AIFI PBMC models `pbmc_clean_AIFI_L1_2024` (9 types), `_L2_2024` (29) and
  `_L3_2024` (71). They take about 5 s together per 25k-cell sample.
- **Also bundled:** `Immune_All_High` and `Immune_All_Low`.
- **Choosing models:** the app panel's dropdown picks one of four sets: AIFI L1, L2, L3 (default),
  AIFI L1 only, `Immune_All_High` and `Immune_All_Low`, or None (no CellTypist). Outside the panel,
  `CELLTYPIST_MODELS` also takes names from this folder or `.pkl` paths, comma-separated.
- **Custom models:** the panel's optional **Custom Celltypist model** field (`CELLTYPIST_PKL`) takes
  `.pkl` files, which replace the dropdown. A relative path is looked up under `/data`. The file
  name names the label columns (`<name>_label`, `<name>_prob`), so a store keeps the same file.
- **How they're applied:** as in `06-label_h5s.py`. One normalized copy (counts per 1e4, log1p) is
  shared by all models, and each cell's label is CellTypist's best match, without majority voting.
  The probability is that label's. Full probability matrices aren't kept.

## Running: the details

### Code Ocean app

`code/run` is the entrypoint. It takes five positional app-panel parameters, in panel order; an
empty value keeps the environment variable or default:

| # | category | parameter | meaning | default |
|---|---|---|---|---|
| 1 | Central Storage | Storage Location | an icechunk repo (data asset, local path, `s3://bucket/prefix`) or a plain zarr folder; created when missing (see below) | `/results/ingest_icechunk` |
| 2 | Central Storage | Origin of s3 (optional) | the writable `s3://` prefix behind a read-only icechunk data asset | none |
| 3 | Sample Information | List of paths to samples | `.h5`/`.h5ad` files, SpatialData `.zarr` folders, folders of them, or globs | `/data/samples` |
| 4 | Run params | Celltypist Model | AIFI L1, L2, L3 · AIFI L1 only · Immune_All High and Low · None | AIFI L1, L2, L3 |
| 5 | Run params | Custom Celltypist model (optional) | `.pkl` file(s), comma-separated, used instead of the dropdown | none |

A **Reproducible Run** therefore ingests every sample in `/data/samples` into a new icechunk store
at `/results/ingest_icechunk`.

**Samples** are separated by commas, semicolons or newlines; quotes, brackets, spaces and trailing
slashes are ignored, so `/data/samples/, /data/x.h5ad` and `["/data/samples", "/data/x.h5ad"]` mean
the same. A folder contributes its `.h5`/`.h5ad` files and SpatialData `.zarr` folders (any case,
hidden files skipped, not recursive), and a glob such as `/data/samples/*SP024*.h5` its matches. A relative path that doesn't
exist is looked up under `/data`. Repeats are read once, and an empty folder, an unmatched glob, a
file of another type or a missing file stops the run before anything is written.

If Code Ocean leaves out an empty optional value, the values after the store are placed by their
shape: a dropdown option is the model choice, an `s3://` URL the origin, `.pkl` paths the custom
model, anything else the samples.

### Where the store lives

- **Icechunk:**
  - A shared store belongs in S3, because Reproducible Runs empty `/scratch`.
  - A data asset linked to that prefix mounts it read-only. Pass the prefix as `origin`, and
    the run writes through the origin. Which samples the store holds, its schema and its genes
    are read from the origin too, because the mount can lag it by several commits.
  - A read-only store without an origin is refused before any work starts.
- **Plain zarr:** appended where it is, so it must be writable (local or S3). A data-asset
  mount is refused.
- **Starting a store:** a storage location that holds no store yet is created at an `s3://`
  prefix (writing it is the permission check), or inside `/results` (kept with the run) or
  `/scratch` (cleared when the run ends), but not as either folder itself. Any other local path
  without a store is an error, so a mistyped path to an existing store can't silently start a new
  one. Set `CREATE=true` to create elsewhere.
  - Creation refuses a folder that holds other files. A local folder holding a plain zarr group
    is appended as a plain zarr even without the `.zarr` suffix; a `.h5`/`.h5ad` file is refused.
- **Appending drops results built for the old cells:** embeddings, graphs and layers (`obsm`,
  `obsp`, `layers`) and `uns/process`, with a warning, recorded in the commit metadata as
  `dropped`. On icechunk they stay in the store's history; on a plain zarr they're gone.

### Other settings

Every setting is an environment variable named after its field in `config.Settings`.
`ingest/.env.example` lists them with their defaults. Outputs go to `/results`:
- `ingest_summary.csv`: per appended sample, cells, mean genes and mito %, doublets, and the
  number of cell types each model found;
- `ingest_run.json`: store, snapshot ids, samples, cell and sample totals, peak memory and
  settings.

### Locally

```bash
cd code/ingest
python main.py /scratch/atlas "" /data/samples "Only AIFI L1 celltypes" ""     # panel order: icechunk
python main.py --samples /data/samples --store /scratch/atlas.zarr               # or flags: plain zarr
python -m pytest tests                                                         # ~1.5 min, synthetic data
```

## The stored object

```
<store>          icechunk repo (the root group is the AnnData) or a plain .zarr
├── X            raw UMI counts, CSR uint32, CHUNK_CELLS cells per chunk
├── obs          sample_id · HISE columns · QC · Scrublet · <model>_label/_prob per CellTypist model
└── var          gene index · mt
```

Open it with `annizarr.Repo(path).open_zarr("r")` (icechunk) or `zarr.open_group(path)`, and
then `anndata.read_zarr` on the group. On icechunk, `Repo.log()` lists the runs, with each
commit's metadata naming the samples it appended and the totals after it.

## Code map

```
code/run              Code Ocean entrypoint: passes the app-panel values on to main.py
code/ingest/
├── main.py           the run: pick new samples, stage them, write them, report
├── config.py         Settings from environment variables and app-panel values; sample-list parsing
├── hise.py           sample sources: HISE .h5 (both layouts), SpatialData tables; gene names; sample IDs
├── annotate.py       QC metrics, CellTypist (several models), Scrublet
├── contract.py       the obs schema, the projection, the schema and gene checks, gene-panel fitting
├── store.py          icechunk and plain-zarr stores behind one interface
└── tests/            unit tests and end-to-end runs on synthetic samples (incl. SpatialData), both store kinds
code/models/          the bundled CellTypist models and their SOURCE.json
code/Icechunk_annizarr_notebook.ipynb   explore a store: history, cells per sample, loading cells, old versions
```

## Environment

- **`environment/`** (`environment.yml`, `requirements.txt`) pins the packages: annizarr, icechunk,
  zarr and anndata, shared with the other capsules, plus scanpy, CellTypist and their compiled
  dependencies.
- **`environment/postInstall`** installs `g++` and then annoy and Scrublet, because annoy has no
  wheel and the packages are installed before `postInstall` runs.
