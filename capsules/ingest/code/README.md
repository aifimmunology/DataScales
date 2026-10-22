# ingest

A Code Ocean capsule that grows one single-cell store, sample by sample.

Each run reads new samples (HISE `.h5`, plain 10x `.h5`, or raw-count `.h5ad`), annotates
each one on its own (QC metrics, CellTypist labels, Scrublet doublet scores), and appends its
cells to an [AnnData](https://anndata.readthedocs.io) store. The store is either an
[Icechunk](https://icechunk.io) repository, where every run lands as one commit, or a plain
AnnData `.zarr`. Nothing is filtered and nothing is computed across the whole store. QC
thresholds, embeddings and clusters come later, from the `qc` and `process`
capsules, on curated cells. All store operations go through
[annizarr](https://pypi.org/project/annizarr/).

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
  file name (`B092-P1_SP02425-01.h5 → SP02425`); a run with nothing new changes nothing.

## The store contract

`contract.project` reshapes every sample into exactly what the store accepts:
- CSR `uint32` raw counts. Fractional or negative values are refused.
- An obs table with a fixed set of columns in a fixed order:
  - `sample_id`;
  - 16 HISE columns, empty for non-HISE inputs;
  - QC metrics;
  - Scrublet: NA where Scrublet failed or found no threshold;
  - `<model>_label` and `<model>_prob` for each CellTypist model, as tenx-scrna-pipeline names them.
- A gene index identical to the store's. A different gene set or order is refused, because an
  append can't reconcile it.

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
- **Choosing models:** pass names from this folder or `.pkl` paths, comma-separated.
- **How they're applied:** as in `06-label_h5s.py`. One normalized copy (counts per 1e4, log1p) is
  shared by all models, and each cell's label is CellTypist's best match, without majority voting.
  The probability is that label's. Full probability matrices aren't kept.

## Running

### Code Ocean app

`code/run` is the entrypoint. It takes five positional app-panel parameters; an empty value
keeps the environment variable or default:

| # | parameter | meaning |
|---|---|---|
| 1 | samples | comma-separated `.h5`/`.h5ad` paths, or one directory of them |
| 2 | store | an icechunk repo (data asset, local path, `s3://bucket/prefix`), or a path ending in `.zarr` |
| 3 | origin | the writable `s3://` prefix behind a read-only icechunk data asset |
| 4 | celltypist models | model names from `code/models` or `.pkl` paths, comma-separated (default: AIFI L1, L2, L3) |
| 5 | create | `true` to start a new store at the store path |

### Where the store lives

- **Icechunk:**
  - A shared store belongs in S3, because Reproducible Runs empty `/scratch`.
  - A data asset linked to that prefix mounts it read-only. Pass the prefix as `origin`, and
    the run reads through the mount and writes through the origin.
  - A read-only store without an origin is refused before any work starts.
- **Plain zarr:** appended where it is, so it must be writable (local or S3). A data-asset
  mount is refused.
- **Starting a store** is always explicit: set `create`.
  - It refuses a path that already holds a store, and a folder that holds other files.
  - Without `create`, a missing store is an error.

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
python main.py --samples /data/samples --store /scratch/atlas --create true   # icechunk
python main.py --samples /data/samples --store /scratch/atlas.zarr --create true
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
code/run              Code Ocean entrypoint: app-panel parameters → main.py flags
code/ingest/
├── main.py           the run: pick new samples, stage them, write them, report
├── config.py         Settings from environment variables and app-panel flags
├── hise.py           HISE .h5 (both layouts) as an annizarr source; sample-ID parsing
├── annotate.py       QC metrics, CellTypist (several models), Scrublet
├── contract.py       the obs schema, the projection, the schema and gene checks
├── store.py          icechunk and plain-zarr stores behind one interface
└── tests/            unit tests and end-to-end runs on synthetic samples, for both store kinds
code/models/          the bundled CellTypist models and their SOURCE.json
```

## Environment

Both files are edited by hand; the shared pins (annizarr, icechunk, zarr, anndata, numpy, …) match
the other capsules' (spec §6.2).
- **`.codeocean/environment.json`** is the environment page. It pins the shared packages plus
  scanpy, CellTypist and their compiled dependencies.
- **`environment/postInstall`** installs `g++` and then annoy and Scrublet, because annoy has no
  wheel and the page installs packages before `postInstall` runs. It ends with the Claude Code
  install shared by every capsule.
