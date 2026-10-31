# qc

A Code Ocean capsule that reviews the quality of a single-cell store and, once a person has
reviewed the result, writes the passing cells as the curated object.

It reads the store the `ingest` capsule builds (an [Icechunk](https://icechunk.io) repository,
read at `main`), or any plain AnnData `.zarr` with the same columns. It also reads
[SpatialData](https://spatialdata.scverse.org) stores saved as a plain `.zarr`, and reviews their
cell table (see [Spatial data](#spatial-data)). Supported inputs: AnnData icechunk repos, AnnData
`.zarr`, and SpatialData `.zarr`. A report run never writes
to the input. Only an `apply` run writes cells, and only with a decisions file that a report
run produced and nobody has changed since.

## The review loop

```
            ┌──────────────── edit qc_decisions.json, attach it, rerun ◄───────────────┐
            ▼                                                                          │
input ──► report run ──► report + qc_decisions.json ──► person reviews ───────────────┘
                                                              │ the report is right
                                                              ▼
               apply run with that run's qc_decisions.json ──► curated cells
                                                              (icechunk: the qc branch; plain zarr: a new zarr)
```

1. Run with mode `report` (the default). Read `report_summary.txt`, the plots, and `qc_decisions.json`.
2. To change anything, edit `qc_decisions.json`, attach it as a data asset, and run with mode `report`
   again. That run previews the edit and writes a fresh `qc_decisions.json`.
3. When the report is right, run with mode `apply` and that run's `qc_decisions.json`.

If an ingest lands between steps 1 and 3, the input's fingerprint changes and `apply` refuses.
Run step 1 again: samples already reviewed keep their entries.

## What a run computes

Every run, report or apply, does the same work. What it checks comes from the store's **QC
profile** (`profiles.py`): `scrna` for AnnData stores, as described here, and `spatial` for
SpatialData tables (see [Spatial data](#spatial-data)). A profile names the metrics, the cell rules
and their defaults. Everything else is shared.

- **Cell metrics.** It reads only the obs columns it needs, one at a time: `sample_id`, the three
  QC metrics, `predicted_doublets`, the group columns, the label column and its `_prob`,
  `hto_category` and `adt_qc_flag`. If `n_genes_by_counts`, `total_counts` or `pct_counts_mt` is
  absent, or NA for any cell, the run stops, unless `compute missing metrics` is set. Then they're computed
  from X in row bands, as `scanpy.pp.calculate_qc_metrics` defines them. Reading X costs about
  16 KB per cell, and X must hold integer counts.
- **Per-sample table** (`sample_qc.csv`): cell count; median and MAD of genes, UMIs and mito %;
  doublet rate and doublet-NA count; per-rule and overall fail %; cell-type composition;
  `hto_category` / `adt_qc_flag` counts; the ambient columns; status; flags.
- **Flags.** They're advisory and never change a decision:

  | Flag | When |
  |---|---|
  | `few_cells` | fewer than 500 cells |
  | `doublet_rate` | a doublet rate of zero, or above 15% |
  | `doublet_unknown` | any cell whose doublet call is NA |
  | `outlier:<genes/umis/mito>` | the sample median is more than 3 MADs from the cohort median (needs ≥ 5 samples; the MAD's floor is 5% of the median) |
  | `ambient:<set>` | a marker set at ≥ 1.5× the cohort median (needs ≥ 5 samples) |

- **Downsampled embedding.** Up to 1,000 cells per sample (200k in total), seed 4242, failed
  cells included. A sample's cells are stored together, so they're taken from a random run of
  consecutive storage chunks, just long enough for the sample's quota. That's one or two chunks
  at ingest's chunk size, instead of reads scattered across the store. Then `normalize_total`,
  `log1p`, 2,000 HVGs (`seurat`), `scale(max_value=10)`, PCA (50), neighbors (15, 30 PCs), UMAP
  and leiden (1.0). There's no batch correction, on purpose: the embedding is there to show batch
  effects. It's for the report only and is never written to the store.
- **Ambient RNA indicator** (detection only, nothing is corrected). In the sampled cells,
  leaving out predicted doublets, it measures each marker set where it shouldn't be expressed:

  | Set | Genes | Measured in |
  |---|---|---|
  | `monocyte` | LYZ, S100A8, S100A9 | T cell, B cell, NK cell |
  | `hemoglobin` | HBB, HBA1, HBA2 | every cell not labelled Erythrocyte |
  | `immunoglobulin` | IGKC, IGHM, IGLC2 | T cell, NK cell, Monocyte |

  Per sample, it reports the median and 90th percentile of the set's share of each cell's UMIs,
  and the median's ratio to the cohort's. It's skipped, with a warning, when the label column is
  absent or a set has none of its genes.
- **Per-cell-type QC** (optional, `per-cell-type QC`): counts, medians, MADs, the label
  probability and fail % per label (`celltype_qc.csv`), and the `cell_types` overrides. If the
  label column is absent, the section is skipped with a warning and the run still succeeds.

## The decisions file (`qc_decisions.json`)

The file a reviewer edits, and the one `apply` writes from:

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
  "previewed_sha256": "…"
}
```

- **Precedence:** `defaults` → `cell_types[label]` → `samples[id]`, key by key; the most specific
  entry wins. `exclude: true` removes all of a sample's cells, and `{}` means reviewed with the
  defaults. A sample entry may set `min_genes`, `max_genes`, `max_mito`, `filter_doublets`,
  `exclude` and `note`. An unknown key (such as a typo) is refused rather than ignored.
- **`cell_types`** applies only when per-cell-type QC is on and the label column exists. The
  run records which column it used in `label_column`.
- **New samples** (in the input but without an entry) are accepted as `{}` if they have no
  flags. Flagged ones go to `pending`. Give a pending sample an entry in `samples` to clear it.
- **`previewed_sha256`:** set by every report run, over the canonical JSON of the file without
  this key. A hand edit changes the content but not the hash, which is how `apply` knows the
  edit hasn't been previewed. Key order and whitespace don't matter.
- **The cell rules,** each evaluated on its own:

  | Rule | A cell fails when |
  |---|---|
  | `sample_excluded` | its sample has `exclude: true` |
  | `doublet` | `filter_doublets` and `predicted_doublets == 1` (NA passes) |
  | `high_mito` | `pct_counts_mt >= max_mito` |
  | `low_genes` | `n_genes_by_counts <= min_genes` |
  | `high_genes` | `n_genes_by_counts >= max_genes` |

  A cell passes when no rule fails. `qc_fail_reason` is the first failing rule in this order;
  the per-rule counts count every failing rule. A rule whose column the store lacks is off,
  with a warning.
- **`profile`** records which profile the file was written for. A file for one profile is
  refused for a store of the other.
- **Where a report run's decisions come from:** the defaults, then (on icechunk) the curation
  branch's own `uns/qc/decisions`, then the given file, key by key. An apply run uses exactly
  the given file and merges nothing.

## Apply

An apply run refuses to start, before any report work, when:

| Condition | Message |
|---|---|
| no decisions file was given | *apply needs the qc_decisions.json of a report run; run in report mode first.* |
| `previewed_sha256` is missing or doesn't match the content | *Decisions edited after their report; rerun in report mode to preview them.* |
| the input's fingerprint isn't `input.fingerprint` | *Input changed since this report (e.g. new samples were ingested); rerun in report mode.* |
| `pending` isn't empty | *Pending samples need a decision: …* |
| icechunk is read-only with no `origin`; the branch is `main`; the plain-zarr output is the input, already holds data, or isn't writable | the location, and what to pass |

A refused run writes nothing to `/results`. Per-cell-type QC and the label column come from the
file's `label_column`, not the panel, so the mask can't drift from the one previewed.

The mask is a deterministic function of the decisions and the input, both checked, so the apply
run recomputes it and writes exactly the cells its report shows as passing. Its `cell_qc.parquet`
equals the report run's.

**What's written,** by annizarr (0.2.4) with its own defaults:
- X and any CSR layers stream from the source's passing rows (`rows.py`, an annizarr reader)
  into annizarr's matrix writer, in bands of about 0.5 GB, one per usable core (annizarr counts
  the container's CPU limit, not the host's cores), at Blosc-zstd level 3. Memory holds a few
  bands, not the store.
- A layer annizarr's `add_layer` made (it records its source matrix, e.g. a CSC copy of X) is
  re-derived from the curated matrix in its own layout. Other layers keep their passing rows.
- obs and obsm are filtered to the passing rows in memory, and unused categories are dropped.
  var, varm, varp and uns are copied unchanged. obsp and `raw` are dropped.
- `uns/qc` is added: `decisions` (a JSON string), `cell_qc_sha256`, `source` (the fingerprint),
  `source_kind`, `source_location`, `created`, `samples` (those with passing cells), per-sample
  before/after `counts`, and `reviewer` if given.

**Where it goes:**
- **icechunk:** the curation branch (default `qc`; another name gives an independent curation).
  The cells are written from scratch in an icechunk session on the fingerprinted `main`
  snapshot, then saved with `flush`, which makes a snapshot without moving any branch:
  `QC curation: 5 samples, 61,204 / 65,766 cells pass (source 1CECHNKR)`, with metadata
  `pipeline, created, source_snapshot, samples, decisions_sha256, reviewer, n_pass, n_cells`.
  Only then is the curation branch created, or reset to that snapshot. A failed or killed write
  leaves no trace.
  `main` is never touched, so ingest appends keep working. The branch holds only the latest
  curation. The previous one, and any `process` commits on it, are no longer on any branch.
  They stay in the repo's storage until you run icechunk's garbage collection yourself: this
  capsule never deletes anything.
- **plain zarr:** a new zarr at the output path (default `/results/<input stem>.qc.zarr`). The
  input is never modified.

## Spatial data

A store is SpatialData when it has a `tables/` group and no root `X`. Only plain `.zarr`
SpatialData is supported; a SpatialData icechunk repo is refused. The run reviews the cell
table `tables/table`, or the only table. Set `TABLE` to pick another. Images, labels, points and
shapes are never read. The `spatial` profile is tuned for 10x Xenium:

| | scRNA (`scrna`) | Spatial (`spatial`) |
|---|---|---|
| Sample | `sample_id` | `sample_id`, else the table's region column (`region_key` in `uns/spatialdata_attrs`) |
| Metrics | genes, UMIs (`total_counts`), mito % | genes, transcripts (`transcript_counts`: X's row sums; Xenium's `total_counts` also counts control probes) |
| Rules (fail when) | `doublet`, `high_mito`, `low_genes` (`<=`), `high_genes` | `low_counts`: `transcript_counts < min_counts` (20) · `low_genes`: `n_genes_by_counts < min_genes` (5) · `no_nucleus`: `nucleus_count < min_nuclei` (1) |
| Missing metrics | stop, unless `compute missing metrics` | always computed from X (panels are small). Only absent columns are computed; the store's own are kept |
| Embedding | up to 1,000 cells per sample | up to 20,000 per sample (a slide is often one sample). A panel of ≤ 2,000 genes uses every gene instead of HVGs |
| Ambient RNA indicator | yes | no |
| Extra plot | — | `plots/spatial/`: each sample's cells at their `obsm/spatial` centroids, failing cells in red |

The defaults come from a 10x Xenium lung panel (289 genes): its 1% / 5% quantiles were 11 / 20
transcripts and 8 / 13 genes per cell. Tune them for other panels. `no_nucleus` matters for older
Xenium output; with nucleus expansion (XOA ≥ 2.0) every cell has a nucleus. Without a `sample_id`
column, each region (for a Xenium slide, its labels element, e.g. `cell_labels`) is a sample, and
the summary says so. The cohort flags need ≥ 5 samples, so add a `sample_id` per slide to get them.

Flags, the decisions file, the embedding and the review loop work the same way. Apply writes
**only the table**: the output holds the root attributes and the curated `tables/<name>`. The
images stay in the input, and the output's table keeps its `region`/`instance_key` links to them.
`spatialdata.read_zarr` reads it, warning that the annotated element (e.g. `cell_labels`) isn't
in the object.

Tested on 10x Xenium lung (11,898 cells × 289 genes): the curated table equals the input's
passing rows.

## Running

### Code Ocean app

The app panel (`.codeocean/app-panel.json`) passes **named** parameters, which `code/run` forwards
to `main.py`. An empty value keeps the environment variable or default. **A first run needs only
the input store**: everything else has a safe default, and the default mode only reads.

| Category | Parameter (`--name`) | Default | Meaning |
|---|---|---|---|
| 1. Data (start here) | Input store (`input`) | — (required) | an icechunk repo (data asset, local path, `s3://bucket/prefix`; reads `main`) or a plain `.zarr` (local or a data asset; S3 is for icechunk only). SpatialData only as `.zarr` |
| | Decisions file (`decisions`) | *(none)* | a `qc_decisions.json` from an earlier run, picked from an attached data asset. Required for `apply` |
| 2. Write the curated cells | Mode (`mode`) | `report` | `report` only reads; `apply` writes the passing cells, after the checks above |
| | Curation branch / output zarr (`target`) | `qc` / `/results/<input stem>.qc.zarr` | icechunk: the curation branch, read for carried-forward decisions and written by `apply`. Plain zarr: a new local zarr path for the copy, used only by `apply` |
| | Writable origin (`origin`) | *(none)* | icechunk only: the writable `s3://` prefix behind a read-only data-asset mount. Needed only for `apply` |
| | Reviewer (`reviewer`) | *(none)* | recorded in `uns/qc` and the commit metadata |
| 3. Advanced | Group columns (`group_columns`) | `batch_id,pool_id,chip_id` | a column that's absent or has a single value is skipped |
| | Label column (`label_column`) | `pbmc_clean_AIFI_L1_2024_label` | groups cells for the ambient indicator and per-cell-type QC; any `<model>_label` column works |
| | Per-cell-type QC (`celltype_qc`) | `no` | `yes` turns on the per-cell-type section and the `cell_types` overrides |
| | Compute missing metrics (`compute_metrics`) | `no` | `yes` allows computing the QC metrics from X |

Other settings are environment variables named after their `config.Settings` field, in upper
case: `RESULTS_DIR` (`/results`), `SEED` (4242), `EMBED_PER_SAMPLE` (1000),
`EMBED_MAX_CELLS` (200000), `BAND_CELLS` (rows per X band when computing metrics, 100000), and `TABLE`
(the SpatialData table to review).

A run that's refused, or fails on bad input, logs one line and exits with code 1. Warnings go to
the log as they're found, and to the top of `report_summary.txt`.

### Locally

```bash
cd code/qc_check
python main.py --input /scratch/atlas                                     # report, icechunk
python main.py --input /scratch/atlas --decisions qc_decisions.json --mode apply --reviewer me
python main.py --input /data/atlas.zarr --decisions qc_decisions.json --mode apply --target /scratch/atlas.qc.zarr
python -m pytest tests                                                     # synthetic stores, both kinds
```

## Outputs (`/results`)

```
qc_decisions.json      the effective decisions, sealed with previewed_sha256 (edit and rerun, or pass to apply)
cell_qc.parquet        one row per cell in store order: sample, qc_pass, qc_fail_reason (dictionary columns),
                       plus the three metrics (float32) if they were computed; zstd, 1M-row groups;
                       file metadata holds the fingerprint, previewed_sha256 and the input
sample_qc.csv          the per-sample table, with flags (;-separated) and status (reviewed / pending / excluded)
celltype_qc.csv        the per-label table (only with per-cell-type QC)
embedding.parquet      the sampled cells: row position, sample_id, UMAP, leiden
plots/samples/         genes, UMIs and mito % per sample (50 per page, flagged first, thresholds drawn);
                       cells before and after; fail % by reason; mito % vs genes for each flagged sample
plots/groups/          the same box plots and fail % for each group column; cells per sample by group
plots/embedding/       UMAP by sample, group, fail reason and label; split by group; density (raw and
                       minus the mean); leiden composition by sample and group
plots/ambient/         each set's median per sample with the 1.5× line; heatmap of the ratios
plots/celltypes/       fail % per label × sample; genes and mito % per label; label probability
report_summary.txt     mode; whether apply would be accepted (or what blocks it); curation status;
                       warnings; counts; flags; pending samples
run.json               parameters, input fingerprint, versions, timestamps, peak memory, the apply result
apply_summary.csv      apply only: per sample, cells before and after, and cells removed by first reason
```

## Code map

```
code/run              Code Ocean entrypoint: app-panel parameters → main.py flags
code/qc_check/
├── main.py           the run: open, read, flag, settle decisions, report, apply
├── config.py         Settings from environment variables and app-panel flags
├── source.py         the input, read-only: AnnData icechunk at main's tip, or a plain AnnData / SpatialData zarr
├── profiles.py       per assay (scrna, spatial): the metrics, the cell rules and their defaults
├── metrics.py        the store's QC metrics, or the X-based fallback
├── samples.py        the per-sample table and its flags
├── decisions.py      the decisions file: validation, merging, sealing, the cell rules
├── embed.py          chunk-aware cell sampling and the downsampled embedding
├── ambient.py        the ambient RNA indicator
├── celltypes.py      the per-cell-type section
├── report.py         the tables, summary and plots under /results
├── plots.py          the figures
├── write.py          apply: the refusal checks, and the copy of the passing cells through annizarr
├── rows.py           the passing rows of a CSR matrix, as an annizarr reader
└── tests/            unit tests, plus report and apply runs on synthetic AnnData stores of both kinds and SpatialData zarrs
```

## Environment

The environment page pins the packages shared with the other capsules (annizarr, icechunk,
zarr, anndata, numpy, pandas, …), plus scanpy, leidenalg, igraph and umap-learn for the
embedding, and pyarrow, matplotlib and seaborn. `environment/postInstall` installs only
Claude Code. The resource class is `medium` for both modes.

The first embedding in a run spends about 40 s compiling UMAP's numba code.
