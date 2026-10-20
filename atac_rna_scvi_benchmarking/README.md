# scvi_pca — re-clustering a cell selection: scVI vs PCA (RNA), PeakVI vs LSI (ATAC)

When a datavis user selects cells, the app re-embeds just that selection and redraws its UMAP. This benchmark asks
whether a learned latent (scVI for RNA, PeakVI for ATAC) beats the current linear path (PCA + Harmony for RNA, LSI
for ATAC) on **runtime** and on **biological signal kept while batch effects are removed**, and whether it is worth
adding to the pipeline.

## What was run

**Datasets** (local zarr stores, one A100 80 GB, 167 GB host RAM):

| Dataset | Modality | Cells | Labels (coarse / fine) |
|---|---|---|---|
| mmrna | RNA | 2,374,117 | AIFI_L2 / AIFI_L3 |
| rna13m | RNA | 13,789,548 | AIFI_L2 / AIFI_L3 |
| mmatac | ATAC | 618,673 | L2 / L2.5 |
| covatac | ATAC | 2,718,917 | L2 / L2.5 |

**Selections**, picked by the store's L1 label the way a user would select in the app:
`all` (every cell), `tnk` (L1 matches "T cell" or "NK cell"), `b` (L1 = "B cell").

**Methods** for each selection:

| Method | RNA | ATAC |
|---|---|---|
| `latent` | slice of a scVI latent trained once on the full dataset | slice of a PeakVI latent trained once on the full dataset |
| `pca` | normalize → log1p → HVG 2000 → scale → PCA 30 → Harmony on `batch_id`, following `dev_tools/rapids-benchmark` | — |
| `retrain` | scVI trained from scratch on the selection | PeakVI trained from scratch on the selection |
| `unintegrated` | (reference only) PCA without Harmony | (reference only) TF-IDF + LSI without batch correction |

Every method then runs the same tail: rapids-singlecell neighbors (ivfflat, k = 20) → UMAP (min_dist 0.45). No
clustering. scVI / PeakVI train a fixed `clamp(2e7 / cells, 10, 100)` epochs with the KL weight warmed up over the
first half (batch_size 1024, scVI n_latent 30 on 4,000 seurat_v3 HVGs, PeakVI n_latent 40 on binarised peaks).
Each run was done twice in a fresh process (rep 1, rep 2); `retrain` on `all` is the full-data model that `latent`
slices.

**Short answer**
- **RNA:** PCA + Harmony keeps as much biology as scVI or more and removes much more batch effect, so keep it as the
  default. The stored scVI latent is the only fast path on big stores (1.2M B cells of 13.8M: 8 s vs 10 min for
  PCA), so it is worth adding only as a fast mode. Retraining scVI per selection is not worth its cost.
- **ATAC:** PeakVI beats uncorrected LSI on biology at every selection. Train it once per store and slice its latent
  (1–18 s); retraining per selection takes 2–14 min for little gain.

Full numbers and caveats: [`report/report.md`](report/report.md).

## umaps/

Rendered from each run's saved UMAP (rep 2) and the store's labels; one PNG per plot, with a `.params.json` beside
each single plot recording the run id, cell count and plot settings.

| File | What it shows |
|---|---|
| `umaps/aifi/<ds>/grid_subset_x_strategy.png` | **Start here.** Rows = selection (all / T + NK / B), columns = method. Coloured by the finest label (AIFI_L3 or L2.5), legend grouped by coarse label, L1 names placed on the clusters. Compare along a row to see methods, down a column to see how a selection's structure opens up. |
| `umaps/aifi/<ds>/<method>/<selection>.png` | The same plots one at a time, larger. |
| `umaps/batch/<ds>/<column>/grid.png` | Same grid layout, coloured by a technical or donor column instead of cell type. Well-mixed colours = batch effect removed; colour islands = residual batch effect. RNA columns: `batch_id`, `pool_id`, `chip_id`, `well_id`, `subject.subjectGuid`, `cohort.cohortGuid`. ATAC: `batch_id`, `pool_id`, `chip_id`, `well_id`, `Sample`, `nFrags`, `TSSEnrichment` (the last two continuous, viridis). |

`<ds>` is mmrna, rna13m, mmatac or covatac; `<method>` is latent, pca or retrain; `<selection>` is all, tnk or b.

## results/

| File | Contents |
|---|---|
| `runs.parquet` | One row per run (60 rows): `run_id` = `<ds>-<method>-<selection>-r<rep>`, `n_cells`, status, and timings in seconds. **`t_e2e` is the headline runtime** (read data → UMAP). It splits into `t_read` (reading counts or the latent), `t_pre_ex_read` (the embedding step itself), `t_neighbors` and `t_umap`. Peak GPU memory (`*_vram_gb`, device-wide) and host memory (`*_rss_gb`) are given per stage and per run. Leiden columns are empty (not run). |
| `setup.parquet` | The full-data model fits behind `latent`, two reps per dataset: training time (`train_s`, `t_pre_s`), epochs, steps, model and latent size on disk, model load time. |
| `metrics/<ds>.parquet` | Quality metrics in long format: `subset`, `strategy` (method, plus `unintegrated` as reference), `level` (`coarse` / `fine` label, blank for label-free rows), `metric`, `value`, `n_eval` (cells scored). |
| `subsets_manifest.parquet` | Cells per L1 label in each selection. |

Metrics, all computed on the same cells for every method of one selection (every labelled cell, or a
sample-stratified 500k when larger):
- **Bio conservation** (higher = more cell-type signal kept): `purity_macro_coarse` / `purity_macro_fine` (share of
  each cell's 20 nearest neighbours with its own label, averaged per cell type), and scib-metrics
  `Isolated labels`, `KMeans NMI`, `KMeans ARI`, `Silhouette label`, `cLISI`, aggregated as `Bio conservation`.
- **Batch removal** (higher = batch effect removed): scib-metrics `BRAS`, `iLISI`, `KBET`, `Graph connectivity`,
  `PCR comparison` (vs the uncorrected embedding), aggregated as `Batch correction`; plus `same_batch` (share of
  neighbours from the cell's own `batch_id`, lower = more mixed; `same_batch_random` is the random-neighbour value).
- **`scib:Total`** = 0.6 × bio conservation + 0.4 × batch correction (scib's default weighting).

## report/

- `report.md` — findings, then per dataset: runtime table and every metric at both label levels.
- `compare_run2.html` — the same results as interactive charts (runtime per selection, bio vs batch scatter, metric
  picker, full table); open in a browser. Built from `compare_run2.json`, `compare_run2_template.html` and
  `compare_run2_findings.html`.

## scripts/

Self-contained: the `scvi_pca` package (`src/`), parameters and dataset definitions (`config/params.toml`,
`config/datasets.toml`) and the pixi environment (`pixi.toml`, `pixi.lock`; default env for the pipeline, `metrics`
env for scib-metrics/JAX) all live here. Large intermediate files go to `/mnt/scvi_pca/run2/` (subsets, models,
embeddings, UMAP coordinates, metric caches).

| Step | Script |
|---|---|
| Build selections | `01_make_subsets.py <ds> ...` |
| Full-data model (setup) | `02_train_full.py <ds> <rep>` |
| One run / a list of runs | `10_run_one.py <run_id>`, `11_run_matrix.sh`, `12_run_list.py` |
| Everything for one dataset | `13_run_dataset.sh <ds>` (setup, runs, metrics, plots; resumable) |
| Metrics | `20_metrics.py <ds>` (+ `21_scib_worker.py` in the metrics env) |
| UMAPs | `30_plots.py <ds>` |
| Report | `40_report.py`, then `41_compare_data.py` and `42_build_compare.py` for the HTML |

```bash
cd scvi_pca/scripts
pixi install -a                      # both envs, from pixi.lock
pixi run python 01_make_subsets.py mmrna rna13m mmatac covatac
# detached so it survives the terminal closing; reruns skip finished runs
setsid nohup bash -c 'for ds in mmatac covatac mmrna rna13m; do bash 13_run_dataset.sh $ds; done' \
  > /mnt/scvi_pca/run2/logs/campaign.log 2>&1 < /dev/null &
pixi run python 40_report.py && pixi run python 41_compare_data.py && pixi run python 42_build_compare.py
```

The pipeline imports `datavis_realtime_analysis/gpu` and `dev_tools/rapids-benchmark` from this repo, needs the GPU
to itself (stop the datavis backend first), and serialises GPU steps through `/mnt/scvi_pca/gpu.lock`. A full run
took about 17 h, about 7 h of it scib metrics.
