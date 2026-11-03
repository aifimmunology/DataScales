# `process` capsule: changes and why

The shared design is in [`../spec.md`](../spec.md) §5. This sheet records what was built or changed in
this capsule, and why. Newest decisions win where they disagree with older text.

## 1. What the capsule is for
- **The RAPIDS environment first.** The capsule exists to hand out a working GPU single-cell environment
  (RAPIDS, rapids-singlecell, Dask-CUDA, scanpy, scvi-tools/torch, squidpy, spatialdata), plus guide
  notebooks for out-of-core and spatial analysis.
  - *Why:* other capsules and people need the environment more than they need one fixed pipeline.
- **No App Panel.** *(Superseded 2026-11-02: there is no `run` any more, see §11; and 2026-11-03: an optional App Panel `run` is back, see §12.)* The pipeline runs from a terminal, `./run --input <store> [--flag value]`. Every setting
  is a flag (`./run --help`) and an upper-case environment variable.
  - *Why:* the analysis is exploratory; a form would freeze choices that belong in a notebook. Plain flags
    keep `run` small.
- **Users bring their own data asset.** Nothing is bundled; notebooks point at `/data/<asset>/...`.

## 2. Layout *(superseded 2026-11-02, see §11)*
```
code/
├── README.md       using the environment and notebooks, and `run`
├── environment/    requirements.txt and pixi.toml
├── notebooks/      out-of-core and spatial guides
├── run             ./run --input <store>
└── process/        the pipeline (config, cluster, steps, store, main) and tests
```
- **Environment files moved into `code/`** (from `capsules/process/environment/`).
  - *Why:* `code/` is what people copy or download from a capsule, so the package list should travel with
    it. `environment/` at the capsule root keeps only `postInstall`, like the other capsules.

## 3. Environment
- **Code Ocean builds from the pip3 list on the environment page.** `requirements.txt` is a copy of that
  list (32 pins) for pasting into another capsule's **pip3 → Edit Bulk**, or for `uv pip install -r`.
  - Base image `codeocean/cuda12-runtime-uv:main-5692e50`, Python 3.13, CUDA 13 wheels.
  - Checked 2026-10-29: every installed version matches its pin. `jupyter==1.1.1` was on the page but
    missing from the files, so it was added.
- **`pixi.toml` is for use outside Code Ocean only** (`CONDA_OVERRIDE_CUDA=13 pixi install`). Code Ocean
  never reads it. **It has not been solved yet**: installing pixi from inside the capsule was blocked.
- **annizarr stays pinned at 0.2.3**, matching the environment page. The code also works on 0.2.4 (§7).
- **Why exact pins:** this is an environment, not a library. The CUDA 13 RAPIDS wheels, torch and
  `cuda-toolkit` have to match each other, and zarr/anndata/spatialdata change fast. annizarr 0.2.4 is an
  example: it renamed `add_expr`. Without pins, a rebuild quietly installs whatever is newest that day.
- **`MPLBACKEND=Agg` is set by the image.** Both notebooks start with `%matplotlib inline`.
  - *Why:* without it, JupyterLab shows no plots.

## 4. Pipeline (`run`) *(superseded 2026-11-02, see §11)*
HVG → scale → PCA → optional harmony → neighbors → UMAP → leiden, plus `layers/lognorm`, on a
Dask-CUDA cluster with managed memory. Defaults follow `10-Rapids_GPU_Analysis.ipynb`.

- **rapids-singlecell 0.18 takes `rng=`** for PCA/neighbors/UMAP/leiden/harmony (`random_state=` is still
  accepted as the legacy name). The notebooks pass `rng=`; `calculate_niche_cellcharter` only takes `random_state=`.
- **lognorm is computed on the GPU and streamed to disk** block by block (anndata writes a dask CSR one
  chunk at a time from the client, so it's safe with an icechunk session). The layer gets a `target_sum`
  attr.
  - *Why:* one writer for every target (icechunk, in place, copies, SpatialData tables). annizarr's
    normalizing `add_expr` only worked on an AnnData at the store root, and is gone in 0.2.4.
- **`obsm/X_scVI` is no longer deleted** by a run.
  - *Why:* this pipeline doesn't make it; another step will.
- **Batch keys with NA values are refused**, and batch keys are only required when integrating.

## 5. Where results go, and the safety fixes *(superseded 2026-11-02, see §11)*
| Input | Results |
|---|---|
| icechunk repo | one commit on `--branch` (default `qc`); a read-only mount needs `--origin` |
| writable `.zarr` | in place |
| read-only `.zarr`, or `--output` given, or `--output-mode results-only` | a copy, default `/results/<stem>.processed.zarr` |
| remote plain `.zarr` | refused; use icechunk on S3 |

Fixes from two independent reviews, each with a test:
- **The store changing during a run.** icechunk: the branch tip is checked before the commit. Plain zarr: a
  fingerprint (X attrs and nnz, obs index; not `column-order`, which this pipeline rewrites) is checked
  before writing.
  - *Why:* results written in another cell order land on the wrong cells, silently.
- **A run that dies midway on a consolidated store.** Consolidated metadata is removed from the root and
  every group down to the table before the first change, written atomically, then the root is always
  consolidated again at the end.
  - *Why:* stale consolidated metadata made deleted elements read back as zeros, and made the "complete"
    marker (`uns/process`, written last) look present.
- **`--output` / `results-only` with a writable input** now always copy.
  - *Why:* they were ignored, so the input was modified.
- **`results-only` keeps everything but the counts** (`X`, `layers`, `raw`), not just obs and var.
- **Copies are built in `<dest>.partial-<pid>` and renamed into place.** A failure removes only its own
  partial copy.
  - *Why:* a failed copy could delete an output another run had just finished.
- **Cluster or NVML shutdown errors are logged, not raised.** `cluster_summary.csv` is cleared at the start
  and written before `run.json`.
  - *Why:* a run could be reported as failed after its results had landed.
- **A failed icechunk write discards its session**, and the branch tip stays where it was.

## 6. SpatialData and icechunk *(partly superseded 2026-11-02: saving, see §11; the icechunk limits still hold)*
- **A SpatialData store is processed through its table.** When the root has no `X`, `run` and both
  notebooks use `tables/<--table>` (default `table`). Only that group is written; images, labels, points
  and shapes are never changed. A read-only full copy copies the whole store; `results-only` gives just
  the table as an AnnData `.zarr`.
- **To edit just the table with history, use an icechunk repo laid out like a SpatialData store.** Tested:
  a `run` commit and a notebook save each touch only `/tables/table` in icechunk's diff. On a plain
  SpatialData `.zarr`, image and shape files were byte-identical after a run.
  - *Why icechunk:* plain zarr also allows table-only edits, but with no history and no way to undo.
- **Why only the table, not the whole object, in icechunk** (spatialdata 0.8.0, checked against earthmover's
  `2026-scverse/icechunk-SpatialData`):
  - `sdata.write()` and `sd.read_zarr()` don't accept an icechunk store; that needs the open spatialdata
    PR #1107 plus earthmover's patches, and their own advice is not to build on it yet.
  - spatialdata's element writers work on an icechunk session for images, labels, tables and zarr-encoded
    shapes. Points and GeoParquet shapes fail.
  - So: write the table with `anndata.io.write_elem`, read it with `read_elem(root["tables/<name>"])`, and
    keep a plain SpatialData `.zarr` for tools that need the whole object (napari, `sd.read_zarr`).
  - annizarr's ops assume the AnnData at the repo root, which is another reason `run` writes lognorm
    itself.

## 7. annizarr 0.2.4 review (2026-10-30)
- **Breaking:** `add_expr` was renamed `add_layer`, and now only copies a matrix in another layout, with no
  normalization or `target_sum`. This capsule no longer calls it, so nothing breaks. The layer attr was
  renamed from `annizarr_target_sum` to `target_sum`, since annizarr no longer writes or reads it.
- **Other changes:** the matrix copy moved into `_writers/_copy.py` (used by `rechunk`), and the CPU count
  respects cgroup quotas (`_core/_system.py`). That suits Code Ocean containers.
- **Not in 0.2.4:** `subset` and `Repo.garbage_collect`, which `../spec.md` §8 planned for qc apply.
- **Checked:** the 38 tests pass against both 0.2.3 and 0.2.4. The `Repo` API this capsule uses (`exists`,
  `create`, `open_zarr`, `log`, `commit`, `discard`, `checkout`, `dirty`) is unchanged.

## 8. Notebooks *(superseded 2026-11-02, see §11)*
- **`rapids_scrna_out_of_core_dask.ipynb`** is rebuilt as 9 short steps matching `run`. It reads an AnnData
  or SpatialData `.zarr`, or an icechunk repo holding either. It only reads.
- **`rapids_spatial_atera_breast_cancer.ipynb`:**
  - It reads the table from a SpatialData `.zarr` or an icechunk repo. The shapes are no longer required,
    and the `nucleus_count` filter is skipped when that column is missing.
  - Integer counts are cast to float32 before the move to the GPU.
    - *Why:* `anndata_to_GPU` refused int32.
  - **New optional §7 save (`SAVE = True`).** It writes the cell types, clusters, niches and embeddings
    into the table under their own names (`spatial_cell_type`, `spatial_leiden`, `spatial_niche`,
    `X_pca_spatial`, `X_umap_spatial`). It writes only those columns, refuses when the icechunk branch
    moved since loading, and checks that the cell IDs are unique and match.
    - *Why:* so it never replaces `run`'s results or puts values on the wrong cells.

## 9. Tested, and not *(superseded 2026-11-02, see §11)*
| What | Result |
|---|---|
| 38 tests (11 full GPU runs): AnnData, SpatialData, icechunk, failure paths | pass on the T4, with annizarr 0.2.3 and 0.2.4 |
| Out-of-core notebook, headless, on a SpatialData `.zarr` and an icechunk repo | every cell runs |
| Spatial notebook, headless, 25,600 synthetic cells, as `.zarr` and icechunk, with saving on | every cell runs, ~56 s, 0.6 GB GPU memory |
| 1M cells × 10k genes with harmony | ~3.5 min, 3.3 GB GPU / 7.3 GB RAM at peak (before the lognorm change) |

**Not tested yet:**
- real data;
- more than one GPU;
- the pixi solve;
- a 1M-cell rerun with the GPU lognorm writer.

## 10. Open
- **scVI / scANVI integration**, in a subprocess (`../spec.md` §5.2 step 5). *(2026-11-03: scVI training is in the optional pipeline, in-process after the cluster is closed, see §12.3. scANVI is not.)*
- **Code Ocean versioned data assets** (AllenNeuralDynamics/aind-code-ocean-info#11; private, not read yet).
  The question is whether a new asset version copies everything, or can point at an icechunk repo on S3,
  where commits cost very little.
- **ingest and qc still pin annizarr 0.2.3.** Neither calls `add_expr`; bumping them is up to their own
  testing.

## 11. Rework: an environment and notebooks capsule (2026-11-02)
Built from [`rework_spec.md`](rework_spec.md), on the user's decisions of 2026-11-02. It replaces §2, §4,
§5, §8 and §9, and the saving parts of §6.

- **Removed `code/run` and `code/process/`** (config, cluster, steps, store, main, 38 tests).
  - *Why:* the capsule's value is the environment and worked examples. The CLI froze choices that belong in
    a notebook, and its guards (refusing read-only inputs, the `qc` branch default) made it a dead end on a
    fresh ingest store. The notebooks now hold the only copy of the steps.
- **Both notebooks follow one pattern**, with the same helper cells (no local package):
  - a settings cell (`STORE`, `TABLE`, `OUTPUT`, `SAVE`, `RESULTS`) with commented icechunk lines (`BRANCH`,
    `ORIGIN`);
  - an open cell: `zarr.open_group(STORE, mode="r")`, using `tables/<TABLE>` when the root has no `X`, with
    a commented icechunk variant;
  - a save section that runs only with `SAVE = True`. Zarr input: a new full object at `OUTPUT` (default
    `/results/<stem>.processed.zarr`). AnnData is copied with `annizarr.rechunk`; SpatialData is copied
    whole (`shutil.copytree`) and the results go in the copy's table, then the root is re-consolidated. It's
    built at `<OUTPUT>.partial` and renamed; an existing output is never overwritten (`<stem>-<n>.zarr`).
    Icechunk: one commit on `BRANCH` through `Repo(STORE, branch=BRANCH, origin=ORIGIN)`, refused only when
    the branch moved since opening.
  - *Why a full copy:* the result stands alone (it can become its own data asset), and the input is never
    changed, so read-only inputs are a non-issue. A save that dies midway can leave a `.partial` folder;
    the user accepted that.
- **Out-of-core notebook** is now what `run` did: it saves `obsm/X_pca` (and `X_pca_harmony`),
  `obsm/X_umap`, `obs/leiden`, `var/highly_variable`, `layers/lognorm` (streamed block by block from the
  GPU, `target_sum` attr) and `uns/process`. It has an optional commented harmony cell (`BATCH_KEYS`), treats
  `""` in a batch key as missing with a warning (ingest writes `""`), and writes `umap.png` and
  `cluster_summary.csv` to `RESULTS`. Same defaults as before. NA batch keys are no longer refused; harmony
  raises its own error on them.
- **Spatial notebook** saves through the same rules: a new SpatialData copy on zarr, a commit on icechunk.
  Result names unchanged, plus `uns/spatial_process`; NA for cells removed by QC. The table is read with
  `read_elem` on both paths (it used `sd.read_zarr` on zarr before).
- **No refusals** beyond the branch-moved check and the spatial cell-ID check; anything else fails with the
  library's own error.
- **Tests:** `code/notebooks/test_notebooks.sh` builds synthetic stores and runs each notebook headless
  with nbconvert, toggling the icechunk and harmony lines by their comment markers. Cases: scRNA on AnnData
  zarr (read-only input, rerun goes to `-1`, lognorm against scanpy), on SpatialData zarr (`sd.read_zarr`,
  images/labels/points/shapes byte-identical), on icechunk with harmony (one commit), branch moved (no
  commit); spatial on SpatialData zarr and icechunk.
  - Passed 2026-11-02 on CPU, with scanpy/squidpy/decoupler stand-ins for rapids-singlecell, cupy, rmm and
    dask-cuda, run as a non-root user so the read-only input was enforced (harness in
    `/scratch/process_rework/`). **Not yet run on the GPU**, so GPU numerics and the `.get()` paths are untested.

### 11.1 Review fixes (2026-11-03)
- **Saving onto an ingest store** failed with `KeyError: 'layers'` (ingest writes no `layers` group). A
  `child` helper creates `layers`, `obsm` and `uns` when missing.
- **Icechunk with `ORIGIN`** only committed once: `SNAPSHOT` came from the mount, the branch check from the
  origin. With `ORIGIN` set, the store is now opened and its snapshot recorded through the origin (as ingest
  reads its state), so load and check see the same branch tip.
- **`OBS_COLUMNS` are optional:** only the columns the store has are read and plotted (spatialdata_io
  tables have no `sample_id`).
- **Icechunk is one block** at the end of the settings cell, which sets `repo`; the open cell and a shared
  `save` helper branch on it. The helper cell is identical in both notebooks.
- `OUTPUT` without `.zarr` gets it added, so reruns give `<stem>-<n>.zarr` like the default. The spatial
  ligrec plot uses the six largest cell types instead of hard-coded names. The spatial notebook passes
  `rng=` (see §4).
- **Tests:** settings are now set in the settings cell itself, and the icechunk block is uncommented as a
  user would. New cases: `helpers_identical`; `scrna_ingest` and `scrna_ingest_no_sample` (stores made with
  `annizarr.convert`: uint32 CSR, categorical obs, `""` strings, no `layers`; the second without
  `sample_id`, with an `OUTPUT` lacking `.zarr`, run twice); `icechunk_origin` (a read-only mount copy and a
  writable origin, two saves in a row, both commit). All 10 cases passed on CPU on 2026-11-03 (`/scratch/process_rework/run_cpu_tests.sh`); not yet run on the GPU.

## 12. Optional App Panel / Reproducible Run pipeline (2026-11-03)
The user decided (2026-11-03) to have a pipeline runnable from the App Panel and as a Reproducible Run as
well. **The capsule stays first an environment and notebook holder**; the README, the panel's first category
and its *Input store* field, and the no-input run all say the pipeline is optional. It partly reverses §11's
removal of `run`. Revised the same day after an independent review (§12.7).

### 12.1 What was built
- `code/run` (`cd` into `process/`, `exec python -u main.py "$@"`; it exports `PROCESS_CWD` first, and every
  path setting is resolved from it, so relative paths mean the folder `run` was started in).
- `code/process/`, small modules modelled on `dev_tools/rapids-benchmark/rapids_benchmark.py`
  (`origin/datavis_analysis`):
  - `config.py`: one frozen `Config` dataclass; every field has a default and is a `--flag` (underscores or
    dashes) and an upper-case environment variable. Defaults < env < flags; an empty value or a bare flag keeps
    the earlier one, so empty panel fields mean "default" however Code Ocean sends them. Paths are resolved
    to absolute ones (also what `uns/process/input` records).
  - `cluster.py`: `LocalCUDACluster` over tcp, one worker per visible GPU, `rmm_mode` managed (default),
    pool or none, and zarr concurrency set on every worker and the client.
  - `steps.py`: the steps in the benchmark's order (load, h2d, normalize/log1p, HVG, rechunk/persist into one
    band per GPU, scale, PCA, harmony, neighbors, UMAP, leiden); the subset by slicing contiguous spans; scVI
    training; one leiden per resolution on one graph.
  - `store.py`: the notebooks' open/save rules. `child`, `put`, `put_column`, `free_path`, `new_output`,
    `save_copy` and `save_commit` are the notebooks' helper cell word for word (the notebooks' `save` wrapper
    passes them its globals); `test_helpers_match_the_notebooks` fails if they drift.
  - `report.py`: per-step wall time and peak host (process tree RSS) / GPU (NVML, device-wide, the `gpus`
    devices) memory, printed and appended to `/results/process_report.txt`.
  - `main.py`: `load` (every input check, before the cluster starts), `embed`, `write`. No input → prints that
    the capsule is mainly an environment with notebooks and how to use the panel, exits 0.
- `.codeocean/app-panel.json`: 14 fields in five categories (Input data, Embedding, Neighbors & UMAP,
  Clustering, Output), `named_parameters: true`, a `param_name` per field equal to the `Config` field. Every
  `pattern` accepts an empty field.
- Results, per the notebooks: the embedding (`X_pca`, `X_pca_harmony` or `X_scVI`), `X_umap`, `obs/leiden`
  for one resolution or `obs/leiden_<res>` (e.g. `leiden_0.5`, `leiden_1.0`) for several, `var/highly_variable`,
  `layers/lognorm` (`target_sum` attr) and `uns/process` (method, basis, batch keys, resolutions, subset,
  every setting, input, snapshot, and the PCA variance ratio or the scVI model path). Plus `umap.png`,
  `cluster_summary[_<key>].csv` and `process_report.txt` in `/results`.
- The notebooks changed only in their helper cell (regenerated with `/scratch/process_rework/build_notebooks.py`):
  `free_path`, and `new_output`/`save_copy`/`save_commit` taking arguments, as in `store.py`. `save_copy` now
  also creates the output's parent folder.

### 12.2 Named parameters, not positional
Code Ocean's App Panel has a **Named Parameters** option: the run script then gets `--<param_name> <value>`.
- Code Ocean docs, "Passing App Panel's Parameters to the Script",
  <https://docs.codeocean.com/user-guide/app-panel-guide/passing-app-panels-parameters-to-the-script.md>:
  "arguments can be accessed by name if the Named Parameters is selected", with an argparse example
  (`parser.add_argument('--plot_title', ...)`).
- The JSON keys come from AllenNeuralDynamics' `auto-app-panel` (PyPI 0.1.3), which writes
  `"named_parameters": true` at the top level and a `"param_name"` per parameter, and from the `codeocean`
  SDK 0.17.0 (`AppPanelParameters.param_name`, "The parameter name/argument key"). The SDK describes the API
  format, not necessarily the file's, so check that the panel loads these keys.
- *Why:* fourteen positional arguments break silently when one is added or reordered. Named parameters
  also make the panel and the terminal use the same flags.
- *Not confirmed:* how an empty field is sent (`--name ""`, a bare `--name`, or left out). All three mean
  "default" (tested).
- The SDK also has a `general` block (title, instructions). The ingest panel's file has no equivalent and the
  key name in `app-panel.json` isn't confirmed, so it was dropped; the "optional" note is in the first
  category's and the *Input store* field's descriptions.

### 12.3 scVI and RAPIDS in one process
scVI trains **in the main process, after the Dask-CUDA cluster is shut down**, not in a subprocess:
1. HVGs are chosen on the GPU cluster as for PCA.
2. The raw counts of those genes for the (subset) cells are computed into host memory as one scipy CSR
   (about 2–3 GB at 1M cells × 2,000 genes, as the user accepted).
3. The cluster is closed, which frees every worker's GPU memory, whatever the RMM mode (a pool holds
   `rmm_pool_size` of each GPU).
4. scVI trains with torch's own allocator on the first GPU of `gpus` (`gpus` sets `CUDA_VISIBLE_DEVICES` for the
   whole run, so that's device 0 for torch and for the client's RMM resource, which is not a pool). Then the
   latent space is copied out, the model saved, and `torch.cuda.empty_cache()` called.
5. Neighbors, UMAP and leiden run in the same process on `X_scVI`. The `lognorm` layer, still a lazy dask graph,
   is streamed at save time by dask's local scheduler on that GPU (the cluster is gone).

- *Why not a subprocess:* the counts would have to go to disk and back, and the latent space be passed back;
  that's more code and failure modes for no gain once the cluster is closed, since the GPU users are then
  sequential, never concurrent.
- *Cost:* on several GPUs, the scVI path trains and writes `lognorm` on one GPU.
- `mg_ivfflat`/`mg_ivfpq` (cuvs multi-GPU) use every GPU the client process sees, so they follow `gpus` too.

### 12.4 Defaults and other choices
- **Defaults follow the out-of-core notebook:** HVG seurat 2000, scale max 10 without centering, PCA 50
  (`covariance_eigh`), neighbors 20 on the first 30 PCs (`ivfflat`), UMAP min_dist 0.45, leiden 1.1.
  `random_seed` 0 and `threads_per_worker` 1, as the notebook gets by default. `leiden_iterations` 100. Harmony
  `max_iter` 30 (from the benchmark; rsc's 10 rarely converges at atlas scale), theta 2, alpha 0.2.
- **scVI defaults** (atlas-style): `n_latent` 30, `n_layers` 2, `n_hidden` 128, dropout 0.1, `nb` likelihood,
  scvi-tools' own epoch rule (`max_epochs` 0 → `None`: 400 below 20,000 cells, `20000/n × 400` above, so 8 at
  1M cells), batch 128, lr 1e-3, train size 0.9, no early stopping. At 1M cells that's ~63,000 steps, which can
  take a long time on a T4; the timing run uses `SCVI_MAX_EPOCHS=5 SCVI_BATCH_SIZE=1024`. They are `SCVI_*`
  settings, not panel fields. Several batch keys: the first is scVI's batch, the rest
  `categorical_covariate_keys`; harmony takes all.
- **Cluster:** tcp only, `rmm_mode` managed | pool | none. The benchmark's `speed` preset (ucx + pool) was
  dropped: UCX needs `distributed-ucxx-cu13`, which isn't in the environment.
- **Benchmark knobs not carried over**, and why:
  - `pca_float64`: the notebook scales in float32; one less option.
  - `label`, `results_txt`: the report always goes to `<results_dir>/process_report.txt`, whose header lists the
    changed settings.
  - `umap_key`, `leiden_key`: names are fixed by the notebooks' rules (`X_umap`, `leiden`/`leiden_<res>`).
  - `write_results`, `subset_dir_name`: results are always saved by the notebooks' rules; a subset is written
    into the full copy (NA elsewhere), not to a separate `subsets/` store.
- **Subset:** cells matching the values are sliced in contiguous spans (one per value on a sorted store).
  Results are written for all cells, with NA/NaN outside the subset, so the copy stays a full object. `lognorm`
  covers every cell.
- **Reruns:** earlier `X_pca`, `X_pca_harmony`, `X_scVI`, `X_umap` and `leiden`/`leiden_<res>` results are
  replaced or removed, so everything in the store comes from one run (`uns/process`).
- **Fail fast, before the cluster starts** (one-line error, exit 1): an input that can't be opened, a SpatialData
  store without `tables/<table>`, a batch key or subset column missing from obs, a subset with no cells, and
  cells without a batch value (missing or `""`), with their count. The notebook only warns about `""`
  batch values; here the run would fail later in harmony or scVI anyway, after the expensive steps. A bad
  setting value exits 2 with its name.
- `output` is ignored, with a message, for icechunk input; `origin` is ignored for a zarr input.

### 12.5 Tested
- **CPU, 2026-11-03:** 49 pytest tests in `code/process/tests` (48 pass, the GPU-only one skips), with stand-ins
  for rapids-singlecell, cupy, rmm, dask-cuda and scvi-tools (`/scratch/process_rework/stubs`;
  `run_cpu_pipeline_tests.sh`, as `nobody`, so read-only inputs are enforced). Config/panel parsing (bare and
  empty flags), env vars, defaults, leiden naming, subset slicing, output naming, full zarr copy, ingest store
  without `layers`, SpatialData copy, icechunk through a read-only mount and its origin twice, branch moved,
  stale results, notebook helper drift, the input errors, the no-input run; and end to end through `code/run`:
  PCA (relative paths), PCA+harmony, scVI, three resolutions with a subset, an ingest-style store with no
  `layers` and no `sample_id` run twice, SpatialData, icechunk via origin twice. Each checks the output reads
  back and that `lognorm` equals scanpy's. The notebook harness (`run_cpu_tests.sh`) passes its 10 cases with
  the regenerated helper cell.
- **Not run on a GPU yet.** GPU-only: `test_the_gpu_libraries_take_our_arguments` (rsc 0.18 / scvi-tools 1.5
  signatures, including harmony's `max_iter_harmony`, `theta`, `alpha`).

### 12.6 GPU capsule test plan
On a GPU machine (the T4 capsule). `.codeocean/resources.json` says `"resource_class": "xsmall"`, left as it
is; a Reproducible Run must be started on a GPU machine class. From `/root/capsule/code`:

- [ ] `./test_gpu.sh` → `notebooks/test_notebooks.sh` prints `PASS` for its 10 cases, pytest reports
  49 passed (no skips), and the script ends with `ALL PASSED`. This covers every case below on tiny stores.
- [ ] Notebooks: `notebooks/test_notebooks.sh` → `PASS` for `helpers_identical`, `scrna_zarr`, `scrna_ingest`,
  `scrna_ingest_no_sample`, `scrna_sdata`, `scrna_icechunk`, `branch_moved`, `icechunk_origin`,
  `spatial_sdata`, `spatial_icechunk`.
- [ ] Signatures: `python -m pytest process/tests/test_steps.py -k gpu_libraries` passes.
- [ ] PCA: `python -m pytest process/tests/test_run.py -k pca_relative_paths` → output with `X_pca`, `X_umap`,
  `leiden`, `lognorm` equal to scanpy's.
- [ ] Harmony: `-k pca_harmony` → `X_pca_harmony`, `basis` `X_pca_harmony` (one batch key, managed memory).
- [ ] scVI: `-k scvi` → `X_scVI` of width 8 (`SCVI_N_LATENT=8`, 5 epochs), the model folder, and no CUDA
  out-of-memory error after the cluster closes.
- [ ] Multi-res + subset: `-k multi_resolution_subset` → `leiden_0.5/1.0/2.0`, NA outside `SP0, SP2`.
- [ ] Ingest store: `-k ingest_store` → `out.zarr` and `out-1.zarr`.
- [ ] SpatialData: `-k spatialdata` → the copy opens with `sd.read_zarr`, results in `tables/table`.
- [ ] Icechunk via origin: `-k icechunk_origin_twice` → two commits on the origin, the mount unchanged.
- [ ] Input errors: `-k input_errors` → exit 1, one line, no cluster started.
- [ ] No-input run: `./run; echo $?` → the usage text and `0`. Also press **Reproducible Run** with an empty
  panel (on a GPU machine class): the run succeeds and its log shows the same text.
- [ ] App Panel: the five categories render and the file's keys load; *Method* is a dropdown; a run with
  *Input store* set shows `--input …` (named) in the log; note how empty fields arrive; leaving them empty
  gives the defaults in `process_report.txt`.
- [ ] Several GPUs (if available): `GPUS=0,1 NEIGHBORS_ALGORITHM=mg_ivfflat ./run --input <store>` → two
  workers in the header; with `--method scvi --gpus 1`, training shows on GPU 1 in `nvidia-smi`.
- [ ] 1M cells: `./test_gpu.sh big` → a 1M × 10k store, then PCA+harmony and scVI (5 epochs, batch 1024) with two
  resolutions. Expect PCA+harmony in a few minutes (≈3.5 min before, §9) and record both runs' wall time and
  peak host/GPU memory from `process_report.txt` here.

### 12.7 Review fixes (2026-11-03)
- The harmony managed-memory patch from the benchmark was removed: rsc 0.18.1 has no `_hc_cl` (it imports
  `_harmony_clustering_cuda`) and passes no joint arrays with one key, so the patch never applied.
- The `speed` preset and `protocol` were dropped (§12.4).
- Flags accept no value (`nargs="?"`), so a bare `--subset_column` from an empty panel field means the default.
- `gpus` sets `CUDA_VISIBLE_DEVICES` for the run, so scVI and `mg_ivfflat` follow it (§12.3).
- Input checks moved before the cluster, as errors (§12.4); empty Leiden resolutions pass the panel pattern;
  the input path is resolved before it's recorded.
- The notebook helper cell and `store.py` share seven functions word for word, with a test.
- Simpler code: `main.run` split into `load`/`embed`/`write`; `report` memory is two functions; no `Source.branch`.
