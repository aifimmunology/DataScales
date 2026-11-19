# Capsules: shared guide

The one guide for every capsule in `capsules/`, for Claude Code and for people. Each capsule's own
design, decisions and findings are in its `spec.md` (`process`: §10 below); its user guide is `code/README.md`. Section
numbers §1, §2, §6, §8, §9 are kept from the former `capsules/spec.md`; §3/§3S/§4/§5 moved to the
capsule specs with their numbers.

| Capsule | Spec | Status |
|---|---|---|
| `anndata_ingest/` | [anndata_ingest/spec.md](anndata_ingest/spec.md) | built |
| `spatial_ingest/` | [spatial_ingest/spec.md](spatial_ingest/spec.md) | built 2026-11-11; robustness rework 2026-11-14 (public-data testing, crash/S3 safety); consolidated to 5 modules (anndata_ingest layout) 2026-11-18; release review in progress |
| `qc/` | [qc/spec.md](qc/spec.md) | design (§4) plus what the build changed |
| `process/` | §10 below (no spec file) | built; environment + notebooks, optional pipeline; consolidated to 5 modules 2026-11-16 |

## Overview

Code Ocean capsules that take HISE scRNA samples from raw files to a curated, processed store, plus a
spatial ingest that builds one SpatialData object from Xenium and Atera bundles:

| # | Capsule | Hardware | Writes | Job |
|---|---|---|---|---|
| 1 | `anndata_ingest/` | CPU, sized to the largest **sample** | `main` | Two input types: raw samples (.h5, .h5ad, SpatialData tables) get per-sample QC metrics, CellTypist, Scrublet; existing AnnData / SpatialData tables go in as they are. Both append to one AnnData store. No whole-store analysis |
| 1S | `spatial_ingest/` | CPU, `large` (64 GB); ≤ ~12 GB peak per sample with the streamed matrix | one joint SpatialData `.zarr` on a local path, or synced to an `s3://` prefix (never icechunk) | Append Xenium/Atera bundles (folders, `.zip`, `.tar.gz`, GEO-style files; kind detected automatically) and existing SpatialData with a table (single- or multi-sample, split per sample): per-sample elements and table, plus one joint single-cell table over the union of genes |
| 2 | `qc/` | CPU; I/O-bound when applying | nothing by default; with `apply`, a commit on `main` (icechunk) or a new zarr | QC report for proposed thresholds, with a downsampled embedding. With `apply` and reviewed decisions, writes the passing cells as the curated object |
| 3 | `process/` | **GPU, CUDA 13** (rapids-singlecell + Dask, scvi-tools) | a new zarr, or a commit (icechunk) | First the shared RAPIDS environment with guide notebooks (out-of-core scRNA, spatial); optionally an App Panel / Reproducible Run pipeline: PCA (± harmony) or scVI, then neighbors, UMAP, leiden |

```
samples ─► ingest ─► main (raw, all cells)
                       │
                       ▼
                  qc (report) ─► report + qc_decisions.json ◄── person reviews, edits, reruns
                       │                     │
                       ▼                     │ the reviewed file
                  qc (apply) ◄───────────────┘
                       │
                       ▼
        main, new commit (curated cells only; raw tagged and kept in history)
                       │
              process notebooks ─► + embeddings, clusters (new zarr, or a commit)
```

- Cell labeling and cluster-level doublet cleanup (standardized notebooks 06–08) come after `process`
  and are a separate stage. Stores built by the old pipeline aren't migrated; they are rebuilt from
  samples with the new ingest.
- Standardized notebooks → capsules: 01 → `ingest`; 02, 03, 05 → `qc` (02 detection only); 10 →
  `process`; 06–09 not covered. Details: [qc/spec.md](qc/spec.md).
- **Anyone can run them on their own data.** Every location (data asset, local path, S3) comes in
  through the app panel or a notebook's settings cell. Nothing is hard-coded; no per-user permissions.
- **Design for first-time users:** simple, no dead ends. Follow the decisions recorded in the specs;
  ask the user before changing one.

## Git and publishing

- Commit and push only DataScales branch `codeocean_capsules`, as the user (Alex Holly). Never
  commit in a Code Ocean capsule checkout such as `/root/capsule`.
- Keep commits extremely sparse: one per body of work. No Claude co-author trailer.
- Work in a capsule checkout (`/root/capsule/code`), then copy it into `capsules/<name>/code`, plus
  `.codeocean/app-panel.json` and `environment/`.
- `origin/codeocean_capsules` is the source of truth (spatial_ingest lives there); pull before starting.
- As of 2026-11-05 this workstation had no gh or git credentials, so pushes failed; the user pushes.
- The user publishes capsules in Code Ocean.
- Before publishing:
  - rebuild the environment so the pins apply (this workstation had annizarr 0.2.3 while 0.2.4 was
    pinned);
  - leave no stray scratch or prototype scripts in `code/`;
  - check `.gitignore` covers `.claude`, `.vscode` and caches.
- `CLAUDE_CONFIG_DIR` is `/root/capsule/code/.claude` on purpose: it persists across CO sessions. It is
  git-ignored; never commit or export it.

## 1. Repo layout and deployment

```
capsules/
├── CLAUDE.md               this guide
├── README.md               environment files, how to keep them in step
├── anndata_ingest/         ┐ (was ingest/)
├── spatial_ingest/         │
├── qc/                     │ each folder is one complete capsule root
└── process/                ┘
    ├── spec.md             the capsule's design, decisions, findings (process has none: §10)
    ├── code/               run (entrypoint), <package>/, tests/, README.md
    ├── environment/        requirements.txt, environment.yml (if conda is needed), pixi.toml, postInstall
    ├── .codeocean/         resources.json, app-panel.json
    ├── metadata/
    └── .gitignore
```

- **Development:** in a full DataScales clone (branch `codeocean_capsules`), where every capsule is
  visible side by side; on a workstation it may be at `/scratch/DataScales`.
- **Updating a capsule checkout:** copy the capsule's `code/` entries, `environment/postInstall` and
  `.codeocean/resources.json` over their namesakes in the checkout (e.g. `/root/capsule`), and load
  `requirements.txt` / `environment.yml` onto the page with **Edit Bulk**. Leave data assets, secrets,
  reference clones and the Dockerfile alone. No reset is needed; an environment change applies on the
  next restart (rebuild).
- **Deployment (the user's step):** each folder goes to its own capsule with
  `git subtree push --prefix=capsules/<name> co-<name> master`. Edits made inside Code Ocean come back
  with `git subtree pull`.
- **Shared code:** there is no shared package. Each capsule depends on `annizarr` for store I/O and
  keeps its own copy of the few column names it needs (§2.2).
- **Clutter:** inside deployed capsules, git-ignore reference clones (`code/DataScales/`,
  `code/standardized_notebooks/`).

## 2. Shared conventions

### 2.1 Inputs
| Input | Opened with | Used by |
|---|---|---|
| icechunk repo (data-asset mount, local path, `s3://bucket/prefix`) | `annizarr.Repo(path, origin=...)`, then `open_zarr("r")` at a branch tip | all three |
| plain AnnData `.zarr` (local path, data-asset mount or `s3://`), not in icechunk | `zarr.open_group(path, mode="r")` | `qc` and `process` |

- **Detecting the kind:** `annizarr.Repo.exists(path)`.
- **Supported locations:** local zarr, data-asset zarr, icechunk on S3, icechunk on a data asset.
- **Where results go:**

  | Input | `qc` with `apply` | `process` notebooks |
  |---|---|---|
  | icechunk | a new commit on `main` with only the curated cells (qc §4.10) | a commit on the branch read |
  | plain zarr (writable or not) | a new zarr at the output path | a new full zarr in `/results` |

  - icechunk writes need write access: S3, a writable local repo, or a data-asset mount given its `origin`.
  - Plain zarr writability is probed, not guessed: write and delete a small `.write_probe` key at the
    store root. Data-asset mounts fail the probe.
  - A `qc` report run never needs write access; writability is checked only when `apply` is set.
- **`.h5ad`** (for qc and process): out of scope. Convert it first with `annizarr convert in.h5ad out.zarr`.
- **Same code after opening:** both kinds give a `zarr.Group` root, and everything after that is shared.

Reading rules:
- **Read obs column by column:** `anndata.io.read_elem(root["obs"][col])` for only the columns needed.
  - Never `read_elem(root["obs"])`: per-cell string columns (`cell_name`, `cell_uuid`,
    `original_barcodes`) make full obs about 1 KB per cell.
  - Convert sample, group and label columns to `category` on read.
- **X is CSR only.** Dense X is an error.
  - CPU reads slice `indptr`, `data` and `indices` directly, in bands of whole storage chunks.
  - Outputs always use an int64 `indptr`; int32 overflows at about 2.1B nonzeros, roughly 1M cells.

### 2.2 Column contract
Written per sample by the ingest:

| Purpose | Column | Type |
|---|---|---|
| Sample key | `sample_id` | categorical |
| Group keys | `batch_id`, `pool_id`, `chip_id` | categorical |
| Genes detected | `n_genes_by_counts` | Int64 |
| Total UMIs | `total_counts` | float64 |
| Mito % | `pct_counts_mt` | float64 |
| Doublet score / call | `scrublet_score`, `predicted_doublets` | float64 / Int64. NA = Scrublet failed or found no threshold |
| Cell-type labels | `<model>_label`, `<model>_prob` per CellTypist model; by default `pbmc_clean_AIFI_L1_2024`, `_L2_2024`, `_L3_2024` | categorical / float64 |
| Hashing / ADT QC (report only) | `hto_category` (multiplexed pools), `adt_qc_flag` (CITE-seq) | categorical. Empty for samples without hashing or CITE-seq |

Written by `process`: `obs/leiden` (name set by `leiden_key`; `<name>_<res>` for several resolutions), `obsm/X_pca`,
`obsm/X_pca_harmony` (if integrated; `X_scVI` from the scVI step), `obsm/X_umap` (name set by `umap_key`), `var/highly_variable`, `layers/lognorm`, `uns/process`.

### 2.3 Entrypoint and code style
- **Entrypoint** (ingest, spatial_ingest, qc; `process`: see §10):
  - `code/run` passes app-panel parameters positionally. An empty value falls back to an env var, then a default.
  - One `main.py`, with a `<Capsule>Error(RuntimeError)` (or a user-error base class, see below) for
    user-facing failures: one log line, exit code 1.
  - Every run writes a run JSON under `/results`: parameters, input fingerprint, versions, timestamps,
    peak memory.
- **Modular within each capsule:** small modules with one job each (module lists in each spec, code
  maps in each README).
- **No comments.** Names and short docstrings on public functions carry the meaning.
- **Not overengineered:** no plugin layers or abstract base classes; no options nobody asked for;
  branch on input kind only where icechunk and plain zarr actually differ.
- **Tests sit beside the code** and run against small synthetic stores.

## Modularity and future-proofing

- **Few files, split by function (user, 2026-11-15).** About five modules, readable top to bottom,
  and few comments (only for a non-obvious why). anndata_ingest is the layout reference: `main.py`
  (the run, top to bottom), `config.py` (settings, panel args, logging), `inputs.py` (reading,
  precheck, picking new inputs), `prepare.py` (annotation, gene policy, contract, staging), `store.py`;
  tests in `test_inputs.py`, `test_prepare.py`, `test_run.py` and `conftest.py`. Imports flow one way
  (config ← inputs ← prepare; store standalone; main imports all). spatial_ingest has the same 5 modules
  (2026-11-18, ~6.5k lines): `main.py`, `config.py` (settings, logging, shared records, `UserError`), `inputs.py`
  (discovery in folders/archives, staging, per-sample inventory, count matrices and cells tables streamed to CSR,
  gene policy), `prepare.py` (precheck, reading a sample: SpatialData objects split per sample and bundle elements,
  per-sample QC/CellTypist), `store.py` (store, joint table, repair, locks, element names, S3 sync and lock);
  `vendor/` holds the vendored Atera reader; tests in `test_inputs.py`, `test_prepare.py`, `test_store.py`,
  `test_run.py`, `conftest.py`. Imports flow config ← inputs ← store ← prepare ← main.
  process is consolidated too (2026-11-16): `main.py`, `config.py`, `steps.py`,
  `store.py` (input, where results go, writers), `runtime.py` (Dask-CUDA cluster, time/memory report); tests in
  `test_config.py`, `test_steps.py`, `test_store.py`, `test_run.py`, `conftest.py`.
- **Precheck:** gather every problem before any write, then stop with one error listing them all.
- **Records:** dataclasses between stages, never tuples.
- **One config dataclass.** Env vars are named after its fields (upper case), then app-panel positional
  values override. A test checks `.codeocean/app-panel.json` against the positional order.
- **Logging:** silence known library noise by exact message or logger name, nothing broader. A clean
  run logs no WARNING line.
- **Errors:** one user-error base class, or an explicit tuple of them. Expected errors give one log line
  and exit 1; bugs give tracebacks.
- **Record library versions** in the run JSON.
- **Private APIs of pinned libraries:** avoid them. If unavoidable, isolate them in one module and
  cover them with an import-canary test.
- **Vendored code:** record source, commit and checksums in a `SOURCE.json`.
- **Genes:** one gene policy module (symbol index, Gene Expression filter, species).
- **Near-copies across capsules** (config parsing, annotate) are candidates for a shared package later.
  Don't create one without asking.

## 7. Testing and review protocol

Run all three tracks before calling work done; subagents can run them in parallel.

1. **pytest** against the pinned versions, plus the installed version if it differs.
2. **End-to-end matrix through `code/run`:**
   - happy paths; append and rerun;
   - local zarr and icechunk;
   - S3: bucket `s3://immunology-datascales-temp-3588`, under a unique test prefix, deleted afterwards;
   - bad inputs: each must give one clean error and a non-zero exit.
3. **Critiques:**
   - software: modularity, future-proofing, correctness, test gaps, publish hygiene;
   - user: README, panel labels and help, log noise, what lands in `/results`.

- Split findings into **clear fix** (apply it) and **decision** (ask the user).
- Point stores and outputs at a scratch directory, never `/results` or `/data`. On a workstation `/data`
  itself can be writable.
- Per-capsule test lists are in each spec. `process` runs its notebooks headless with
  `code/notebooks/test_notebooks.sh` and its pipeline with `code/test_gpu.sh`, on the GPU capsule.

## 6. Environments

### 6.1 How a capsule's environment is built
- **Files, edited by hand.** Packages are `environment/requirements.txt` (pip) and, when conda packages
  are needed, `environment/environment.yml`. Load them onto the environment page with its bulk editor
  (gear icon → **Edit Bulk**, paste or drop the file). `environment/postInstall` installs only Claude
  Code. The base image and VS Code extensions are set on the page. There is no generator.
- **Code Ocean generates `environment/Dockerfile` from them:** one `RUN` per page installer, in this
  order: `conda install -y …` (conda-forge on the mambaforge image), `pip3 install -U …`, the VS Code
  extensions, then `COPY postInstall` + `RUN /postInstall`. Its `# hash:` line is Code Ocean's own, so
  never edit the Dockerfile by hand.
- **Only `postInstall` is copied into the build.** Anything it needs (e.g. a pinned package list) is
  written inside the script.
- **Everything installs into the image at build time.** A change applies only after a rebuild (a
  restart). Page packages install before `postInstall` runs. Prefer a prebuilt conda package over a pip
  source build: `annoy` has no Linux wheel, so it comes from conda (`python-annoy`) before pip installs
  scrublet, and no `g++` is needed.
- **Same path, different storage.** A build installs into the image's Python (`/opt/conda` on the CPU
  image, `/opt/venv` on the GPU image) as read-only image layers. A `pip install` in a running capsule
  lands in the container's writable layer: about 5 GB, reset every run, absent from Reproducible Runs.
  Anything a capsule needs goes in its environment, never a runtime install or `/scratch`.
- **Hand edits:** copying a changed `environment.json` into a running capsule's checkout does **not**
  change the page (verified 2026-10-22: the rebuild used the page and overwrote the checkout file). Make
  changes on the page, then update `requirements.txt` / `environment.yml` to match. The repo keeps no
  copy of `environment.json`.
- **Keep the files in step** every so often, and before any release: the page can change without them
  (an edit on the page, or packages Code Ocean picks up from a session).
  1. Read the capsule's current packages: the page's **Edit Bulk** text for pip3 and conda, or
     `/root/capsule/.codeocean/environment.json` in a running capsule (regenerated from the page on
     every build, so editing it changes nothing).
  2. Update `<capsule>/environment/requirements.txt`, `environment.yml` and `pixi.toml` to match
     (`process`: `code/environment/`). File roles and use outside Code Ocean: [README.md](README.md).
- **Getting them into a capsule:** copy them into the checkout (§1), or subtree-push the capsule folder.

### 6.2 Packages
- **Shared pins**, the same everywhere: `annizarr 0.2.4`, `icechunk 2.3.0`, `boto3`, `zarr 3.4.1`,
  `anndata 0.13.4`, `numpy 2.4.6`, `pandas 3.0.6`, `scipy 1.18.1`, `h5py 3.16.0`, `pytest`. Change one
  in every capsule at once (annizarr moved from 0.2.3 to 0.2.4 by 2026-11-15).
- **What each capsule adds:**

  | Capsule | Installed by | Adds |
  |---|---|---|
  | ingest | page (`requirements.txt`, `environment.yml`) | scanpy, celltypist, scrublet (pip); `python-annoy`, `gh` (conda) |
  | qc | page (`requirements.txt`) | scanpy, leidenalg, igraph, umap-learn (downsampled embedding); pyarrow, matplotlib, seaborn |
  | process | page (`requirements.txt`) | RAPIDS, torch and scvi-tools, pinned (§10); squidpy, spatialdata, omnipath for the tutorial notebooks |
  | spatial_ingest | page (`requirements.txt`) | spatialdata 0.8.0, spatialdata-io 0.7.1 (+ vendored Atera reader), scanpy, celltypist, geopandas, pyarrow, s3fs (spatial_ingest spec §3S.11) |

- **CPU capsules:** transitive dependencies aren't locked. A released image is built once and frozen; if
  a rebuild breaks, pin the offending package on the page.
- **`process`: CUDA 13, from PyPI only, on the page.** Pins and verification: §10.
- **numpy 2.4.6 everywhere:** cudf 26.10 needs `numba < 0.65`, and numba 0.64 supports only
  `numpy < 2.5`. The CPU capsules resolve with it too.

### 6.3 Base images
- **CPU capsules:** `codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04` (conda Python 3.12 in `/opt/conda`).
- **`process`:** `codeocean/cuda12-runtime-uv:main-5692e50` (uv, Python 3.13 in the venv `/opt/venv`);
  driver and GPU details in §10.
- **annizarr:** 0.2.4 from PyPI (0.2.3 before). It imports on Python 3.12 (the `TypeIs` fix is in), and the ingest's
  63 tests pass against it with numpy 2.4.6.

### 6.4 Dev tooling (every image)
- **On the page:** `pytest` (every capsule's `requirements.txt`) and the VS Code extensions.
- **In `postInstall`, last in every capsule:** Claude Code.
  - `postInstall` runs at image build time, before `/root/capsule` is mounted, so anything it writes
    under `/root/capsule` is hidden at runtime. The binary installs to the image's `~/.local/bin`.
  - `~` is part of the image layer and is reset every run, so Claude Code's state (settings, history,
    sessions) goes to `CLAUDE_CONFIG_DIR=/root/capsule/code/.claude`, exported in `~/.bashrc`, on the
    persistent capsule volume. Git-ignore `code/.claude/` in every deployed capsule.

### 6.5 Resource classes
| Capsule | Class |
|---|---|
| ingest | `medium`; it depends on the largest sample, not the store. About 1 minute per 30k-cell sample with CellTypist and Scrublet |
| qc | `medium`, for both modes. The report reads only a few obs columns plus a fixed-size embedding sample; apply streams X out of core (qc §4.10). Apply is I/O-bound, roughly 12 minutes per 1M cells from local disk, slower from S3 |
| spatial_ingest | `large` (64 GB). One sample in memory at a time; the matrix is streamed (measured peaks 8–11 GB on Xenium 5K / Atera kidney samples). `/scratch` needs the S3 store copy plus about 2.3× the largest zipped bundle |
| process | A GPU machine (the T4 has 15 GB). One GPU with ≥ 15 GB VRAM handled 5M cells in about 15 minutes in the notebook's benchmark, on a faster GPU than a T4; above about 13M cells, ≥ 30 GB VRAM. Host RAM must absorb the spill |

## 8. Build order
1. annizarr 0.2.3 on PyPI, with the `TypeIs` fix. **Done.**
2. `ingest`: remove the whole-store analysis **(done)**, move its environment to the page (§6)
   **(copied into the checkout; rebuild pending)**, and rebuild a store from samples.
3. `qc`, report mode: icechunk first, then plain zarr. Review the report on the rebuilt store.
4. annizarr `subset` and `Repo.garbage_collect`: **not in 0.2.4**; qc apply streams its own subset
   meanwhile (qc §4.10).
5. `qc`, apply mode: icechunk (a commit on `main`) first, then plain zarr.
6. `process`: built as a pipeline on 2026-10-30, **reworked into an environment + notebooks capsule on
   2026-11-02** (§10).
7. Subtree-push each capsule to its own Code Ocean capsule.
8. `anndata_ingest`: SpatialData-table inputs, gene-panel fitting, source paths in commits, custom
   `.pkl` **(done 2026-10-30)**; S3 creation without `CREATE`, dropping results on append **(done 2026-11-02)**.
9. `spatial_ingest`: **built 2026-11-11** (spatial_ingest spec §3S.0); robustness rework and crash/concurrency safety 2026-11-14 (b0fbe83, 806fa97); consolidated 2026-11-16; release review (software + user critiques) 2026-11-16.

## 9. Open questions
- **annizarr 0.2.4 shipped without `subset` and `Repo.garbage_collect`** (checked 2026-10-30). It
  renamed `add_expr` to `add_layer`, which now only copies a matrix in another layout (no
  normalization), split the matrix copy into `_writers/_copy.py`, and caps its CPU count by the cgroup
  quota. `process` works on it unchanged; qc apply still needs the two below, now for a later release.
- **Planned for annizarr:** `subset(store, *, output, keep, branch, commit)` mirrors `rechunk`'s
  streaming copy with a row mask; `Repo.garbage_collect(older_than)` wraps icechunk's. Both keep the
  capsules free of store internals (no private `_writer_repo()`).
- **Append new genes:** the user will add it to annizarr; until then ingests fit samples to the store's genes.
- **First GPU run:** the environment builds and works on the T4. The reworked notebooks still need a
  headless run there (`code/notebooks/test_notebooks.sh`).
- **Using RAPIDS in other capsules:** a capsule with its own environment could install
  `process/code/environment/requirements.txt` into a separate venv (`/opt/rapids`) from `postInstall`,
  with a Jupyter kernel. Untested whether Code Ocean exposes the GPU to a non-GPU base image.

## 10. The process capsule

Design summary; the file-level code map is in `process/code/README.md`. Earlier change logs are in git history
(`process/spec.md`, `process/rework_spec.md`, removed 2026-11-16).

- **What it is, first:** the pinned RAPIDS GPU environment (`code/environment/`, copyable to other capsules)
  and two interactive notebooks (out-of-core scRNA, spatial walkthrough, adapted for one T4). An environment
  and notebooks capsule, not automation; first-time users get no dead ends.
- **Optional pipeline:** `code/run`, from the App Panel (named parameters) or a terminal: PCA (± harmony) or
  scVI (train), then neighbors, UMAP and leiden at one or more resolutions, optionally on an obs subset. One
  `Config` dataclass; the panel sets the common fields, the rest (cluster, GPUs, scVI knobs) are flags or
  upper-case environment variables. A Reproducible Run with no input exits 0 with usage.
- **Five modules:** `main.py` (run; `EMBEDDERS` registry), `config.py`, `steps.py`, `store.py` (input, output
  planning, writers), `runtime.py` (Dask-CUDA cluster, step report); tests beside them.
- **Output (one box, `output` + `output_format`):** empty or the placeholder `s3://bucket/prefix` = a copy in
  `/results` (`<stem>.processed`, never overwritten); a bare name or a path under `/results` = a copy there;
  `s3://` or any other path = written to directly: icechunk input commits onto its own repo, zarr input gets a
  new store or results written into an existing zarr with the same cells and genes. Checked before any GPU work;
  S3 needs an AWS secret with write access (a put/delete probe runs first). If the icechunk branch gets another
  commit during the run, nothing is saved (`BranchMoved`).
- **Results:** `umap_key` / `leiden_key` set the names (custom names are refused if the store already has
  them from something else). scVI: the Dask cluster is closed before training; the model goes to
  `/results/scvi_model/`.
- **Environment:** CUDA 13 wheels from PyPI only, on the page. `requirements.txt` (+ `pixi.toml`, kept in
  `code/environment/`) pins the shared pins, RAPIDS 26.10 cu13, `rapids-singlecell-cu13 0.18.1`, torch,
  `scvi-tools 1.5.1`, and the packages that must agree (`numba`, `numba-cuda`, `cupy-cuda13x`, `cuda-toolkit`,
  `dask`, `distributed`); the rest resolves at build time (246 packages with squidpy, spatialdata, omnipath).
  Base image `codeocean/cuda12-runtime-uv:main-5692e50` (Python 3.13 in `/opt/venv`); the wheels bring CUDA 13,
  so the host driver matters (measured 610.57.04 on a Tesla T4, 15 GB, compute capability 7.5, the oldest CUDA
  13 supports). The page's `pip3` is `/opt/venv/bin/pip3`.
- **Tests:** `python -m pytest process/tests` (101, GPU capsule); `./test_gpu.sh` adds the notebooks and the
  pipeline end to end; `notebooks/test_notebooks.sh` runs each notebook headless. Real S3 is untested.
- **Open:** `resources.json` is still `"xsmall"` and needs a GPU class; gather every precheck problem into one
  error; one user-error base class; `Results` (the pipeline's schema) could leave `store.py` if a second output
  type appears.
