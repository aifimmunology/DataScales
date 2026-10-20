# Single-cell curation capsules — spec

Three Code Ocean capsules that take HISE scRNA samples from raw files to a QC-curated store:

| Capsule | Writes? | Job |
|---|---|---|
| `ingest/` | yes (`main` branch) | Append samples into the raw, uncurated icechunk store (exists today) |
| `qc_check/` | **never** | Per-sample and per-cell-type QC report for proposed thresholds; emits the decisions + cell mask |
| `qc_apply/` | yes (`qc` branch, or a new file) | Writes exactly what a reviewed `qc_check` run produced, nothing else |

The person in the loop sits between the two QC capsules: run `qc_check`, read the report, edit the
thresholds, rerun `qc_check` until satisfied, then run `qc_apply` on that run's output. Cell
labeling and cluster-level doublet cleanup (standardized notebooks 06–08) are a later, separate stage.

Source notebooks: `standardized_notebooks/scRNA/03-dataset-qc.ipynb` (report) and
`05-apply_qc_filters.ipynb` (thresholds). Those notebooks use the column names `pct_counts_mito` and
`predicted_doublet`. The capsules use the store's names below; don't copy their filter code verbatim.

---

## 1. Repo layout and deployment

```
capsules/
├── spec.md                 this file
├── ingest/                 ┐
├── qc_check/               ├ each folder is one complete capsule root
└── qc_apply/               ┘
    ├── code/               run (entrypoint), <package>/, tests/
    ├── environment/        postInstall (and the Dockerfile Code Ocean generates)
    ├── .codeocean/         environment.json (the environment page), resources.json
    ├── metadata/
    └── .gitignore
```

- **Development:** develop in a full DataScales clone, where every capsule is visible side by side.
- **Deployment:** each folder goes to its own capsule with
  `git subtree push --prefix=capsules/<name> co-<name> master`.
  - The capsule therefore holds only its own code.
  - Edits made inside Code Ocean come back with `git subtree pull`.
- **Shared code:** there is no shared package. Each capsule depends on `annizarr` for store I/O and keeps its own copy of the few constants it shares with the others (§2).
  - Each copy carries a comment `# keep in sync with capsules/ingest/code/icechunk_ingest/contract.py`.
  - A repo-level CI check that the copies match is optional.
- **Clutter:** inside the deployed capsules, git-ignore any reference clones (e.g. `code/DataScales/`, `code/standardized_notebooks/`).

---

## 2. Shared conventions

### Inputs
Every capsule accepts one store or file:

| Input | How it's opened |
|---|---|
| icechunk repo (data-asset mount, local path, `s3://bucket/prefix`) | `annizarr.Repo(path, origin=...)`, read at a branch tip (default `main`) |
| `.zarr` AnnData store | `annizarr.open_source` |
| `.h5ad` | `annizarr.open_source` |

Rules for reading:
- **Read obs only, unless something is missing.** QC needs only obs, plus `obsm/X_umap` for plots if present.
  - Open lazily and load `obs` alone.
  - Only read X when per-cell metrics are missing (§3.3).
  - This keeps `qc_check` cheap on a store of millions of cells.
- **Never load X in full.** When X must be read, use row chunks.

### Column contract
Copied from the ingest `contract.py`:

| Purpose | Column | Type |
|---|---|---|
| Sample key | `sample_id` | str/categorical |
| Genes detected | `n_genes_by_counts` | Int64 |
| Total UMIs | `total_counts` | float64 |
| Mito % | `pct_counts_mt` | float64 |
| Doublet score / call | `scrublet_score`, `predicted_doublets` | float64 / Int64 (NA = Scrublet failed) |
| Cell-type label | `celltypist_label` (+ `celltypist_prob`, `celltypist_model`) | str/categorical |
| Embedding | `obsm/X_umap` | float32 |

Each capsule takes the sample column as a parameter (default `sample_id`), so h5ad files from other sources work too.

### Entrypoint style
These follow the ingest capsule:
- `code/run` passes app-panel parameters positionally. An empty value falls back to an env var, then to a default.
- One `main.py`, with a `<Capsule>Error(RuntimeError)` for user-facing failures. These are logged as one line and give exit code 1.
- Outputs go under `/results`. Each run writes `run.json`, holding the parameters, input fingerprint, versions and timestamps.

---

## 3. `qc_check` — the report (read-only)

### 3.1 App panel
| # | Parameter | Default | Notes |
|---|---|---|---|
| 1 | input | — | Store or file (§2) |
| 2 | decisions | *(none)* | Path to a `qc_decisions.json` (an attached data asset, or a previous run's results). None → defaults, with any decisions already stored in the input carried forward (§3.5) |
| 3 | sample column | `sample_id` | |
| 4 | per-cell-type QC | `false` | Optional in v1. Turns on §3.4 and the `cell_types` overrides |
| 5 | label column | `celltypist_label` | Used only when per-cell-type QC is on; the section runs only if this column exists (§3.4) |
| 6 | branch | `main` | icechunk only |
| 7 | outlier MADs | `3` | Sample-level outlier flag (§3.3) |

Default thresholds, from notebook 05: `min_genes 200`, `max_genes 5000`, `max_mito 10`, `filter_doublets true`.

### 3.2 Decisions file (`qc_decisions.json`)
This file is the contract between `qc_check` and `qc_apply`.

```json
{
  "version": 1,
  "input": {"kind": "icechunk", "location": "s3://…", "branch": "main",
            "snapshot": "1CECHNKREP0F1RSTCMT0", "file_sha256": null},
  "sample_column": "sample_id",
  "label_column": "celltypist_label",
  "defaults": {"min_genes": 200, "max_genes": 5000, "max_mito": 10, "filter_doublets": true},
  "cell_types": {"Platelets": {"min_genes": 50}},
  "samples": {
    "SP09002": {"exclude": true, "note": "low viability, see per-sample report"},
    "SP09003": {"max_mito": 15}
  },
  "pending": ["SP09010", "SP09011"],
  "reviewer": "", "note": ""
}
```

Meaning of the fields:
- **Precedence:** `defaults` → `cell_types[label]` → `samples[id]`. The most specific entry wins, key by key.
  - `exclude: true` on a sample removes all its cells.
  - `cell_types` entries apply only when per-cell-type QC is on and the label column exists. Otherwise they're ignored and the report says so.
- **`pending`:** samples present in the input but not yet reviewed. They get the defaults in the preview.
- **Cell rules:** these follow notebook 05's strict inequalities.
  - A cell **passes** if `n_genes_by_counts > min_genes`, `n_genes_by_counts < max_genes` and `pct_counts_mt < max_mito`.
  - With `filter_doublets`, it must also not have `predicted_doublets == 1`.
  - `predicted_doublets` NA (Scrublet failed) **passes**. Those cells are counted and shown in the report as "doublet status unknown".
- **`input` block:** the fingerprint. For icechunk it's the snapshot id of the branch tip; for files it's the SHA-256.

### 3.3 Per-sample section
- **Per-cell metrics.** If `n_genes_by_counts`, `total_counts` or `pct_counts_mt` are missing (a typical external h5ad), they are computed from X in row chunks.
  - The definitions match `scanpy.pp.calculate_qc_metrics`, with mito genes being those whose names start with `MT-`.
  - They go into `cell_qc.parquet` (§3.6) so `qc_apply` never recomputes anything.
  - If X isn't raw counts, stop with an error, as the ingest does.
- **Per-sample metrics** (`sample_qc.csv`):
  - cell count;
  - median and MAD of genes, UMIs and mito %;
  - doublet rate;
  - % of cells failing each rule and % failing overall under the current decisions;
  - cell-type composition (if labels exist);
  - status (`reviewed` / `pending` / `excluded`).
- **Outlier flag:** a sample is flagged when any of its medians, its doublet rate or its fail % lies more than `outlier MADs` from the cohort median. Flags are suggestions only; they never change a decision.
- **Plots** (PNG, one figure per metric, samples on the x-axis, sorted by flag then id):
  - violins of genes, UMIs and mito %, with the threshold lines drawn;
  - cells per sample, before and after filtering;
  - fail % per sample stacked by reason;
  - doublet rate;
  - mito % against genes for each flagged sample;
  - a UMAP coloured by pass/fail reason, if `X_umap` exists. Downsample to ≤ 200k points.

### 3.4 Per-cell-type section (optional in v1)
- **When per-cell-type QC is off (the default):** skip the section silently. The per-sample report and `qc_apply` don't depend on it.
- **When it's on but the label column is absent:**
  - skip the section;
  - log a warning and write it at the top of `report_summary.txt`: *"No `celltypist_label` column; per-cell-type QC skipped. Label the store first (ingest `--annotate-existing`, §5) or pass a label column."*
  - the run still succeeds.
- **When present** (`celltype_qc.csv`, plus plots):
  - per label: cell count, the same medians and MADs, and fail % by reason under the current decisions;
  - a heatmap of fail % for each label × sample;
  - violins of genes and mito % per label, with that label's effective thresholds drawn;
  - `celltypist_prob` distribution per label, which flags labels the model is unsure about.
- **Why this section matters:** it catches thresholds that are wrong for a whole cell type, for example a `min_genes` that wipes out platelets or a `max_mito` that strips one type disproportionately. The fix is a `cell_types` entry in the decisions file.

### 3.5 Incremental runs (icechunk)
- **If the repo has a `qc` branch:** its `uns/qc/decisions` are the starting point.
  - Every reviewed sample keeps its entry.
  - Samples on `main` that `qc` hasn't seen go to `pending`.
- **If a decisions file is passed:** it overrides the stored decisions key by key.
- **The input fingerprint is always the current `main` tip**, so the output is ready to apply against what's there now.

### 3.6 Outputs (`/results`)
```
qc_decisions.json      the effective decisions (edit this, rerun; or hand to qc_apply)
cell_qc.parquet        per cell: obs index, sample, label, the 3 metrics, doublet call,
                       qc_pass (bool), qc_fail_reason (str: "" | low_genes | high_genes |
                       high_mito | doublet | sample_excluded; first rule that failed, in this order)
sample_qc.csv          §3.3
celltype_qc.csv        §3.4 (absent when skipped)
plots/samples/*.png
plots/celltypes/*.png
report_summary.txt     headline counts, warnings, flagged samples, pending samples
run.json
```
- `cell_qc.parquet` carries the SHA-256 of the canonical `qc_decisions.json` and the input fingerprint in its file metadata. `qc_apply` checks both.
- To feed `qc_apply`, capture the reviewed run's results as a data asset.

---

## 4. `qc_apply` — write the curated version

### 4.1 App panel
| # | Parameter | Default | Notes |
|---|---|---|---|
| 1 | input | — | The same store or file `qc_check` read |
| 2 | qc_check results | — | Data asset holding `qc_decisions.json` + `cell_qc.parquet` |
| 3 | output | `qc` (icechunk) / `<input stem>.qc.h5ad` or `.qc.zarr` | Branch name, or output file name |
| 4 | reviewer | — | Required; recorded in the commit/metadata |

### 4.2 Refuse to run when
The point of these checks is that what was reviewed is what gets applied:
- **The input changed since the report.** The fingerprint doesn't match: the icechunk branch tip moved, or the file hash differs. *"Input changed since this report; rerun qc_check."*
- **The decisions file was edited after the report.** Its hash doesn't match the one in `cell_qc.parquet`. *"Decisions edited after the report was made; rerun qc_check to preview them."*
- **There are pending samples.** `pending` is non-empty. Apply needs every sample reviewed, so the reviewer moves each into `samples` (an empty `{}` means "defaults are fine") or excludes it.
- **The cells don't match.** The obs index in `cell_qc.parquet` must match the input's exactly.

`qc_apply` doesn't evaluate thresholds itself. It writes `cell_qc.parquet` as produced, so the two capsules can never disagree about which cells pass.

### 4.3 Output
- **icechunk:** the `qc` branch is always **`main`'s tip plus one QC commit**.
  - Create or reset `qc` to point at the fingerprinted `main` snapshot, then commit on it:
    - obs columns `qc_pass` (bool) and `qc_fail_reason` (categorical);
    - `uns/qc` = {decisions, per-sample counts, `cell_qc` sha, reviewer, source snapshot}.
  - The commit metadata records `pipeline: qc_apply`, the source snapshot, the decisions sha, the reviewer and the pass/total counts.
  - Tag each applied version `qc-<YYYYMMDD>-<n>`, so earlier curations stay addressable after `qc` is reset.
  - X and the other obs are never touched.
  - Columns go on a branch, not `main`, because the ingest's `check_schema` refuses appends when `main` has columns the ingest didn't write.
- **h5ad / zarr:** write a **new** file in `/results`. The input is never modified, and data assets are read-only anyway.
  - The file holds the passing cells only, with `qc_pass` and `qc_fail_reason` dropped and `uns/qc` added.
  - A `removed_cells.csv` lists the obs of every removed cell, with its reason.
  - A row-chunked subset through annizarr avoids loading X in full.
- **Both:** `apply_summary.csv` (per sample: before, after, removed by reason) and `run.json`.

### 4.4 Consumers
Downstream code reads the `qc` branch and subsets on `qc_pass`. A later export step (or labeling-stage capsule) can produce a physically filtered h5ad from it when someone needs a plain file.

---

## 5. Ingest addition: label an existing store (`--annotate-existing`)

Not possible today: CellTypist only runs inside `prepare_sample` for new samples, and a store
without the `celltypist_*` columns fails `check_schema` when `RUN_CELLTYPIST=true`. This is the
case for `sample_linked_icechunk`.

Proposed mode, run with no new samples (app-panel toggle "annotate existing cells"):
1. **Find the cells.** Open `main` and find the cells without labels: all of them if the columns are absent, otherwise those whose `celltypist_label == ""`.
2. **Label them one sample at a time,** which keeps memory bounded:
   - read that sample's rows of X;
   - normalize to 1e4 and log1p, exactly as `annotate.run_celltypist` does, by calling it;
   - collect the labels.
3. **Write the columns in the same single ingest commit.**
   - `celltypist_label`, `celltypist_prob` and `celltypist_model` are written for every cell, added to `column-order`.
   - The commit metadata records `annotated_samples`.
   - From then on the store's schema includes CellTypist, so normal runs with `RUN_CELLTYPIST=true` append cleanly.
4. **Mixing models:** if the model differs from the one already in the store, refuse unless `--relabel` is given, which relabels every cell.

This also answers the pending CellTypist question for the production store, and it unblocks
`qc_check`'s per-cell-type section.

---

## 6. Environments

### 6.0 How a capsule's environment is built
- **The environment page and `.codeocean/environment.json` are the same data.** The page's installers (pip3, apt, VS Code extensions) are stored in that file.
- **Code Ocean generates `environment/Dockerfile` from it.**
  - The file starts with a `# hash:` header.
  - It has one `RUN pip3 install …` per installer.
  - It ends with `COPY postInstall` + `RUN /postInstall`.
- **Everything is installed into the Docker image at build time, not when a run starts.**
  - Page packages and `postInstall` alike install into the image's conda Python (`/opt/conda/lib/python3.12/site-packages`).
  - Runs and workstations start containers from that image.
  - So a change to either takes effect only after the environment is rebuilt.
- **Editing `environment.json` by hand** (e.g. through a subtree push) should show up on the page and trigger a rebuild. Verify this once on the first push, as with the push itself.

### 6.1 Recommendation: Code Ocean environment page, exact pins, one per capsule
- **Packages go on the environment page.** It writes `.codeocean/environment.json`, so it's versioned in git and travels with each capsule folder through the subtree.
  - Use the `pip3` installer with exact versions.
  - Use the `apt` installer for system packages.
- **`postInstall` shrinks to what the page can't do** (§6.4).
- **Transitive dependencies aren't locked.** That's acceptable because a released capsule's image is built once and frozen. Pins only matter on rebuild, and exact direct pins keep rebuild drift small. If a rebuild ever breaks, pin the offending transitive package on the page.
  - pixi plus lockfiles is the alternative if exact rebuilds become a requirement.
- **Package extras:** list them explicitly instead of relying on the page to support `[extras]`. `annizarr[icechunk]` = `annizarr` + `icechunk` + `boto3`.
- **Base image:** all three use the same one, `codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04`.
  - annizarr is installed from PyPI like any other pin. 0.2.3 must import on Python 3.12, and PR #14's `from typing import TypeIs` doesn't, so fix it before releasing. Until 0.2.3 is published, the pin won't resolve and the image won't build.

### 6.2 Core pins — identical in every capsule
These three capsules read and write the same store, so these versions move together. Change them in
one PR that touches all three `environment.json` files, and rebuild all three.

| Package | Version |
|---|---|
| annizarr | 0.2.3 |
| icechunk | 2.3.0 |
| boto3 | *(pin to what 0.2.3 resolves to on first build)* |
| zarr | 3.4.1 |
| anndata | 0.13.4 |
| numpy | 2.5.4 |
| pandas | 3.0.6 |
| scipy | 1.18.1 |
| h5py | 3.16.0 |
| pyarrow | *(pin on first build; needed for `cell_qc.parquet`)* |
| awscli | 1.46.1 (already on the page) |

### 6.3 Per-capsule packages
| Capsule | pip3 | apt |
|---|---|---|
| ingest | scanpy 1.12.4, scrublet 0.2.3, annoy 1.17.3, celltypist 1.7.1, harmonypy 2.1.0, leidenalg 0.12.0, igraph 1.0.0, umap-learn 0.5.12 | g++ (annoy has no wheel and compiles at install) |
| qc_check | matplotlib, seaborn *(pin on first build)* | — |
| qc_apply | — (core only) | — |

`qc_check` and `qc_apply` deliberately don't need scanpy. The metrics fallback is plain scipy (§3.3),
so these images stay small and build quickly.

### 6.4 Dev tooling — kept in every image
- **On the page:**
  - `pytest` (≥8) on the pip3 installer;
  - the VS Code extensions already listed.
- **In `postInstall`:**
  - Claude Code only.
  - Pin it: `curl -fsSL https://claude.ai/install.sh | bash -s <version>`. Bump the version deliberately, like any other pin.
- **The ingest's current `postInstall` becomes:** the apt line moves to the page, the pip block moves to the page, and only the Claude Code install remains.

### 6.5 Open environment questions
- Resource class:
  - `qc_check` on obs alone fits `2xsmall`.
  - The metrics fallback on a big h5ad and the h5ad/zarr output in `qc_apply` may need more.
  - Start at `small` and measure.

---

## 7. Tests (per capsule, `code/tests`)

**`qc_check`**
- precedence (defaults → cell type → sample), including `exclude`;
- the strict-inequality boundaries;
- NA doublet calls pass and are counted;
- with per-cell-type QC off, the section and `cell_types` overrides are ignored; with it on and no label column, the section is skipped and the warning is written;
- the metrics fallback equals `scanpy.pp.calculate_qc_metrics`. That is a test-only comparison, with scanpy pinned in a test fixture or comparing against stored values;
- the decisions sha is written into `cell_qc.parquet`;
- incremental runs carry stored decisions forward and new samples land in `pending`;
- the run never writes to the input. Run it against a read-only copy.

**`qc_apply`**
- refuses when:
  - the snapshot moved;
  - the decisions were edited;
  - samples are pending;
  - the obs index doesn't match;
- the `qc` branch's parent is the fingerprinted `main` snapshot;
- `main` is untouched, and an ingest append to `main` still works afterwards;
- the tag is created;
- h5ad output has exactly the passing cells, and `removed_cells.csv` holds the complement;
- a rerun after new samples reach `main` resets `qc` onto the new tip.

**`ingest`**
- `--annotate-existing` on a store without CellTypist columns:
  - adds the columns in one commit;
  - a later normal run with `RUN_CELLTYPIST=true` appends cleanly;
  - a model mismatch without `--relabel` is refused.

---

## 8. Build order

1. Ingest: `--annotate-existing` (§5), then move the environment to the page (§6). Tag annizarr 0.2.3 with the `TypeIs` fix.
2. `qc_check` on h5ad and icechunk; review the report on `sample_linked_icechunk`.
3. `qc_apply`, using the icechunk path first.
4. Subtree-push each capsule to its own Code Ocean capsule. Optionally, add a GitHub Action that pushes on merge.

## 9. Open questions
- **Branch name and permissions:** is `qc` the right name, and should only certain people be allowed to run `qc_apply` (Code Ocean sharing on that capsule)?
- **Pending samples:** should `qc_apply` really refuse while any sample is pending (current spec), or allow applying with pending samples left out of the `qc` branch until reviewed?
