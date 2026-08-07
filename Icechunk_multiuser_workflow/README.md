# Icechunk Multi-User Workflow

Workflows and experiments for concurrent, versioned access to Zarr stores via Icechunk —
multiple users reading/writing snapshots, branching, and committing against a shared repo
(local now, GCS-backed later).

## `icechunk_rapids_demo.py`

Icechunk + Zarr + RAPIDS demo: two "users" do versioned, branch-based GPU
single-cell work on one large dataset, with unchanged data shared across snapshots
(no copies). GPU/CUDA only — runs on the server, not the laptop.

- **Step 0 — ingest.** Create the repo and stream an existing anndata CSR zarr (raw
  counts) onto `main`.
- **Step 1 — User 1 normalizes.** Branch, normalize + log1p *all* cells (multi-GPU
  dask, streamed block-by-block so host RAM ≈ one block), write to a **new** layer
  `layers/norm` — raw counts stay in `X` untouched. Commit, push to `main`.
- **Step 2 — User 2 subsets.** Branch, pull one cell type from `layers/norm`,
  run HVG → PCA → neighbors → UMAP → Leiden in-memory on a single GPU, and commit
  results into a scoped `analyses/<cell_type>` group (`X`/`layers` never rewritten).
- **Step 3 — show.** Open `main` and print the final store layout + commit history.

Reruns are incremental: an existing `REPO_PATH` is reused and any step already
committed on `main` is skipped. Delete `REPO_PATH` to force a fresh run.

Run with `pixi run python icechunk_rapids_demo.py` on the GPU node. Paths, cell-type
column/value, and parallelism knobs are set in the config block at the top of the script.
