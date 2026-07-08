---
name: icechunk
description: >-
  Use when reading or writing Zarr data through Icechunk — opening a Repository, working in a
  Session, committing/branching/tagging snapshots, configuring Storage (local/S3/GCS), or
  benchmarking Icechunk vs. plain Zarr stores. Invoke before writing any `icechunk.*` call.
  NOTE: icechunk source is vendored (v2.0.6) but it is NOT yet a pixi/pyproject dependency —
  adding it is still a real setup step. Always verify the API against the vendored source,
  since the Python binding over the Rust core changes quickly.
---

# Icechunk — transactional, versioned Zarr storage backend

Icechunk is a transactional storage engine that presents as a Zarr store: chunks are written
into an object store with git-like snapshots/branches/tags and atomic commits. DataScale is
**adopting it as a backend** — treat its conventions as first-class alongside zarr.

> ⚠️ **Setup gap:** the source is vendored (v2.0.6) but `icechunk` is **not** a pixi/pyproject
> dependency yet — `import icechunk` will fail in the current env. Add it to `pyproject.toml`
> + pixi before writing runnable code; until then the vendored source is read-only reference.

## Where to look (ground truth)

Vendored from `github.com/earth-mover/icechunk` @ **v2.0.6** (Python binding over a Rust core).

- **Python binding source:** `.claude/vendor/icechunk/src/icechunk/` — typed Python API
- **Type stub:** `.claude/vendor/icechunk/src/icechunk/_icechunk_python.pyi` (signatures the
  Rust extension exposes — often the fastest place to read the real API surface)
- **Rust core (behavior of record):** `.claude/vendor/icechunk/src/rust-core/`
- **Docs:** `.claude/vendor/icechunk/docs/docs/` (mkdocs site source) + `docs/README.md`
- **Installed check (once it's a dep):** `pixi run python -c "import icechunk; print(icechunk.__version__)"`

Python modules (real, v2.0.6):

| Concept | File |
|---------|------|
| `Repository` (open/create, branches, tags, snapshots) | `src/icechunk/repository.py` |
| `Session` (read/write view, `commit`) | `src/icechunk/session.py` |
| Zarr store implementation | `src/icechunk/store.py` |
| Storage backends (s3/gcs/local/...) | `src/icechunk/storage.py` |
| Repo/store config | `src/icechunk/config.py` |
| Credentials | `src/icechunk/credentials.py` |
| Conflict detection / resolution | `src/icechunk/conflicts.py` |
| dask / distributed write integration | `src/icechunk/{dask,distributed}.py` |
| Snapshots / history | `src/icechunk/snapshots.py` |

### Grep recipes

```bash
V=.claude/vendor/icechunk/src/icechunk
rg -n "class Repository|def open|def create|def writable_session|def readonly_session" $V/repository.py
rg -n "class Session|def commit|def rebase|def merge" $V/session.py
rg -n "def .*storage|class .*Storage" $V/storage.py        # storage constructors
rg -n "def commit|def writable_session" $V/_icechunk_python.pyi   # exact signatures
```

## Model (confirmed in v2.0.6 source — still verify specifics before coding)

- **Repository** — points at a `Storage` (local dir / S3 / GCS). Open existing or create.
- **Session** — a working view at a branch/snapshot. A **read** session is a snapshot; a
  **writable** session stages changes you then `commit()` (returns a snapshot id). Branches
  and tags name snapshots.
- **Zarr integration** — the session exposes a zarr-compatible store; you `zarr.open_group`/
  `create_array` against it exactly like a plain store. So the `zarr` and `anndata` skills
  still apply for layout/encoding — icechunk only changes *where/how* chunks land and how
  versions are tracked.

## Performance & correctness principles (version-independent)

- **Batch into few, large commits.** A commit per chunk/row-batch is pathological — staging
  and commit have fixed overhead. Write a whole array (or store) in one session, then commit.
- **Separate open + first-read from steady-state** when benchmarking — snapshot resolution and
  manifest fetch are one-time costs.
- **Hold layout constant when comparing** icechunk vs. plain zarr (same chunks, codecs, shards,
  dataset) so you measure the *engine*, not the layout.
- **Concurrency / conflicts:** concurrent writers can conflict at commit; understand the
  rebase/conflict-resolution story in the docs before parallelizing writes.
- **Don't double-count threads:** icechunk + zarr + dask + BLAS all parallelize. See the
  "Silent performance killers" list in [CLAUDE.md](../../../CLAUDE.md).

## ✍️ Maintainer notes

icechunk **2.0.6** is a real dependency now (`icechunk>=2.0.6` in `pyproject.toml`,
importable in the pixi env). The output backend is selected by `cfg.io.backend`
(`"zarr"` default / `"icechunk"`); all the wiring lives in
[datascale/storage.py](../../../datascale/storage.py).

- **Exact version adopted:** 2.0.6 (`pyproject.toml` `dependencies`). `import icechunk` is
  **lazy** (inside `storage.py` functions) so the default zarr path never imports it.
- **Storage target:** **local filesystem only** for now —
  `icechunk.local_filesystem_storage(str(output_path))`. GCS is *scaffolded* in
  `_icechunk_storage` (config fields `icechunk_storage="gcs"`, `gcs_bucket`, `gcs_prefix`)
  but raises until wired; the intended entry point is
  `icechunk.gcs_storage(bucket=..., prefix=...)` (keyword-only).
- **Repo/branch conventions:** one repo == one output store (the `--output` path). Writes
  always go to branch `main` (hard-coded in `open_output_store`; not configurable). No
  tagging yet.
- **Commit granularity:** **one commit per conversion** (after X + metadata + any
  `uns/datascale_sort_index` are written). `finalize()` calls `session.commit(...)`. Do not
  commit per chunk/row-batch.
- **Confirmed API (v2.0.6), as used in `storage.py`:**
  ```python
  import icechunk, zarr
  repo = icechunk.Repository.open_or_create(icechunk.local_filesystem_storage(str(path)))
  session = repo.writable_session("main")
  root = zarr.open_group(store=session.store, mode="w")   # write exactly like a plain store
  # … write_elem / da.store / require_array against `root` …
  snapshot_id = session.commit("datascale convert → …")
  # read side:
  repo = icechunk.Repository.open(icechunk.local_filesystem_storage(str(path)))
  g = zarr.open_group(store=repo.readonly_session(branch="main").store, mode="r")
  ```
- **Parallelism / known constraints:**
  - **No `--backed` with icechunk** — the backed writers fan out to worker *processes* that
    reopen the store by filesystem path (`store_path.store.root`), which `IcechunkStore`
    doesn't expose. `_resolve_backend_cfg` rejects backed + icechunk.
  - In-process writes only: `_resolve_backend_cfg` **forces `cpus=1`** for icechunk so the
    single in-process session isn't written from multiple threads. For true parallel writes
    use `Session.fork()` + `icechunk.dask.store_dask` (changeset merge across workers) — not
    wired yet.
  - LocalFileSystem prints a "not safe for concurrent commits" warning — benign for our
    single-writer path.
- **Conflict / retry:** single-writer, so none yet. `commit(..., rebase_with=, rebase_tries=)`
  exists for when concurrent writers are introduced.
- **Vendored doc pages worth reading first:** `docs/docs/icechunk-python/` —
  `parallel.md` (fork/distributed writes), `version-control.md` (branches/commits).
