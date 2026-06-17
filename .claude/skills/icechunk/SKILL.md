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

## ✍️ Maintainer notes — ADD YOURS (high priority — this skill is the thinnest)

<!-- TODO(you): source is vendored (v2.0.6); fill these once it's a real dependency. -->

- **Exact version adopted** and the `pyproject`/pixi entry you added:
- **Storage target** (local path? S3/GCS bucket + region + auth method?):
- **Repo/branch/tag conventions** for DataScale stores (naming, when to tag a release):
- **Commit granularity policy** (one commit per conversion? per file in a concat?):
- **Confirmed API snippets** (open repo, writable session, write X, commit) — paste working code here:
- **Conflict / retry strategy** for concurrent writers:
- **Vendored doc pages worth reading first:**
