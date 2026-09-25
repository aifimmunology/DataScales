# scizarr-ic — design context for the next agent / maintainer

Decisions made while building this package (Sept 2026), what was tried and dropped,
and how it is used from the Allen Institute Code Ocean capsule. Read this before
extending or merging scizarr-ic into another DataScales tool.

## What the package is

A thin git-like layer over Icechunk for single-cell Zarr stores: `init` (import a zarr
store), `create` (empty repo), `log`, `tree`, `checkout [-b]`, `cherrypick` (branch
reset to a snapshot), `copy` (clone a whole repo), plus a Python `Repo` with
`writable()` / `commit()` / `discard()` / `root()`. It must stay generic:
local directories and `s3://` / `gs://` URIs, credentials from the environment, nothing
tied to one deployment. Keep it small; every command should map to one Icechunk idea.

## Read here, write there — the one non-obvious feature

`Repo(path, origin=None)`: reads use `path`; writes use `origin` if given (argument,
CLI `--origin`, or `SCIZARR_IC_ORIGIN`), else `path`. The origin is opened lazily on
the first write. Once opened it also serves reads in that process, because a read-only
mirror can lag behind a fresh write. `_branch_exists` consults the origin when the read
path lacks a branch HEAD names (stale mirror). A read-only `path` with no origin is
reads-only, and the error names `--origin` and `copy`.

HEAD is a scizarr-ic concept (Icechunk has no current branch): a `scizarr_head` file in
a writable local repo, otherwise a per-user sidecar under `$SCIZARR_IC_HOME`
(default `~/.cache/scizarr_ic`), keyed by the origin so a mirror and its origin share it.

### Tried and removed (do not reintroduce without a reason)

- **Stamping `origin_url` into repo metadata** so a read-only mount could find its
  own bucket. Dropped: outside Code Ocean nobody needs it, and inside, internal
  assets carried a stamp that pointed at a bucket the mount would never reflect.
  The writable location is now always explicit.
- **Frozen-mount detection** via `/proc/mounts` filesystem type (NFS = frozen copy).
  Dropped: deployment heuristic, not package logic.
- **`origin` subcommand**, `SCIZARR_IC_READONLY_PREFIXES`, `Repo.read_only`,
  a `.path` note beside HEAD sidecars, `numpy` as a runtime dependency, live
  Code Ocean tests inside the package.
- **`aws s3 sync` for `copy`**; replaced by boto3 (optional extra `scizarr-ic[s3]`).
- **`Repo.session()`** exposing the raw icechunk session. Everything callers needed was
  the zarr group (`writable()`, `root()`) and `commit()`; the ingest was refactored to
  pass the `Repo` around instead. Keep the icechunk session private.

## How Icechunk behaves (verified, drives the API shape)

- A writable session stages **metadata** in memory but uploads **chunk data eagerly**:
  writing a 1000×100 array put 10 chunk objects in the store before any commit.
  `commit()` then writes a manifest, a snapshot and moves the branch ref. Other
  readers see nothing until commit. `discard()` leaves the uploaded chunks as
  unreferenced garbage (Icechunk's `garbage_collect` removes them).
- Consequence: there is no "write locally, then upload" mode for a remote repo.
  A safe scratch workflow is `Repo.copy` to a local dir, experiment and commit
  there, then re-apply the wanted edits to the real repo. There is no `push`
  between repos; Icechunk snapshots are not portable across stores.
- The repo layout is content-addressed object files (`chunks/`, `manifests/`,
  `snapshots/`, `transactions/`, `overwritten/`, `repo`), so a byte-for-byte copy is
  a faithful clone with identical snapshot ids. `copy` relies on this.
- Local-filesystem storage is not safe for concurrent commits (Icechunk warns);
  the CLI silences that warning unless `ICECHUNK_LOG` is set.
- `Repository.exists` on a local path that does not exist raises, hence the
  `os.path.exists` guard in `Repo.exists`.

## The Code Ocean capsule this was built in (for context only)

Capsule `icechunk_ingest` (code at `code/icechunk_ingest`, this package vendored under
`code/DataScales/tools/scizarr_IC`). Facts about that environment:

- Data assets mount read-only under `/data`; `os.access(W_OK)` is False for root,
  so `is_readonly_path` works with no configuration.
- Two asset kinds. *Linked* external S3 assets (`/data/sample_linked_icechunk`, a
  FUSE mount) are live views of `s3://immunology-datascales-temp-3588/DataScales/
  sample_linked_icechunk`; writes to the prefix appear in the mount. *Internal*
  assets (`/data/sample_icechunk`, an EFS/NFS mount) are one-time copies that
  never update; read or `copy` them, there is nothing to write back to.
- The capsule's AWS role can write that bucket. `run`, `icechunk_demo.sh`,
  `icechunk_demo.py` and `.env.example` set `SCIZARR_IC_ORIGIN` to the linked
  asset's S3 prefix; `SCIZARR_IC_HOME=/scratch/scizarr_ic` keeps HEAD out of
  `/data`. `/scratch` is emptied after every Reproducible Run.
- The ingest pipeline's `store_utils.py` imports `scizarr_ic` (`Repo.exists`,
  `Repo.create`, `Repo(path, branch="main").writable()` / `.commit()`); its earlier
  private helpers `icechunk_tools.py` and `scizarr.py` were deleted so there is one
  implementation. Its pixi manifest tracks the `icechunk_helper_clean` branch of
  DataScales; after changing this package, push and run `pixi lock` there.
- The Python env with icechunk lives at `/scratch/pixi-envs/icechunk_ingest-*/envs/
  default`; there is no `/opt/pipeline/activate.sh`. Tests run with
  `PYTHONPATH=src <that python> -m pytest`.

## Tests

28 hermetic tests, no network, no credentials, ~1.5 s: `test_repo.py` (core API),
`test_cli.py`, `test_copy.py` (`copy_group` + `Repo.copy`), `test_origin.py`
(read from one location, write to another, using a stale local copy as the "mirror"
and a local dir as the "origin", which exercises the same code paths as a
mount + S3 pair). Keep it that size; add a test only for a new command or a bug.

## Ideas not done

- `push`/`pull` between repos (needs Icechunk-level snapshot transplanting).
- Caching `list_branches` in `Repo.__init__` (called up to three times; each is one
  object-store request).
- Tags (`Repository.create_tag`) if releases of an atlas need names.
