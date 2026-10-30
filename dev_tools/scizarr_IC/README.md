# scizarr-ic

> **Legacy copy, kept for tracking.** This tool was merged into **AnniZarr**, which now lives in its own
> repository: https://github.com/A-Jolly-Holly/annizarr (`pip install annizarr` once it is published). The code
> below is frozen at its last standalone version (0.3.0) so the history stays reproducible; use annizarr
> for anything new.
>
> Mapping: `from scizarr_ic import Repo` → `from annizarr.ic import Repo` (same API; `ScizarrError` →
> `annizarr.errors.RepoError`). Writes from the shell go through `annizarr convert --ic` / `--branch` / `-m`.

Git-like version control for Zarr stores, backed by [Icechunk](https://icechunk.io).

Point it at an existing zarr store and it becomes a versioned Icechunk repository:
commit changes, branch, inspect history, and reset the store to any past snapshot, from
a notebook or a script. Repos live in a local directory or an object store
(`s3://bucket/prefix`, `gs://bucket/prefix`).

## Install

```bash
pip install .            # from tools/scizarr_IC
pip install '.[s3]'      # adds boto3, only needed for Repo.copy to/from s3://
```

Needs Python 3.12+, Icechunk 2.2.2+ and Zarr v3 (pulled in automatically).

## Usage

A `Repo` is one Icechunk repository opened on a branch (`main` unless you say
otherwise). `branches()`, `log()` and `tree()` return data but display git-style, so a
bare expression in a notebook cell and `print()` in a script show the same thing.

```python
from scizarr_ic import Repo

repo = Repo.init("data.zarr", "data.icechunk")    # import a zarr store (one commit on main)
repo = Repo.create("s3://bucket/empty")           # or start empty
repo = Repo("s3://bucket/store", branch="dev")    # open an existing repo on a branch
#repo = Repo("gs://bucket/store", anonymous=True)  # public bucket, no credentials needed
Repo.exists("s3://bucket/empty")                  # True / False

repo                                              # location, branch, tip commit
repo.branches()                                   # list branches, * marks the current branch
repo.log()                                        # all commits from the current branch
repo.log()[0].id                                 
repo.tree()                                       # every branch as one commit graph

z = repo.open_zarr("w")                           # writable zarr.Group at the branch tip
z.attrs["step"] = "lognorm"                       # edit like any zarr group
repo.commit("normalize")                        

repo.checkout("experiment", create=True)          # branch off the current tip (this object only)
z = repo.open_zarr("w"); z["X"][:] *= 2
repo.commit("scaled X")

root = repo.open_zarr("r")                        # read-only zarr.Group at the branch tip
root                                              # prints location, branch, snapshot, members (subgroups keep zarr's repr)
old  = repo.open_zarr("r", snapshot_id=some_id)   # or a past snapshot (read-only)
repo.cherrypick(some_id)                          # reset the current branch to a snapshot

mine = repo.copy("/work/store")                   # clone (all branches, ids intact)
```

Batch writes into **few, large commits**; a commit per chunk/row-batch is pathological for
Icechunk. `open_zarr("w")` reuses one open session until you `commit()` or `discard()`.

### Credentials

Object-store credentials come from the environment: `gcloud auth application-default login`
for `gs://`, `aws sso login` / `AWS_*` variables / an instance role for `s3://`.
`anonymous=True` reads a public bucket unsigned. A credentials problem at open raises a
`ScizarrError` that says so, rather than "no repository".

### Read here, write there

When the path you read is not the place you can write, such as a read-only local mirror of
an `s3://` prefix, name the writable location: `Repo("/mnt/store", origin="s3://bucket/store")`.
Reads stay on the path; `checkout(create=True)`, `cherrypick` and `commit` go to the origin,
which then also serves reads in that process, since a mirror can lag behind fresh writes.
A read-only path with no origin is reads-only; `repo.copy(dest)` takes a fully writable clone.

## Notes

- `init` copies the source store's layout (chunks, shards, codecs, fill value, attrs)
  verbatim and streams data in chunk-aligned bands, so the import stays memory-bounded
  and anndata-readable.
- `copy` replicates the repo's object tree byte for byte (Icechunk is content-addressed),
  so every branch and snapshot id is preserved. Local→local needs nothing; anything
  touching `s3://` needs `boto3`.
- One repo == one store. Writes are single-writer (in-process session); concurrent /
  distributed writes are not wired up yet.
