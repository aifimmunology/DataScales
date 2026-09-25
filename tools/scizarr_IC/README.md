# scizarr-ic

Git-like version control for Zarr stores, backed by [Icechunk](https://icechunk.io).

Point it at an existing zarr store and it becomes a versioned Icechunk repository:
commit changes, branch, inspect history, and reset the store to any past snapshot —
with a CLI that reads similar to git and a small Python API for scripted edits. Repos live
in a local directory or an object store (`s3://bucket/prefix`, `gs://bucket/prefix`).

## Install

```bash
pip install .            # from tools/scizarr_IC; gives the `scizarr-ic` / `scz` commands
pip install '.[s3]'      # adds boto3, only needed for `copy` to/from s3://
```

Needs Python 3.10–3.12, Icechunk 2.1.2+ and Zarr v3 (pulled in automatically).

## CLI

```bash
scizarr-ic init SRC.zarr OUT.icechunk -m "import atlas v1"   # seed a repo from a zarr store
scizarr-ic log     -C OUT.icechunk [--oneline] [-b BRANCH]   # history of a branch, newest first
scizarr-ic tree    -C OUT.icechunk                           # every branch + its commits
scizarr-ic checkout -C OUT.icechunk BRANCH [-b]              # switch branch (-b to create)
scizarr-ic cherrypick -C OUT.icechunk SNAPSHOT_ID           # reset current branch to a snapshot
scizarr-ic copy    -C OUT.icechunk DEST                      # clone (all branches, ids intact)
```

The current branch is remembered per repo,
so `-C` is all you need between commands. Repo paths may be local directories or
`s3://` / `gs://` URIs; object-store credentials come from the environment (`AWS_*`
variables, a profile, or an instance/container role).

`commit` is **Python-API only** — Icechunk stages edits in a session's memory, so there
is nothing for a fresh CLI process to commit.

## Python API

```python
from scizarr_ic import Repo

repo = Repo.init("data.zarr", "data.icechunk")   # import a zarr store (one commit on main)
repo = Repo.create("s3://bucket/empty")          # or start an empty repo (pipelines fill it)
repo = Repo('s3://bucket/store')
#repo = Repo("/mnt/store", origin="s3://bucket/store")   #Special case when writing to repo is different from reading

Repo.exists("s3://bucket/empty")                 # True or false

z = repo.writable()                              # returns zarr editable object for this repo/branch
z.attrs["step"] = "lognorm"                      # ... edit like any zarr group ...
repo.commit("normalize")                         # stage - durable snapshot

repo.checkout("experiment", create=True)         # branch off the current tip
z = repo.writable(); z["X"][:] *= 2
repo.commit("scaled X... other comments here for commit"). #commit to branch, for others to see, and be tracked

for snap in repo.log():                          # show commits log of repo
    print(snap.id, snap.message)

repo.cherrypick(some_snapshot_id)                # reset current branch to a snapshot
root = repo.root()                               # read-only zarr.Group at the branch tip

mine = repo.copy("/work/store")                  # or clone it somewhere writable
```

Batch writes into **few, large commits** — a commit per chunk/row-batch is pathological
for Icechunk. `writable()` reuses one open session until you `commit()` or `discard()`.


### Read here, write there

When the path you read is not the place you can write — a read-only local mirror of an
`s3://` prefix, for instance — name the writable location with `--origin` (or
`SCIZARR_IC_ORIGIN`). Reads stay on the path; `checkout -b`, `cherrypick` and
`Repo.commit` go to the origin, which then also serves reads in that process, since a
mirror can lag behind fresh writes.

```bash
scizarr-ic log      -C /mnt/store                                   # reads, no credentials
scizarr-ic checkout -C /mnt/store dev -b --origin s3://bucket/store # writes go to S3
export SCIZARR_IC_ORIGIN=s3://bucket/store                          # same, for a whole session
```

A read-only path with no origin is reads-only; `copy` takes a fully writable clone
(`scizarr-ic copy -C /mnt/store /work/store`, or to an `s3://` prefix with boto3).

HEAD lives locally, like git's: a `scizarr_head` file inside a *writable* local repo,
otherwise a per-user sidecar under `$SCIZARR_IC_HOME` (default `~/.cache/scizarr_ic`),
keyed by the repo's origin. Nothing is ever written into a read-only repo.


## Notes

- Icechunk **2.1.2+** and Zarr **v3**. `init` copies the source store's layout
  (chunks, shards, codecs, fill value, attrs) verbatim and streams data in
  chunk-aligned bands, so the import stays memory-bounded and anndata-readable.
- `copy` replicates the repo's object tree byte for byte (Icechunk is
  content-addressed), so every branch and snapshot id is preserved. Local→local
  needs nothing; anything touching `s3://` needs `boto3`.
- One repo == one store. Writes are single-writer (in-process session); concurrent /
  distributed writes are not wired up yet.
- This tool is meant to compose with the other DataScale tools (e.g. `zarrsmith`,
  which can already write directly to an Icechunk backend).
