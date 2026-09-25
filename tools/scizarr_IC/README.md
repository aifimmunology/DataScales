# scizarr-ic

Git-like version control for Zarr stores, backed by [Icechunk](https://icechunk.io).

Point it at an existing zarr store and it becomes a versioned Icechunk repository:
commit changes, branch, inspect history, and reset the store to any past snapshot —
with a CLI that reads like `git` and a small Python API for scripted edits.

## Install

```bash
cd tools/scizarr_IC
pixi install -e dev          # dev env includes pytest
pixi run -e dev pytest       # run the test suite
```

## CLI

```bash
scizarr-ic init SRC.zarr OUT.icechunk -m "import atlas v1"   # seed a repo from a zarr store
scizarr-ic log     -C OUT.icechunk [--oneline] [-b BRANCH]   # history of a branch, newest first
scizarr-ic tree    -C OUT.icechunk                           # every branch + its commits
scizarr-ic checkout -C OUT.icechunk BRANCH [-b]              # switch branch (-b to create)
scizarr-ic cherrypick -C OUT.icechunk SNAPSHOT_ID           # reset current branch to a snapshot
```

`scz` is a shorter alias for `scizarr-ic`. The current branch is remembered per repo
(a `scizarr_head` file), so `-C` is all you need between commands. Repo paths may be
local directories or `s3://bucket/prefix` / `gs://bucket/prefix` URIs (object-store
credentials read from the environment).

`commit` is **Python-API only** — Icechunk stages edits in a session's memory, so there
is nothing for a fresh CLI process to commit.

## Python API

```python
from scizarr_ic import Repo

repo = Repo.init("data.zarr", "data.icechunk")   # import a zarr store (one commit on main)

g = repo.writable()                              # zarr.Group on a writable session
g.attrs["step"] = "lognorm"                      # ... edit like any zarr group ...
repo.commit("normalize")                         # stage → durable snapshot

repo.checkout("experiment", create=True)         # branch off the current tip
g = repo.writable(); g["X"][:] *= 2
repo.commit("scale X")

for snap in repo.log():                          # newest first
    print(snap.id, snap.message)

repo.cherrypick(some_snapshot_id)                # reset current branch to a snapshot
root = repo.root()                               # read-only zarr.Group at the branch tip
```

Batch writes into **few, large commits** — a commit per chunk/row-batch is pathological
for Icechunk. `writable()` reuses one open session until you `commit()` or `discard()`.

## Notes

- Icechunk **2.1.2+** and Zarr **v3**. `init` copies the source store's layout
  (chunks, shards, codecs, fill value, attrs) verbatim and streams data in
  chunk-aligned bands, so the import stays memory-bounded and anndata-readable.
- One repo == one store. Writes are single-writer (in-process session); concurrent /
  distributed writes are not wired up yet.
- This tool is meant to compose with the other DataScale tools (e.g. `zarrsmith`,
  which can already write directly to an Icechunk backend).
