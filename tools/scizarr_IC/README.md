# scizarr-ic

Git-like version control for Zarr stores, backed by [Icechunk](https://icechunk.io).

Point it at an existing zarr store and it becomes a versioned Icechunk repository:
commit changes, branch, inspect history, and reset the store to any past snapshot —
with a CLI that reads like `git` and a small Python API for scripted edits.

## Install

```bash
cd tools/scizarr_IC
pixi install -e dev          # dev env includes pytest
pixi run -e dev pytest       # hermetic suite (no network, no credentials)

# live checks against a real read-only mount (Code Ocean data asset):
SCIZARR_IC_LIVE_REPO=/data/my_store pixi run -e dev pytest tests/test_live_codeocean.py
SCIZARR_IC_LIVE_WRITE=1 ...  # also round-trips a scratch branch at the s3:// origin
```

## CLI

```bash
scizarr-ic init SRC.zarr OUT.icechunk -m "import atlas v1"   # seed a repo from a zarr store
scizarr-ic log     -C OUT.icechunk [--oneline] [-b BRANCH]   # history of a branch, newest first
scizarr-ic tree    -C OUT.icechunk                           # every branch + its commits
scizarr-ic checkout -C OUT.icechunk BRANCH [-b]              # switch branch (-b to create)
scizarr-ic cherrypick -C OUT.icechunk SNAPSHOT_ID           # reset current branch to a snapshot
scizarr-ic origin  -C OUT.icechunk [URL]                    # show (or stamp) where writes go
scizarr-ic copy    -C SRC.icechunk DEST                     # writable clone, full history
```

`scz` is a shorter alias for `scizarr-ic`. The current branch is remembered per repo,
so `-C` is all you need between commands. Repo paths may be local directories or
`s3://bucket/prefix` / `gs://bucket/prefix` URIs (object-store credentials read from
the environment).

### Read-only mounts (Code Ocean data assets)

A data asset mounts read-only under `/data`, but the S3 prefix behind it is writable.
`init` stamps the repo's writable location into its metadata (`origin_url`), so:

```bash
scizarr-ic origin   -C /data/my_store            # path (read-only) / stamped origin / where writes go
scizarr-ic log      -C /data/my_store            # reads: straight from the mount, no credentials
scizarr-ic checkout -C /data/my_store dev -b     # writes: go to the s3:// origin (env credentials)
scizarr-ic origin   -C s3://bucket/prefix s3://bucket/prefix   # retro-stamp an older repo
```

`--origin URL` (or `SCIZARR_IC_ORIGIN`) overrides the stamped origin. Once a write has
opened the origin, the same process reads from it too (a mount can lag behind fresh
writes); a fresh process that finds its HEAD branch missing from the mount also checks
the origin before giving up.

HEAD lives locally, like git's: a `scizarr_head` file inside a *writable* local repo,
otherwise a per-user sidecar under `$SCIZARR_IC_HOME` (default `~/.cache/scizarr_ic`),
keyed by the repo's origin. Nothing is ever written into a read-only repo.

**Two kinds of data asset.** A *linked* (external S3) asset is a live view of its
bucket prefix, so writes to the stamped origin show up in the mount shortly after. An
*internal* asset is a frozen copy on EFS: its stamped origin may still be writable, but
the mount will never reflect those writes. When you cannot write back — no AWS secret,
or a frozen asset — take a copy instead:

```bash
scizarr-ic copy -C /data/my_store /results/my_store     # or s3://bucket/prefix (uses `aws s3 sync`)
scizarr-ic checkout -C /results/my_store analysis -b    # the copy is its own origin
```

`copy` replicates the repo's object tree byte for byte, so every branch and snapshot id
is preserved, then stamps the copy as its own `origin_url` and carries the current
branch over. The destination must be empty (or a prefix holding no repo).

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

mine = Repo("/data/my_store").copy("/results/my_store")   # writable clone of a read-only asset
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
