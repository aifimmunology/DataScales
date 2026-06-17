# Vendored library source + docs (read-only ground truth)

This tree holds **read-only** copies of the source and documentation for the libraries
DataScale depends on. It exists so the agent can verify APIs against the *exact* version in
use instead of relying on (often stale) memory. See the "Ground-truth research protocol" in
[../../CLAUDE.md](../../CLAUDE.md) and the per-library skills in [../skills/](../skills/).

## Expected layout

```
.claude/vendor/
  zarr-python/        src/   docs/
  anndata/            src/   docs/
  dask/               src/   docs/
  icechunk/           src/   docs/
  rapids-singlecell/  src/   docs/
```

- `src/` — the package's source tree (e.g. `zarr-python/src/zarr/...`). For projects that
  don't ship a `src/` layout upstream (dask), keep the upstream layout and note it in that
  library's skill file.
- `docs/` — prose docs/guides (Markdown/rst). Used for *intent and recommended usage*; the
  source is authoritative for *signatures and defaults*.

> The directory names above are what the skill files grep against. **If you vendor under a
> different name, update the `<pkg>` path in that library's `.claude/skills/<lib>/SKILL.md`.**

## How to populate (your call on the mechanism)

Pin each to the version in use (see CLAUDE.md): zarr 3.1.6 · dask 2026.3.0 · anndata 0.12.10.
`icechunk` and `rapids-singlecell` aren't installed yet — vendor them when adopted.

Options: shallow `git clone --depth 1 --branch <version>`, a git submodule, or a plain copy
of the source tree. Keep it read-only; never edit vendored code.

## What's vendored (provenance)

Pulled via shallow `git clone --depth 1 --branch <tag>` from the upstream repos, then
`src/` + docs copied in (no `.git`). Vendored **2026-06-16**.

| Package | Tag | Upstream | Notes on layout |
|---------|-----|----------|-----------------|
| zarr-python | `v3.1.6` | github.com/zarr-developers/zarr-python | `src/zarr/`, `docs/` |
| anndata | `0.12.10` | github.com/scverse/anndata | `src/anndata/`, `docs/` |
| dask | `2026.3.0` | github.com/dask/dask | package placed at `src/dask/` (no upstream `src/`), `docs/` |
| icechunk | `v2.0.6` | github.com/earth-mover/icechunk | Python pkg at `src/icechunk/`, Rust core at `src/rust-core/`, `docs/` |
| rapids-singlecell | `v0.15.2` | github.com/scverse/rapids_singlecell | `src/rapids_singlecell/`, `docs/` |

`icechunk` and `rapids-singlecell` are vendored as reference but are **not** installed in the
pixi env (icechunk = backend not yet a dependency; rapids = GPU/CUDA-only). The other three
match the installed/pinned versions.

## ✍️ Maintainer notes — ADD YOURS

<!-- TODO(you): keep provenance current when you refresh -->

- **Refresh cadence / when to re-vendor** (e.g. on each pinned-version bump):
- **Any local edits or trims** to the vendored trees (should normally be none — keep read-only):
