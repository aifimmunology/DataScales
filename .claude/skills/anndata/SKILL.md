---
name: anndata
description: >-
  Use when reading/writing AnnData (.h5ad or zarr), calling the private I/O spec API
  (`anndata._io.specs.write_elem` / `read_elem`), handling backed sparse datasets
  (`_CSRDataset` / `_CSCDataset`), dealing with obs/var/X/layers/raw/obsm encoding, or
  matching the on-disk encoding-type/encoding-version attrs that keep a zarr store
  anndata-readable. Invoke before touching any anndata I/O internals. Verifies against the
  vendored anndata 0.12.10 source because these are private APIs that move between versions.
---

# AnnData (0.12.10) — I/O internals & encoding

DataScale calls **private** anndata APIs (`anndata._io.specs`) and hand-writes the encoding
attrs that anndata's reader expects. These are unstable across versions — **always confirm
signatures and the IOSpec registry against the vendored source.**

## Where to look (ground truth)

- **Source:** `.claude/vendor/anndata/src/anndata/`
- **Docs:** `.claude/vendor/anndata/docs/`
- **Installed fallback:** `pixi run python -c "import anndata; print(anndata.__version__)"`

| Question | File |
|----------|------|
| `write_elem` / `read_elem` public-ish entry points & signatures | `src/anndata/_io/specs/registry.py` |
| Per-type encoders/decoders + **the encoding-type/version each writes** | `src/anndata/_io/specs/methods.py` |
| Lazy/partial reads | `src/anndata/_io/specs/{lazy_methods,registry}.py` |
| Backed sparse datasets (`_CSRDataset` / `_CSCDataset`, `.indptr`, slicing) | `src/anndata/_core/sparse_dataset.py` |
| h5ad read/write path | `src/anndata/_io/h5ad.py` |
| zarr read/write path | `src/anndata/_io/zarr.py` |
| AnnData object (X/obs/var/layers/raw/obsm semantics) | `src/anndata/_core/anndata.py` |

### Grep recipes

```bash
V=.claude/vendor/anndata/src/anndata
rg -n "def write_elem|def read_elem" $V/_io/specs/registry.py    # exact signatures
rg -n "encoding-type|encoding_version|IOSpec" $V/_io/specs/methods.py  # what attrs get written
rg -n "class _CSRDataset|class _CSCDataset|def indptr|def __getitem__" $V/_core/sparse_dataset.py
rg -n "csr_matrix|csc_matrix" $V/_io/specs/methods.py            # sparse group spec
```

## Project-relevant essentials

- **`write_elem(group, key, value, dataset_kwargs=...)`** is how DataScale writes obs/var/
  uns/obsm/etc. and (for already-dense X) the matrix itself. It's imported lazily inside
  functions in [converter.py](../../../datascale/converter.py). Confirm the keyword name
  for chunking (`dataset_kwargs`) hasn't changed in 0.12.x.
- **Backed sparse (`_CSRDataset`/`_CSCDataset`):** returned by `read_h5ad(..., backed="r")`.
  These are **not** scipy sparse — they lack `.indices`/`.tocsc()`, expose `.format`, and
  must be sliced to materialize. h5py is **not thread-safe** → backed reads run on the
  synchronous dask scheduler. DataScale reads `indptr` via `matrix.group["indptr"]` for
  backed input; verify that attribute path in `sparse_dataset.py`.
- **Encoding attrs that must match the reader** (DataScale sets these manually — verify the
  exact strings in `methods.py`):
  - root: `encoding-type="anndata"`, `encoding-version="0.1.0"`
  - dense array: `encoding-type="array"`, `encoding-version="0.2.0"`
  - sparse: `encoding-type="csr_matrix"`/`"csc_matrix"`, `encoding-version="0.1.0"`,
    `shape=[n_obs, n_vars]`, children `data`/`indices`/`indptr`.
- **Round-trip is the acceptance test:** a store DataScale writes must reopen via
  `anndata.read_zarr` / `read_elem` with identical values. When in doubt, write tiny and
  re-read (the test suite does exactly this in `tmp_path`).

## Read/query access patterns (analysis reads)

DataScale also *reads* these stores for analysis; how anndata loads them sets query cost.

- **Lazy / dask reads chunk only the major axis.** `read_elem_lazy` → `read_sparse_as_dask`
  (`src/anndata/_io/specs/lazy_methods.py`) builds a dask array whose blocks stride the
  **major** axis only (default `_DEFAULT_STRIDE = 1000`); the **minor axis is always one
  full-width chunk** — it raises *"Only the major axis can be chunked"* if you try otherwise. So:
  - **CSR** → major = obs/cells → cheap **row / cell-subset** queries; a gene-only query still
    reads full rows per block.
  - **CSC** → major = vars/genes → cheap **gene / column** queries; a cell-subset query reads
    full columns per block.
  → For sparse, **format choice is the dominant query lever**, more than chunk shape.
- **Backed slicing** (`src/anndata/_core/sparse_dataset.py`):
  `BaseCompressedSparseDataset.__getitem__` (~L466) and `_get_sliceXslice` (CSR ~L190, CSC
  ~L225). Slicing *along the major axis* is a contiguous `indptr` range (cheap); slicing the
  minor axis must scan all major vectors. h5py is not thread-safe → backed reads stay
  single-threaded (see CLAUDE.md).
- These blocks flow straight to the GPU via rapids `X_to_GPU` (see the `rapids` skill); the
  stored format/chunk becomes the dask block and the GPU transfer unit.

```bash
V=.claude/vendor/anndata/src/anndata
rg -n "Only the major axis|_DEFAULT_STRIDE|def read_sparse_as_dask" $V/_io/specs/lazy_methods.py
rg -n "def __getitem__|_get_sliceXslice|get_compressed_vectors" $V/_core/sparse_dataset.py
```

## Gotchas / version notes (0.12.10)

- `write_elem`/`read_elem` are re-exported in a few places; `_io/specs/registry.py` is the
  definition. Don't rely on an import path you remember — grep it.
- Encoding-version strings are easy to get wrong and fail silently (store opens but a
  downstream tool rejects it). Copy them from `methods.py`, don't transcribe from memory.
- `anndata.io` (public) exists in 0.12.x; prefer documented public API where it covers the
  need, and only drop to `_io.specs` for the streaming/element-level control DataScale needs.

## ✍️ Maintainer notes — ADD YOURS

<!-- TODO(you): fill in / delete prompts as needed -->

- **Which anndata fields DataScale intentionally drops** (e.g. concat ignores layers/raw/uns) and why:
- **Validation rules** that mirror anndata expectations (link to `datascale/validation.py`):
- **Any private API you've pinned to a specific 0.12.x behavior** (note the risk if bumped):
- **Vendored doc pages worth reading first:**
