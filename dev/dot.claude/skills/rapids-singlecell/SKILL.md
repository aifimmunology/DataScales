---
name: rapids-singlecell
description: >-
  Use when running GPU-accelerated single-cell work (the scanpy-equivalent API on GPU) —
  moving AnnData to/from device, CuPy dense / cupyx CSR-CSC sparse matrices, or benchmarking
  GPU vs CPU pipelines. Invoke before writing any `rapids_singlecell` (rsc) call. NOTE:
  rapids-singlecell is GPU/CUDA-only and is NOT installed in the local CPU env — it runs on an
  HPC/GPU node. Verify the API against vendored source; do not import it in CPU code paths or tests.
---

# rapids-singlecell — GPU single-cell (scanpy-compatible)

GPU-accelerated analogue of scanpy/anndata ops, built on RAPIDS (CuPy, cuDF, cuML) and
cupyx sparse. DataScale uses it for GPU pipelines/benchmarks on CUDA hardware.

> ⚠️ **Environment:** CUDA-only; **not installed in the local osx/CPU pixi env**. It belongs
> to a separate GPU/HPC environment. Never import it from CPU converter code or the default
> test suite. If asked to run it locally, flag that it needs the GPU env.

## Where to look (ground truth)

Vendored from `github.com/scverse/rapids_singlecell` @ **v0.15.2**.

- **Source:** `.claude/vendor/rapids-singlecell/src/rapids_singlecell/`
- **Docs:** `.claude/vendor/rapids-singlecell/docs/` (start with `usage_principles.md`)
- **Out-of-core & multi-GPU (prose, resolvable):** `.claude/vendor/rapids-singlecell/docs/out_of_core.md`
  (Dask-CUDA cluster setup + processing data larger than VRAM) and
  `.claude/vendor/rapids-singlecell/docs/memory_management.md` (RMM pool / managed memory /
  VRAM spilling) cover the CUDA + multi-GPU workflows central to our implementations.
  ⚠️ The `docs/notebooks/*.ipynb` tutorials (incl. `06_out-of-core`, `07_multi_gpu`) are
  **broken symlinks** in this vendored tree — read the prose docs above instead.
- **Installed check (GPU env only):** `python -c "import rapids_singlecell as rsc; print(rsc.__version__)"`

Real namespaces (v0.15.2 — top-level subpackages):

| Namespace | Dir | Scanpy analogue |
|-----------|-----|-----------------|
| `rsc.get` (host↔device movement) | `src/rapids_singlecell/get/` | `sc.get` |
| `rsc.pp` (preprocessing) | `src/rapids_singlecell/preprocessing/` | `sc.pp` |
| `rsc.tl` (tools) | `src/rapids_singlecell/tools/` | `sc.tl` |
| `squidpy_gpu` / `decoupler_gpu` / `pertpy_gpu` | same-named dirs | spatial / decoupler / pertpy |

### Grep recipes

```bash
V=.claude/vendor/rapids-singlecell/src/rapids_singlecell
rg -n "^def " $V/get/_anndata.py     # anndata_to_GPU / anndata_to_CPU / X_to_GPU / X_to_CPU
rg -n "^def " $V/preprocessing/*.py  # pp.* equivalents
rg -n "cupy|cupyx|csr_matrix|csc_matrix" $V   # device array / sparse types
```

## Model (confirmed in v0.15.2 source)

- API mirrors scanpy namespaces: `rsc.get`, `rsc.pp`, `rsc.tl` (plus `squidpy_gpu`,
  `decoupler_gpu`, `pertpy_gpu`). Don't assume parity with a scanpy version from memory — grep.
- **Data lives on the GPU as CuPy (dense) or cupyx.scipy.sparse (CSR/CSC) arrays.** Move AnnData
  to device with `rsc.get.anndata_to_GPU(adata)` and back with `rsc.get.anndata_to_CPU(adata)`;
  single matrices via `X_to_GPU` / `X_to_CPU` (all defined in `get/_anndata.py`).
- Interop with DataScale: convert/move on GPU, then bring `X` back to host (scipy/numpy) before
  handing to the zarr/anndata writers — those expect CPU arrays. The `anndata` and `zarr`
  skills own the write side.

## Performance principles (the GPU-specific killers)

- **Host↔device transfer dominates** if you ping-pong arrays across the PCIe bus per op. Move
  to GPU once, do all the work, move back once. Measure transfer time separately in benchmarks.
- **Record device provenance** in every benchmark: GPU model, CUDA + driver version, CuPy/RAPIDS
  versions, and resident vs. transferred bytes.
- **GPU Usage** Use source code as a guide when looking into bottlenecks for areas of throughput and memory to find which part of steps could be holding up the parallel, multi-gpu usage.
- **GPU memory ceiling** is the hard limit — a matrix that fits in host RAM may not fit in VRAM.
  Know the device memory budget; consider chunked/streamed GPU passes.
- **Don't compare CPU vs GPU on different layouts/dtypes** — hold them identical (see
  benchmarking best practices in [CLAUDE.md](../../../CLAUDE.md)).

## Reading/streaming a store onto the GPU

The query path ends on the device: **zarr chunk → anndata major-axis block → dask block → GPU
block**. The layout you wrote upstream sets the GPU transfer + VRAM granularity.

- **`X_to_GPU` (`src/rapids_singlecell/get/_anndata.py`, ~L79):** for a `DaskArray` it does
  `X.map_blocks(X_to_GPU)` — **preserves block structure**, moving one block at a time to the
  device (cupyx CSR/CSC, or CuPy dense). For in-memory scipy CSR/CSC it ships
  `data`/`indices`/`indptr` to the GPU. So the stored chunk/format **is** the per-block VRAM
  cost: for sparse that's `major-axis stride × full minor width` (see the `anndata` skill).
- **Dask GPU kernels** under `src/rapids_singlecell/_cuda/qc_dask/` run per block — block size
  is both the parallelism unit and the memory granularity. Choose the stored chunking against
  the device memory budget (see the GPU memory ceiling above).
- **On-device format conversion** exists (`src/rapids_singlecell/_utils/_csr_to_csc.py`) but a
  CSR↔CSC transpose is expensive — prefer *storing* in the format your dominant query wants
  over converting per query (ties to the row-vs-column tradeoff in the `zarr` skill).

```bash
V=.claude/vendor/rapids-singlecell/src/rapids_singlecell
rg -n "def X_to_GPU|map_blocks|csr_matrix_gpu|csc_matrix_gpu" $V/get/_anndata.py
rg -n "def |csr|csc" $V/_utils/_csr_to_csc.py | head
```

## ✍️ Maintainer notes — ADD YOURS (high priority — GPU env not local)

<!-- TODO(you): fill once running on the GPU node -->

- **GPU environment definition** (separate pixi env / module load / container image):
- **Exact rapids-singlecell + CUDA + CuPy versions** on the target node:
- **Which operations you actually run on GPU** (and which stay on CPU) and why:
- **Confirmed host↔device helper names** (paste working snippet):
- **VRAM budget** of the target GPU(s) and the largest matrix shape that fits:
- **Vendored doc pages worth reading first:**
