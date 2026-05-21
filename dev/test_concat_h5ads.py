"""End-to-end test for convert_h5ads_to_zarr.

Splits an existing .h5ad along obs into N parts, runs the multi-file concat into
a single zarr, then verifies the result equals what a single-file conversion
would have produced.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
import zarr

from datascale.config import AppConfig, ChunkConfig, IOConfig, ValidationConfig
from datascale.converter import convert_h5ad_to_zarr, convert_h5ads_to_zarr


SRC_H5AD = Path(__file__).parent.parent / "data" / "scanpy-pbmc3k.h5ad"


def _split_h5ad(src: Path, out_dir: Path, n_parts: int) -> list[Path]:
    """Split an h5ad along obs into n_parts pieces."""
    adata = ad.read_h5ad(src)
    n = adata.n_obs
    bounds = np.linspace(0, n, n_parts + 1, dtype=int)
    paths = []
    for i in range(n_parts):
        sub = adata[bounds[i] : bounds[i + 1]].copy()
        # ensure CSR
        if not sp.isspmatrix_csr(sub.X):
            sub.X = sub.X.tocsr() if sp.issparse(sub.X) else sp.csr_matrix(sub.X)
        p = out_dir / f"part_{i}.h5ad"
        sub.write_h5ad(p)
        paths.append(p)
    return paths


def _cfg(x_storage: str, backed: bool = False) -> AppConfig:
    return AppConfig(
        io=IOConfig(overwrite=True, x_storage=x_storage, backed=backed),
        chunks=ChunkConfig(x_row_chunk=128, x_col_chunk=128, sparse_flat_chunk=10_000, cpus=1),
        validation=ValidationConfig(reject_spatial=False),
    )


def _read_x_from_zarr(path: Path) -> np.ndarray:
    g = zarr.open_group(str(path), mode="r")
    x = g["X"]
    if isinstance(x, zarr.Group):
        # sparse
        data = np.asarray(x["data"][:])
        indices = np.asarray(x["indices"][:])
        indptr = np.asarray(x["indptr"][:])
        shape = tuple(x.attrs["shape"])
        enc = x.attrs.get("encoding-type")
        if enc == "csr_matrix":
            return sp.csr_matrix((data, indices, indptr), shape=shape).toarray()
        return sp.csc_matrix((data, indices, indptr), shape=shape).toarray()
    return np.asarray(x[:])


def _read_obs_index(path: Path):
    g = zarr.open_group(str(path), mode="r")
    obs = g["obs"]
    idx_key = obs.attrs.get("_index", "_index")
    return np.asarray(obs[idx_key][:])


def main() -> int:
    if not SRC_H5AD.exists():
        print(f"Source h5ad not found: {SRC_H5AD}", file=sys.stderr)
        return 1

    tmp = Path(tempfile.mkdtemp(prefix="datascale_concat_test_"))
    print(f"Workdir: {tmp}", file=sys.stderr)
    try:
        # Reference: full source AnnData, X coerced to CSR -> dense for comparison.
        ref_adata = ad.read_h5ad(SRC_H5AD)
        if sp.issparse(ref_adata.X):
            ref_X = ref_adata.X.toarray()
        else:
            ref_X = np.asarray(ref_adata.X)
        ref_idx = np.asarray(ref_adata.obs_names)

        results: dict[str, bool] = {}

        for n_parts in (2, 3):
            for storage in ("sparse-csr", "dense"):
                for backed in (False, True):
                    label = f"parts={n_parts}, storage={storage}, backed={backed}"
                    print(f"\n=== {label} ===", file=sys.stderr)

                    parts_dir = tmp / f"parts_{n_parts}_{storage}_b{int(backed)}"
                    parts_dir.mkdir()
                    parts = _split_h5ad(SRC_H5AD, parts_dir, n_parts)

                    out_zarr = tmp / f"out_{n_parts}_{storage}_b{int(backed)}.zarr"
                    warns = convert_h5ads_to_zarr(
                        [str(p) for p in parts], str(out_zarr),
                        _cfg(storage, backed=backed),
                    )
                    print(f"  warnings: {warns}", file=sys.stderr)

                    got_X = _read_x_from_zarr(out_zarr)
                    got_idx = _read_obs_index(out_zarr)

                    x_ok = np.array_equal(got_X, ref_X)
                    idx_ok = np.array_equal(got_idx, ref_idx)
                    print(f"  X equal: {x_ok}, obs index equal: {idx_ok}", file=sys.stderr)
                    print(f"  shapes: got={got_X.shape}, ref={ref_X.shape}", file=sys.stderr)
                    results[label] = x_ok and idx_ok

        # Negative cases
        print("\n=== negative: var mismatch ===", file=sys.stderr)
        bad_dir = tmp / "bad_var"
        bad_dir.mkdir()
        parts = _split_h5ad(SRC_H5AD, bad_dir, 2)
        # mutate part 1 var
        a1 = ad.read_h5ad(parts[1])
        a1 = a1[:, :-1].copy()  # drop one var
        a1.write_h5ad(parts[1])
        try:
            convert_h5ads_to_zarr(
                [str(p) for p in parts], str(tmp / "bad_var.zarr"), _cfg("sparse-csr"),
            )
            results["var_mismatch_raises"] = False
            print("  FAIL: expected ConversionError", file=sys.stderr)
        except Exception as e:
            print(f"  raised as expected: {type(e).__name__}: {e}", file=sys.stderr)
            results["var_mismatch_raises"] = True

        print("\n=== negative: obs schema mismatch ===", file=sys.stderr)
        bad_dir = tmp / "bad_obs"
        bad_dir.mkdir()
        parts = _split_h5ad(SRC_H5AD, bad_dir, 2)
        a1 = ad.read_h5ad(parts[1])
        a1.obs["extra_col"] = 1
        a1.write_h5ad(parts[1])
        try:
            convert_h5ads_to_zarr(
                [str(p) for p in parts], str(tmp / "bad_obs.zarr"), _cfg("sparse-csr"),
            )
            results["obs_mismatch_raises"] = False
            print("  FAIL: expected ConversionError", file=sys.stderr)
        except Exception as e:
            print(f"  raised as expected: {type(e).__name__}: {e}", file=sys.stderr)
            results["obs_mismatch_raises"] = True

        print("\n=== Summary ===", file=sys.stderr)
        all_ok = True
        for k, v in results.items():
            print(f"  [{'PASS' if v else 'FAIL'}] {k}", file=sys.stderr)
            all_ok = all_ok and v
        return 0 if all_ok else 2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
