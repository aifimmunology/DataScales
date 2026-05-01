#!/usr/bin/env python3
"""Naive in-memory dense conversion — for benchmarking against datascale's dask path.

Loads the h5ad fully into memory, calls .toarray() on X to materialise the full
dense matrix at once, then writes to zarr using anndata's write_zarr.
No chunked streaming — peak memory = full sparse load + full dense array simultaneously.

Usage:
    python naive_dense_convert.py --input data/pbmc3k_raw.h5ad --output /tmp/naive_dense.zarr
    python naive_dense_convert.py --input data/pbmc3k_raw.h5ad --output /tmp/naive_dense.zarr --overwrite
"""

import argparse
import shutil
from pathlib import Path

import anndata as ad
import scipy.sparse as sp


def main() -> None:
    parser = argparse.ArgumentParser(description="Naive in-memory dense zarr conversion")
    parser.add_argument("--input", required=True, help="Path to input .h5ad file")
    parser.add_argument("--output", required=True, help="Path to output .zarr directory")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        if not args.overwrite:
            raise SystemExit(f"Output already exists: {output}. Use --overwrite to replace.")
        shutil.rmtree(output)

    print(f"Loading {args.input} into memory...")
    adata = ad.read_h5ad(args.input)  # full eager load — no backed mode

    print(f"Converting X to dense (shape={adata.X.shape})...")
    if sp.issparse(adata.X):
        adata.X = adata.X.toarray()  # full dense materialisation — peak mem here

    print(f"Writing dense zarr to {output}...")
    ad.settings.zarr_write_format = 3
    adata.write_zarr(str(output))

    print("Done.")


if __name__ == "__main__":
    main()
