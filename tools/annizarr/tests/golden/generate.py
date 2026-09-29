"""Regenerate ``tests/golden/*.tar.gz`` golden fixtures using the OLD ``convert-to-zarr`` CLI.

Standalone script — stdlib + numpy/scipy/anndata only (no ``annizarr`` import: this must run
in the OLD tool's pixi env, which doesn't have annizarr installed). It builds tiny,
deterministic 100 x 50 AnnData fixtures, converts each with the old CLI via ``subprocess``,
then packs ``input/*.h5ad`` + ``expected/<out>.zarr`` + ``case.json`` (the new-style
``annizarr.convert`` parameters) into one tarball per case.

Regenerate with (see ../../tests/golden/README.md for the policy):

    cd tools/convert-to-zarr && pixi run -e dev python \
        ../annizarr/tests/golden/generate.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

GOLDEN_DIR = Path(__file__).resolve().parent
N_OBS, N_VARS = 100, 50
CELL_TYPES = ("B", "D", "A", "C")  # deliberately non-alphabetical + unsorted assignment
BATCHES = ("batch1", "batch0")

# convert-to-zarr's x_storage spelling -> annizarr's (see final_buildspec.md §8 unit 3.0)
_X_STORAGE_OLD = {"csr": "sparse-csr", "csc": "sparse-csc", "dense": "dense"}


def _var(n_vars: int = N_VARS) -> pd.DataFrame:
    return pd.DataFrame(index=[f"gene{i}" for i in range(n_vars)])


def _obs(n_obs: int, seed: int, prefix: str = "cell") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "cell_type": pd.Categorical(rng.choice(CELL_TYPES, n_obs), categories=CELL_TYPES),
            "batch": pd.Categorical(rng.choice(BATCHES, n_obs), categories=BATCHES),
        },
        index=[f"{prefix}{i}" for i in range(n_obs)],
    )


def _csr_float(n_obs: int, n_vars: int, seed: int, density: float = 0.3) -> sp.csr_matrix:
    x = sp.random(n_obs, n_vars, density=density, format="csr", dtype=np.float32, random_state=seed)
    x.data = (np.abs(x.data) + 1.0).astype(np.float32)
    return x


def _csr_int(n_obs: int, n_vars: int, seed: int, density: float = 0.3, high: int = 50) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)

    def _rvs(n: int) -> np.ndarray:
        return rng.integers(1, high, size=n).astype(np.int64)

    x = sp.random(n_obs, n_vars, density=density, format="csr", dtype=np.int64, random_state=seed, data_rvs=_rvs)
    return x.astype(np.int32)


def _base_adata(seed: int, n_obs: int = N_OBS, n_vars: int = N_VARS, prefix: str = "cell") -> AnnData:
    """A CSR float32 AnnData with cell_type (4 levels) + batch (2 levels) obs and an X_pca obsm."""
    x = _csr_float(n_obs, n_vars, seed)
    obs = _obs(n_obs, seed, prefix=prefix)
    var = _var(n_vars)
    rng = np.random.default_rng(seed + 1000)
    obsm = {"X_pca": rng.random((n_obs, 2)).astype(np.float32)}
    return AnnData(X=x, obs=obs, var=var, obsm=obsm)


def build_inputs(input_dir: Path) -> dict[str, Path]:
    """Write every distinct h5ad fixture once; return name -> path."""
    input_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    a = _base_adata(seed=0)
    paths["a"] = input_dir / "a.h5ad"
    a.write_h5ad(paths["a"])

    # second CSR file, identical var, for concat — different n_obs/seed, same obs schema.
    b = _base_adata(seed=1, n_obs=60, prefix="cellb")
    paths["b"] = input_dir / "b.h5ad"
    b.write_h5ad(paths["b"])

    # dense-X variant (same shape/obs shape, independent seed).
    dense = _base_adata(seed=2)
    dense.X = np.asarray(dense.X.todense())
    paths["dense"] = input_dir / "dense.h5ad"
    dense.write_h5ad(paths["dense"])

    # layers["counts"] + raw + obsm variant.
    layered = _base_adata(seed=3)
    layered.layers["counts"] = _csr_int(N_OBS, N_VARS, seed=13)
    raw_adata = AnnData(X=_csr_float(N_OBS, N_VARS, seed=23), var=_var(N_VARS))
    layered.raw = raw_adata
    paths["layers_raw"] = input_dir / "layers_raw.h5ad"
    layered.write_h5ad(paths["layers_raw"])

    # integer-counts variant.
    int_counts = _base_adata(seed=4)
    int_counts.X = _csr_int(N_OBS, N_VARS, seed=4)
    paths["int_counts"] = input_dir / "int_counts.h5ad"
    int_counts.write_h5ad(paths["int_counts"])

    return paths


def _old_cli_args(case: dict[str, Any], input_paths: list[Path], output: Path) -> list[str]:
    """Translate a case's NEW-style params dict into OLD `convert-to-zarr` CLI flags."""
    args: list[str] = ["convert-to-zarr"]
    if len(input_paths) > 1:
        args += ["concat-h5ads", "--inputs", *[str(p) for p in input_paths]]
    else:
        args += ["convert-h5ad", "--input", str(input_paths[0])]
    args += ["--output", str(output)]
    args += ["--x-storage", _X_STORAGE_OLD[case["x_storage"]]]
    if case.get("backed"):
        args.append("--backed")
    if case.get("cpus") is not None:
        args += ["--cpus", str(case["cpus"])]
    if case.get("x_row_chunk") is not None:
        args += ["--x-row-chunk", str(case["x_row_chunk"])]
    if case.get("x_col_chunk") is not None:
        args += ["--x-col-chunk", str(case["x_col_chunk"])]
    if case.get("sparse_flat_chunk") is not None:
        args += ["--sparse-flat-chunk", str(case["sparse_flat_chunk"])]
    if case.get("x_shard_factor") is not None:
        args += ["--x-shard-factor", str(case["x_shard_factor"])]
    if case.get("sort_by"):
        args += ["--sort-by", *case["sort_by"]]
    return args


# Each case: `inputs` names the h5ad fixture(s) (see build_inputs); the rest are
# annizarr-style `apply_cli_overrides` kwargs, packed verbatim into case.json for unit 3.1's
# test to feed straight into `annizarr.convert(..., cfg=...)`.
CASES: dict[str, dict[str, Any]] = {
    "csr_eager": {"inputs": ["a"], "x_storage": "csr"},
    "csr_backed": {"inputs": ["a"], "x_storage": "csr", "backed": True},
    "csc_eager": {"inputs": ["a"], "x_storage": "csc"},
    "dense_from_sparse_eager": {"inputs": ["a"], "x_storage": "dense"},
    "dense_from_sparse_backed": {"inputs": ["a"], "x_storage": "dense", "backed": True},
    "dense_from_dense_eager": {"inputs": ["dense"], "x_storage": "dense"},
    "dense_sharded": {
        "inputs": ["a"],
        "x_storage": "dense",
        "x_row_chunk": 20,
        "x_col_chunk": 10,
        "x_shard_factor": 2,
    },
    "dense_small_chunks": {
        "inputs": ["a"],
        "x_storage": "dense",
        "x_row_chunk": 20,
        "x_col_chunk": 10,
    },
    "csr_small_flat_chunk": {"inputs": ["a"], "x_storage": "csr", "sparse_flat_chunk": 200},
    "concat_csr": {"inputs": ["a", "b"], "x_storage": "csr"},
    "concat_dense": {"inputs": ["a", "b"], "x_storage": "dense"},
    "sorted_eager": {"inputs": ["a"], "x_storage": "csr", "sort_by": ["cell_type", "batch"]},
    "sorted_backed": {
        "inputs": ["a"],
        "x_storage": "csr",
        "sort_by": ["cell_type", "batch"],
        "backed": True,
    },
    "csr_layers_raw_obsm": {"inputs": ["layers_raw"], "x_storage": "csr"},
    "csr_int_counts": {"inputs": ["int_counts"], "x_storage": "csr"},
    "csr_cpus2": {"inputs": ["a"], "x_storage": "csr", "cpus": 2},
}


def _run_case(name: str, case: dict[str, Any], fixture_paths: dict[str, Path], work: Path) -> Path:
    case_dir = work / name
    input_dir = case_dir / "input"
    expected_dir = case_dir / "expected"
    input_dir.mkdir(parents=True)
    expected_dir.mkdir(parents=True)

    input_names = case["inputs"]
    local_inputs = []
    for n in input_names:
        src = fixture_paths[n]
        dst = input_dir / src.name
        shutil.copyfile(src, dst)
        local_inputs.append(dst)

    out_name = f"{name}.zarr"
    output_path = expected_dir / out_name
    cli_args = _old_cli_args(case, local_inputs, output_path)
    result = subprocess.run(cli_args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"case {name!r} failed ({' '.join(cli_args)}):\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    case_json = {
        "x_storage": case["x_storage"],
        "backed": case.get("backed", False),
        "cpus": case.get("cpus"),
        "x_row_chunk": case.get("x_row_chunk"),
        "x_col_chunk": case.get("x_col_chunk"),
        "sparse_flat_chunk": case.get("sparse_flat_chunk"),
        "x_shard_factor": case.get("x_shard_factor"),
        "sort_by": case.get("sort_by"),
        "inputs": [p.name for p in local_inputs],
        "output": out_name,
    }
    (case_dir / "case.json").write_text(json.dumps(case_json, indent=2, sort_keys=True) + "\n")

    tarball = GOLDEN_DIR / f"{name}.tar.gz"
    with tarfile.open(tarball, "w:gz") as tar:
        for p in sorted(case_dir.rglob("*")):
            if p.is_file():
                tar.add(p, arcname=str(p.relative_to(case_dir)))
    return tarball


def main() -> None:
    import tempfile

    with tempfile.TemporaryDirectory(prefix="annizarr-golden-") as tmp:
        work = Path(tmp)
        fixture_paths = build_inputs(work / "_fixtures")

        sizes = []
        for name, case in CASES.items():
            tarball = _run_case(name, case, fixture_paths, work)
            size = tarball.stat().st_size
            sizes.append((name, size))
            print(f"{name}: {tarball.name} ({size / 1024:.1f} KiB)", file=sys.stderr)

        total = sum(s for _, s in sizes)
        print(f"\n{len(sizes)} cases, {total / 1024:.1f} KiB total", file=sys.stderr)


if __name__ == "__main__":
    main()
