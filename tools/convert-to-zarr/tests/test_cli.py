"""CLI-surface tests: the repo layout and flag set downstream consumers install against.

Consumers outside this repo (e.g. the umap app) clone DataScales, install this tool
from ``tools/convert-to-zarr``, and shell out to the ``convert-to-zarr`` console script. That
coupling is invisible to the Python-API tests: a moved directory, a renamed entry point, or a
dropped flag fails at their runtime, not at import. So these drive the real CLI via subprocess.
"""
from __future__ import annotations

import shutil
import subprocess
import tomllib
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import zarr

TOOL_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TOOL_DIR.parents[1]
TOOL_SUBDIR = "tools/convert-to-zarr"
CLI_NAME = "convert-to-zarr"

N_OBS, N_VARS = 12, 7
COL_CHUNK = 5
SHARD_FACTOR = 32


def _input_h5ad(path: Path) -> np.ndarray:
    """Write a small CSR h5ad. ``X[i, j] == (i + 1) * 100 + j`` so a column read is checkable
    against its column index. Returns the expected dense X."""
    dense = np.array(
        [[(i + 1) * 100 + j for j in range(N_VARS)] for i in range(N_OBS)], dtype=np.float32
    )
    adata = ad.AnnData(
        X=sp.csr_matrix(dense),
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(N_OBS)]),
        var=pd.DataFrame(index=[f"GENE{j}" for j in range(N_VARS)]),
    )
    adata.write_h5ad(path)
    return dense


def _cli() -> str:
    exe = shutil.which(CLI_NAME)
    assert exe, (
        f"The `{CLI_NAME}` console script is not on PATH. Consumers invoke it by name via "
        f"`pixi run --manifest-path .../pyproject.toml {CLI_NAME}`; keep the [project.scripts] "
        f"entry point. (Run the suite with `pixi run -e dev pytest`.)"
    )
    return exe


def test_tool_path_and_entry_point() -> None:
    """The install path and script name consumers hardcode."""
    manifest = REPO_ROOT / TOOL_SUBDIR / "pyproject.toml"
    assert manifest.is_file(), f"This tool must stay installable as `{TOOL_SUBDIR}/pyproject.toml`."
    scripts = tomllib.loads(manifest.read_text()).get("project", {}).get("scripts", {})
    assert CLI_NAME in scripts


@pytest.mark.parametrize("backed", [False, True], ids=["eager", "backed"])
def test_dense_convert_flag_set(tmp_path: Path, backed: bool) -> None:
    """The full dense-conversion flag set, on both the eager and ``--backed`` paths.

    Renaming or dropping any of these flags is a breaking change for CLI consumers.
    """
    expected = _input_h5ad(tmp_path / "in.h5ad")
    out = tmp_path / "out.zarr"
    row_chunk = N_OBS if not backed else 4

    cmd = [
        _cli(), "convert-h5ad",
        "--input", str(tmp_path / "in.h5ad"),
        "--output", str(out),
        "--x-storage", "dense",
        "--overwrite",
        "--x-row-chunk", str(row_chunk),
        "--x-col-chunk", str(COL_CHUNK),
        "--x-shard-factor", str(SHARD_FACTOR),
        "--cpus", "2",
    ]
    if backed:
        cmd.append("--backed")

    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"cmd: {' '.join(cmd)}\nstderr:\n{proc.stderr}"

    z = zarr.open(str(out), mode="r")
    xa = z["X"]
    assert isinstance(xa, zarr.core.array.Array), "--x-storage dense must yield a dense array."
    assert xa.shape == (N_OBS, N_VARS)
    # The chunk flags must be honored exactly: the inner chunk is the read granularity, so a
    # widened column chunk turns a single-column read into a much larger one.
    assert xa.chunks == (row_chunk, COL_CHUNK)
    np.testing.assert_array_equal(xa[:], expected)
    np.testing.assert_array_equal(xa[:, 2], expected[:, 2])


def test_overwrite_flag(tmp_path: Path) -> None:
    """``--overwrite`` is required for re-runs into an existing output path."""
    _input_h5ad(tmp_path / "in.h5ad")
    out = tmp_path / "out.zarr"
    base = [_cli(), "convert-h5ad", "--input", str(tmp_path / "in.h5ad"), "--output", str(out)]

    assert subprocess.run(base, capture_output=True, text=True).returncode == 0
    assert subprocess.run(base, capture_output=True, text=True).returncode != 0
    assert subprocess.run(base + ["--overwrite"], capture_output=True, text=True).returncode == 0
