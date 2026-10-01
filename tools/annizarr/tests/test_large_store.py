"""Memory-bounded proof: every conversion/edit path stays far below full materialisation
on a synthetic 50 000 x 30 000 (~1% density, ~15M nnz) CSR store.

Every step runs the real ``annizarr`` CLI in a fresh subprocess (``sys.executable -c``, a
tiny wrapper around ``annizarr._cli.main``) so ``resource.getrusage(RUSAGE_SELF)`` inside
that subprocess measures exactly the CLI call's own footprint — no cross-step
contamination, unlike ``RUSAGE_CHILDREN`` on the long-lived pytest process (a running
*maximum* across every reaped child, never reset between steps).

Ceiling formula (see ``CEILINGS`` below): peak RSS is baseline import/runtime overhead
(numpy/scipy/anndata/zarr/h5py, ~300-400MB) plus a small constant multiple of
``_layout.BATCH_BYTES`` (256 MiB default) — one input band, one output reorder copy, and
the disk-backed bucket temp files touched by the streamed transpose/lognorm engine — not a
function of total dataset size (a 10x larger store would need more *bands*, not more RAM
per band). That is far below the ~2.4-6GB a dense 50k x 30k float32 materialisation (or
even a fully in-memory sparse round-trip) would need; a run on a 12-core dev box measured
0.2-1.5GB across the five steps (see the agent report for the exact numbers), comfortably
under ``CEILINGS``.

Run explicitly: ``pixi run -e default pytest tests/test_large_store.py -m slow -q
-p no:cacheprovider --no-cov -s``. Excluded from the default suite by
``addopts = "-m 'not slow'"``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import zarr
from anndata.io import sparse_dataset

pytestmark = pytest.mark.slow

N_OBS, N_VARS = 50_000, 30_000
DENSITY = 0.01
N_APPEND = 2_000
CATEGORIES = ("A", "B", "C", "D")

# Peak-RSS ceilings (bytes), one per step below — see the module docstring for the
# formula. All are far below the ~2.4-6GB a dense/full materialisation would need.
CEILINGS = {
    "convert_csr": 900 * 1024**2,
    "convert_csc": 1600 * 1024**2,
    "add_expr_csc": 1800 * 1024**2,
    "sort": 1900 * 1024**2,
    "append": 450 * 1024**2,
}

_RUNNER = """
import json, resource, sys, time
from annizarr._cli import main
argv = json.loads(sys.argv[1])
t0 = time.perf_counter()
rc = main(argv)
wall = time.perf_counter() - t0
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
peak_bytes = peak if sys.platform == "darwin" else peak * 1024
print(json.dumps({"rc": rc, "wall_s": wall, "peak_rss_bytes": peak_bytes}))
"""


def _run_cli_measured(argv: list[str]) -> dict[str, Any]:
    """Run ``annizarr`` argv in a fresh subprocess; return {rc, wall_s, peak_rss_bytes}."""
    proc = subprocess.run(
        [sys.executable, "-c", _RUNNER, json.dumps(argv)],
        capture_output=True,
        text=True,
        check=False,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    assert lines, f"runner produced no output; stderr:\n{proc.stderr}"
    result: dict[str, Any] = json.loads(lines[-1])
    assert result["rc"] == 0, f"annizarr {argv} failed (rc={result['rc']}); stderr:\n{proc.stderr}"
    return result


def _build_csr(
    n_obs: int, n_vars: int, density: float, seed: int, name_prefix: str
) -> tuple[sp.csr_matrix, pd.DataFrame]:
    """Build data/indices/indptr directly (no per-row loop): oversample random (row, col)
    pairs, sort, dedupe — one vectorised pass rather than ~1000s of ``rng.choice`` calls."""
    rng = np.random.default_rng(seed)
    target = int(n_obs * n_vars * density)
    rows = rng.integers(0, n_obs, size=target, dtype=np.int64)
    cols = rng.integers(0, n_vars, size=target, dtype=np.int64)
    order = np.lexsort((cols, rows))
    rows, cols = rows[order], cols[order]
    keep = np.empty(len(rows), dtype=bool)
    keep[0] = True
    keep[1:] = (rows[1:] != rows[:-1]) | (cols[1:] != cols[:-1])
    rows, cols = rows[keep], cols[keep]
    indptr = np.searchsorted(rows, np.arange(n_obs + 1)).astype(np.int64)
    data = rng.random(len(rows)).astype(np.float32) * 10 + 0.1
    X = sp.csr_matrix((data, cols.astype(np.int32), indptr), shape=(n_obs, n_vars))
    obs = pd.DataFrame(
        {"cell_type": pd.Categorical(rng.choice(CATEGORIES, n_obs), categories=CATEGORIES)},
        index=[f"{name_prefix}_{i}" for i in range(n_obs)],
    )
    return X, obs


@pytest.fixture(scope="session")
def large_h5ad(tmp_path_factory: pytest.TempPathFactory) -> Path:
    import anndata as ad

    base = tmp_path_factory.mktemp("large_store_src")
    path = base / "in.h5ad"
    X, obs = _build_csr(N_OBS, N_VARS, DENSITY, seed=0, name_prefix="cell")
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(N_VARS)])
    ad.AnnData(X=X, obs=obs, var=var).write_h5ad(path)
    return path


@pytest.fixture(scope="session")
def eager_max_bytes_config(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A low eager_max_bytes so convert's backed=None auto-select picks backed (X is
    ~120MB on disk here; well above this threshold)."""
    path = tmp_path_factory.mktemp("large_store_cfg") / "low_eager.toml"
    path.write_text("[io]\neager_max_bytes = 50000000\n")
    return path


def _sample_rows(n: int, k: int, seed: int) -> np.ndarray:
    return np.sort(np.random.default_rng(seed).choice(n, size=k, replace=False))


def test_large_store_pipeline_is_memory_bounded(
    tmp_path_factory: pytest.TempPathFactory, large_h5ad: Path, eager_max_bytes_config: Path
) -> None:
    out_dir = tmp_path_factory.mktemp("large_store_out")
    csr_store = out_dir / "csr.zarr"
    csc_store = out_dir / "csc.zarr"
    sorted_store = out_dir / "sorted.zarr"
    results: list[tuple[str, float, int, int]] = []

    def _step(name: str, argv: list[str]) -> dict[str, Any]:
        r = _run_cli_measured(argv)
        ceiling = CEILINGS[name]
        results.append((name, r["wall_s"], r["peak_rss_bytes"], ceiling))
        assert r["peak_rss_bytes"] < ceiling, (
            f"{name}: peak RSS {r['peak_rss_bytes'] / 1024**2:.1f}MB exceeds ceiling {ceiling / 1024**2:.0f}MB"
        )
        return r

    # 1. convert -> csr (backed, auto-selected via the low eager_max_bytes config)
    _step(
        "convert_csr",
        ["-q", "convert", str(large_h5ad), "-o", str(csr_store), "--config", str(eager_max_bytes_config)],
    )
    rows = _sample_rows(N_OBS, 5, seed=42)
    with h5py.File(large_h5ad) as f:
        expected = sparse_dataset(f["X"])[rows].toarray()
    got = sparse_dataset(zarr.open_group(str(csr_store), mode="r")["X"])[rows].toarray()
    np.testing.assert_array_equal(got, expected)

    # 2. convert --x-storage csc (backed CSR -> csc: the new streamed transpose, never
    #    materialising X — see write_transposed_sparse)
    _step(
        "convert_csc",
        [
            "-q",
            "convert",
            str(large_h5ad),
            "-o",
            str(csc_store),
            "--config",
            str(eager_max_bytes_config),
            "--x-storage",
            "csc",
        ],
    )
    got_csc = sparse_dataset(zarr.open_group(str(csc_store), mode="r")["X"])[rows].toarray()
    np.testing.assert_array_equal(got_csc, expected)

    # 3. add-expr --format csc on the csr store (streamed lognorm CSR -> csc via the same
    #    write_transposed_sparse engine, with row_scale)
    _step("add_expr_csc", ["-q", "add-expr", str(csr_store), "--format", "csc"])
    root = zarr.open_group(str(csr_store), mode="r")
    gexp = sparse_dataset(root["layers/gexp"])[rows].toarray()
    x_rows = sparse_dataset(root["X"])[rows].toarray().astype(np.float64)
    sums = x_rows.sum(axis=1)
    factor = np.divide(1e4, sums, out=np.zeros_like(sums), where=sums > 0)
    expected_gexp = np.log1p(x_rows * factor[:, None]).astype(np.float32)
    np.testing.assert_allclose(gexp, expected_gexp, rtol=1e-4)

    # 4. sort --by cell_type on the csr store (now carrying a gexp layer — sort re-derives
    #    a lone gexp layer on the sorted output)
    _step(
        "sort",
        ["-q", "sort", str(csr_store), "-o", str(sorted_store), "--by", "cell_type"],
    )
    sorted_root = zarr.open_group(str(sorted_store), mode="r")
    from anndata.io import read_elem

    sorted_obs = read_elem(sorted_root["obs"])
    codes = sorted_obs["cell_type"].cat.codes.to_numpy()
    assert np.all(np.diff(codes) >= 0), "sort did not produce contiguous cell_type blocks"
    assert "gexp" in list(sorted_root["layers"])

    # 5. append a 2 000-cell slice (append requires another *converted* AnnData zarr store
    #    with matching var/obs schema, not a raw h5ad)
    extra_h5ad = out_dir / "extra.h5ad"
    import anndata as ad

    X_extra, obs_extra = _build_csr(N_APPEND, N_VARS, DENSITY, seed=1, name_prefix="extra")
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(N_VARS)])
    ad.AnnData(X=X_extra, obs=obs_extra, var=var).write_h5ad(extra_h5ad)
    extra_store = out_dir / "extra.zarr"
    assert _run_cli_measured(["-q", "convert", str(extra_h5ad), "-o", str(extra_store)])["rc"] == 0

    _step(
        "append",
        ["-q", "append", str(csr_store), str(extra_store), "--drop-derived"],
    )
    appended_root = zarr.open_group(str(csr_store), mode="r")
    n_obs_after, _ = appended_root["X"].attrs["shape"]
    assert n_obs_after == N_OBS + N_APPEND
    appended_tail = sparse_dataset(appended_root["X"])[np.arange(N_OBS, N_OBS + 5)].toarray()
    expected_tail = X_extra[:5].toarray()
    np.testing.assert_array_equal(appended_tail, expected_tail)

    print("\nstep                 wall_s   peak_rss_MB  ceiling_MB")
    for name, wall_s, peak, ceiling in results:
        print(f"{name:<20} {wall_s:>7.2f}  {peak / 1024**2:>10.1f}  {ceiling / 1024**2:>9.0f}")
