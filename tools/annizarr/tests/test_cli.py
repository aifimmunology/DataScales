from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import anndata as ad
import h5py
import numpy as np
import pytest

from _builders import make_adata, make_h5ad
from _readable import assert_anndata_readable
from annizarr._cli import main
from annizarr._version import __version__


def _make_10x_h5(base: Path) -> Path:
    """A minimal Cell Ranger v3 HDF5 file (3 genes x 2 barcodes)."""
    h5_path = base / "matrix.h5"
    with h5py.File(h5_path, "w") as f:
        m = f.create_group("matrix")
        m.create_dataset("barcodes", data=np.array([b"CELL1-1", b"CELL2-1"]))
        m.create_dataset("data", data=np.array([1.0, 3.0, 2.0], dtype=np.float32))
        m.create_dataset("indices", data=np.array([0, 2, 1], dtype=np.int32))
        m.create_dataset("indptr", data=np.array([0, 2, 3], dtype=np.int32))
        m.create_dataset("shape", data=np.array([3, 2], dtype=np.int32))
        feat = m.create_group("features")
        feat.create_dataset("id", data=np.array([b"ENSG001", b"ENSG002", b"ENSG003"]))
        feat.create_dataset("name", data=np.array([b"GENEA", b"GENEB", b"GENEC"]))
        feat.create_dataset("feature_type", data=np.array([b"Gene Expression"] * 3))
    return h5_path


# ── convert ──────────────────────────────────────────────────────────────────


def test_convert_sniffs_h5ad_content_regardless_of_extension(tmp_path: Path) -> None:
    # h5ad content under a non-.h5ad extension
    h5 = make_h5ad(tmp_path, "in.h5", adata=make_adata(n_obs=4, n_vars=3))
    out = tmp_path / "out.zarr"
    assert main(["convert", str(h5), "-o", str(out)]) == 0
    assert_anndata_readable(out)
    assert ad.read_zarr(str(out)).n_obs == 4


def test_convert_sniffs_10x_v3(tmp_path: Path) -> None:
    h5 = _make_10x_h5(tmp_path)
    out = tmp_path / "out.zarr"
    assert main(["convert", str(h5), "-o", str(out)]) == 0
    got = ad.read_zarr(str(out))
    assert got.n_obs == 2 and got.n_vars == 3


def test_convert_from_override(tmp_path: Path) -> None:
    h5 = _make_10x_h5(tmp_path)
    out = tmp_path / "out.zarr"
    assert main(["convert", str(h5), "-o", str(out), "--from", "10x"]) == 0
    assert ad.read_zarr(str(out)).n_vars == 3


def test_convert_concatenates_multiple_inputs(tmp_path: Path) -> None:
    a = make_h5ad(tmp_path, "a.h5ad", adata=make_adata(n_obs=4, seed=0))
    b = make_h5ad(tmp_path, "b.h5ad", adata=make_adata(n_obs=3, seed=1))
    out = tmp_path / "out.zarr"
    assert main(["convert", str(a), str(b), "-o", str(out)]) == 0
    assert ad.read_zarr(str(out)).n_obs == 7


def test_obs_columns_requires_two_inputs(tmp_path: Path) -> None:
    h5 = make_h5ad(tmp_path, "a.h5ad")
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "convert",
                str(h5),
                "-o",
                str(tmp_path / "out.zarr"),
                "--obs-columns",
                "cell_type",
            ]
        )
    assert exc.value.code == 2


def test_backed_and_eager_are_mutually_exclusive(tmp_path: Path) -> None:
    h5 = make_h5ad(tmp_path, "in.h5ad")
    with pytest.raises(SystemExit) as exc:
        main(["convert", str(h5), "-o", str(tmp_path / "out.zarr"), "--backed", "--eager"])
    assert exc.value.code == 2


def test_eager_flag_forces_eager_load(tmp_path: Path) -> None:
    """--eager overrides auto-select even when eager_max_bytes would otherwise pick backed."""
    h5 = make_h5ad(tmp_path, "in.h5ad")
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text("[io]\neager_max_bytes = 0\n")
    out = tmp_path / "out.zarr"
    with patch("anndata.read_h5ad", wraps=ad.read_h5ad) as mocked:
        assert main(["convert", str(h5), "-o", str(out), "--eager", "--config", str(cfg_file)]) == 0
    assert "backed" not in mocked.call_args_list[0].kwargs


def test_overwrite_gating(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    h5 = make_h5ad(tmp_path, "in.h5ad")
    out = tmp_path / "out.zarr"
    assert main(["convert", str(h5), "-o", str(out)]) == 0
    capsys.readouterr()

    rc = main(["convert", str(h5), "-o", str(out)])
    assert rc == 1
    assert "error:" in capsys.readouterr().err

    assert main(["convert", str(h5), "-o", str(out), "--overwrite"]) == 0


def test_missing_input_path(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["convert", str(tmp_path / "missing.h5ad"), "-o", str(tmp_path / "out.zarr")])
    assert rc == 1
    assert "error:" in capsys.readouterr().err


def test_remote_output_without_ic_requires_flag(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    h5 = make_h5ad(tmp_path, "in.h5ad")
    rc = main(["convert", str(h5), "-o", "s3://bucket/out.zarr"])
    assert rc == 1
    assert "--ic" in capsys.readouterr().err


def test_quiet_silences_info(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    h5 = make_h5ad(tmp_path, "in.h5ad")

    assert main(["convert", str(h5), "-o", str(tmp_path / "loud.zarr")]) == 0
    assert "wrote" in capsys.readouterr().err

    assert main(["-q", "convert", str(h5), "-o", str(tmp_path / "quiet.zarr")]) == 0
    assert "wrote" not in capsys.readouterr().err


# ── add-expr / rechunk / sort / append ──────────────────────────────────────


def test_add_expr_happy_path(tmp_path: Path) -> None:
    h5 = make_h5ad(tmp_path, "in.h5ad")
    out = tmp_path / "out.zarr"
    assert main(["convert", str(h5), "-o", str(out)]) == 0
    assert main(["add-expr", str(out), "--chunk-elems", "16"]) == 0
    assert_anndata_readable(out)
    assert "gexp" in ad.read_zarr(str(out)).layers


def test_rechunk_happy_path(tmp_path: Path) -> None:
    h5 = make_h5ad(tmp_path, "in.h5ad", adata=make_adata(n_obs=8, n_vars=5))
    store = tmp_path / "store.zarr"
    assert main(["convert", str(h5), "-o", str(store)]) == 0
    out = tmp_path / "rechunked.zarr"
    assert main(["rechunk", str(store), "-o", str(out), "--sparse-flat-chunk", "4"]) == 0
    got = ad.read_zarr(str(out))
    assert got.n_obs == 8


def test_sort_happy_path(tmp_path: Path) -> None:
    # non-decreasing codes after sort is checked regardless of the input's initial order,
    # so make_adata's random "cell_type" categorical column exercises the same path
    h5 = make_h5ad(tmp_path, "in.h5ad", adata=make_adata(n_obs=6, n_vars=4, seed=2))
    store = tmp_path / "store.zarr"
    assert main(["convert", str(h5), "-o", str(store)]) == 0
    out = tmp_path / "sorted.zarr"
    assert main(["sort", str(store), "-o", str(out), "--by", "cell_type"]) == 0
    got = ad.read_zarr(str(out))
    codes = got.obs["cell_type"].cat.codes.to_numpy()
    assert (np.diff(codes) >= 0).all()


def test_append_happy_path(tmp_path: Path) -> None:
    # store side (a) must carry no obsm/obsp/layers, or append needs --drop-derived/-y
    # (see test_append_without_consent_errors_and_exits_1); the cells side (b) is unaffected
    a_adata = make_adata(n_obs=4, seed=0)
    a_adata.obsm.clear()
    a = make_h5ad(tmp_path, "a.h5ad", adata=a_adata)
    b = make_h5ad(tmp_path, "b.h5ad", adata=make_adata(n_obs=2, seed=1))
    sa, sb = tmp_path / "a.zarr", tmp_path / "b.zarr"
    assert main(["convert", str(a), "-o", str(sa)]) == 0
    assert main(["convert", str(b), "-o", str(sb)]) == 0
    assert main(["append", str(sa), str(sb), "-y"]) == 0
    assert ad.read_zarr(str(sa)).n_obs == 6


def test_append_without_consent_errors_and_exits_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a = make_h5ad(tmp_path, "a.h5ad", adata=make_adata(n_obs=4, seed=0))  # has obsm -> append would drop it
    b = make_h5ad(tmp_path, "b.h5ad", adata=make_adata(n_obs=2, seed=1))
    sa, sb = tmp_path / "a.zarr", tmp_path / "b.zarr"
    assert main(["convert", str(a), "-o", str(sa)]) == 0
    assert main(["convert", str(b), "-o", str(sb)]) == 0

    # non-TTY stdin (the pytest default): no prompt, just an error naming the escape hatches
    assert main(["append", str(sa), str(sb)]) == 1
    err = capsys.readouterr().err
    assert "error:" in err and "--drop-derived" in err and "-y" in err
    assert ad.read_zarr(str(sa)).n_obs == 4  # nothing was mutated


def test_append_with_yes_drops_obsm_and_succeeds(tmp_path: Path) -> None:
    a = make_h5ad(tmp_path, "a.h5ad", adata=make_adata(n_obs=4, seed=0))
    b = make_h5ad(tmp_path, "b.h5ad", adata=make_adata(n_obs=2, seed=1))
    sa, sb = tmp_path / "a.zarr", tmp_path / "b.zarr"
    assert main(["convert", str(a), "-o", str(sa)]) == 0
    assert main(["convert", str(b), "-o", str(sb)]) == 0

    assert main(["append", str(sa), str(sb), "-y"]) == 0
    got = ad.read_zarr(str(sa))
    assert got.n_obs == 6
    assert "X_pca" not in got.obsm


def test_append_with_drop_derived_succeeds(tmp_path: Path) -> None:
    a = make_h5ad(tmp_path, "a.h5ad", adata=make_adata(n_obs=4, seed=0))
    b = make_h5ad(tmp_path, "b.h5ad", adata=make_adata(n_obs=2, seed=1))
    sa, sb = tmp_path / "a.zarr", tmp_path / "b.zarr"
    assert main(["convert", str(a), "-o", str(sa)]) == 0
    assert main(["convert", str(b), "-o", str(sb)]) == 0

    assert main(["append", str(sa), str(sb), "--drop-derived"]) == 0
    assert ad.read_zarr(str(sa)).n_obs == 6


# ── --version / entry points ─────────────────────────────────────────────────


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == f"annizarr {__version__}"


@pytest.mark.parametrize("exe", ["annizarr", "anz"])
def test_console_script_version(exe: str) -> None:
    path = shutil.which(exe)
    if path is None:
        pytest.skip(f"the `{exe}` console script is not on PATH")
    proc = subprocess.run([path, "--version"], capture_output=True, text=True)
    assert proc.returncode == 0
    assert proc.stdout.strip() == f"annizarr {__version__}"
