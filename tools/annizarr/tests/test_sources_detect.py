from __future__ import annotations

from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pytest
import scipy.sparse as sp
import zarr

from _builders import make_adata, make_h5ad
from _readable import assert_anndata_readable
from annizarr.config import AppConfig, ChunkConfig, IOConfig
from annizarr.errors import ConversionError
from annizarr.ops import convert
from annizarr.sources import Source, detect_format, open_source, register_source

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _cfg(**io) -> AppConfig:
    return AppConfig(io=IOConfig(**io), chunks=ChunkConfig(x_row_chunk=4, x_col_chunk=4, sparse_flat_chunk=64))


def _make_10x_v3(path: Path) -> None:
    """Minimal Cell Ranger v3 HDF5 (matrix/{...,features/{...}})."""
    with h5py.File(path, "w") as f:
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
        feat.create_dataset("genome", data=np.array([b"GRCh38"] * 3))


def _make_10x_v2(path: Path, genome: str = "GRCh38") -> None:
    """Minimal Cell Ranger v2 HDF5 (one <genome> group per top-level key)."""
    with h5py.File(path, "w") as f:
        g = f.create_group(genome)
        g.create_dataset("data", data=np.array([1.0, 3.0, 2.0], dtype=np.float32))
        g.create_dataset("indices", data=np.array([0, 2, 1], dtype=np.int32))
        g.create_dataset("indptr", data=np.array([0, 2, 3], dtype=np.int32))
        g.create_dataset("shape", data=np.array([3, 2], dtype=np.int32))
        g.create_dataset("barcodes", data=np.array([b"CELL1-1", b"CELL2-1"]))
        g.create_dataset("genes", data=np.array([b"ENSG001", b"ENSG002", b"ENSG003"]))
        g.create_dataset("gene_names", data=np.array([b"GENEA", b"GENEB", b"GENEC"]))


# ---------------------------------------------------------------------------
# detect_format
# ---------------------------------------------------------------------------


def test_detect_h5ad_by_attrs_and_by_content_not_extension(tmp_path: Path) -> None:
    p = make_h5ad(tmp_path, "in.h5ad", adata=make_adata(n_obs=4, n_vars=3))
    assert detect_format(p) == "h5ad"
    # a .h5-named file with h5ad content is still sniffed as h5ad, by content not extension
    p_h5 = make_h5ad(tmp_path, "in.h5", adata=make_adata(n_obs=4, n_vars=3))
    assert detect_format(p_h5) == "h5ad"


def test_detect_10x_v3_and_v2(tmp_path: Path) -> None:
    p_v3 = tmp_path / "matrix.h5"
    _make_10x_v3(p_v3)
    assert detect_format(p_v3) == "10x"

    p_v2 = tmp_path / "raw_gene_bc_matrices.h5"
    _make_10x_v2(p_v2)
    assert detect_format(p_v2) == "10x"


def test_detect_zarr_icechunk_dirs_and_remote_uri(tmp_path: Path) -> None:
    zarr_dir = tmp_path / "store.zarr"
    zarr.open_group(str(zarr_dir), mode="w")
    assert detect_format(zarr_dir) == "zarr"

    ic_dir = tmp_path / "repo.icechunk"
    ic_dir.mkdir()
    (ic_dir / "repo").write_text("stub")
    (ic_dir / "snapshots").mkdir()
    assert detect_format(ic_dir) == "icechunk"

    assert detect_format("s3://bucket/prefix") == "icechunk"
    assert detect_format("gs://bucket/prefix") == "icechunk"


def test_detect_guards_unrecognised_and_missing_inputs(tmp_path: Path) -> None:
    plain_dir = tmp_path / "plain_dir"
    plain_dir.mkdir()
    with pytest.raises(ConversionError, match="--from"):
        detect_format(plain_dir)  # a directory that's neither a zarr nor an icechunk store

    mystery_h5 = tmp_path / "mystery.h5"
    with h5py.File(mystery_h5, "w") as f:
        f.create_dataset("foo", data=np.arange(4))
    with pytest.raises(ConversionError, match="not recognised as h5ad or 10x"):
        detect_format(mystery_h5)  # valid HDF5, but neither h5ad nor 10x shaped

    not_hdf5 = tmp_path / "notreal.h5ad"
    not_hdf5.write_bytes(b"not an hdf5 file at all")
    with pytest.raises(ConversionError, match="not a readable HDF5"):
        detect_format(not_hdf5)

    with pytest.raises(ConversionError, match="does not exist"):
        detect_format(tmp_path / "nope.h5ad")


# ---------------------------------------------------------------------------
# open_source
# ---------------------------------------------------------------------------


def test_open_source_dispatches_by_kind_and_rejects_unknown(tmp_path: Path) -> None:
    h5ad_path = make_h5ad(tmp_path, "in.h5ad", adata=make_adata(n_obs=4, n_vars=3))
    src = open_source(str(h5ad_path), _cfg())
    assert isinstance(src, Source)
    assert src.kind == "h5ad" and src.adata.n_obs == 4 and not src.backed
    src.close()
    src.close()  # safe twice

    tenx_path = tmp_path / "matrix.h5"
    _make_10x_v3(tenx_path)
    src = open_source(str(tenx_path), _cfg())
    assert src.kind == "10x" and src.adata.n_obs == 2
    src.close()

    adata = ad.AnnData(X=sp.csr_matrix(np.eye(3, dtype=np.float32)))
    src = open_source(adata, _cfg())
    assert src.kind == "anndata" and not src.backed
    assert src.adata is adata
    src.close()

    with pytest.raises(ConversionError, match="registered kinds"):
        open_source(str(h5ad_path), _cfg(), fmt="not-a-kind")


def test_open_source_anndata_reflects_backed_state(tmp_path: Path) -> None:
    """The AnnData-passthrough source (load_anndata) must read the backed state off the
    object itself, not hardcode eager."""
    h5ad_path = make_h5ad(tmp_path, "in.h5ad", adata=make_adata(n_obs=4, n_vars=3))

    backed = ad.read_h5ad(h5ad_path, backed="r")
    src = open_source(backed, _cfg())
    assert src.backed is True
    src.close()

    eager = ad.AnnData(X=sp.csr_matrix(np.eye(3, dtype=np.float32)))
    src = open_source(eager, _cfg())
    assert src.backed is False
    src.close()


def test_open_source_h5ad_auto_selects_eager_or_backed_by_threshold(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """backed=None (default): a huge threshold picks eager (X is tiny), a zero threshold
    picks backed (any X exceeds it) -- asserted both via caplog and via Source.backed."""
    caplog.set_level("INFO", logger="annizarr")
    p = make_h5ad(tmp_path, "in.h5ad", adata=make_adata(n_obs=4, n_vars=3))

    src = open_source(str(p), _cfg(backed=None, eager_max_bytes=2 * 1024**3))
    assert not src.backed
    assert any("Auto-selected eager load" in r.message for r in caplog.records)
    src.close()

    caplog.clear()
    src = open_source(str(p), _cfg(backed=None, eager_max_bytes=0))
    assert src.backed
    assert src.adata.isbacked
    assert any("Auto-selected backed load" in r.message for r in caplog.records)
    src.close()


def test_load_h5ad_no_x_raises_conversion_error(tmp_path: Path) -> None:
    """load_h5ad's auto-select peek (_peek_x_nbytes) raises ConversionError, not a bare
    KeyError, for an h5ad-shaped file with no X."""
    from annizarr._sources._h5ad import load_h5ad

    p = tmp_path / "no_x.h5ad"
    with h5py.File(p, "w") as f:
        f.create_group("obs")
        f.create_group("var")
    with pytest.raises(ConversionError, match="has no X"):
        load_h5ad(p, _cfg(backed=None, eager_max_bytes=2 * 1024**3))


# ---------------------------------------------------------------------------
# register_source + convert() dispatch
# ---------------------------------------------------------------------------


def _toy_sniffer(path: Path) -> str | None:
    return "toy" if path.suffix == ".toy" else None


def _toy_loader(path, cfg) -> Source:
    adata = ad.AnnData(X=sp.csr_matrix(np.eye(5, 3, dtype=np.float32)))
    return Source(adata=adata, kind="toy", backed=False, warnings=("toy loaded",))


def test_registered_source_round_trips_through_convert(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    register_source("toy", _toy_loader, sniffer=_toy_sniffer)

    p = tmp_path / "input.toy"
    p.write_text("toy marker")
    out = tmp_path / "out.zarr"
    with caplog.at_level("WARNING", logger="annizarr"):
        result = convert(str(p), output=str(out), cfg=_cfg())

    assert any("toy loaded" in rec.message for rec in caplog.records)
    assert result.n_obs == 5 and result.n_vars == 3
    got = ad.read_zarr(str(out))
    assert got.n_obs == 5 and got.n_vars == 3


def test_convert_dispatch_single_multi_and_anndata_inputs(tmp_path: Path) -> None:
    """The public convert() dispatches: a single-element sequence like a bare path, two+
    h5ad paths through concat, and an in-memory AnnData through convert_adata."""
    p = make_h5ad(tmp_path, "in.h5ad", adata=make_adata(n_obs=4, n_vars=3))
    out_single = tmp_path / "single.zarr"
    convert([str(p)], output=str(out_single), cfg=_cfg())
    assert ad.read_zarr(str(out_single)).n_obs == 4

    a = make_h5ad(tmp_path, "a.h5ad", adata=make_adata(n_obs=4, n_vars=3, seed=0))
    b = make_h5ad(tmp_path, "b.h5ad", adata=make_adata(n_obs=3, n_vars=3, seed=1))
    out_concat = tmp_path / "concat.zarr"
    convert([str(a), str(b)], output=str(out_concat), cfg=_cfg())
    assert_anndata_readable(out_concat)
    assert ad.read_zarr(str(out_concat)).n_obs == 7

    adata = make_adata(n_obs=6, n_vars=4, seed=3)
    out_adata = tmp_path / "adata.zarr"
    result = convert(adata, output=str(out_adata), cfg=_cfg())
    assert result.path == str(out_adata)
    assert result.n_obs == 6
    assert result.n_vars == 4
    assert result.snapshot_id is None


def test_convert_rejects_mixed_formats_empty_sequence_and_existing_store(tmp_path: Path) -> None:
    a = make_h5ad(tmp_path, "a.h5ad")
    b = tmp_path / "b_matrix.h5"
    _make_10x_v3(b)
    with pytest.raises(ConversionError, match="concat supports h5ad inputs only"):
        convert([str(a), str(b)], output=str(tmp_path / "out.zarr"), cfg=_cfg())

    with pytest.raises(ConversionError, match="at least one input"):
        convert([], output="unused.zarr", cfg=_cfg())

    store = tmp_path / "existing.zarr"
    zarr.open_group(str(store), mode="w")
    with pytest.raises(ConversionError, match="already a store"):
        convert(str(store), output=str(tmp_path / "out2.zarr"), cfg=_cfg())


def test_convert_fmt_overrides_detection(tmp_path: Path) -> None:
    """fmt='10x' forces the 10x loader on h5ad content, so it fails the way a 10x
    parse would (proving detection was overridden, not just re-confirmed)."""
    p = make_h5ad(tmp_path, "in.h5ad")
    out = tmp_path / "out.zarr"
    with pytest.raises(ConversionError, match="Failed to read 10x H5 file"):
        convert(str(p), output=str(out), cfg=_cfg(), fmt="10x")


def test_public_facade_exports() -> None:
    from annizarr import sources

    for name in ("Source", "Loader", "Sniffer", "open_source", "register_source", "detect_format"):
        assert hasattr(sources, name)
