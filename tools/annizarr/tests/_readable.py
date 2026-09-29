from __future__ import annotations

import zarr


def assert_anndata_readable(store) -> None:
    """Open ``store`` with ``ad.read_zarr``, and assert encoding-type/-version attrs on
    every array/group in it.

    ``store`` is a path/URI, or an already-opened ``zarr.Group`` (e.g. an icechunk-backed
    store from ``open_input_group``, which isn't a plain zarr path ``ad.read_zarr`` can open).
    """
    import anndata as ad

    root = store if isinstance(store, zarr.Group) else zarr.open_group(str(store), mode="r")
    got = ad.read_zarr(root)
    assert got.n_obs >= 0
    assert got.n_vars >= 0

    assert root.attrs.get("encoding-type") == "anndata"
    assert root.attrs.get("encoding-version")

    def _walk(node) -> None:
        enc = node.attrs.get("encoding-type")
        assert enc, f"missing encoding-type on {node.path!r}"
        assert node.attrs.get("encoding-version"), f"missing encoding-version on {node.path!r}"
        if enc in ("csr_matrix", "csc_matrix"):
            return  # data/indices/indptr are raw structural children, not individually encoded
        if isinstance(node, zarr.Group):
            for child_name in node:
                _walk(node[child_name])

    for top_name in root:
        _walk(root[top_name])
