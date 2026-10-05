from __future__ import annotations

from typing import TYPE_CHECKING

from annizarr._core._runtime import stage
from annizarr._core._zarr import get_group
from annizarr._sources._readers import as_reader
from annizarr._writers._encoding import autoshard_setting, set_anndata_root_attrs, set_raw_group_attrs, write_elem
from annizarr._writers._matrix import write_matrix
from annizarr._writers._sparse import _local_tmp_dir

if TYPE_CHECKING:
    import anndata as ad
    import zarr

    from annizarr._core._config import AppConfig


def write_adata(adata: ad.AnnData, store: zarr.Group, cfg: AppConfig) -> None:
    set_anndata_root_attrs(store)

    with autoshard_setting(cfg.chunks.auto_shard), stage("Writing metadata (obs/var/uns/obsm/varm/obsp/varp)"):
        write_elem(store, "obs", adata.obs)
        write_elem(store, "var", adata.var)
        write_elem(store, "uns", dict(adata.uns))
        write_elem(store, "obsm", dict(adata.obsm))
        write_elem(store, "varm", dict(adata.varm))
        write_elem(store, "obsp", dict(adata.obsp))
        write_elem(store, "varp", dict(adata.varp))

    tmp_dir = _local_tmp_dir(store)
    x_reader = as_reader(adata.X, cfg=cfg, tmp_dir=tmp_dir)
    try:
        with stage(f"Writing X (shape={x_reader.shape}, {cfg.io.x_storage})"):
            write_matrix(store, "X", x_reader, cfg)
    finally:
        x_reader.close()

    if adata.layers:
        with autoshard_setting(cfg.chunks.auto_shard):
            write_elem(store, "layers", {})
        layers_group = get_group(store, "layers")
        for name, data in adata.layers.items():
            reader = as_reader(data, cfg=cfg, tmp_dir=tmp_dir)
            try:
                with stage(f"Writing layers/{name} (shape={reader.shape})"):
                    write_matrix(layers_group, name, reader, cfg)
            finally:
                reader.close()

    if adata.raw is not None:
        raw_group = store.require_group("raw")
        set_raw_group_attrs(raw_group)
        with autoshard_setting(cfg.chunks.auto_shard):
            write_elem(raw_group, "var", adata.raw.var)
            write_elem(raw_group, "varm", dict(adata.raw.varm))
        reader = as_reader(adata.raw.X, cfg=cfg, tmp_dir=tmp_dir)
        try:
            with stage(f"Writing raw/X (shape={reader.shape})"):
                write_matrix(raw_group, "X", reader, cfg)
        finally:
            reader.close()
