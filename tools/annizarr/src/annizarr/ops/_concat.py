from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anndata as ad

from annizarr._core._config import AppConfig, load_config, resolve_backend_cfg
from annizarr._core._runtime import configure_runtime, stage
from annizarr._core._validation import validate_single_cell_anndata
from annizarr._sources import close_backed_if_needed, ensure_csr, load_h5ad
from annizarr._storage import check_output_target, open_output_store
from annizarr._writers import _write_concatenated_csr, _write_concatenated_dense
from annizarr._writers._encoding import autoshard_setting, set_anndata_root_attrs, write_elem
from annizarr.errors import AnzError, ConversionError
from annizarr.ops._result import OpResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    from annizarr.typing import PathLike

logger = logging.getLogger(__name__)


def concat(
    paths: Sequence[PathLike],
    *,
    output: PathLike,
    cfg: AppConfig | None = None,
    branch: str | None = None,
    message: str | None = None,
) -> OpResult:
    """Concatenate multiple .h5ad files along obs (rows) into a single zarr store (X/obs/var only)."""
    import pandas as pd

    if cfg is None:
        cfg = load_config()
    if not paths:
        raise ConversionError("concat requires at least one input file.")

    if cfg.io.x_storage == "csc":
        raise ConversionError("x_storage='csc' is not supported for multi-h5ad concat. Use 'csr' or 'dense'.")

    cfg = resolve_backend_cfg(cfg)
    configure_runtime(cfg.chunks.cpus)
    if cfg.grouping.enabled:
        raise ConversionError("grouping (sort_by) is only supported by convert for now.")

    inputs = [Path(p) for p in paths]
    output_path = Path(output)
    check_output_target(output_path, cfg)
    ad.settings.zarr_write_format = 3

    adatas: list[ad.AnnData] = []
    try:
        for p in inputs:
            adata, _ = load_h5ad(p, cfg)
            adatas.append(adata)

        ref_var_names = adatas[0].var_names
        ref_var = adatas[0].var
        n_vars = adatas[0].n_vars
        for i, a in enumerate(adatas[1:], start=1):
            if a.n_vars != n_vars or not (a.var_names == ref_var_names).all():
                raise ConversionError(
                    f"var mismatch in {inputs[i]}: expected {n_vars} vars matching "
                    f"{inputs[0].name}, got {a.n_vars} (names+order must be identical)."
                )

        obs_columns = list(cfg.concat.obs_columns)
        if obs_columns:
            for i, a in enumerate(adatas):
                missing = [c for c in obs_columns if c not in a.obs.columns]
                if missing:
                    raise ConversionError(
                        f"obs columns not found in {inputs[i].name}: {missing}. "
                        f"Requested via obs_columns; available: {list(a.obs.columns)}."
                    )
                dropped = [c for c in a.obs.columns if c not in obs_columns]
                if dropped:
                    logger.warning(
                        f"[{inputs[i].name}] dropping {len(dropped)} obs column(s) not in obs_columns: {dropped}."
                    )
            # a categorical column with mismatched categories/dtype across inputs would
            # silently coerce to a string array on concat (dropping the compact `codes`
            # encoding); fail loudly instead.
            for c in obs_columns:
                is_cat = [isinstance(a.obs[c].dtype, pd.CategoricalDtype) for a in adatas]
                if not any(is_cat):
                    continue
                if not all(is_cat):
                    have = [inputs[i].name for i, v in enumerate(is_cat) if v]
                    lack = [inputs[i].name for i, v in enumerate(is_cat) if not v]
                    raise ConversionError(
                        f"obs column '{c}' is categorical in {have} but not in {lack}; "
                        f"concatenating would coerce it to a string array (dropping the "
                        f"categorical encoding). Make '{c}' categorical in all inputs, or "
                        f"drop it from obs_columns."
                    )
                cat0 = set(adatas[0].obs[c].cat.categories)
                bad = [inputs[i].name for i, a in enumerate(adatas) if set(a.obs[c].cat.categories) != cat0]
                if bad:
                    raise ConversionError(
                        f"obs column '{c}' has mismatched categorical categories across "
                        f"inputs ({bad} differ from {inputs[0].name}); concatenating would "
                        f"coerce it to a string array (dropping the categorical encoding). "
                        f"Reconcile the categories (union them) across inputs, or drop "
                        f"'{c}' from obs_columns."
                    )
        else:
            ref_obs_cols = list(adatas[0].obs.columns)
            for i, a in enumerate(adatas[1:], start=1):
                if list(a.obs.columns) != ref_obs_cols:
                    raise ConversionError(
                        f"obs schema mismatch in {inputs[i]}: expected columns {ref_obs_cols}, "
                        f"got {list(a.obs.columns)}."
                    )

        for i, a in enumerate(adatas):
            _validate_and_warn(a, cfg, inputs[i].name)

        x_matrices: list[Any] = []
        x_dtype = None
        for i, a in enumerate(adatas):
            x, warn = ensure_csr(a.X, inputs[i].name, eager_max_bytes=cfg.io.eager_max_bytes)
            if warn:
                logger.warning(warn)
            if x_dtype is None:
                x_dtype = x.dtype
            elif x.dtype != x_dtype:
                raise ConversionError(f"X dtype mismatch: {inputs[i].name} has {x.dtype}, expected {x_dtype}.")
            x_matrices.append(x)

        n_obs_each = [a.n_obs for a in adatas]
        n_obs_total = sum(n_obs_each)

        if obs_columns:
            obs_concat = pd.concat([a.obs[obs_columns] for a in adatas], axis=0)
            for c in obs_columns:
                in_dtypes = {str(a.obs[c].dtype) for a in adatas}
                out_dtype = str(obs_concat[c].dtype)
                if in_dtypes != {out_dtype}:
                    logger.warning(f"obs column '{c}' coerced on concat: {sorted(in_dtypes)} -> {out_dtype}.")
        else:
            obs_concat = pd.concat([a.obs for a in adatas], axis=0)

        logger.info(
            f"Concatenating {len(inputs)} h5ads → {output_path} "
            f"(n_obs={n_obs_total}, n_vars={n_vars}, {cfg.io.x_storage})"
        )
        t0 = time.perf_counter()

        commit_message = message or f"annizarr concat → {output_path.name}"
        out = open_output_store(
            output_path, cfg, commit_message=commit_message, branch=branch, expected_shape=(n_obs_total, n_vars)
        )
        try:
            set_anndata_root_attrs(out.root)

            with autoshard_setting(cfg.chunks.auto_shard):
                with stage("Writing metadata (obs, var, empty obsm/varm/uns/obsp/varp)"):
                    write_elem(out.root, "obs", obs_concat)
                    write_elem(out.root, "var", ref_var)
                    write_elem(out.root, "uns", {})
                    write_elem(out.root, "obsm", {})
                    write_elem(out.root, "varm", {})
                    write_elem(out.root, "obsp", {})
                    write_elem(out.root, "varp", {})

                with stage(f"Writing X (n_obs={n_obs_total}, n_vars={n_vars}, {cfg.io.x_storage})"):
                    if cfg.io.x_storage == "dense":
                        _write_concatenated_dense(out.root, "X", x_matrices, n_obs_each, n_vars, x_dtype, cfg)
                    else:  # csr
                        _write_concatenated_csr(out.root, "X", x_matrices, n_obs_each, n_vars, x_dtype, cfg)

            snapshot_id = out.finalize()
        except BaseException:
            out.abort()
            raise
        logger.info(f"Done in {time.perf_counter() - t0:.1f}s")
        return OpResult(path=str(output_path), n_obs=n_obs_total, n_vars=n_vars, snapshot_id=snapshot_id)

    except AnzError:
        raise
    except Exception as e:
        raise ConversionError(f"Failed to concatenate h5ads: {e}") from e
    finally:
        for a in adatas:
            close_backed_if_needed(a)


def _validate_and_warn(adata: ad.AnnData, cfg: AppConfig, label: str) -> None:
    result = validate_single_cell_anndata(adata, cfg.validation)
    for w in result.warnings:
        logger.warning(f"[{label}] {w}")
