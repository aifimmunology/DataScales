from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import anndata as ad

from ..config import AppConfig, _resolve_backend_cfg
from ..engine import _stage, configure_runtime
from ..errors import ConversionError
from ..sources import _close_backed_if_needed, _ensure_csr, _load_h5ad_for_conversion
from ..storage import open_output_store
from ..validation import validate_single_cell_anndata
from ..writers import _write_concatenated_csr, _write_concatenated_dense


def convert_h5ads_to_zarr(
    input_h5ads: list[str], output_zarr: str, cfg: AppConfig
) -> list[str]:
    """Concatenate multiple .h5ad files along obs (rows) into a single zarr store.

    Requirements:
      - All inputs must share the same `var` (gene names + order, strict match).
      - obs columns: by default all inputs must share an identical obs schema
        (same names + order). If ``cfg.concat.obs_columns`` is set, each input must
        instead merely *contain* those columns; obs is projected to exactly those
        (in that order) and all other columns are dropped before concatenation. A
        selected column that is categorical must have the same categories in every
        input, else it would degrade to a string array on concat — a hard error.
      - Only X, obs, var are written. layers/raw/uns/obsm/etc. are ignored.

    Sparse output uses CSR; dense output is supported. CSC output is not supported
    for multi-file concat (would require costly transpose).
    """
    from anndata._io.specs import write_elem
    import pandas as pd

    if not input_h5ads:
        raise ConversionError("convert_h5ads_to_zarr requires at least one input file.")

    if cfg.io.x_storage == "sparse-csc":
        raise ConversionError(
            "x_storage='sparse-csc' is not supported for multi-h5ad concat. "
            "Use 'sparse-csr' or 'dense'."
        )

    cfg = _resolve_backend_cfg(cfg)
    configure_runtime(cfg.chunks.cpus)
    if cfg.grouping.enabled:
        raise ConversionError("grouping (sort_by) is only supported by convert for now.")

    inputs = [Path(p) for p in input_h5ads]
    output_path = Path(output_zarr)
    # Fail fast on a pre-existing output before the (expensive) multi-file load; the actual
    # prepare/overwrite happens in open_output_store below.
    if output_path.exists() and not cfg.io.overwrite:
        raise ConversionError(
            f"Output path already exists: {output_path}. "
            "Use overwrite=true in config or --overwrite flag."
        )
    ad.settings.zarr_write_format = 3

    adatas: list[ad.AnnData] = []
    all_warnings: list[str] = []
    try:
        # ── Pass 1: load + validate ───────────────────────────────────────────
        for p in inputs:
            adata, _ = _load_h5ad_for_conversion(p, cfg)
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
            # Explicit selection: every input must contain the named columns; obs is then
            # projected down to exactly these (in this order) at concat time — all other
            # columns are dropped. Lets files with differing *extra* columns be joined.
            for i, a in enumerate(adatas):
                missing = [c for c in obs_columns if c not in a.obs.columns]
                if missing:
                    raise ConversionError(
                        f"obs columns not found in {inputs[i].name}: {missing}. "
                        f"Requested via obs_columns; available: {list(a.obs.columns)}."
                    )
                dropped = [c for c in a.obs.columns if c not in obs_columns]
                if dropped:
                    all_warnings.append(
                        f"[{inputs[i].name}] dropping {len(dropped)} obs column(s) not in "
                        f"obs_columns: {dropped}."
                    )
            # Categorical columns must line up across inputs. If they don't (mixed
            # categorical/non-categorical, or differing category *sets*), pandas coerces the
            # column to a string (object) array on concat — dropping the compact categorical
            # `codes` encoding and making per-cell-type access far slower. Fail loudly instead.
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
                bad = [inputs[i].name for i, a in enumerate(adatas)
                       if set(a.obs[c].cat.categories) != cat0]
                if bad:
                    raise ConversionError(
                        f"obs column '{c}' has mismatched categorical categories across "
                        f"inputs ({bad} differ from {inputs[0].name}); concatenating would "
                        f"coerce it to a string array (dropping the categorical encoding). "
                        f"Reconcile the categories (union them) across inputs, or drop "
                        f"'{c}' from obs_columns."
                    )
        else:
            # Default: strict identical obs schema (names + order) against file 0.
            ref_obs_cols = list(adatas[0].obs.columns)
            for i, a in enumerate(adatas[1:], start=1):
                if list(a.obs.columns) != ref_obs_cols:
                    raise ConversionError(
                        f"obs schema mismatch in {inputs[i]}: expected columns "
                        f"{ref_obs_cols}, got {list(a.obs.columns)}."
                    )

        for i, a in enumerate(adatas):
            vr = validate_single_cell_anndata(a, cfg.validation)
            all_warnings.extend(f"[{inputs[i].name}] {w}" for w in vr.warnings)

        # ── Ensure CSR for X; verify common dtype ─────────────────────────────
        x_matrices: list[Any] = []
        x_dtype = None
        for i, a in enumerate(adatas):
            x, warn = _ensure_csr(a.X, inputs[i].name)
            if warn:
                all_warnings.append(warn)
            if x_dtype is None:
                x_dtype = x.dtype
            elif x.dtype != x_dtype:
                raise ConversionError(
                    f"X dtype mismatch: {inputs[i].name} has {x.dtype}, "
                    f"expected {x_dtype}."
                )
            x_matrices.append(x)

        n_obs_each = [a.n_obs for a in adatas]
        n_obs_total = sum(n_obs_each)

        # ── Concat obs (small; pandas) ────────────────────────────────────────
        if obs_columns:
            # Project each obs to the selected columns (fixes output order), then concat.
            # (Categorical mismatches already errored out above; any coercion left here is
            # numeric, e.g. int+float -> float — harmless, but worth a heads-up.)
            obs_concat = pd.concat([a.obs[obs_columns] for a in adatas], axis=0)
            for c in obs_columns:
                in_dtypes = {str(a.obs[c].dtype) for a in adatas}
                out_dtype = str(obs_concat[c].dtype)
                if in_dtypes != {out_dtype}:
                    all_warnings.append(
                        f"obs column '{c}' coerced on concat: {sorted(in_dtypes)} -> {out_dtype}."
                    )
        else:
            obs_concat = pd.concat([a.obs for a in adatas], axis=0)

        print(
            f"Concatenating {len(inputs)} h5ads → {output_path} "
            f"(n_obs={n_obs_total}, n_vars={n_vars}, {cfg.io.x_storage})",
            flush=True, file=sys.stderr,
        )
        t0 = time.perf_counter()

        store, finalize = open_output_store(
            output_path, cfg, commit_message=f"convert-to-zarr concat → {output_path.name}",
        )
        store.attrs["encoding-type"] = "anndata"
        store.attrs["encoding-version"] = "0.1.0"

        with _stage("Writing metadata (obs, var, empty obsm/varm/uns/obsp/varp)"):
            write_elem(store, "obs", obs_concat)
            write_elem(store, "var", ref_var)
            write_elem(store, "uns", {})
            write_elem(store, "obsm", {})
            write_elem(store, "varm", {})
            write_elem(store, "obsp", {})
            write_elem(store, "varp", {})

        with _stage(f"Writing X (n_obs={n_obs_total}, n_vars={n_vars}, {cfg.io.x_storage})"):
            if cfg.io.x_storage == "dense":
                _write_concatenated_dense(
                    store, "X", x_matrices, n_obs_each, n_vars, x_dtype, cfg,
                )
            else:  # sparse-csr
                _write_concatenated_csr(
                    store, "X", x_matrices, n_obs_each, n_vars, x_dtype, cfg,
                )

        finalize()
        print(
            f"Done in {time.perf_counter() - t0:.1f}s",
            flush=True, file=sys.stderr,
        )
        return all_warnings

    except Exception as e:
        raise ConversionError(f"Failed to concatenate h5ads: {e}") from e
    finally:
        for a in adatas:
            _close_backed_if_needed(a)
