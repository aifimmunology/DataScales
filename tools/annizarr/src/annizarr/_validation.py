from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from annizarr.errors import ValidationError

if TYPE_CHECKING:
    from anndata import AnnData

    from annizarr._config import ValidationConfig

__all__ = ["ValidationResult", "validate_single_cell_anndata"]


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Outcome of :func:`validate_single_cell_anndata`.

    Parameters
    ----------
    ok
        Always ``True`` — a failed validation raises instead of returning ``ok=False``.
    warnings
        Non-fatal issues found (e.g. duplicate obs/var names); callers log these.
    """

    ok: bool
    warnings: list[str]


def _has_spatial_markers(adata: AnnData) -> bool:
    if "spatial" in adata.uns:
        return True

    # Common conventions in AnnData objects carrying spatial coordinates.
    for key in adata.obsm:
        if "spatial" in key.lower() or key.lower().startswith("x_spatial"):
            return True

    return False


def validate_single_cell_anndata(adata: AnnData, cfg: ValidationConfig) -> ValidationResult:
    """Validate that ``adata`` is a non-spatial, single-cell AnnData ready to convert.

    Parameters
    ----------
    adata
        Input to validate.
    cfg
        Thresholds and toggles for the checks below.

    Returns
    -------
    ValidationResult
        ``ok=True`` with any non-fatal warnings (duplicate obs/var names).

    Raises
    ------
    ValidationError
        Too few observations/variables, no ``X``, or spatial markers detected while
        ``cfg.reject_spatial`` is set.
    """
    warnings: list[str] = []

    if cfg.require_non_empty:
        if adata.n_obs < cfg.min_obs:
            raise ValidationError(f"AnnData has too few observations: {adata.n_obs} < {cfg.min_obs}")
        if adata.n_vars < cfg.min_vars:
            raise ValidationError(f"AnnData has too few variables: {adata.n_vars} < {cfg.min_vars}")

    if adata.X is None:
        raise ValidationError("AnnData has no expression matrix in X.")

    if cfg.reject_spatial and _has_spatial_markers(adata):
        raise ValidationError(
            "Input appears spatial (detected spatial markers in uns/obsm), "
            "but this converter run is configured for non-spatial single-cell AnnData only."
        )

    if adata.obs_names.has_duplicates:
        warnings.append("obs_names contain duplicates; downstream tools may require uniqueness.")

    if adata.var_names.has_duplicates:
        warnings.append("var_names contain duplicates; downstream tools may require uniqueness.")

    return ValidationResult(ok=True, warnings=warnings)
