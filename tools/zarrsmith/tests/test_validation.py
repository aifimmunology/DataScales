import numpy as np
from anndata import AnnData

from zarrsmith.config import ValidationConfig
from zarrsmith.validation import ValidationError, validate_single_cell_anndata


def test_validate_non_spatial_ok() -> None:
    adata = AnnData(X=np.ones((3, 4)))
    cfg = ValidationConfig()

    result = validate_single_cell_anndata(adata, cfg)

    assert result.ok


def test_validate_reject_spatial_from_uns() -> None:
    adata = AnnData(X=np.ones((3, 4)))
    adata.uns["spatial"] = {"library": {}}

    cfg = ValidationConfig(reject_spatial=True)

    try:
        validate_single_cell_anndata(adata, cfg)
        raised = False
    except ValidationError:
        raised = True

    assert raised


def test_validate_reject_empty() -> None:
    adata = AnnData(X=np.ones((0, 4)))
    cfg = ValidationConfig(require_non_empty=True)

    try:
        validate_single_cell_anndata(adata, cfg)
        raised = False
    except ValidationError:
        raised = True

    assert raised
