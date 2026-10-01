from __future__ import annotations

import logging

import pytest
from anndata import AnnData

# `from conftest import ...` is unsafe (tests/ic/conftest.py shares the bare module name
# "conftest" under pytest's default import mode), so the builders other test files import
# live in their own uniquely-named module; re-exported here for the `adata` fixture.
from _builders import make_adata, make_h5ad, make_store  # noqa: F401
from _readable import assert_anndata_readable  # noqa: F401  # re-exported for discovery


@pytest.fixture
def adata() -> AnnData:
    return make_adata()


@pytest.fixture(autouse=True)
def _annizarr_warning_capture(caplog):
    """Capture annizarr's logged warnings (ops no longer return warning lists)."""
    caplog.set_level(logging.WARNING, logger="annizarr")
