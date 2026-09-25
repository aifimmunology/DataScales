"""scizarr-ic — git-like version control for Zarr stores, backed by Icechunk.

Python API entry point is :class:`Repo`::

    from scizarr_ic import Repo
    repo = Repo.init("data.zarr", "data.icechunk")   # import a zarr store
    g = repo.writable(); ...; repo.commit("edit obs")  # stage + commit
    repo.checkout("experiment", create=True)           # branch
    for snap in repo.log(): print(snap.id, snap.message)

The command line (``scizarr-ic`` / ``scz``) exposes ``init``, ``commit``, ``log``,
``tree``, ``checkout`` and ``cherrypick``.
"""
from .errors import ScizarrError
from .repo import DEFAULT_BRANCH, Repo

__all__ = ["Repo", "ScizarrError", "DEFAULT_BRANCH"]
__version__ = "0.1.0"
