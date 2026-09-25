"""scizarr-ic — git-like version control for Zarr stores, backed by Icechunk.

Python API entry point is :class:`Repo`::

    from scizarr_ic import Repo
    repo = Repo.init("data.zarr", "data.icechunk")   # import a zarr store
    g = repo.writable(); ...; repo.commit("edit obs")  # stage + commit
    repo.checkout("experiment", create=True)           # branch
    for snap in repo.log(): print(snap.id, snap.message)

    # a read-only data-asset mount reads in place; writes resolve to its s3:// origin
    repo = Repo("/data/my_store"); print(repo.origin)
    mine = repo.copy("/results/my_store")             # or take a writable clone

The command line (``scizarr-ic`` / ``scz``) exposes ``init``, ``log``, ``tree``,
``checkout``, ``cherrypick``, ``origin`` and ``copy``; ``commit`` is Python-API only.
"""
from .errors import ScizarrError
from .repo import DEFAULT_BRANCH, Repo

__all__ = ["Repo", "ScizarrError", "DEFAULT_BRANCH"]
__version__ = "0.2.0"
