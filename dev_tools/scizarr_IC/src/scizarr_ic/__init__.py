"""scizarr-ic: git-like version control for Zarr stores, backed by Icechunk.

Everything goes through :class:`Repo`, from a notebook or a script::

    from scizarr_ic import Repo
    repo = Repo.init("data.zarr", "data.icechunk")      # import a zarr store
    g = repo.open_zarr("w"); ...; repo.commit("edit obs")  # stage + commit
    repo.checkout("experiment", create=True)              # branch
    repo.log(); repo.branches(); repo.tree()              # history, shown git-style

    Repo("gs://bucket/store", anonymous=True)             # public bucket, no credentials
    Repo("/mnt/store", origin="s3://bucket/store")        # read a mirror, write to the bucket
"""
from .errors import ScizarrError
from .repo import DEFAULT_BRANCH, Repo

__all__ = ["Repo", "ScizarrError", "DEFAULT_BRANCH"]
__version__ = "0.3.0"
