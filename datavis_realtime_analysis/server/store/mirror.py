from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..config import Location
from .gcs import bucket, delete_gcs_prefix, media_type
from .local import LocalStore

MIRROR_THREADS = 16


class MirrorError(Exception):
    pass


class Mirror:
    def __init__(self, loc: Location, local: LocalStore, client=None):
        self.loc, self.uri, self.local, self._client = loc, loc.uri, local, client

    def _bucket(self):
        return self._client.bucket(self.loc.bucket) if self._client else bucket(self.loc)

    def delete_prefix(self, rel: str) -> int:
        return len(delete_gcs_prefix(self.loc, rel, self._client))

    def copy(self, *rels: str) -> int:
        """Chunks go before metadata, deepest zarr.json first, so a bucket reader never finds a node without its data."""
        files = [f for rel in rels for f in self.local.files(rel)]
        chunks = [f for f in files if f.name != "zarr.json"]
        metas = sorted((f for f in files if f.name == "zarr.json"), key=lambda f: -len(f.parts))
        try:
            b = self._bucket()

            def upload(f: Path) -> None:
                rel = f.relative_to(self.local.root).as_posix()
                b.blob(self.loc.key(rel)).upload_from_filename(str(f), content_type=media_type(rel))

            with ThreadPoolExecutor(MIRROR_THREADS) as pool:
                list(pool.map(upload, chunks))
            for f in metas:
                upload(f)
        except Exception as e:
            raise MirrorError(f"{len(files)} files -> {self.loc.uri}: {e}") from e
        return len(files)

    def replace(self, rel: str) -> int:
        n = self.copy(rel)
        keep = {self.loc.key(f.relative_to(self.local.root).as_posix()) for f in self.local.files(rel)}
        try:
            prefix = self.loc.key(rel).rstrip("/") + "/"
            for blob in [b for b in self._bucket().list_blobs(prefix=prefix) if b.name not in keep]:
                blob.delete()
        except Exception as e:
            raise MirrorError(f"pruning {self.loc.uri}/{rel}: {e}") from e
        return n
