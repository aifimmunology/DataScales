"""Backend configuration from the environment.

LOCAL_DATA    AnnData zarr group on disk, served and written directly
GCS_DATA      the same group in a bucket: alone it is the store, with LOCAL_DATA it is the mirror
EMBED_METHOD  pca | latent | latent:<obsm key>; empty lets the pipeline pick per store
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Location:
    uri: str

    @property
    def gcs(self) -> bool:
        return self.uri.startswith("gs://")

    @property
    def bucket(self) -> str:
        return self.uri[len("gs://"):].partition("/")[0]

    @property
    def prefix(self) -> str:
        return self.uri[len("gs://"):].partition("/")[2]

    @property
    def root(self) -> str:
        return self.uri

    def key(self, rel: str) -> str:
        return f"{self.prefix}/{rel}" if self.prefix else rel


def _location(value: str) -> Location | None:
    return Location(value.rstrip("/")) if value else None


LOCAL = _location(os.environ.get("LOCAL_DATA", ""))
GCS = _location(os.environ.get("GCS_DATA", ""))
if LOCAL and LOCAL.gcs:
    raise RuntimeError("LOCAL_DATA must be a filesystem path")
if GCS and not GCS.gcs:
    raise RuntimeError("GCS_DATA must be a gs:// prefix")

SOURCE = LOCAL or GCS
MIRROR = GCS if LOCAL and GCS else None
EMBED_METHOD = os.environ.get("EMBED_METHOD", "")
MAX_JOBS = 50
