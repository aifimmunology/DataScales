from ..config import GCS, LOCAL
from .base import Store
from .gcs import GcsStore
from .local import LocalStore
from .mirror import Mirror, MirrorError

__all__ = ["STORE", "MIRROR", "Store", "LocalStore", "GcsStore", "Mirror", "MirrorError", "build"]


def build(local, gcs) -> tuple[Store | None, Mirror | None]:
    if local:
        store = LocalStore(local)
        return store, Mirror(gcs, store) if gcs else None
    return (GcsStore(gcs) if gcs else None), None


STORE, MIRROR = build(LOCAL, GCS)
