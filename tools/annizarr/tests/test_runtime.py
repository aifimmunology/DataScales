from __future__ import annotations

import os

from annizarr._core._runtime import configure_runtime


def test_configure_runtime_reapplies_zarr_thread_sizing_every_call() -> None:
    """A second op in the same process must not keep the first call's thread sizing."""
    import zarr

    configure_runtime(2)
    configure_runtime(4)
    assert zarr.config.get("threading.max_workers") == max(4, os.cpu_count() or 1)
