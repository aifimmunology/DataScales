from __future__ import annotations

import os
from typing import Literal

__all__ = ["Backend", "PathLike", "XStorage"]

type XStorage = Literal["csr", "csc", "dense"]
type Backend = Literal["zarr", "icechunk"]
type PathLike = str | os.PathLike[str]
