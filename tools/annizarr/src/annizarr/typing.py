from __future__ import annotations

import os
from typing import Literal, TypeAlias

__all__ = ["Backend", "PathLike", "XStorage"]

XStorage: TypeAlias = Literal["csr", "csc", "dense"]
Backend: TypeAlias = Literal["zarr", "icechunk"]
PathLike: TypeAlias = str | os.PathLike[str]
