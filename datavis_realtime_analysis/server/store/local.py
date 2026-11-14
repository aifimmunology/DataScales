import ctypes
import hashlib
import json
import os
import re
import shutil
from collections.abc import Callable
from pathlib import Path

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, Response

from ..config import Location
from .base import Store
from .gcs import media_type

EAGER_BYTES = 8 << 20
STAGING = ".datavis_staging"
_SAFE_SEGMENT = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]*")
_RENAME_EXCHANGE = 2


def exchange(a: Path, b: Path) -> bool:
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        rc = libc.renameat2(-100, os.fsencode(a), -100, os.fsencode(b), _RENAME_EXCHANGE)
    except (OSError, AttributeError):
        return False
    return rc == 0


def _etag(st) -> str:
    return f'"{hashlib.md5(f"{st.st_mtime_ns}-{st.st_size}".encode(), usedforsecurity=False).hexdigest()}"'


class LocalStore(Store):
    is_gcs = False

    def __init__(self, loc: Location | str):
        self.uri = loc.uri if isinstance(loc, Location) else str(loc).rstrip("/")
        self.root = Path(self.uri).resolve()

    def serve(self, path: str, request: Request) -> Response:
        if not all(_SAFE_SEGMENT.fullmatch(s) for s in path.split("/")):
            raise HTTPException(404, "Not found")
        try:
            file = (self.root / path).resolve()
            ok = file.is_relative_to(self.root) and file.is_file()
            st = file.stat() if ok else None
        except (OSError, ValueError):
            ok = False
        if not ok:
            raise HTTPException(404, "Not found")
        headers = {"ETag": _etag(st), "Cache-Control": "no-cache"}
        if headers["ETag"] in request.headers.get("if-none-match", ""):
            return Response(status_code=304, headers=headers)
        if st.st_size <= EAGER_BYTES and "range" not in request.headers:
            try:
                return Response(file.read_bytes(), media_type=media_type(path), headers=headers)
            except FileNotFoundError:
                raise HTTPException(404, "Not found")
        return FileResponse(file, media_type=media_type(path), headers=headers, stat_result=st)

    def list_children(self, rel: str) -> list[str]:
        d = self.root / rel
        return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.is_dir() else []

    def read_attrs(self, rel: str) -> dict | None:
        try:
            return json.loads((self.root / rel / "zarr.json").read_bytes()).get("attributes", {})
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as e:
            raise HTTPException(503, f"could not read {rel}: {e}")

    def delete_prefix(self, rel: str) -> int:
        d = self.root / rel
        n = sum(1 for p in d.rglob("*") if p.is_file()) if d.is_dir() else 0
        shutil.rmtree(d, ignore_errors=True)
        return n

    def files(self, rel: str) -> list[Path]:
        p = self.root / rel
        if p.is_dir():
            return [q for q in p.rglob("*") if q.is_file()]
        return [p] if p.is_file() else []

    def replace_group(self, rel: str, write: Callable) -> None:
        import zarr

        staging = self.root / STAGING
        staging.mkdir(exist_ok=True)
        tag = rel.replace("/", "~")
        new, old, final = staging / f"{tag}.new", staging / f"{tag}.old", self.root / rel
        for p in (new, old):
            shutil.rmtree(p, ignore_errors=True)
        write(zarr.open_group(str(new), mode="w"))
        final.parent.mkdir(parents=True, exist_ok=True)
        if not final.exists():
            new.rename(final)
        elif exchange(new, final):
            shutil.rmtree(new, ignore_errors=True)
        else:
            final.rename(old)
            new.rename(final)
            shutil.rmtree(old, ignore_errors=True)
