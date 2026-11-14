"""Materialises what discovery found into scratch under canonical 10x names: plain files are symlinked,
archive members extracted, per-file gzip (GEO) decompressed. Only the files asked for are touched, so
a precheck can read a matrix header without extracting a 40 GB image.

Speed on shared filesystems: all members asked for from one .tar(.gz) come out of ONE pass over the stream
(a tar has no index, so reading members one by one re-reads the archive each time); a SpatialData store in a
.zip is extracted by several threads (many small chunk files, latency-bound). Archives over 1 GB get one INFO
line before the work starts. Sizes in messages are GB = 10^9 bytes."""
from __future__ import annotations

import gzip
import logging
import os
import shutil
import tarfile
import threading
import zipfile
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from contracts import Candidate, Member

logger = logging.getLogger(__name__)

CHUNK = 1 << 24
BIG = 10 ** 9                       # archives larger than this get a progress line
ZIP_THREADS = 8


def gb(size: int) -> str:
    """A byte count as decimal gigabytes for messages (`4.9 GB`)."""
    return f"{size / 1e9:.1f} GB"


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _note_big(container: Path, what: str) -> None:
    size = _size(container)
    if size > BIG:
        logger.info("%s from %s (%s); this can take a few minutes …", what, container.name, gb(size))


def _is_zip(path: Path) -> bool:
    return path.is_file() and zipfile.is_zipfile(path)


def _open_member(member: Member):
    """A binary stream of the member's bytes as stored (still gzipped when `member.gz`)."""
    if member.inner is None:
        return open(member.container, "rb")
    if _is_zip(member.container):
        zf = zipfile.ZipFile(member.container)
        stream = zf.open(member.inner)
        stream._spatial_ingest_owner = zf  # keep the archive open as long as the stream
        return stream
    tf = tarfile.open(member.container, "r:*")
    stream = tf.extractfile(member.inner)
    if stream is None:
        tf.close()
        raise FileNotFoundError(f"{member.display} is not a regular file")
    stream._spatial_ingest_owner = tf
    return stream


def _write(src, member: Member, target: Path) -> None:
    """Copy a raw member stream to `target` (gunzipped when `member.gz`), atomically via `.partial`."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".partial")
    data = gzip.GzipFile(fileobj=src) if member.gz else src
    with open(tmp, "wb") as dst:
        shutil.copyfileobj(data, dst, CHUNK)
    tmp.rename(target)


def _copy(member: Member, target: Path) -> None:
    with _open_member(member) as raw:
        _write(raw, member, target)


def _is_tar_member(member: Member) -> bool:
    return member.inner is not None and not _is_zip(member.container)


def fetch_member(member: Member, target: Path) -> Path:
    """One member at `target`: a symlink for a plain, uncompressed file, else extracted/decompressed."""
    if target.exists() or target.is_symlink():
        return target
    if member.inner is None and not member.gz:
        target.parent.mkdir(parents=True, exist_ok=True)
        if member.container.is_dir():
            target.symlink_to(member.container.resolve(), target_is_directory=True)
        else:
            target.symlink_to(member.container.resolve())
        return target
    _copy(member, target)
    return target


def _fetch_tar(container: Path, wanted: dict[str, list[tuple[Member, Path]]]) -> None:
    """Every wanted member of one tar in a single streaming pass (stops once all are out)."""
    _note_big(container, f"Extracting {sum(map(len, wanted.values()))} file(s)")
    left = dict(wanted)
    with tarfile.open(container, "r|*") as tf:
        for info in tf:
            jobs = left.pop(info.name, None)
            if not jobs or not info.isfile():
                continue
            _write(tf.extractfile(info), *jobs[0])
            for member, target in jobs[1:]:          # the same member wanted under a second name
                _copy(member, target)
            if not left:
                break
    if left:
        missing = ", ".join(sorted(left)[:3])
        raise FileNotFoundError(f"{container}: no member {missing} (the archive changed since it was listed?)")


def fetch(candidate: Candidate, names: Iterable[str], dest: Path) -> dict[str, Path]:
    """The named canonical files of a bundle, under `dest`. A name ending in `/` stages every file
    under that folder (e.g. `morphology_focus/`). Members of one .tar(.gz) come out in one pass."""
    todo: dict[str, Member] = {}
    for name in names:
        if name.endswith("/"):
            todo.update({k: m for k, m in candidate.files.items() if k.startswith(name)})
        elif name in candidate.files:
            todo[name] = candidate.files[name]
    out: dict[str, Path] = {}
    tars: dict[Path, dict[str, list[tuple[Member, Path]]]] = {}
    for key, member in todo.items():
        target = dest / key
        out[key] = target
        if target.exists() or target.is_symlink():
            continue
        if _is_tar_member(member):
            tars.setdefault(member.container, {}).setdefault(member.inner, []).append((member, target))
        else:
            fetch_member(member, target)
    for container, wanted in tars.items():
        _fetch_tar(container, wanted)
    return out


def bundle(candidate: Candidate, dest: Path, skip: Iterable[str] = ()) -> Path:
    """The whole bundle as a canonical folder at `dest` (files in `skip` left out)."""
    skip = set(skip)
    dest.mkdir(parents=True, exist_ok=True)
    fetch(candidate, [k for k in candidate.files if k not in skip], dest)
    return dest


def _inner(name: str) -> str:
    """An archive member name as discovery lists it (`./` and slashes stripped, `\\` as `/`)."""
    name = name.replace("\\", "/")
    while name.startswith("./"):
        name = name[2:]
    return name.strip("/")


def _junk(rel: str) -> bool:
    """macOS litter inside an archived store (`__MACOSX/`, `._*` AppleDouble files, `.DS_Store`), and member
    names that would land outside the target folder (`..`)."""
    parts = rel.split("/")
    return "__MACOSX" in parts or ".." in parts or any(p.startswith("._") or p == ".DS_Store" for p in parts)


def _in_store(raw: str, prefix: str) -> str | None:
    """The path inside the store of an archive member under the store's prefix, else None."""
    name = _inner(raw)
    if name.startswith(prefix) and len(name) > len(prefix) and not _junk(name[len(prefix):]):
        return name[len(prefix):]
    return None


def _unzip_store(container: Path, prefix: str, tmp: Path) -> None:
    """Only the store's members, by several threads (each with its own handle on the zip)."""
    with zipfile.ZipFile(container) as zf:
        jobs = [(info.filename, rel) for info in zf.infolist()
                if not info.is_dir() and (rel := _in_store(info.filename, prefix)) is not None]
    for folder in {os.path.dirname(rel) for _, rel in jobs}:
        (tmp / folder).mkdir(parents=True, exist_ok=True)
    local = threading.local()
    handles: list[zipfile.ZipFile] = []

    def one(job: tuple[str, str]) -> None:
        if not hasattr(local, "zf"):
            local.zf = zipfile.ZipFile(container)
            handles.append(local.zf)
        raw, rel = job
        with local.zf.open(raw) as src, open(tmp / rel, "wb") as dst:
            shutil.copyfileobj(src, dst, CHUNK)

    try:
        with ThreadPoolExecutor(ZIP_THREADS) as pool:
            list(pool.map(one, jobs))
    finally:
        for handle in handles:
            handle.close()


def _untar_store(container: Path, prefix: str, tmp: Path) -> None:
    with tarfile.open(container, "r|*") as tf:
        for info in tf:
            rel = _in_store(info.name, prefix) if info.isfile() else None
            if rel is None:
                continue
            target = tmp / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with tf.extractfile(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, CHUNK)


def spatialdata(candidate: Candidate, dest: Path) -> Path:
    """A readable SpatialData store root: the folder itself, or the store's members of an archive extracted to
    `dest` (other members and macOS junk left out)."""
    member = candidate.zarr
    if member is None:
        raise ValueError(f"{candidate.sid} is not a SpatialData candidate")
    if member.inner is None and member.container.is_dir():
        return member.container
    root = dest / (Path(member.inner.rstrip("/")).name if member.inner else candidate.sid + ".zarr")
    if root.exists():
        return root
    prefix = (_inner(member.inner) + "/") if member.inner else ""
    tmp = root.with_name(root.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    _note_big(member.container, f"Extracting the SpatialData object {candidate.sid}")
    if _is_zip(member.container):
        _unzip_store(member.container, prefix, tmp)
    else:
        _untar_store(member.container, prefix, tmp)
    tmp.rename(root)
    return root


def cleanup(dest: Path) -> None:
    shutil.rmtree(dest, ignore_errors=True)
