"""Where the current branch ("HEAD") is remembered between CLI invocations.

HEAD is a scizarr-ic concept — icechunk itself has no notion of a current branch —
so it is stored locally, like git's ``.git/HEAD``:

* a **writable local repo** keeps a ``scizarr_head`` file at its root (moves with the dir);
* anything else — a read-only path, an ``s3://``/``gs://`` URI — uses a per-user
  sidecar under ``$SCIZARR_IC_HOME/heads/`` (default ``~/.cache/scizarr_ic``), keyed
  by the repo's canonical location. Nothing is ever written into a read-only repo.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

ENV_HOME = "SCIZARR_IC_HOME"
HEAD_FILE = "scizarr_head"


def home_dir() -> Path:
    override = os.environ.get(ENV_HOME)
    if override:
        return Path(override).expanduser()
    return Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser() / "scizarr_ic"


class HeadStore:
    """Read/write the HEAD branch name for one repo location.

    ``in_repo_dir`` is the repo directory when it is local and writable (HEAD lives
    inside it); otherwise pass ``None`` and give ``key`` (the canonical location) so the
    sidecar is used.
    """

    def __init__(self, *, key: str, in_repo_dir: str | None = None) -> None:
        self.key = key
        self._in_repo = Path(in_repo_dir) / HEAD_FILE if in_repo_dir is not None else None

    @property
    def path(self) -> Path:
        if self._in_repo is not None:
            return self._in_repo
        digest = hashlib.sha1(self.key.encode()).hexdigest()[:20]
        return home_dir() / "heads" / digest

    @property
    def is_sidecar(self) -> bool:
        return self._in_repo is None

    def load(self) -> str | None:
        p = self.path
        if p.is_file():
            return p.read_text().strip() or None
        return None

    def save(self, branch: str) -> None:
        p = self.path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(branch + "\n")
