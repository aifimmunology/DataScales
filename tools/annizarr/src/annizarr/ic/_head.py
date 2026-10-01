from __future__ import annotations

import hashlib
import os
from pathlib import Path

ENV_HOME = "ANNIZARR_HOME"
HEAD_FILE = "annizarr_head"


def home_dir() -> Path:
    override = os.environ.get(ENV_HOME)
    if override:
        return Path(override).expanduser()
    return Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser() / "annizarr"


class HeadStore:
    """Persists the current branch ("HEAD") locally, like git's .git/HEAD, since icechunk
    itself has no notion of one: `in_repo_dir` for a writable local repo (an annizarr_head
    file at its root), else a per-user sidecar under `$ANNIZARR_HOME/heads/` keyed by `key`."""

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
