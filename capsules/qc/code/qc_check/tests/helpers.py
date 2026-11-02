import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

import main
from config import load_settings
from conftest import env


def run(results: Path, **settings) -> dict:
    results.mkdir(parents=True, exist_ok=True)
    return main.run(load_settings([], env(results, **settings)))


def exit_code(results: Path, **settings) -> int:
    results.mkdir(parents=True, exist_ok=True)
    from unittest import mock
    with mock.patch.dict("os.environ", env(results, **settings), clear=False):
        return main.main([])


def decisions(results: Path) -> dict:
    return json.loads((results / "qc_decisions.json").read_text())


def write_json(path: Path, data: dict) -> str:
    path.write_text(json.dumps(data, indent=2))
    return str(path)


def accept_all(results: Path, out: Path, **samples) -> str:
    """The run's decisions with every pending sample given an entry, plus any `samples` overrides."""
    d = decisions(results)
    for sid in d["pending"]:
        d["samples"].setdefault(sid, {})
    d["samples"].update(samples)
    return write_json(out, d)


def tree_digest(path: str) -> str:
    digest = hashlib.sha256()
    for f in sorted(Path(path).rglob("*")):
        if f.is_file():
            digest.update(str(f.relative_to(path)).encode())
            digest.update(f.read_bytes())
    return digest.hexdigest()


def cell_qc(results: Path):
    return pq.read_table(results / "cell_qc.parquet").to_pandas()


def summary(results: Path) -> str:
    return (results / "report_summary.txt").read_text()
