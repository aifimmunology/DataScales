"""runs.parquet: one row per run id, upserted atomically."""
import os

import pandas as pd

from . import config

STAGES = ["t_load", "t_pre", "t_neighbors", "t_umap", "t_leiden"]
GB = 1024 ** 3


def runs_path():
    return config.REPO / "results" / "runs.parquet"


def flat_row(run_json):
    r = run_json
    row = {k: r.get(k) for k in ("run_id", "ds", "strategy", "subset", "rep", "n_cells", "status", "rep_name", "n_clusters",
                             "t_pre", "t_neighbors", "t_umap", "t_leiden", "t_total", "t_load", "started_utc")}
    for k in ("t_read", "t_pre_ex_read", "t_e2e", "eager"):
        row[k] = r.get(k)
    row["overrides"] = repr(r["overrides"]) if r["overrides"] else ""
    row["harmony_alpha"] = r.get("harmony_alpha")
    row["git_sha"] = r["git"]["sha"]
    for s in STAGES:
        st = r.get("stages", {}).get(s)
        row[f"{s}_vram_gb"] = st["peak_vram_bytes"] / GB if st else None
        row[f"{s}_rss_gb"] = st["peak_rss_bytes"] / GB if st else None
    row["peak_vram_gb"] = r.get("peak_vram_bytes", 0) / GB
    row["peak_rss_gb"] = r.get("peak_rss_bytes", 0) / GB
    return row


def upsert(run_json):
    path = runs_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([flat_row(run_json)])
    if path.exists():
        old = pd.read_parquet(path)
        old = old[old.run_id != run_json["run_id"]]
        new = pd.concat([old, new], ignore_index=True)
    tmp = path.with_suffix(".tmp")
    new.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def rebuild():
    """Regenerate runs.parquet from every non-explore runs/**/run.json."""
    import json
    rows = [flat_row(json.loads(f.read_text())) for f in sorted((config.REPO / "runs").rglob("run.json"))]
    df = pd.DataFrame(rows)
    tmp = runs_path().with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, runs_path())
    return df


def is_ok(run_id):
    path = runs_path()
    if not path.exists():
        return False
    df = pd.read_parquet(path)
    return bool(((df.run_id == run_id) & (df.status == "ok")).any())
