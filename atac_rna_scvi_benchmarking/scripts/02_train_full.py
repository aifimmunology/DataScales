"""Setup run S-<ds>-r<rep> = retrain @ f100: 02_train_full.py <ds> <rep> [--override k=v ...].

Runs `<ds>-retrain-all-r<rep>` through the normal runner, then saves model/ and latent.npy under
/mnt/scvi_pca/models/<ds>/full_r<rep>/ and upserts the setup.parquet row (sizes, model-load time, pull time).
"""
import argparse
import json
import shutil
import sys
import time

import pandas as pd

from scvi_pca import config, runner
from scvi_pca.strategies import scvi_train


def dir_bytes(path):
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def pull_seconds(ds):
    pulls = config.out_root() / "logs" / "pull.json"
    if not pulls.exists():
        return None
    store = config.dataset(ds)["store"]
    for entry in json.loads(pulls.read_text()).values():
        if isinstance(entry, dict) and store.startswith(entry.get("local_path", "\0")):
            return entry["pull_seconds"]
    return None


def timed_load(model_dir, modality, accelerator):
    import scvi
    from scvi_pca import measure
    cls = scvi.model.SCVI if modality == "rna" else scvi.model.PEAKVI
    measure.sync()
    t0 = time.perf_counter()
    cls.load(str(model_dir), adata=None, accelerator=accelerator)
    measure.sync()
    return time.perf_counter() - t0


def upsert_setup(row):
    path = config.REPO / "results" / "setup.parquet"
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_parquet(path)
        new = pd.concat([old[~((old.ds == row["ds"]) & (old.rep == row["rep"]))], new], ignore_index=True)
    new.to_parquet(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ds")
    ap.add_argument("rep", type=int)
    ap.add_argument("--post-only", action="store_true", help="skip training/saving; re-measure existing artifacts")
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()
    rid = f"{args.ds}-retrain-all-r{args.rep}"
    if not args.post_only and runner.execute(rid, args.override) != "ok":
        sys.exit(1)
    p = config.params(config.parse_overrides(args.override))
    run = json.loads((config.REPO / "runs" / args.ds / "retrain" / "all" / f"r{args.rep}" / "run.json").read_text())
    dest = config.model_latent_path(args.ds, p, args.rep).parent
    dest.mkdir(parents=True, exist_ok=True)
    if not args.post_only:
        scvi_train.LAST["model"].save(str(dest / "model"), overwrite=True, save_anndata=False)
        scvi_train.LAST.clear()
        shutil.copy(config.out_root(p) / "runs" / args.ds / "retrain" / "all" / f"r{args.rep}" / "embedding.npy", dest / "latent.npy")
    load_s = timed_load(dest / "model", config.dataset(args.ds)["modality"], p["train"]["accelerator"])
    row = {"ds": args.ds, "rep": args.rep, "run_id": rid, "train_s": run["sub_timers"]["t_train"], "t_pre_s": run["t_pre"],
           "epochs": run["epochs_run"], "total_steps": run["total_steps"], "max_epochs_budget": run["max_epochs_budget"],
           "early_stopped": run["early_stopped"], "model_bytes": dir_bytes(dest / "model"),
           "latent_bytes": (dest / "latent.npy").stat().st_size, "load_s": load_s, "pull_s": pull_seconds(args.ds)}
    upsert_setup(row)
    print(json.dumps(row, indent=1, default=str))


if __name__ == "__main__":
    main()
