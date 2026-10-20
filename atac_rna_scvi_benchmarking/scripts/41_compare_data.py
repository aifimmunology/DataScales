"""Join run-2 runtimes and metrics into report/compare_run2.json for the comparison page (rerunnable)."""
import glob
import json

import pandas as pd

from scvi_pca import config

OUT = config.REPO / "report" / "compare_run2.json"


def main():
    runs = pd.read_parquet(config.REPO / "results" / "runs.parquet")
    runs = runs[runs.status == "ok"]
    m = pd.concat([pd.read_parquet(f) for f in glob.glob(str(config.REPO / "results" / "metrics" / "*.parquet"))])
    setup = pd.read_parquet(config.REPO / "results" / "setup.parquet")
    rows = []
    for (ds, sub, strat), g in m.groupby(["ds", "subset", "strategy"]):
        rec = {"ds": ds, "subset": sub, "strategy": strat, "n_eval": int(g.n_eval.max())}
        for x in g.itertuples():
            key = f"{x.level}|{x.metric}" if x.level else x.metric
            rec[key] = round(float(x.value), 4)
        t = runs[(runs.ds == ds) & (runs.subset == sub) & (runs.strategy == strat)]
        for rep in (1, 2):
            tr = t[t.rep == rep]
            if not tr.empty:
                rec[f"t_e2e_r{rep}"] = round(float(tr.iloc[0].t_e2e), 2)
                rec["n_cells"] = int(tr.iloc[0].n_cells)
        rows.append(rec)
    setups = [{"ds": r.ds, "rep": int(r.rep), "t_pre_s": round(float(r.t_pre_s)), "epochs": int(r.epochs)}
              for r in setup.itertuples()]
    OUT.write_text(json.dumps({"rows": rows, "setup": setups}))
    print(f"wrote {OUT}: {len(rows)} rows, datasets {sorted(set(r['ds'] for r in rows))}")


if __name__ == "__main__":
    main()
