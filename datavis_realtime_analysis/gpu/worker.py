"""Warm pipeline worker, spawned by the backend on first submit. Does the heavy
imports + CUDA init once, then runs one job per stdin JSON line
({id, store, selection, out}), reporting `stage:` / `done:` / `error:` lines on
stdout. Exits after WORKER_IDLE_S without a job so GPU memory frees when idle.
"""

import gc
import json
import os
import select
import sys

IDLE_S = float(os.environ.get("WORKER_IDLE_S", "900"))


def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    print("stage: warming up pipeline (imports + CUDA)", flush=True)
    import rerun_umap_on_selection as pipeline
    pipeline.warm_up()

    while True:
        ready, _, _ = select.select([sys.stdin], [], [], IDLE_S)
        if not ready:
            return
        line = sys.stdin.readline()
        if not line:
            return
        job = json.loads(line)
        try:
            pipeline.run(job["store"], job["selection"], job["out"])
            print(f"done: {job['id']}", flush=True)
        except Exception as e:
            print(f"error: {str(e)[:300]}", flush=True)
        finally:
            gc.collect()


if __name__ == "__main__":
    main()
