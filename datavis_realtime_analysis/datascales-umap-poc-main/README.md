# DataScales UMAP POC

A React + deck.gl + zarrita proof-of-concept, built with Vite and Bun. A FastAPI backend serves the zarr store to the frontend at `/api/data`.

![DataScales UMAP POC](public/datascales-umap-poc.png)

---

## Prerequisites

### Install Bun

```bash
curl -fsSL https://bun.sh/install | bash
```

Then restart your terminal (or source your shell profile) so `bun` is on your PATH.

### Install Docker

Download and install Docker Desktop for your platform from https://docs.docker.com/get-docker/, then start the Docker Desktop app.

---

## Data

`DATA_DIR` points at the root of an AnnData zarr v3 store — the directory (or GCS prefix) containing `zarr.json` and `obsm/`, with UMAP coordinates at `obsm/X_umap` (shape `(n_obs, 2)`, the standard layout from scanpy's `sc.tl.umap`). It can be:

- a local path, e.g. `./data/soundlife-other-tiny.zarr`
- a private GCS store, e.g. `gs://my-bucket/path/store.zarr` — read with your Google credentials (see [Docker deployment](#docker-deployment-gcs))

Optionally set `RAPIDS_DIR` to a second store (e.g. the CSR store RAPIDS consumes): coords, labels, barcodes, and views then come from `RAPIDS_DIR` (served at `/api/rapids-data`), `DATA_DIR` answers only gene-expression reads, and selection exports/submissions record `RAPIDS_DIR` as their `store`. The two stores must share cell order. Unset, one store serves everything.

In the app: lasso a cell selection, name it, and hit "GPU run" — the backend queues the job to the GPU box over `gcloud compute ssh` (warm runner `../gpu_runner.py`: RAPIDS + obs stay loaded; ~20s per run), the view lands in the rapids store's `umap_views/` on GCS, and the runs panel marks it ready in the View picker. The runs panel shows queued/running with a live timer and pipeline stage; views are deletable from the View picker (✕). A gene dropdown colors the UMAP by that gene's expression, resolved in order: `layers/gexp` (csc or dense, the `zarrsmith add-expr` setup) → dense `X` → CSC `X`.

GPU runs: config via `GPU_INSTANCE` (defaults to the datavis GPU box; set `""` for the 10s simulator), `GPU_ZONE`, `GPU_DATA` (store path the pipeline reads on the box, default `/mnt/subset3M_megazarr_v1.0.zarr`). Works in docker (the backend image ships gcloud + your mounted credentials/ssh key) and in dev mode. The runner caches obs at warm-up — after changing the store, `gcloud compute ssh <box> --command 'pkill -f gpu_runner'` and the next submit restarts it fresh.

---

## Local dev

### Python setup (first time)

```bash
cd datascales-umap-poc
pip install -r server/requirements.txt
```

### dev Run

```bash
DATA_DIR=./data/soundlife-other-tiny.zarr bun run dev
```

This starts both servers concurrently:
- Vite (frontend) → http://localhost:3000
- FastAPI (backend) → http://localhost:8000

Frontend requests to `/api/*` are proxied to the FastAPI server.

To run the servers separately:

```bash
bun run dev:frontend
DATA_DIR=./data/soundlife-other-tiny.zarr uvicorn server.main:app --reload
```

---

## Docker deployment (GCS)

Two services via compose: nginx serves the built frontend and proxies `/api`; the FastAPI backend reads the store from GCS.

### 1. Authenticate (once per machine)

```bash
gcloud auth application-default login
```

This writes ADC credentials under `~/.config/gcloud`, which compose mounts read-only into the backend container. Your account needs read access on the bucket (`roles/storage.objectViewer`). If your ADC has no default project, also export `GOOGLE_CLOUD_PROJECT=<project-id>`.

### 2. Build and run
using 1 store, X/ is used for rapids runs (best if csr), layers/gexp is used for gene highlighting (best if dense/csc) (zarrsmith can create the layers/gexp for a zarr easily for datavis use)
```bash
DATA_DIR=gs://MY_BUCKET/stores/soundlife-other-tiny.zarr docker compose up --build -d
```
or if two different stores:
```
DATA_DIR=gs://bucket/vis-csc-store.zarr RAPIDS_DIR=gs://bucket/rapids-csr.zarr docker compose up --build
```

App: http://localhost:3000

### 3. Verify

```bash
curl http://localhost:8000/api/health        # {"status":"ok"}
curl http://localhost:8000/api/config        # echoes the gs:// DATA_DIR
curl -sI http://localhost:3000/api/data/zarr.json | head -1   # 200 through the full chain
docker compose logs -f backend               # request log / GCS errors
```

### Troubleshooting

- `503 GCS auth failed` — re-run step 1 on the host; confirm `~/.config/gcloud/application_default_credentials.json` exists.
- `404` on `/api/data/zarr.json` — `DATA_DIR` must point at the store root (the prefix containing `zarr.json`).
- `port is already allocated` — something else is publishing 3000/8000; `docker ps`, then stop it.

`docker compose down` stops the stack. For a local store instead of GCS, uncomment the data volume in `docker-compose.yml` and run with `DATA_DIR=/data`.
