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

`DATA_DIR` points at the root of an AnnData zarr v3 store — the directory (or GCS prefix) containing `zarr.json`. It can be a local path (`./data/soundlife-other-tiny.zarr`) or a private GCS store (`gs://my-bucket/path/store.zarr`, read with your Google credentials — see [Docker deployment](#docker-deployment-gcs)).

One store serves everything:

- `obsm/X_umap` — the coordinates the viewer renders (`(n_obs, 2)`, scanpy layout)
- `X/` (best if csr) — what the GPU pipeline consumes
- `layers/gexp` (csc or dense; `zarrsmith add-expr` creates it) — gene-expression highlighting, resolved in order `layers/gexp` → dense `X` → CSC `X`
- `umap_views/`, `groups.json`, `jobs/` — written by the app: returned views, the view listing, and the GPU job queue + status objects

In the app: lasso a cell selection, name it, and hit "GPU run". The backend writes the job to `jobs/submitted/<id>.json` in the store, then dispatches one **cold run** on the GPU box over `gcloud compute ssh`: `../gpu_job.sh` sets up the pixi env fresh, runs `../rerun_umap_on_selection.py` against the store, and uploads the view to `umap_views/<slug>`. The script reports each stage to `jobs/status/<id>.json`; the runs panel polls it (live timer + stage) and marks the view ready in the View picker — no auto-switch. Views are deletable from the picker (✕).

GPU runs: config via `GPU_INSTANCE` (defaults to the datavis GPU box; set `""` for the 10s simulator), `GPU_ZONE`, `GPU_PIXI_DIR` (pixi project on the box, default `/mnt/DataScales/rapids_user_notebook`). Works in docker (the backend image ships gcloud + your mounted credentials/ssh key) and in dev mode. One-time on the box: `gcloud auth application-default login` so python can read `gs://` stores.

---

## Local dev

### Python setup (first time)

```bash
cd datascales-umap-poc-main
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

Put the connection config in `.env` next to `docker-compose.yml` (auto-loaded, gitignored) — `DATA_DIR`, and `GPU_INSTANCE`/`GPU_ZONE`/`GPU_PIXI_DIR` when overriding the defaults — then:

```
docker compose up --build -d
```

(or inline: `DATA_DIR=gs://MY_BUCKET/store.zarr docker compose up --build -d`)

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
