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

`DATA_DIR` points at the root of an AnnData zarr v3 store — the directory (or GCS prefix) containing `zarr.json`. It can be a local path (`./data/soundlife-other-tiny.zarr`) or a private GCS store (`gs://my-bucket/path/store.zarr`, read with the VM's service account — see [Deploy on the GPU VM](#deploy-on-the-gpu-vm)).

One store serves everything:

- `obsm/X_umap` — the coordinates the viewer renders (`(n_obs, 2)`, scanpy layout)
- `X/` (best if csr) — what the GPU pipeline consumes
- `layers/gexp` (csc or dense; `zarrsmith add-expr` creates it) — gene-expression highlighting, resolved in order `layers/gexp` → dense `X` → CSC `X`
- `umap_views/`, `groups.json`, `jobs/history/` — written by the app: returned views, the view listing, and one record per finished GPU job

In the app: lasso a cell selection, name it, and hit "Generate New UMAP". The backend queues the job in memory and runs it in its own GPU container: a **warm pipeline process** (`gpu/worker.py`) does the heavy imports + CUDA init once, runs `gpu/rerun_umap_on_selection.py`'s pipeline per job, and writes the view straight to `umap_views/<slug>` in the store. Stage updates stream over the worker's stdout into the runs panel (live timer + stage); the view goes ready in the View picker — no auto-switch. The worker exits after 15 min idle (GPU memory frees) and respawns on the next submit. Views are deletable from the picker (✕); running jobs are cancellable via `DELETE /api/jobs/<id>`.

---

## Local dev

### Python setup (first time)

```bash
cd datavis_realtime_analysis
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

## Deploy on the GPU VM

Everything runs on the GPU VM ([deploy-spec.md](deploy-spec.md) phase 1): nginx serves the built frontend and proxies `/api` to the FastAPI backend (compose, one published port — **8000**). The backend container has the GPU and runs the rapids pipeline itself — env baked into the image from `server/pixi.toml`. The VM's service account is the only credential — no `gcloud auth login` anywhere.

### 0. One-time bucket grant

Grant the VM's service account access to the team bucket (views, labels, and job records all ride it):

```bash
gcloud storage buckets add-iam-policy-binding gs://MY_BUCKET \
  --member="serviceAccount:<vm-service-account>" --role="roles/storage.objectAdmin"
```

### 1. VM setup

Install docker + the compose plugin, plus the NVIDIA container toolkit (once):

```bash
sudo apt install nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
```

Clone the repo, then from `datavis_realtime_analysis/`:

- **Config** — set `DATA_DIR` as instance metadata:
  `gcloud compute instances add-metadata $VM --zone=$ZONE --metadata=DATA_DIR=gs://...`
  — or put it in `.env` in the repo. Metadata wins at boot (`deploy/write-env.sh`).
- **Build** — `docker compose build` (first build downloads the rapids env — expect a while and a large image)
- **systemd unit** —

```bash
sudo cp deploy/datavis-app.service /etc/systemd/system/
# edit the paths to your checkout
sudo systemctl daemon-reload
sudo systemctl enable --now datavis-app
```

### 2. Access from your laptop

```bash
deploy/tunnel.sh          # IAP TCP tunnel → http://localhost:8000
deploy/tunnel.sh --ssh    # fallback: forward over IAP ssh — needs only ssh access
```

The direct tunnel needs `roles/iap.tunnelResourceAccessor` on the VM plus a firewall rule allowing `35.235.240.0/20 → tcp:8000`; if you can't get the firewall rule, the `--ssh` fallback rides the existing ssh rule (port 22).

### 3. Verify & troubleshoot

- A red **GPU runs** rail badge → the tab shows the failing step with fix commands: bucket access (SA grant missing) or the container can't see the GPU (toolkit/compose device reservation). Hit *Re-check* after fixing.
- Everything logs in one place: `docker compose logs -f backend` (submits, pipeline stages, worker stderr).
- `sudo systemctl stop datavis-app` (or `docker compose down`) stops the stack. For a local store instead of GCS, mount it into the backend and run with `DATA_DIR=/data`.
