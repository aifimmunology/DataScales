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
- `umap_views/`, `groups.json`, `jobs/` — written by the app: returned views, the view listing, and the GPU job queue + status objects

In the app: lasso a cell selection, name it, and hit "Generate New UMAP". The backend writes the job to `jobs/submitted/<id>.json` in the store; the watcher on the GPU host (`gpu/job_watcher.sh`) claims it and runs one **cold run** — `gpu/gpu_job.sh` sets up the pixi env fresh, runs `gpu/rerun_umap_on_selection.py` against the store, and uploads the view to `umap_views/<slug>`. The script reports each stage to `jobs/status/<id>.json`; the runs panel polls it (live timer + stage) and marks the view ready in the View picker — no auto-switch. Views are deletable from the picker (✕). No ssh anywhere: the app and the watcher talk only through the store, using the VM's service account.

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

Everything runs on the GPU VM ([deploy-spec.md](deploy-spec.md) phase 1): nginx serves the built frontend and proxies `/api` to the FastAPI backend (compose, one published port — **8000**), and a host-side watcher runs GPU jobs in the rapids pixi env. The VM's service account is the only credential — no `gcloud auth login` anywhere.

### 0. One-time bucket grant

Grant the VM's service account access to the team bucket (jobs, views, and labels all ride it):

```bash
gcloud storage buckets add-iam-policy-binding gs://MY_BUCKET \
  --member="serviceAccount:<vm-service-account>" --role="roles/storage.objectAdmin"
```

### 1. VM setup

Install docker + the compose plugin, clone the repo, then from `datavis_realtime_analysis/`:

- **Config** — set `DATA_DIR` (and optionally `GPU_PIXI_DIR`) as instance metadata:
  `gcloud compute instances add-metadata $VM --zone=$ZONE --metadata=DATA_DIR=gs://...`
  — or put them in `.env` in the repo. Metadata wins at boot (`deploy/write-env.sh`).
- **Build** — `docker compose build`
- **systemd units** —

```bash
sudo cp deploy/datavis-app.service deploy/datavis-watcher.service /etc/systemd/system/
# edit both: paths to your checkout; watcher User= to the account whose pixi env runs jobs
sudo systemctl daemon-reload
sudo systemctl enable --now datavis-app datavis-watcher
```

### 2. Access from your laptop

```bash
deploy/tunnel.sh          # IAP TCP tunnel → http://localhost:8000
deploy/tunnel.sh --ssh    # fallback: forward over IAP ssh — needs only ssh access
```

The direct tunnel needs `roles/iap.tunnelResourceAccessor` on the VM plus a firewall rule allowing `35.235.240.0/20 → tcp:8000`; if you can't get the firewall rule, the `--ssh` fallback rides the existing ssh rule (port 22).

### 3. Verify & troubleshoot

- A red **GPU runs** rail badge → the tab shows the failing step with fix commands: bucket access (SA grant missing) or the job watcher (not running / stale heartbeat at `jobs/watcher.json`). Hit *Re-check* after fixing.
- On the VM: `sudo systemctl status datavis-app datavis-watcher`, `journalctl -u datavis-watcher -f`, `docker compose logs -f backend`; per-job logs land in `/tmp/datavis_job_<id>.log`.
- `sudo systemctl stop datavis-app` (or `docker compose down`) stops the stack. For a local store instead of GCS, uncomment the data volume in `docker-compose.yml` and run with `DATA_DIR=/data`.
