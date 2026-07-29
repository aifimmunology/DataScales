# DataScales UMAP POC

A React + deck.gl + zarrita proof-of-concept, built with Vite and Bun.

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

The app loads UMAP coordinates from an AnnData zarr v3 store. `DATA_DIR` must point directly at the zarr store root — the directory that contains `zarr.json` and `obsm/`. The store must have UMAP coordinates at `obsm/X_umap` (shape `(n_obs, 2)`), which is the standard AnnData layout produced by scanpy's `sc.tl.umap`.

---

## Local dev

### Python setup (first time)

```bash
cd datascales-umap-poc
pip install -r server/requirements.txt
```

### Run

```bash
DATA_DIR=./data/soundlife-other-tiny.zarr bun run dev
```

This starts both servers concurrently:
- Vite (frontend) → http://localhost:3000
- FastAPI (backend) → http://localhost:8000

Frontend requests to `/api/*` are proxied to the FastAPI server — no CORS configuration needed. API docs are available at http://localhost:8000/docs.

To run the servers separately:

```bash
DATA_DIR=./data/soundlife-other-tiny.zarr bun run dev:frontend
uvicorn server.main:app --reload
```

---

## Docker build

Mount your data directory into the container at runtime with `-v`:

```bash
cd datascales-umap-poc
docker build -t datascales-umap-poc .
docker run -p 3000:3000 -v /path/to/your/data:/data -e DATA_DIR=/data datascales-umap-poc
```

The app is served at http://localhost:3000.
