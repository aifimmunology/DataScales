"""Route assembly. Sections live in their own modules:
config (env), storage (zarr proxy + store JSON), views (groups.json), gpu (jobs).
"""

from fastapi import FastAPI, Request

from . import gpu, labels, storage, views
from .config import DATA_DIR

app = FastAPI(redoc_url=None)  # /docs = interactive API console


@app.get("/")
def root():
    return {"this": "datavis backend API", "app_ui": "http://localhost:3000",
            "api_console": "/docs", "health": "/api/health"}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/config")
def config():
    return {"store": DATA_DIR}


@app.api_route("/api/data/{path:path}", methods=["GET", "HEAD"])
def data(path: str, request: Request):
    return storage.serve(path, request)


@app.post("/api/submit")
def submit(artifact: dict):
    return gpu.submit(artifact)


@app.get("/api/jobs")
def jobs():
    return gpu.list_jobs()


@app.delete("/api/views/{view_id}")
def delete_view(view_id: str):
    return views.delete_view(view_id)


@app.post("/api/labels")
def save_labels(payload: dict):
    return labels.save_labels(payload)
