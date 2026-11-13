from fastapi import FastAPI, HTTPException, Request

from . import embedding, gpu, labels, store, validate, views

app = FastAPI(redoc_url=None)


@app.get("/")
def root():
    return {"this": "datavis backend API",
            "app_ui": "nginx on :8000 (deployed) / vite on :3000 (dev)",
            "api_console": "/docs", "health": "/api/health"}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/config")
def config():
    out = {"store": store.STORE.uri if store.STORE else "", "mirror": store.MIRROR.uri if store.MIRROR else None}
    try:
        out["embedding"] = embedding.describe()
    except Exception as e:
        out["error"] = f"store unreadable: {e}"
    return out


@app.get("/api/validate")
def validate_store(deep: bool = False):
    if store.STORE is None:
        return {"ok": False, "errors": ["no store configured"], "warnings": [], "info": {}, "capabilities": {}}
    return validate.inspect_store(store.STORE.uri, deep)


@app.api_route("/api/data/{path:path}", methods=["GET", "HEAD"])
def data(path: str, request: Request):
    if store.STORE is None:
        raise HTTPException(503, "no store configured: set LOCAL_DATA and/or GCS_DATA")
    if not path or path.startswith("/") or ".." in path.split("/"):
        raise HTTPException(404, "Not found")
    return store.STORE.serve(path, request)


@app.post("/api/submit")
def submit(payload: dict):
    return gpu.submit(payload)


@app.get("/api/jobs")
def jobs():
    return gpu.list_jobs()


@app.get("/api/gpu/health")
def gpu_health(refresh: bool = False):
    return gpu.health(refresh)


@app.get("/api/views")
def list_views():
    return views.list_views()


@app.delete("/api/views/{view_id}")
def delete_view(view_id: str):
    return views.delete_view(view_id)


@app.post("/api/labels")
def save_labels(payload: dict):
    return labels.save_labels(payload)
