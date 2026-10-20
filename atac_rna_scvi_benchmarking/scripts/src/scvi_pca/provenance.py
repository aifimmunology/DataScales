"""Versions, git SHA and hardware description recorded in every run.json."""
import importlib.metadata as md
import os
import platform
import subprocess
import sys

from . import config

PACKAGES = {"rapids_singlecell": ["rapids-singlecell", "rapids-singlecell-cu13"],
            "cupy": ["cupy-cuda13x", "cupy-cuda12x", "cupy"], "torch": ["torch"], "scvi": ["scvi-tools"],
            "anndata": ["anndata"], "zarr": ["zarr"], "dask_cuda": ["dask-cuda"], "cuml": ["cuml-cu13", "cuml-cu12", "cuml"]}


def _version(dists):
    for d in dists:
        try:
            return md.version(d)
        except md.PackageNotFoundError:
            continue
    return None


def versions():
    return {"python": platform.python_version(), **{k: _version(v) for k, v in PACKAGES.items()}}


def git_state():
    def git(*a):
        return subprocess.run(["git", "-C", str(config.REPO), *a], capture_output=True, text=True).stdout.strip()
    return {"sha": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--", ":!runs", ":!runs_explore", ":!results"))}


def hardware():
    import psutil
    out = {"cpu_count": os.cpu_count(), "host_ram_bytes": psutil.virtual_memory().total, "python_exe": sys.executable}
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(h)
        cuda = pynvml.nvmlSystemGetCudaDriverVersion()
        out.update(gpu_model=name.decode() if isinstance(name, bytes) else name,
                   gpu_vram_bytes=pynvml.nvmlDeviceGetMemoryInfo(h).total,
                   gpu_driver=str(pynvml.nvmlSystemGetDriverVersion()),
                   cuda_driver_version=f"{cuda // 1000}.{cuda % 1000 // 10}")
    except Exception as e:
        out["gpu_error"] = repr(e)
    return out
