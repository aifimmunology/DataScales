# Capsules

Sources for the three Code Ocean capsules (`ingest`, `qc`, `process`). The design is in [spec.md](spec.md).

## Environments

Each capsule's environment is set on its Code Ocean **environment page**. These files are copies, so the
environment can be read, reused in other environments, and loaded onto a new capsule:

| File | Holds | Loaded with |
|---|---|---|
| `<capsule>/environment/requirements.txt` | pip packages | page → pip3 gear → **Edit Bulk** |
| `<capsule>/environment/environment.yml` | conda packages (only where needed, e.g. `python-annoy`) | page → conda gear → **Edit Bulk** |
| `<capsule>/environment/pixi.toml` | the same packages, for use outside Code Ocean (no lock is kept) | `pixi install --manifest-path <capsule>/environment/pixi.toml` |
| `<capsule>/environment/postInstall` | Claude Code only, no packages | copied into the capsule's `environment/` |

The base image and VS Code extensions are set on the page only.

### Keep the files in step with the real capsules

The page can change without these files changing (an edit on the page, or packages Code Ocean picks up
from a session). These files are what gets copied out, so **every so often, and before any release**, bring
them up to date with what each capsule really installs:

1. Read the capsule's current packages: the page's **Edit Bulk** text for pip3 and conda, or
   `/root/capsule/.codeocean/environment.json` in a running capsule. Code Ocean regenerates that file from
   the page on every build, so editing it changes nothing.
2. Update `<capsule>/environment/requirements.txt`, `environment.yml` and `pixi.toml` here to match.

### process: pins that matter

`process/environment/requirements.txt` pins its top-level packages plus the parts of the CUDA stack that
must agree with each other (`numpy`, `numba`, `numba-cuda`, `cupy-cuda13x`, `cuda-toolkit`, `torch`,
`dask`, `distributed`, exact RAPIDS 26.10 versions). The resolver picks everything else at build time.
If a rebuild breaks, pin the offending package. To see what a change resolves to before rebuilding:

```bash
uv pip compile process/environment/requirements.txt --python-version 3.13 \
    --python-platform x86_64-manylinux_2_28 --no-annotate --no-header
```

On 2026-10-23 this gave 247 packages (torch 2.14.1, CUDA 13.0 wheels), including squidpy, spatialdata and omnipath
for the interactive tutorial notebooks in `process/code/notebooks/`.

### Using the environments outside Code Ocean

`pixi install --manifest-path <capsule>/environment/pixi.toml` builds the capsule's environment anywhere pixi runs.
`ingest` and `qc` solve for `linux-64` and `osx-arm64`. `process` is `linux-64` only and needs a CUDA 13 driver; to
solve it on a machine without a GPU, set `CONDA_OVERRIDE_CUDA=13`. Put pixi's cache somewhere large
(`PIXI_CACHE_DIR`): the process environment is about 5 GB to download.
