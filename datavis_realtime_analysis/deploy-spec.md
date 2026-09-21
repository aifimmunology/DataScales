# GPU-centralized deployment — work spec

## End goal

A user points at a zarr in the team bucket and gets the app running on the GPU VM from
their browser: view, label, submit UMAP reruns. The VM's service account is the only
credential anywhere — no user ever runs `gcloud auth login` for the app. The VM starts
on demand and stops itself when idle.

## Layout

- **GPU VM** (`$GPU_INSTANCE` / `$GPU_ZONE`, values in `.env`) — everything runs here: the
  frontend + backend containers (compose, up at boot). The backend container has the GPU
  (nvidia-container-toolkit) and runs the pipeline itself via a warm worker process; the
  job queue lives in backend memory. All GCS access rides the VM's service account.
- **Store** (the team bucket, per-zarr prefix) — data, labels, views, and per-job
  `jobs/history/` records live in the zarr's prefix; the app only touches the prefix it
  was booted with.
- **Access** — IAP TCP tunnel from the user's laptop to the VM (`localhost:8000`).
  Team-wide browser access via Cloudflare/Tailscale is parked for later.
- **Launcher** (phase 2) — small always-on Cloud Run service: starts the VM with the chosen
  `DATA_DIR`, serves the warming-up page until the app is healthy.
- **Idle-stop** — the VM shuts itself down after 15 min without user-interface interaction,
  never while a job is running.

## Decisions made

- SA grant on the team bucket for the VM SA: approved; all zarrs live in this bucket.
- App exposure is scoped to the booted `DATA_DIR` prefix — already enforced (every
  read/write goes through `storage.key()`); bucket-wide SA is acceptable inside one team bucket.
- ~~rapids pixi env stays on the VM host; jobs run there via a queue watcher~~ — superseded:
  the env is baked into the backend image (`server/pixi.toml`, NVIDIA cu13 wheels) and jobs
  run in-container through a warm worker (`gpu/worker.py`, idle exit after 15 min). The
  store-side queue existed for remote submitters; with the app VM-only it moved in-process.
- Cold start gets a "warming up" page from the launcher; no pretending it's instant.
- Image freshness: build on the VM for now; CI → Artifact Registry later.
- **Access: option A (IAP tunnel) now.** Team access later via Cloudflare Tunnel + Access or
  Tailscale — the institute has used these before, so follow that precedent; LB + IAP only
  as fallback. Far out, not blocking anything.
- **Idle = 15 min without UI interaction.** Frontend heartbeat sent only while the tab is
  visible and the user recently interacted; backend polling, health checks, and probes do
  not count as activity. Never stop while a job is running.

## Phase 0 — prereq (techdev)

- [ ] Grant the VM's default compute service account `roles/storage.objectAdmin`
      on the team bucket

## Phase 1 — app moves onto the GPU VM (this alone ends daily reauth)

Code side done (this branch); on-VM install + verify remain. One published port: nginx
serves UI + `/api` on **8000** — the tunnel target.

- [ ] docker + compose + nvidia-container-toolkit on the VM; build images there
      (first backend build pulls the rapids env — slow once, cached after)
- [x] systemd unit: `docker compose up -d` at boot, `DATA_DIR` read from instance metadata
      (`deploy/datavis-app.service` + `deploy/write-env.sh`; metadata wins, repo `.env` fallback)
- [x] `gpu.py`: drop ssh + the store-side queue; in-memory queue dispatches to a warm worker
      (`gpu/worker.py` — imports + CUDA once, idle exit at 15 min), stages stream over stdout,
      one `jobs/history/<id>.json` record per job; health = SA store access + GPU visible;
      cancel via `DELETE /api/jobs/<id>`
- [x] backend image: pixi-built GPU env (`server/pixi.toml` + committed lock, rapids cu13
      wheels); compose grants the GPU, no credential mounts — ADC from the metadata server
- [x] pipeline: importable `run()` (view written straight to `gs://`, cluster closed per run);
      standalone `RERUN_*` entry kept
- [ ] verify on the VM: store proxy, labels, views, submit round-trip (eager + dask paths),
      warm second submit, idle exit, all via SA only
- [x] document the tunnel command for users: `deploy/tunnel.sh` (IAP tunnel to 8000; `--ssh`
      fallback forwards over IAP ssh when there's no firewall rule for the IAP range → 8000)

## Phase 2 — lifecycle

- [ ] frontend heartbeat: ping backend while tab visible + interaction in the last 15 min
- [ ] backend `/api/idle` → `{idle_s, jobs_running}` (heartbeats only; polling/health excluded)
- [ ] host systemd timer: idle > 15 min and no running job → `shutdown -h now`
      (VM self-stop needs no IAM; disk persists)
- [ ] launcher: small Cloud Run service, own SA with `compute.instances.start` on this VM;
      takes `store=gs://...` param → writes instance metadata → starts VM → warming page
      polling `/api/health`
- [ ] hard-cap stop (e.g. 12 h) as backstop

## Phase 3 — team browser access (parked, far out)

- [ ] ask techdev which is the supported pattern: Cloudflare Tunnel + Access, or Tailscale
- [ ] Cloudflare: `cloudflared` on the VM + Access policy (Google Workspace SSO) — pure
      browser, no client install. Tailscale: stable tailnet hostname, but every user
      installs the client and joins the tailnet.
- [ ] fit the launcher into the flow (the tunnel dies with the VM, so "visit URL →
      auto-start" needs the always-on launcher in front)

## Phase 4 — later

- [ ] CI: build + push images to Artifact Registry, VM pulls at boot
- [ ] per-user label/job attribution
- [ ] runtime store switching (v2 — config/SOURCE are boot-time globals today)

## Questions for techdev

- Cloudflare Tunnel/Access vs Tailscale — which does the institute support today?
- GPU quota/capacity risk when start/stopping the VM on demand in its zone?
