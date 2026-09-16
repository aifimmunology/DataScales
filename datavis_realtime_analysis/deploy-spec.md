# GPU-centralized deployment — work spec

## End goal

A user points at a zarr in the team bucket and gets the app running on the GPU VM from
their browser: view, label, submit UMAP reruns. The VM's service account is the only
credential anywhere — no user ever runs `gcloud auth login` for the app. The VM starts
on demand and stops itself when idle.

## Layout

- **GPU VM** (`$GPU_INSTANCE` / `$GPU_ZONE`, values in `.env`) — everything runs here: the
  frontend + backend containers (compose, up at boot), and the job watcher on the host
  running `gpu_job.sh` in the rapids pixi env. All GCS access rides the VM's service account.
- **Store** (the team bucket, per-zarr prefix) — data, labels, views, and the `jobs/` queue live in
  the zarr's prefix; the app only touches the prefix it was booted with.
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
- rapids pixi env stays on the VM host; jobs run there via a queue watcher, not in-container.
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

- [ ] docker + compose on the VM; build images there
- [ ] systemd unit: `docker compose up -d` at boot, `DATA_DIR` read from instance metadata
- [ ] `gpu.py`: drop `_ssh_cmd`/`_ship` + the probe section; dispatch = write job json (already done)
      and let the watcher pick it up
- [ ] host job watcher (systemd service): poll `jobs/submitted/`, run `gpu_job.sh` in the pixi env
- [ ] compose: remove the `~/.config/gcloud` / `~/.ssh` mounts and `entrypoint.sh` copy — ADC
      comes from the metadata server
- [ ] verify: store proxy, labels, views, job round-trip, all via SA only
- [ ] document the tunnel command for users:
      `gcloud compute start-iap-tunnel $GPU_INSTANCE 8000 --local-host-port=localhost:8000 --zone=$GPU_ZONE`

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
