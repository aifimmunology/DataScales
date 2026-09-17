#!/bin/bash
# GPU-host queue watcher (systemd: datavis-watcher.service). Polls the store's
# jobs/submitted/, claims each job, and runs gpu_job.sh in the rapids pixi env.
# Heartbeats to jobs/watcher.json so the app can report watcher liveness.
#   job_watcher.sh [store gs://...] [pixi_dir]   (defaults: $DATA_DIR, $GPU_PIXI_DIR)
set -u
STORE=${1:-${DATA_DIR:?set DATA_DIR or pass the store as arg 1}}
PIXI_DIR=${2:-${GPU_PIXI_DIR:?set GPU_PIXI_DIR or pass the pixi dir as arg 2}}
HERE=$(cd "$(dirname "$0")" && pwd)
POLL_S=5

# heartbeat runs independently of the job loop, which blocks for a whole gpu_job.sh run
( while true; do
    printf '{"ts":"%s","host":"%s"}' "$(date -u +%FT%TZ)" "$(hostname)" \
      | gcloud -q storage cp - "$STORE/jobs/watcher.json" 2>/dev/null
    sleep 30
  done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null' EXIT

echo "[watcher] store=$STORE pixi_dir=$PIXI_DIR" >&2
while true; do
  for url in $(gcloud -q storage ls "$STORE/jobs/submitted/*.json" 2>/dev/null); do
    id=$(basename "$url" .json)
    f=/tmp/datavis_job_$id.json
    gcloud -q storage cp "$url" "$f" 2>/dev/null || continue
    gcloud -q storage rm "$url" 2>/dev/null  # claim — a single watcher owns the queue
    slug=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("slug",""))' "$f")
    [ -n "$slug" ] || slug=view_$id
    echo "[watcher] running job $id (slug=$slug)" >&2
    bash "$HERE/gpu_job.sh" "$STORE" "$id" "$slug" "$PIXI_DIR" "$f"
  done
  sleep "$POLL_S"
done
