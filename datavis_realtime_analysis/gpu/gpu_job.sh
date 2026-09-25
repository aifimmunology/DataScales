#!/bin/bash
# Cold per-job GPU run, dispatched by the datavis backend over ssh.
#   gpu_job.sh <store gs://...> <job_id> <slug> <pixi_dir>
# Fetches jobs/submitted/<id>.json from the store, sets up the pixi env fresh,
# runs rerun_umap_on_selection.py against the store, uploads the view to
# umap_views/<slug>, and reports every stage to jobs/status/<id>.json.
set -u
STORE=$1; ID=$2; SLUG=$3; PIXI_DIR=$4
PIXI=~/.pixi/bin/pixi
LOG=/tmp/datavis_job_$ID.log

status() { # status <state> <stage> — stage text sanitized for JSON
  local stage=${2//[\"\\]/}
  local view=null
  [ "$1" = done ] && view="\"umap_views/$SLUG\""
  printf '{"status":"%s","stage":"%s","view":%s,"updated":"%s"}' \
    "$1" "$stage" "$view" "$(date -u +%FT%TZ)" \
    | gcloud -q storage cp - "$STORE/jobs/status/$ID.json" 2>/dev/null
}
fail() { status failed "$1"; exit 1; }

status running "fetching job from queue"
gcloud -q storage cp "$STORE/jobs/submitted/$ID.json" "/tmp/datavis_job_$ID.json" \
  || fail "could not fetch job json from the store"

status running "setting up GPU env"
[ -x "$PIXI" ] || {  # fresh user account on the box: bootstrap pixi first
  status running "installing pixi"
  curl -fsSL https://pixi.sh/install.sh | bash >> "$LOG" 2>&1 || fail "pixi install failed"
}
cd "$PIXI_DIR" || fail "pixi dir $PIXI_DIR missing"
$PIXI install >> "$LOG" 2>&1 || fail "pixi env setup failed (see $LOG on the box)"

OUT=/tmp/datavis_view_$ID
rm -rf "$OUT"
set -o pipefail
RERUN_DATA=$STORE RERUN_SELECTION=/tmp/datavis_job_$ID.json RERUN_OUT=$OUT RERUN_GPUS=0 \
  $PIXI run python /tmp/rerun_umap_on_selection.py 2>&1 | tee -a "$LOG" \
  | while IFS= read -r line; do
      case "$line" in stage:*) status running "${line#stage: }";; esac
    done
[ $? -eq 0 ] || fail "pipeline failed (see $LOG on the box)"

status running "uploading view to store"
gcloud -q storage rsync -r "$OUT" "$STORE/umap_views/$SLUG" >> "$LOG" 2>&1 \
  || fail "view upload failed"
rm -rf "$OUT" "/tmp/datavis_job_$ID.json"
status done "done"
