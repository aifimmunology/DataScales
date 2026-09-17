#!/bin/bash
# Boot-time config resolution (run as root by datavis-app.service): instance
# metadata wins, the repo .env is the fallback. Writes /etc/datavis.env — the
# EnvironmentFile for the watcher unit and compose's --env-file.
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
[ -f "$REPO/.env" ] && . "$REPO/.env"

md() {
  curl -sf -H 'Metadata-Flavor: Google' \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"
}
v=$(md DATA_DIR) && DATA_DIR=$v
v=$(md GPU_PIXI_DIR) && GPU_PIXI_DIR=$v

[ -n "${DATA_DIR:-}" ] || {
  echo "DATA_DIR is set in neither instance metadata nor $REPO/.env" >&2
  exit 1
}
printf 'DATA_DIR=%s\nGPU_PIXI_DIR=%s\n' "$DATA_DIR" "${GPU_PIXI_DIR:-}" > /etc/datavis.env
