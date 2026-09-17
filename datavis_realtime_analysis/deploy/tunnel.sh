#!/bin/bash
# Laptop-side access: IAP tunnel to the app on the GPU VM -> http://localhost:8000
# Reads GPU_INSTANCE / GPU_ZONE from the repo .env. The direct tunnel needs
# roles/iap.tunnelResourceAccessor plus a firewall rule allowing
# 35.235.240.0/20 -> tcp:8000. No firewall rule? `tunnel.sh --ssh` forwards the
# port over IAP ssh (port 22) instead — works with plain ssh access.
set -eu
REPO=$(cd "$(dirname "$0")/.." && pwd)
. "$REPO/.env"
: "${GPU_INSTANCE:?GPU_INSTANCE missing from .env}" "${GPU_ZONE:?GPU_ZONE missing from .env}"
LOCAL_PORT=${LOCAL_PORT:-8000}

if [ "${1:-}" = "--ssh" ]; then
  exec gcloud compute ssh "$GPU_INSTANCE" --zone="$GPU_ZONE" --tunnel-through-iap \
    -- -N -L "$LOCAL_PORT:localhost:8000"
fi
exec gcloud compute start-iap-tunnel "$GPU_INSTANCE" 8000 \
  --local-host-port="localhost:$LOCAL_PORT" --zone="$GPU_ZONE"
