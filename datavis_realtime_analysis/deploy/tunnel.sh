#!/bin/bash
set -eu

GPU_INSTANCE=""
GPU_ZONE=""
LOCAL_PORT=8000

#for now ssh IAP tunnel
exec gcloud compute ssh "$GPU_INSTANCE" --zone="$GPU_ZONE" --tunnel-through-iap -- -N -L "$LOCAL_PORT:localhost:8000"

#IAP tunnel needs specific access..
#exec gcloud compute start-iap-tunnel "$GPU_INSTANCE" 8000 --local-host-port="localhost:$LOCAL_PORT" --zone="$GPU_ZONE"
