#!/bin/sh
# gcloud needs a writable config dir; the host's is mounted read-only at /gcloud-ro
if [ -d /gcloud-ro ]; then
  mkdir -p /root/.config
  cp -r /gcloud-ro /root/.config/gcloud
fi
if [ -f /ssh-ro/google_compute_engine ]; then
  mkdir -p /root/.ssh
  cp /ssh-ro/google_compute_engine* /root/.ssh/
  chmod 600 /root/.ssh/google_compute_engine
fi
exec uvicorn server.main:app --host 0.0.0.0 --port 8000
