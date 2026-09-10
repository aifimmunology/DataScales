#!/bin/sh
# gcloud needs a writable config dir; the host's is mounted read-only at /gcloud-ro.
# rm first: on container RESTART the dir already exists and cp -r would nest the
# fresh copy inside it, leaving stale credentials on top
if [ -d /gcloud-ro ]; then
  rm -rf /root/.config/gcloud
  mkdir -p /root/.config
  cp -r /gcloud-ro /root/.config/gcloud
fi
if [ -f /ssh-ro/google_compute_engine ]; then
  mkdir -p /root/.ssh
  cp /ssh-ro/google_compute_engine* /root/.ssh/
  chmod 600 /root/.ssh/google_compute_engine
fi
exec uvicorn server.main:app --host 0.0.0.0 --port 8000
