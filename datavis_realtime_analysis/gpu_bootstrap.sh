#!/bin/bash
# Bootstrap any GPU box for datavis UMAP jobs: pixi -> repo -> RAPIDS env -> warm runner.
# Run ON the box:  bash gpu_bootstrap.sh
# Env overrides:   REPO_URL REPO_DIR BRANCH RUNNER_DATA
set -euo pipefail

REPO_URL="${REPO_URL:-git@github.com:AlexHolly-AllenOps/DataScale.git}"
REPO_DIR="${REPO_DIR:-/mnt/DataScales}"
BRANCH="${BRANCH:-datavis_analysis}"
RUNNER_DATA="${RUNNER_DATA:-/mnt/subset3M_megazarr_v1.0.zarr}"

export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null 2>&1 || curl -fsSL https://pixi.sh/install.sh | bash
export PATH="$HOME/.pixi/bin:$PATH"

if [ -d "$REPO_DIR/.git" ]; then
  echo "repo exists at $REPO_DIR — leaving its branch alone (runner script must be present)"
else
  git clone -b "$BRANCH" "$REPO_URL" "$REPO_DIR"
fi

RUNNER="$REPO_DIR/datavis_realtime_analysis/gpu_runner.py"
[ -f "$RUNNER" ] || { echo "missing $RUNNER (wrong branch?)"; exit 1; }

echo "installing RAPIDS env (first run takes a while)…"
cd "$REPO_DIR/rapids_user_notebook"
pixi install

pkill -f "gpu_runner.py" 2>/dev/null || true
nohup env RUNNER_DATA="$RUNNER_DATA" bash -c \
  "cd $REPO_DIR/rapids_user_notebook && pixi run python $RUNNER" \
  >> /tmp/datavis_runner.log 2>&1 < /dev/null &
echo "warm runner starting (log: /tmp/datavis_runner.log); ready when the heartbeat appears:"
echo "  watch -n2 'ls -la /tmp/datavis_runner.alive'"
