#!/bin/bash
# Sync results from RunPod to local muon_speedrun_results/runs/
# Usage: ./sync.sh
# Requires SSH key at ~/.ssh/id_ed25519

REMOTE="root@103.207.149.54"
PORT="15860"
KEY="$HOME/.ssh/id_ed25519"
REMOTE_LOG_DIR="/workspace/modded-nanogpt/logs"
LOCAL_DIR="$(dirname "$0")/runs"

mkdir -p "$LOCAL_DIR"

echo "Syncing from $REMOTE:$REMOTE_LOG_DIR ..."
scp -r -i "$KEY" -P "$PORT" -o StrictHostKeyChecking=no \
    "$REMOTE:$REMOTE_LOG_DIR/" "$LOCAL_DIR/"

echo "Done. Files in $LOCAL_DIR:"
ls "$LOCAL_DIR"
