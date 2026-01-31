#!/usr/bin/env bash
set -euo pipefail

BASE="$HOME/cerbero-coscienza"
LOGDIR="$BASE/logs"
mkdir -p "$LOGDIR"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

echo "[$(ts)] run_cycle START" >> "$LOGDIR/data_factory.log"

cd "$BASE"

# venv
source "$BASE/.venv/bin/activate"

# env Twelve Data
set -a
source "$BASE/secrets/twelvedata.env"
set +a

# 1) ingest RAW locale
python "$BASE/data_factory/01_ingest_twelvedata.py" >> "$LOGDIR/data_factory.log" 2>>"$LOGDIR/data_factory.err"

# 2) features locali
python "$BASE/data_factory/02_features_local.py" >> "$LOGDIR/data_factory.log" 2>>"$LOGDIR/data_factory.err"

echo "[$(ts)] run_cycle DONE" >> "$LOGDIR/data_factory.log"
