#!/bin/bash
# Deploy the Dash dashboard to Vercel (project: capitaledge).
# Stages a lean copy: the app, its assets, and the processed CSVs it reads
# (data/ is gitignored, so this runs from a local checkout that has the data).
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
STAGE="${TMPDIR:-/tmp}/capitaledge-vercel"
rm -rf "$STAGE" && mkdir -p "$STAGE/src" "$STAGE/data/processed"
cp "$ROOT/src/app.py" "$STAGE/src/"
cp -R "$ROOT/src/assets" "$STAGE/src/"
for f in master_dataset.csv mortgage_rates.csv undervalued_zips.csv; do
  [ -f "$ROOT/data/processed/$f" ] && cp "$ROOT/data/processed/$f" "$STAGE/data/processed/"
done
[ -d "$ROOT/models" ] && cp -R "$ROOT/models" "$STAGE/"

cat > "$STAGE/requirements.txt" <<'REQ'
dash==4.1.0
flask
plotly==6.7.0
pandas==3.0.2
numpy
scikit-learn==1.8.0
requests
REQ

cat > "$STAGE/app.py" <<'PY'
from src.app import server as app
PY

cd "$STAGE"
[ "$1" = "--stage-only" ] && { echo "$STAGE"; exit 0; }
vercel link --yes --project capitaledge
vercel deploy --prod --yes
