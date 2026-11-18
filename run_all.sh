#!/usr/bin/env bash
# End-to-end pipeline after data/db/*.parquet has been built. Usage: bash run_all.sh <studentID>
set -e
SID=${1:-studentID}
PY=${PY:-.venv/Scripts/python}
[ -x "$PY" ] || PY=.venv/bin/python
cd src
$PY prepare.py
$PY train_lgbm.py --variant temporal
$PY train_lgbm.py --variant full
$PY train_gnn.py --user_feats none
$PY train_gnn.py --user_feats full --no_time
$PY train_gnn.py --user_feats full
$PY ensemble.py --student_id "$SID"
$PY draw_graphs.py
$PY plots.py
