#!/usr/bin/env bash
set -e
python tools/prepare_ml1m.py --in_dir data/raw/ml-1m --out_dir data/processed/ml1m
python -m src.main --config configs/ml1m_saa.yaml
