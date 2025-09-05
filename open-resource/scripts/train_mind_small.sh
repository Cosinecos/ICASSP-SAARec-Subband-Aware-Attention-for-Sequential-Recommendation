#!/usr/bin/env bash
set -e
python tools/prepare_mind_small.py --train_dir data/raw/MINDsmall_train --dev_dir data/raw/MINDsmall_dev --out_dir data/processed/mind_small
python -m src.main --config configs/mind_small_saa.yaml
