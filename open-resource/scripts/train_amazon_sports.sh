#!/usr/bin/env bash
set -e
python tools/prepare_amazon.py --in_file data/raw/amazon/Sports_and_Outdoors.json.gz --out_dir data/processed/amazon_sports
python -m src.main --config configs/amazon_sports_saa.yaml
