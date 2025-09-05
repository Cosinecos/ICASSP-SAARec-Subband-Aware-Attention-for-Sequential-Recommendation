#!/usr/bin/env bash
set -e
python tools/prepare_amazon.py --in_file data/raw/amazon/All_Beauty.json.gz --out_dir data/processed/amazon_beauty
python -m src.main --config configs/amazon_beauty_saa.yaml
