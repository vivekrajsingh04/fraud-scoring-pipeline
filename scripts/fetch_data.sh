#!/usr/bin/env bash
# Fetch a real labelled dataset. We do not generate transactions.
#
# IEEE-CIS requires a Kaggle account and accepting the competition rules, so it
# cannot be downloaded unattended without credentials. Set KAGGLE_USERNAME and
# KAGGLE_KEY (or have ~/.kaggle/kaggle.json in place) before running.
set -euo pipefail

DATASET="${1:-ieee}"
DEST="$(cd "$(dirname "$0")/.." && pwd)/data"
mkdir -p "$DEST"

case "$DATASET" in
  ieee)
    command -v kaggle >/dev/null || { echo "pip install kaggle first"; exit 1; }
    echo "downloading IEEE-CIS Fraud Detection (~590k labelled transactions)"
    kaggle competitions download -c ieee-fraud-detection -f train_transaction.csv -p "$DEST"
    unzip -o "$DEST/train_transaction.csv.zip" -d "$DEST" 2>/dev/null || true
    echo "-> $DEST/train_transaction.csv"
    ;;
  sparkov)
    command -v kaggle >/dev/null || { echo "pip install kaggle first"; exit 1; }
    echo "downloading Sparkov simulated credit card transactions"
    kaggle datasets download -d kartik2112/fraud-detection -p "$DEST" --unzip
    echo "-> $DEST/fraudTrain.csv"
    ;;
  *)
    echo "usage: $0 [ieee|sparkov]" >&2
    exit 2
    ;;
esac
