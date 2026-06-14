#!/usr/bin/env python
"""Load epiTCR RF pickle with sklearn 1.2.x and write predict_proba[:,1].

Run in an env with ``pip install scikit-learn==1.2.2`` when the main env
has sklearn>=1.3 (pickle dtype incompatibility).
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    x = np.load(args.features)
    model = pickle.load(open(args.model, "rb"))
    scores = model.predict_proba(x)[:, 1]
    np.save(args.output, scores)


if __name__ == "__main__":
    main()
