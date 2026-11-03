"""Rerun the original training-only screen, including optional CatBoost models."""
import os
for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(variable, "1")

import argparse
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from house_models import comparable_guard, make_model, training_mask


def evaluate(configuration, X, y, folds):
    raw = np.empty(len(y))
    guarded = np.empty(len(y))
    for training, validation in folds:
        keep = training_mask(X.iloc[training], y[training], configuration["policy"])
        model = make_model(configuration["spec"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            model.fit(X.iloc[training].loc[keep], y[training][keep])
            prediction = np.maximum(1, model.predict(X.iloc[validation]))
        raw[validation] = prediction
        guarded[validation] = comparable_guard(
            X.iloc[training], y[training], X.iloc[validation], prediction, k=3
        )
    return {
        "name": configuration["name"],
        "family": configuration["spec"]["family"],
        "rmse": float(np.sqrt(np.mean((raw - y) ** 2))),
        "guarded_rmse": float(np.sqrt(np.mean((guarded - y) ** 2))),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-run subset")
    args = parser.parse_args()
    configurations = json.loads((ROOT / "experiments" / "screen_configurations.json").read_text())
    if args.limit is not None:
        configurations = configurations[:args.limit]
    train = pd.read_csv(ROOT / "data" / "train.csv")
    X = train.drop(columns="SalePrice")
    y = train.SalePrice.to_numpy(float)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=20261103).split(
        X, pd.qcut(y, 10, labels=False, duplicates="drop")
    ))
    result = Parallel(n_jobs=args.jobs)(delayed(evaluate)(c, X, y, folds) for c in configurations)
    output = ROOT / "outputs" / "rerun_search_scores.csv"
    output.parent.mkdir(exist_ok=True)
    pd.DataFrame(result).sort_values("guarded_rmse").to_csv(output, index=False)
    print(pd.DataFrame(result).sort_values("guarded_rmse").to_string(index=False))
