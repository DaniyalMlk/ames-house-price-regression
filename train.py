"""Fit the frozen ensemble and create model-generated submissions.

This entry point reads train.csv, test.csv, and sample_soln.csv only.
"""
from __future__ import annotations

import os
for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(variable, "1")

import argparse
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from house_models import comparable_guard, make_model, training_mask

ROOT = Path(__file__).resolve().parent


def load_data(data_dir: Path):
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    template = pd.read_csv(data_dir / "sample_soln.csv")
    if "SalePrice" not in train or "SalePrice" in test:
        raise ValueError("SalePrice must be present in train.csv and absent from test.csv.")
    if set(train.columns) - {"SalePrice"} != set(test.columns):
        raise ValueError("Training and test feature names differ.")
    if len(test) != len(template) or not template["Id"].is_unique:
        raise ValueError("The submission template has invalid IDs or row count.")
    return train, test, template


def generate(data_dir: Path | None = None, output_dir: Path | None = None):
    data_dir = Path(data_dir) if data_dir else ROOT / "data"
    output_dir = Path(output_dir) if output_dir else ROOT / "outputs"
    train, test, template = load_data(data_dir)
    config_path = ROOT / "config" / "final_model.json"
    config = json.loads(config_path.read_text())
    X = train.drop(columns="SalePrice")
    y = train["SalePrice"].to_numpy(float)
    weighted_predictions = []
    core_predictions = []
    core_weights = []
    for component in config["models"]:
        model = make_model(component["spec"])
        keep = training_mask(X, y, component["policy"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            model.fit(X.loc[keep], y[keep])
            prediction = np.maximum(1, model.predict(test))
        weighted_predictions.append(component["weight"] * prediction)
        if component["spec"]["family"] != "lightgbm":
            core_predictions.append(component["weight"] * prediction)
            core_weights.append(component["weight"])
        print(f"Fitted {component['name']} ({component['weight']:.2%})")
    raw = np.sum(weighted_predictions, axis=0)
    candidates = {
        "submission_model.csv": comparable_guard(X, y, test, raw, k=config["guard_k"]),
        "submission_nearest_comparable.csv": comparable_guard(X, y, test, raw, k=1),
        "submission_sklearn_only.csv": comparable_guard(
            X, y, test, np.sum(core_predictions, axis=0) / sum(core_weights), k=config["guard_k"]
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprints = {}
    for name, prediction in candidates.items():
        if not np.isfinite(prediction).all() or (prediction <= 0).any():
            raise ValueError(f"Invalid predictions for {name}")
        output = pd.DataFrame({"Id": template["Id"].to_numpy(), "SalePrice": prediction})
        path = output_dir / name
        output.to_csv(path, index=False, float_format="%.8f")
        fingerprints[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    audit = {
        "rows": len(test),
        "input_test_id_range": [int(test.Id.min()), int(test.Id.max())],
        "submission_id_range": [int(template.Id.min()), int(template.Id.max())],
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "submission_sha256": fingerprints,
        "target_source": "train.csv only; no reference-answer input",
    }
    (output_dir / "prediction_audit.json").write_text(json.dumps(audit, indent=2))
    return candidates, audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    _, audit = generate(args.data_dir, args.output_dir)
    print(json.dumps(audit, indent=2))
