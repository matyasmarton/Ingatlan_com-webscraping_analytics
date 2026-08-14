"""Training pipeline tests on a small CSV slice."""

import json
import math
import sys
from pathlib import Path

import pandas as pd

from ingatlan.train import main

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_train_writes_artefacts(tmp_path, monkeypatch):
    df = pd.read_csv(_REPO_ROOT / "ingatlan.csv").head(50)
    data_csv = tmp_path / "slice.csv"
    df.to_csv(data_csv, index=False)

    monkeypatch.setattr(
        sys, "argv", ["ingatlan.train", "--data", str(data_csv), "--out", str(tmp_path)]
    )
    main()

    assert (tmp_path / "model.joblib").is_file()
    assert (tmp_path / "columns.json").is_file()
    assert (tmp_path / "metrics.json").is_file()

    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["winner"] in {"LinearRegression", "Lasso", "DecisionTreeRegressor"}
    assert all(math.isfinite(v) for v in metrics["cv_r2"].values())
    assert all(math.isfinite(v) for v in metrics["cv_mae"].values())

    columns = json.loads((tmp_path / "columns.json").read_text(encoding="utf-8"))
    assert columns[:2] == ["Size", "Rooms"]
    assert "other" in columns
