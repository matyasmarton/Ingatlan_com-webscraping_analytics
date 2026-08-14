"""Training CLI: rebuild the canonical artefacts from the CSV dump.

Usage: ``python -m ingatlan.train --data ingatlan.csv --out artifacts/``

Writes exactly three files into ``--out``:

- ``model.joblib`` — the winning scikit-learn estimator (joblib, never pickle),
- ``columns.json`` — ``["Size", "Rooms", <sorted district labels incl. "other">]``,
- ``metrics.json`` — winner, per-model CV R2/MAE, sklearn version, date.

The model search mirrors the legacy notebook (``GridSearchCV`` over
``LinearRegression``/``Lasso``/``DecisionTreeRegressor`` with
``ShuffleSplit(n_splits=5, test_size=0.2, random_state=10)``) using modern
scikit-learn parameter grids.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.tree import DecisionTreeRegressor

from ingatlan.features import (
    collapse_rare_labels,
    parse_district,
    parse_price_huf,
    parse_rooms,
    parse_size_sqm,
)

_MODELS: dict[str, tuple[object, dict]] = {
    "LinearRegression": (LinearRegression(), {}),
    "Lasso": (
        Lasso(random_state=10, max_iter=5000),
        {"alpha": [1, 2], "selection": ["random", "cyclic"]},
    ),
    "DecisionTreeRegressor": (
        DecisionTreeRegressor(random_state=10),
        {
            "criterion": ["squared_error", "absolute_error"],
            "splitter": ["best", "random"],
        },
    ),
}


def build_dataset(csv_path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Parse the raw CSV into the cleaned feature matrix and price target."""
    df = pd.read_csv(csv_path)
    df["Price"] = df["Price"].map(parse_price_huf)
    df["Size"] = df["Size"].map(parse_size_sqm)
    df["Rooms"] = df["Rooms"].map(parse_rooms)
    df["Address"] = df["Address"].map(parse_district)
    df = df.dropna(subset=["Price", "Size"])
    df["Address"] = collapse_rare_labels(df["Address"])
    dummies = pd.get_dummies(df["Address"], dtype=int)
    # The API routes unknown locations to the "other" column, so it must exist
    # in the feature matrix even when no label is rare enough to be collapsed.
    if "other" not in dummies.columns:
        dummies["other"] = 0
    X = pd.concat([df["Size"], df["Rooms"], dummies], axis=1)
    return X, df["Price"].astype(int)


def train(X: pd.DataFrame, y: pd.Series) -> tuple[object, dict]:
    """Grid-search the candidate models; return the winner refitted on all data."""
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=10)
    cv_r2: dict[str, float] = {}
    cv_mae: dict[str, float] = {}
    best: tuple[float, str, object] | None = None

    for name, (estimator, params) in _MODELS.items():
        grid = GridSearchCV(
            estimator=estimator,
            param_grid=params,
            scoring={"r2": "r2", "mae": "neg_mean_absolute_error"},
            refit="r2",
            cv=cv,
        )
        grid.fit(X, y)
        idx = grid.best_index_
        cv_r2[name] = float(grid.best_score_)
        cv_mae[name] = float(-grid.cv_results_["mean_test_mae"][idx])
        if best is None or grid.best_score_ > best[0]:
            best = (float(grid.best_score_), name, grid.best_estimator_)

    assert best is not None, "no model was trained"
    _, winner, final_model = best
    metrics = {"winner": winner, "cv_r2": cv_r2, "cv_mae": cv_mae}
    return final_model, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="path to the ingatlan.com CSV")
    parser.add_argument("--out", required=True, help="output directory for artefacts")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    X, y = build_dataset(args.data)
    model, metrics = train(X, y)
    columns = ["Size", "Rooms", *sorted(X.columns[2:])]

    joblib.dump(model, out / "model.joblib")
    (out / "columns.json").write_text(json.dumps(columns, indent=2) + "\n", encoding="utf-8")

    metrics.update(
        {
            "sklearn_version": sklearn.__version__,
            "date": datetime.date.today().isoformat(),
        }
    )
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"winner={metrics['winner']} cv_r2={metrics['cv_r2']} cv_mae={metrics['cv_mae']}")
    print(f"wrote {out / 'model.joblib'}, {out / 'columns.json'}, {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
