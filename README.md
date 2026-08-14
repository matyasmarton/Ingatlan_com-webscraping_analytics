# Ingatlan Budapest Price

Budapest apartment price estimation: a cleaning pipeline, a scikit-learn model search, and a FastAPI microservice, trained on ingatlan.com listings (2020–2021).

## Architecture

- `src/ingatlan/features.py` — parsing and cleaning of the raw CSV (`Price`, `Address`, `Size`, `Rooms`): prices to HUF, sizes to m², combined rooms as `n + m/2`, and canonical roman-numeral district labels (`"XIII"`).
- `src/ingatlan/train.py` — CLI that grid-searches LinearRegression / Lasso / DecisionTreeRegressor under shuffled cross-validation and writes the canonical artefacts `artifacts/model.joblib`, `artifacts/columns.json`, `artifacts/metrics.json`.
- `src/ingatlan/api.py` — FastAPI service: `POST /predict`, `GET /locations`, `GET /healthz`, plus a vanilla-JS estimator frontend in `static/`.

No scraper is included: ingatlan.com is protected by Cloudflare, so the dataset is committed as `ingatlan.csv` and scraping stays out of this repo.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run uvicorn ingatlan.api:app --reload
```

Open http://localhost:8000 for the estimator UI.

### Docker

```bash
docker build -t ingatlan .
docker run -d -p 8000:8000 ingatlan
```

### Retrain

```bash
uv run python -m ingatlan.train --data ingatlan.csv --out artifacts/
```

## API

`GET /healthz` → `{"status": "ok"}`

`GET /locations` → the districts the model knows, e.g. `["IV", "V", "XIII", "XIX", "XX", "other"]`

`POST /predict` with `{"location": "XIII", "sqm": 61, "rooms": 2.5}`:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"location": "XIII", "sqm": 61, "rooms": 2.5}'
```

→ `{"estimated_price_huf": 36450000, "model_version": "2026-08-14"}`

Unknown locations fall back to the `"other"` district and still return 200; invalid inputs (e.g. `sqm <= 0`) return 422.

## Testing

```bash
uv run pytest -q
```

## Deployment

Azure Container Apps deployment lands in a follow-up; the live-demo URL will be added here.

## Legacy version

The pre-refactor state — the original Flask server, the scraping notebooks, and the hand-pickled model — is preserved at [`legacy-v0`](https://github.com/matyasmarton/Ingatlan_com-webscraping_analytics/tree/legacy-v0). What started as a scraping experiment with a pickled model behind a Flask app grew into a reproducible pipeline: parsers, model search, tests, and a containerized API, all under version control.
