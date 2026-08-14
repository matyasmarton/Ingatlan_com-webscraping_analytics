"""API integration tests against the committed artefacts."""

import pytest
from fastapi.testclient import TestClient

from ingatlan.api import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_predict_known_district(client):
    resp = client.post("/predict", json={"location": "XIII", "sqm": 61, "rooms": 2.5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["estimated_price_huf"] > 0
    assert isinstance(body["model_version"], str)


def test_predict_unknown_location_falls_back_to_other(client):
    resp = client.post("/predict", json={"location": "UNKNOWN_FAKE", "sqm": 60, "rooms": 2})
    assert resp.status_code == 200


def test_predict_invalid_input_422(client):
    resp = client.post("/predict", json={"location": "XIII", "sqm": -5, "rooms": 2})
    assert resp.status_code == 422


def test_locations_contains_roman_district(client):
    resp = client.get("/locations")
    assert resp.status_code == 200
    locations = resp.json()
    assert locations
    assert "XIII" in locations


def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
