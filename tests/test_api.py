"""End-to-end tests of the annotation loop through the HTTP API."""

import os
import sys
import tempfile

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

DB = os.path.join(tempfile.mkdtemp(), "test.db")
os.environ["DATABASE_URL"] = f"sqlite:///{DB}"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_ui_is_served(client):
    r = client.get("/ui")
    assert r.status_code == 200 and "Active Learning" in r.text


def test_query_before_initialize_is_rejected(client):
    assert client.post("/api/v1/query", json={"strategy": "entropy"}).status_code == 400


def test_invalid_strategy_is_rejected(client):
    assert client.post("/api/v1/query", json={"strategy": "magic"}).status_code == 422


def test_labels(client):
    assert client.get("/api/v1/labels").json()["labels"] == ["negative", "neutral", "positive"]


def test_full_active_learning_cycle(client):
    r = client.post("/api/v1/initialize", json={"seed_size": 12, "strategy": "entropy", "batch_size": 5})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["seed_size"] == 12 and body["test_size"] == 60
    assert 0 <= body["metrics"]["f1_macro"] <= 1

    s = client.get("/api/v1/status").json()
    assert s["labeled"] == 12 and s["cycle"] == 0
    pool_before = s["unlabeled_pool"]

    q = client.post("/api/v1/query", json={"strategy": "entropy", "batch_size": 5}).json()
    assert len(q["selected"]) == 5
    scores = [d["uncertainty"] for d in q["selected"]]
    assert scores == sorted(scores, reverse=True)

    ann = [{"document_id": d["id"], "label": d["predicted_label"]} for d in q["selected"]]
    a = client.post("/api/v1/annotate", json={"annotations": ann})
    assert a.status_code == 200, a.text
    assert a.json()["cycle"] == 1 and a.json()["accepted"] == 5

    s = client.get("/api/v1/status").json()
    assert s["labeled"] == 17 and s["unlabeled_pool"] == pool_before - 5

    # labelling the same documents twice is rejected
    assert client.post("/api/v1/annotate", json={"annotations": ann}).status_code == 400

    m = client.get("/api/v1/evaluate").json()
    assert set(m) == {"accuracy", "precision_macro", "recall_macro", "f1_macro"}


@pytest.mark.parametrize("strategy", ["random", "least_confidence", "margin"])
def test_other_strategies_query(client, strategy):
    q = client.post("/api/v1/query", json={"strategy": strategy, "batch_size": 3})
    assert q.status_code == 200 and len(q.json()["selected"]) == 3
