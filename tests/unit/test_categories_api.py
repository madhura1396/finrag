"""The ADR 0005 endpoints' error mapping, with categorization mocked."""

import pytest
from fastapi.testclient import TestClient

from app import categorization, main
from app.sessions import get_session


@pytest.fixture
def client():
    main.app.dependency_overrides[get_session] = lambda: object()
    yield TestClient(main.app, raise_server_exceptions=False, client=("127.0.0.1", 50000))
    main.app.dependency_overrides.clear()


def test_preview_returns_the_effect_and_token(client, monkeypatch):
    monkeypatch.setattr(categorization, "preview", lambda db, payload: {"token": "1:abc", "data_version": 1,
                                                                        "effect": {"rows_changed": 2}})
    response = client.post("/changes/preview", json={"kind": "create_rule", "params": {"pattern": "X"}})
    assert response.status_code == 200
    assert response.json()["token"] == "1:abc"


def test_invalid_change_is_422_with_reason(client, monkeypatch):
    def reject(db, payload):
        raise categorization.ChangeRejected("bad pattern", {"field": "pattern"})
    monkeypatch.setattr(categorization, "preview", reject)
    response = client.post("/changes/preview", json={"kind": "create_rule", "params": {}})
    assert response.status_code == 422
    assert response.json()["detail"] == {"reason": "bad pattern", "detail": {"field": "pattern"}}


def test_stale_preview_is_409(client, monkeypatch):
    def stale(db, payload, token):
        raise categorization.StalePreview("data changed")
    monkeypatch.setattr(categorization, "apply", stale)
    monkeypatch.setattr(main, "run_maintenance", lambda: None)
    response = client.post("/changes/apply", json={"kind": "create_rule", "params": {}, "token": "0:x"})
    assert response.status_code == 409


def test_unknown_change_kind_is_rejected_by_the_schema(client):
    assert client.post("/changes/preview", json={"kind": "drop_everything", "params": {}}).status_code == 422


def test_apply_schedules_background_maintenance(client, monkeypatch):
    ran = []
    monkeypatch.setattr(categorization, "apply", lambda db, payload, token: {"change_id": 1, "effect": {}})
    monkeypatch.setattr(main, "run_maintenance", lambda: ran.append(True))
    assert client.post("/changes/apply", json={"kind": "pin", "params": {}}).status_code == 200
    assert ran == [True]


def test_upload_reports_rows_pending_when_labelling_fails(monkeypatch, tmp_path):
    from datetime import date
    monkeypatch.setattr(main, "UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(main, "extract_from_pdf", lambda path, name: {
        "statement_id": 1, "period_start": date(2025, 12, 3), "period_end": date(2026, 1, 2),
        "total_transactions": 4})

    def boom(statement_id):
        raise RuntimeError("model blew up")
    monkeypatch.setattr(main, "enrich_transactions", boom)
    monkeypatch.setattr(main, "run_maintenance", lambda: None)
    client = TestClient(main.app, raise_server_exceptions=False, client=("127.0.0.1", 50000))
    response = client.post("/upload", files={"file": ("s.pdf", b"%PDF-1.4", "application/pdf")})
    assert response.status_code == 200
    assert response.json()["pending"] == 4
