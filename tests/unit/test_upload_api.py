"""/upload error mapping, with the pipeline mocked.

UPLOAD_DIR is redirected to tmp_path in every test: app/main.py saves the file
before the pipeline runs, and tests must never write into uploads/.
"""

import os
import uuid
from datetime import date

import pytest
from fastapi.testclient import TestClient

from app import main
from app.statement import DuplicateStatement, Mismatch, StatementRejected


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(main, "enrich_transactions", lambda sid: {"enriched": 2, "needs_review": 0, "pending": 0})
    monkeypatch.setattr(main, "run_maintenance", lambda: None)
    # raise_server_exceptions=False so an unmapped exception shows up as the 500 a
    # real client would get, instead of being re-raised into the test.
    # The API is loopback-only (ADR 0005); present as a local client.
    return TestClient(main.app, raise_server_exceptions=False, client=("127.0.0.1", 50000))


def upload(client, name="synthetic.pdf"):
    return client.post("/upload", files={"file": (name, b"%PDF-1.4 synthetic", "application/pdf")})


def pipeline_raises(monkeypatch, exc):
    def fake(filepath, filename):
        raise exc
    monkeypatch.setattr(main, "extract_from_pdf", fake)


def test_rejected_statement_returns_422_with_mismatch_list(client, monkeypatch):
    mismatch = Mismatch("section_total", "purchases", "35.50", "35.51")
    pipeline_raises(monkeypatch, StatementRejected([mismatch]))

    response = upload(client)

    assert response.status_code == 422
    assert response.json() == {
        "detail": {
            "reason": "statement did not reconcile",
            "mismatches": [
                {"check": "section_total", "section": "purchases", "expected": "35.50", "actual": "35.51"}
            ],
        }
    }


def test_duplicate_returns_409(client, monkeypatch):
    pipeline_raises(monkeypatch, DuplicateStatement("Statement for 2025-12-03 → 2026-01-02 already uploaded"))
    response = upload(client)
    assert response.status_code == 409
    assert "already uploaded" in response.json()["detail"]


def test_unrelated_value_error_is_not_reported_as_conflict(client, monkeypatch):
    pipeline_raises(monkeypatch, ValueError("something else broke"))
    assert upload(client).status_code == 500


def test_success_returns_statement_summary(client, monkeypatch):
    monkeypatch.setattr(main, "extract_from_pdf", lambda filepath, filename: {
        "statement_id":       7,
        "period_start":       date(2025, 12, 3),
        "period_end":         date(2026, 1, 2),
        "total_transactions": 5,
    })
    response = upload(client)
    assert response.status_code == 200
    assert response.json() == {
        "statement_id":       7,
        "period_start":       "2025-12-03",
        "period_end":         "2026-01-02",
        "total_transactions": 5,
        "enriched":           2,
        "needs_review":       0,
        "pending":            0,
    }


def test_upload_is_written_to_the_redirected_dir_not_uploads(client, monkeypatch, tmp_path):
    seen = {}

    def fake(filepath, filename):
        seen["path"] = filepath
        raise DuplicateStatement("dup")

    monkeypatch.setattr(main, "extract_from_pdf", fake)
    name = f"unit-test-{uuid.uuid4().hex}.pdf"

    upload(client, name)

    assert seen["path"] == os.path.join(str(tmp_path), name)
    assert (tmp_path / name).exists()
    # Checks one invented path only; never lists or reads uploads/.
    assert not os.path.exists(os.path.join("uploads", name))


def test_non_loopback_clients_are_refused(monkeypatch):
    remote = TestClient(main.app, raise_server_exceptions=False, client=("203.0.113.7", 50000))
    response = remote.get("/categories")
    assert response.status_code == 403


def test_ipv6_loopback_is_allowed(client, monkeypatch, tmp_path):
    local6 = TestClient(main.app, raise_server_exceptions=False, client=("::1", 50000))
    pipeline_raises(monkeypatch, DuplicateStatement("dup"))
    assert upload(local6).status_code == 409
