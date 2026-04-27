from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from app.config import Settings
from app.db import DatabaseClient
from app.server import AppServices, create_app


class FakeVanna:
    def get_training_data(self):
        return pd.DataFrame(
            [
                {
                    "id": "1-doc",
                    "question": None,
                    "content": "schema notes",
                    "training_data_type": "documentation",
                }
            ]
        )


class FakeDatabase(DatabaseClient):
    def __init__(self):
        self.settings = None

    def close(self) -> None:
        return None

    def test_connection(self) -> None:
        return None


def build_test_app() -> TestClient:
    settings = Settings(
        _env_file=None,
        db_name="analytics",
        db_user="analyst",
        db_password="secret",
    )
    services = AppServices(
        settings=settings,
        db=FakeDatabase(),
        vn=FakeVanna(),
        vanna_v2_chat_handler=object(),
        started_at=datetime.now(UTC),
    )
    app = create_app(services=services, settings=settings)
    return TestClient(app)


def test_health_endpoint_returns_ok():
    with build_test_app() as client:
        response = client.get("/health")
        payload = response.json()

    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert "uptime_seconds" in payload


def test_ready_endpoint_returns_success_with_mocked_ollama():
    with build_test_app() as client:
        with patch(
            "app.health.inspect_ollama",
            return_value={
                "ready": True,
                "reachable": True,
                "configured_host": "http://localhost:11434",
                "requested_models": ["llama3.2:latest"],
                "installed_models": ["llama3.2:latest"],
                "missing_models": [],
                "error": None,
            },
        ):
            response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
