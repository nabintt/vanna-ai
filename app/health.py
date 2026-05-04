from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from app.agent import inspect_glm
from app.config import Settings
from app.db import DatabaseClient, DatabaseConnectionError
from app.training import summarize_training


def build_health_payload(started_at: datetime) -> dict[str, Any]:
    now = datetime.now(UTC)
    return {
        "status": "ok",
        "timestamp": now.isoformat(),
        "started_at": started_at.isoformat(),
        "uptime_seconds": int((now - started_at).total_seconds()),
    }


def build_ready_payload(settings: Settings, db: DatabaseClient, vn: Any) -> tuple[bool, dict[str, Any]]:
    glm_status = inspect_glm(settings)

    try:
        db.test_connection()
        database_status = {
            "ready": True,
            "target": settings.database_target,
            "error": None,
        }
    except DatabaseConnectionError as exc:
        database_status = {
            "ready": False,
            "target": settings.database_target,
            "error": str(exc),
        }

    training_status = summarize_training(vn)
    training_payload = {
        "ready": training_status["total_entries"] > 0 or not settings.train_on_start,
        "required": settings.train_on_start,
        "available": training_status["total_entries"] > 0,
        **training_status,
    }

    ready = bool(glm_status["ready"] and database_status["ready"] and training_payload["ready"])
    payload = {
        "status": "ready" if ready else "not_ready",
        "timestamp": datetime.now(UTC).isoformat(),
        "checks": {
            "glm": glm_status,
            "database": database_status,
            "training": training_payload,
        },
    }
    return ready, payload
