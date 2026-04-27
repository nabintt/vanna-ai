from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent import connect_vanna_to_database, create_vanna_agent, ensure_ollama_ready
from app.config import get_settings
from app.db import DatabaseClient
from app.logging_config import configure_logging
from app.training import TrainingBootstrapError, bootstrap_training


def main() -> int:
    settings = get_settings()
    configure_logging(settings.log_level)

    try:
        ensure_ollama_ready(settings)
        db = DatabaseClient(settings)
        db.test_connection()
        vn = create_vanna_agent(settings)
        connect_vanna_to_database(vn, settings)
        summary = bootstrap_training(vn=vn, db=db, settings=settings)
        print(json.dumps(summary, indent=2))
        db.close()
        return 0
    except Exception as exc:  # pragma: no cover - CLI guard
        message = str(exc)
        if isinstance(exc, TrainingBootstrapError):
            message = f"Bootstrap failed: {exc}"
        print(message, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
