from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent import inspect_ollama
from app.config import get_settings


def main() -> int:
    settings = get_settings()
    diagnostics = inspect_ollama(settings)

    if diagnostics["ready"]:
        print(
            json.dumps(
                {
                    "status": "ready",
                    "configured_host": diagnostics["configured_host"],
                    "requested_models": diagnostics["requested_models"],
                    "installed_models": diagnostics["installed_models"],
                },
                indent=2,
            )
        )
        return 0

    if diagnostics["reachable"] is False:
        print(diagnostics["error"], file=sys.stderr)
        return 1

    pull_commands = "\n".join(
        f"ollama pull {model.removesuffix(':latest')}" for model in diagnostics["missing_models"]
    )
    print(
        json.dumps(
            {
                "status": "not_ready",
                "configured_host": diagnostics["configured_host"],
                "requested_models": diagnostics["requested_models"],
                "installed_models": diagnostics["installed_models"],
                "missing_models": diagnostics["missing_models"],
                "hint": pull_commands,
            },
            indent=2,
        ),
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
