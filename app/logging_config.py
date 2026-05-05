from __future__ import annotations

import logging
from logging.config import dictConfig

from app.config import Settings


def configure_logging(level: str = "INFO") -> None:
    normalized = level.upper()
    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                }
            },
            "root": {"level": normalized, "handlers": ["console"]},
            "loggers": {
                "uvicorn": {"level": normalized},
                "uvicorn.error": {"level": normalized},
                "uvicorn.access": {"level": normalized},
            },
        }
    )


def log_startup_banner(settings: Settings, training_total: int) -> None:
    logger = logging.getLogger("app.startup")
    logger.info("Starting %s", settings.app_name)
    backend = settings.normalized_llm_backend
    if backend == "glm":
        logger.info("  LLM backend: GLM")
        logger.info("  GLM model: %s", settings.normalized_glm_model)
    else:
        logger.info("  LLM backend: Ollama")
        logger.info("  Ollama host: %s", settings.ollama_host)
        logger.info("  Ollama model: %s", settings.normalized_ollama_model)
    logger.info("  Database target: %s", settings.database_target)
    logger.info("  Chroma path: %s", settings.chroma_path)
    logger.info("  Training entries detected: %s", training_total)
