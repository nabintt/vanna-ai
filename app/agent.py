from __future__ import annotations

import logging
from typing import Any

from vanna.legacy.chromadb import ChromaDB_VectorStore
from vanna.legacy.ZhipuAI import ZhipuAI_Chat

from app.config import Settings

logger = logging.getLogger(__name__)


class GLMNotReadyError(RuntimeError):
    """Raised when the GLM API key is not configured or invalid."""


def inspect_glm(settings: Settings) -> dict[str, Any]:
    api_key = settings.glm_api_key.strip() if settings.glm_api_key else ""
    model = settings.normalized_glm_model

    if not api_key:
        return {
            "ready": False,
            "configured": False,
            "configured_model": model,
            "error": "GLM_API_KEY is not set. Please set it in your .env file.",
        }

    return {
        "ready": True,
        "configured": True,
        "configured_model": model,
        "api_url": settings.glm_api_url or "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        "error": None,
    }


def ensure_glm_ready(settings: Settings) -> dict[str, Any]:
    diagnostics = inspect_glm(settings)
    if not diagnostics["ready"]:
        raise GLMNotReadyError(diagnostics["error"])
    return diagnostics


class LocalChromaGLMVanna(ChromaDB_VectorStore, ZhipuAI_Chat):
    def __init__(self, config: dict[str, Any]):
        ChromaDB_VectorStore.__init__(self, config=config)
        ZhipuAI_Chat.__init__(self, config=config)


def create_vanna_agent(settings: Settings) -> LocalChromaGLMVanna:
    ensure_glm_ready(settings)

    config: dict[str, Any] = {
        "api_key": settings.glm_api_key,
        "model": settings.normalized_glm_model,
        "path": str(settings.chroma_path),
        "n_results": settings.vanna_top_k,
        "initial_prompt": (
            "You are a careful SQL generation assistant. "
            "Return only executable SQL for the configured database dialect."
        ),
    }

    if settings.glm_api_url:
        config["api_url"] = settings.glm_api_url

    logger.info("Creating Vanna agent with local Chroma persistence at %s", settings.chroma_path)
    logger.info("Using GLM model: %s", settings.normalized_glm_model)
    return LocalChromaGLMVanna(config=config)


def connect_vanna_to_database(vn: LocalChromaGLMVanna, settings: Settings) -> None:
    if settings.normalized_db_type == "postgres":
        vn.connect_to_postgres(
            host=settings.db_host,
            dbname=settings.db_name,
            user=settings.db_user,
            password=settings.db_password,
            port=settings.db_port,
        )
        return

    if settings.normalized_db_type == "mysql":
        vn.connect_to_mysql(
            host=settings.db_host,
            dbname=settings.db_name,
            user=settings.db_user,
            password=settings.db_password,
            port=settings.db_port,
        )
        return

    raise ValueError(
        f"Unsupported DB_TYPE '{settings.db_type}'. Supported values are 'postgres' and 'mysql'."
    )
