from __future__ import annotations

import logging
from typing import Any

import httpx
from vanna.chromadb import ChromaDB_VectorStore
from vanna.ollama import Ollama

from app.config import Settings

logger = logging.getLogger(__name__)


class OllamaNotReadyError(RuntimeError):
    """Raised when the local Ollama runtime is unavailable or missing models."""


class OllamaEmbeddingFunction:
    def __init__(self, host: str, model: str, timeout: float):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    def __call__(self, input: list[str]) -> list[list[float]]:
        response = httpx.post(
            f"{self.host}/api/embed",
            json={"model": self.model, "input": input},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        embeddings = payload.get("embeddings")
        if not embeddings:
            raise OllamaNotReadyError(
                f"Ollama did not return embeddings for model '{self.model}'."
            )
        return embeddings


def _collect_model_names(models: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for model_info in models:
        for key in ("name", "model"):
            value = model_info.get(key)
            if not value:
                continue
            names.add(value)
            if value.endswith(":latest"):
                names.add(value[: -len(":latest")])
    return names


def inspect_ollama(settings: Settings) -> dict[str, Any]:
    requested_models = [settings.normalized_ollama_model]
    if settings.normalized_ollama_embed_model:
        requested_models.append(settings.normalized_ollama_embed_model)

    try:
        response = httpx.get(
            f"{settings.ollama_host.rstrip('/')}/api/tags",
            timeout=settings.ollama_timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPError as exc:
        return {
            "ready": False,
            "reachable": False,
            "configured_host": settings.ollama_host,
            "requested_models": requested_models,
            "installed_models": [],
            "missing_models": requested_models,
            "error": f"Unable to reach Ollama at {settings.ollama_host}: {exc}",
        }

    installed_models = payload.get("models", [])
    installed_names = sorted(_collect_model_names(installed_models))
    missing_models = [model for model in requested_models if model not in installed_names]
    return {
        "ready": not missing_models,
        "reachable": True,
        "configured_host": settings.ollama_host,
        "requested_models": requested_models,
        "installed_models": installed_names,
        "missing_models": missing_models,
        "error": None,
    }


def ensure_ollama_ready(settings: Settings) -> dict[str, Any]:
    diagnostics = inspect_ollama(settings)
    if diagnostics["reachable"] is False:
        raise OllamaNotReadyError(diagnostics["error"])
    if diagnostics["missing_models"]:
        pull_instructions = ", ".join(
            f"ollama pull {model.removesuffix(':latest')}"
            for model in diagnostics["missing_models"]
        )
        raise OllamaNotReadyError(
            "Missing required Ollama model(s): "
            + ", ".join(diagnostics["missing_models"])
            + f". Install them with: {pull_instructions}"
        )
    return diagnostics


class FailFastOllama(Ollama):
    @staticmethod
    def _Ollama__pull_model_if_ne(ollama_client: Any, model: str) -> None:
        model_response = ollama_client.list()
        # Handle both dict (older client) and object (newer client) responses.
        if isinstance(model_response, dict):
            models_list = model_response.get("models", [])
        else:
            models_list = getattr(model_response, "models", []) or []
            models_list = [m.__dict__ if hasattr(m, "__dict__") else m for m in models_list]
        installed_names = _collect_model_names(models_list)
        short_model = model.removesuffix(":latest")
        if model not in installed_names and short_model not in installed_names:
            raise OllamaNotReadyError(
                f"Ollama model '{model}' is not installed. Run: ollama pull {short_model}"
            )


class LocalChromaOllamaVanna(ChromaDB_VectorStore, FailFastOllama):
    def __init__(self, config: dict[str, Any]):
        ChromaDB_VectorStore.__init__(self, config=config)
        FailFastOllama.__init__(self, config=config)


def create_vanna_agent(settings: Settings) -> LocalChromaOllamaVanna:
    ensure_ollama_ready(settings)

    config: dict[str, Any] = {
        "model": settings.normalized_ollama_model,
        "ollama_host": settings.ollama_host,
        "ollama_timeout": settings.ollama_timeout,
        "keep_alive": settings.ollama_keep_alive,
        "options": {"num_ctx": settings.ollama_num_ctx},
        "path": str(settings.chroma_path),
        "n_results": settings.vanna_top_k,
        "initial_prompt": (
            "You are a careful SQL generation assistant. "
            "Return only executable SQL for the configured database dialect."
        ),
    }

    if settings.normalized_ollama_embed_model:
        config["embedding_function"] = OllamaEmbeddingFunction(
            host=settings.ollama_host,
            model=settings.normalized_ollama_embed_model,
            timeout=settings.ollama_timeout,
        )

    logger.info("Creating Vanna agent with local Chroma persistence at %s", settings.chroma_path)
    return LocalChromaOllamaVanna(config=config)


def connect_vanna_to_database(vn: LocalChromaOllamaVanna, settings: Settings) -> None:
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
