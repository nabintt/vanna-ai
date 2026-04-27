from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.agent import connect_vanna_to_database, create_vanna_agent
from app.config import Settings, get_settings
from app.db import DatabaseClient, DatabaseConnectionError
from app.health import build_health_payload, build_ready_payload
from app.logging_config import configure_logging, log_startup_banner
from app.training import (
    TrainingBootstrapError,
    add_question_sql_if_missing,
    bootstrap_training,
    ensure_training_ready,
    summarize_training,
)

logger = logging.getLogger(__name__)


@dataclass
class AppServices:
    settings: Settings
    db: DatabaseClient
    vn: Any
    started_at: datetime


class QuestionSqlPairInput(BaseModel):
    question: str = Field(min_length=3)
    sql: str = Field(min_length=3)


class TrainRequest(BaseModel):
    documentation: list[str] = Field(default_factory=list)
    question_sql_pairs: list[QuestionSqlPairInput] = Field(default_factory=list)


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    auto_train: bool = True
    allow_llm_to_see_data: bool = False
    max_rows: int = Field(default=200, ge=1, le=1000)


class GenerateSqlRequest(BaseModel):
    question: str = Field(min_length=3)
    allow_llm_to_see_data: bool = False


class RunSqlRequest(BaseModel):
    sql: str = Field(min_length=1)
    max_rows: int = Field(default=200, ge=1, le=1000)


def build_services(settings: Settings | None = None) -> AppServices:
    settings = settings or get_settings()
    settings.ensure_directories()
    configure_logging(settings.log_level)

    db = DatabaseClient(settings)
    db.test_connection()

    vn = create_vanna_agent(settings)
    connect_vanna_to_database(vn, settings)
    training_state = ensure_training_ready(vn, db, settings)

    training_total = training_state["training"]["total_entries"]
    log_startup_banner(settings, training_total)
    return AppServices(
        settings=settings,
        db=db,
        vn=vn,
        started_at=datetime.now(UTC),
    )


def create_app(
    services: AppServices | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if services is None:
            app.state.services = build_services(settings=settings)
        else:
            app.state.services = services
        try:
            yield
        finally:
            current_services = getattr(app.state, "services", None)
            if current_services is not None:
                current_services.db.close()

    if settings is not None:
        effective_settings = settings
    elif services is not None:
        effective_settings = services.settings
    else:
        effective_settings = get_settings()

    app = FastAPI(
        title=effective_settings.app_name,
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        app_services: AppServices = request.app.state.services
        return build_health_payload(app_services.started_at)

    @app.get("/ready")
    async def ready(request: Request) -> dict[str, Any]:
        app_services: AppServices = request.app.state.services
        is_ready, payload = build_ready_payload(
            app_services.settings,
            app_services.db,
            app_services.vn,
        )
        if not is_ready:
            return JSONResponse(status_code=503, content=payload)
        return payload

    @app.post("/train")
    async def train_endpoint(request: Request, body: TrainRequest) -> dict[str, Any]:
        app_services: AppServices = request.app.state.services
        try:
            summary = bootstrap_training(
                vn=app_services.vn,
                db=app_services.db,
                settings=app_services.settings,
                extra_documentation=body.documentation,
                extra_question_sql_pairs=[item.model_dump() for item in body.question_sql_pairs],
            )
            return summary
        except (TrainingBootstrapError, DatabaseConnectionError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/generate_sql")
    async def generate_sql_endpoint(
        request: Request,
        body: GenerateSqlRequest,
    ) -> dict[str, Any]:
        app_services: AppServices = request.app.state.services
        try:
            sql = app_services.vn.generate_sql(
                question=body.question,
                allow_llm_to_see_data=body.allow_llm_to_see_data,
            )
            return {
                "question": body.question,
                "sql": sql,
                "training": summarize_training(app_services.vn),
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"SQL generation failed: {exc}") from exc

    @app.post("/run_sql")
    async def run_sql_endpoint(request: Request, body: RunSqlRequest) -> dict[str, Any]:
        app_services: AppServices = request.app.state.services
        try:
            result = app_services.db.execute_sql(body.sql, max_rows=body.max_rows)
            return result.to_dict()
        except (DatabaseConnectionError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/ask")
    async def ask_endpoint(request: Request, body: AskRequest) -> dict[str, Any]:
        app_services: AppServices = request.app.state.services
        try:
            sql = app_services.vn.generate_sql(
                question=body.question,
                allow_llm_to_see_data=body.allow_llm_to_see_data,
            )
            result = app_services.db.execute_sql(sql, max_rows=body.max_rows)
            training_saved = False
            if body.auto_train:
                training_saved = add_question_sql_if_missing(
                    app_services.vn,
                    question=body.question,
                    sql=sql,
                )
            return {
                "question": body.question,
                "sql": sql,
                "result": result.to_dict(),
                "auto_trained": training_saved,
            }
        except (DatabaseConnectionError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Ask endpoint failed")
            raise HTTPException(status_code=500, detail=f"Ask failed: {exc}") from exc

    return app


def get_app() -> FastAPI:
    """Lazy factory so that importing the module does not trigger full startup."""
    return create_app()


app = get_app()
