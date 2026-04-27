from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from vanna.legacy.types import TrainingPlan, TrainingPlanItem

from app.config import Settings
from app.db import DatabaseClient, build_bootstrap_examples, build_ddl_statements, list_tables_from_columns

logger = logging.getLogger(__name__)

TRAINING_COLUMNS = ["id", "question", "content", "training_data_type"]


class TrainingBootstrapError(RuntimeError):
    """Raised when bootstrap training could not produce usable Vanna state."""


@dataclass
class ExistingTrainingIndex:
    ddl: set[str]
    documentation: set[str]
    question_sql: set[tuple[str, str]]

    @classmethod
    def from_dataframe(cls, training_df: pd.DataFrame) -> "ExistingTrainingIndex":
        ddl: set[str] = set()
        documentation: set[str] = set()
        question_sql: set[tuple[str, str]] = set()

        if training_df.empty:
            return cls(ddl=ddl, documentation=documentation, question_sql=question_sql)

        for _, row in training_df.fillna("").iterrows():
            training_type = str(row["training_data_type"]).strip().lower()
            content = normalize_training_text(str(row["content"]))
            question = normalize_training_text(str(row.get("question", "")))
            if training_type == "ddl":
                ddl.add(content)
            elif training_type == "documentation":
                documentation.add(content)
            elif training_type == "sql":
                question_sql.add((question, content))

        return cls(ddl=ddl, documentation=documentation, question_sql=question_sql)


def normalize_training_text(value: str) -> str:
    return " ".join(value.strip().split())


def get_training_dataframe(vn: Any) -> pd.DataFrame:
    training_df = vn.get_training_data()
    if training_df is None or training_df.empty:
        return pd.DataFrame(columns=TRAINING_COLUMNS)

    for column in TRAINING_COLUMNS:
        if column not in training_df.columns:
            training_df[column] = None
    return training_df[TRAINING_COLUMNS].copy()


def summarize_training(vn: Any) -> dict[str, Any]:
    training_df = get_training_dataframe(vn)
    by_type = {
        training_type: int(count)
        for training_type, count in training_df["training_data_type"].value_counts().sort_index().items()
    }
    return {
        "total_entries": int(len(training_df)),
        "by_type": by_type,
    }


def training_data_exists(vn: Any) -> bool:
    return summarize_training(vn)["total_entries"] > 0


def add_question_sql_if_missing(vn: Any, question: str, sql: str) -> bool:
    training_df = get_training_dataframe(vn)
    index = ExistingTrainingIndex.from_dataframe(training_df)
    fingerprint = (normalize_training_text(question), normalize_training_text(sql))
    if fingerprint in index.question_sql:
        return False
    vn.add_question_sql(question=question, sql=sql)
    return True


def _load_glossary_documents(path: Path) -> list[str]:
    if not path.exists():
        return []

    raw_text = path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return []

    parts = [part.strip() for part in raw_text.split("\n---\n") if part.strip()]
    return parts or [raw_text]


def _load_example_pairs(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TrainingBootstrapError(
            f"Expected a JSON array in {path}, but found {type(payload).__name__}."
        )

    pairs: list[dict[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise TrainingBootstrapError(
                f"Invalid example question/SQL entry in {path}: {item!r}"
            )
        question = str(item.get("question", "")).strip()
        sql = str(item.get("sql", "")).strip()
        if question and sql:
            pairs.append({"question": question, "sql": sql})
    return pairs


def _persist_example_pairs(path: Path, pairs: Iterable[dict[str, str]]) -> None:
    path.write_text(json.dumps(list(pairs), indent=2) + "\n", encoding="utf-8")


def _apply_training_plan(vn: Any, plan: TrainingPlan, existing: ExistingTrainingIndex) -> tuple[int, int, int]:
    added_ddl = 0
    added_documentation = 0
    added_pairs = 0

    for item in plan._plan:
        if item.item_type == TrainingPlanItem.ITEM_TYPE_DDL:
            fingerprint = normalize_training_text(item.item_value)
            if fingerprint not in existing.ddl:
                vn.add_ddl(item.item_value)
                existing.ddl.add(fingerprint)
                added_ddl += 1
        elif item.item_type == TrainingPlanItem.ITEM_TYPE_IS:
            fingerprint = normalize_training_text(item.item_value)
            if fingerprint not in existing.documentation:
                vn.add_documentation(item.item_value)
                existing.documentation.add(fingerprint)
                added_documentation += 1
        elif item.item_type == TrainingPlanItem.ITEM_TYPE_SQL:
            fingerprint = (
                normalize_training_text(item.item_name),
                normalize_training_text(item.item_value),
            )
            if fingerprint not in existing.question_sql:
                vn.add_question_sql(question=item.item_name, sql=item.item_value)
                existing.question_sql.add(fingerprint)
                added_pairs += 1

    return added_ddl, added_documentation, added_pairs


def bootstrap_training(
    vn: Any,
    db: DatabaseClient,
    settings: Settings,
    extra_documentation: list[str] | None = None,
    extra_question_sql_pairs: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    logger.info("Starting bootstrap training for %s", settings.database_target)
    settings.ensure_directories()

    columns_df = db.fetch_information_schema_columns()
    if columns_df.empty:
        raise TrainingBootstrapError(
            "No tables were found in the configured database. Bootstrap training requires at least one visible table."
        )

    constraints_df = db.fetch_table_constraints()
    plan = vn.get_training_plan_generic(columns_df)
    plan_items = len(plan._plan)

    before_df = get_training_dataframe(vn)
    existing = ExistingTrainingIndex.from_dataframe(before_df)
    before_total = len(before_df)

    plan_added_ddl, plan_added_docs, plan_added_pairs = _apply_training_plan(vn, plan, existing)

    ddl_statements = build_ddl_statements(settings.normalized_db_type, columns_df, constraints_df)
    ddl_added = 0
    for ddl in ddl_statements:
        fingerprint = normalize_training_text(ddl)
        if fingerprint in existing.ddl:
            continue
        vn.add_ddl(ddl)
        existing.ddl.add(fingerprint)
        ddl_added += 1

    glossary_documents = _load_glossary_documents(settings.business_glossary_path)
    if extra_documentation:
        glossary_documents.extend(extra_documentation)

    documentation_added = 0
    for document in glossary_documents:
        fingerprint = normalize_training_text(document)
        if fingerprint in existing.documentation:
            continue
        vn.add_documentation(document)
        existing.documentation.add(fingerprint)
        documentation_added += 1

    example_pairs = _load_example_pairs(settings.example_pairs_path)
    if not example_pairs and settings.allow_bootstrap_sample_data:
        example_pairs = build_bootstrap_examples(settings.normalized_db_type, columns_df)
        _persist_example_pairs(settings.example_pairs_path, example_pairs)

    if extra_question_sql_pairs:
        example_pairs.extend(extra_question_sql_pairs)

    example_pairs_added = 0
    for pair in example_pairs:
        question = str(pair["question"]).strip()
        sql = str(pair["sql"]).strip()
        fingerprint = (normalize_training_text(question), normalize_training_text(sql))
        if fingerprint in existing.question_sql:
            continue
        vn.add_question_sql(question=question, sql=sql)
        existing.question_sql.add(fingerprint)
        example_pairs_added += 1

    after_summary = summarize_training(vn)
    if after_summary["total_entries"] <= 0:
        raise TrainingBootstrapError(
            "Bootstrap completed but no training data was saved. Check Chroma persistence and Ollama embedding configuration."
        )

    added_total = after_summary["total_entries"] - before_total
    skipped_duplicates = max(
        plan_items
        + len(ddl_statements)
        + len(glossary_documents)
        + len(example_pairs)
        - added_total,
        0,
    )

    summary = {
        "status": "ready",
        "database_target": settings.database_target,
        "tables_discovered": len(list_tables_from_columns(columns_df)),
        "training_plan_items": plan_items,
        "added": {
            "plan_ddl": plan_added_ddl,
            "plan_documentation": plan_added_docs,
            "plan_question_sql": plan_added_pairs,
            "ddl": ddl_added,
            "documentation": documentation_added,
            "question_sql": example_pairs_added,
        },
        "skipped_duplicates": skipped_duplicates,
        "training": after_summary,
        "persisted_paths": {
            "chroma_path": str(settings.chroma_path),
            "bootstrap_state_path": str(settings.bootstrap_state_path),
            "glossary_path": str(settings.business_glossary_path),
            "example_pairs_path": str(settings.example_pairs_path),
        },
        "generated_at": datetime.now(UTC).isoformat(),
    }

    settings.bootstrap_state_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Bootstrap training finished with %s training entries",
        after_summary["total_entries"],
    )
    return summary


def ensure_training_ready(vn: Any, db: DatabaseClient, settings: Settings) -> dict[str, Any]:
    current_summary = summarize_training(vn)
    if current_summary["total_entries"] > 0:
        return {
            "status": "ready",
            "source": "existing",
            "training": current_summary,
        }

    if settings.train_on_start:
        logger.info("No training data found. TRAIN_ON_START=true, running bootstrap.")
        return bootstrap_training(vn=vn, db=db, settings=settings)

    raise TrainingBootstrapError(
        "No Vanna training data was found in the local Chroma store. "
        "Run `python scripts/bootstrap_training.py` first or set TRAIN_ON_START=true."
    )
