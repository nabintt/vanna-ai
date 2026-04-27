from __future__ import annotations

import json

import pandas as pd
import pytest
from vanna.types import TrainingPlan, TrainingPlanItem

from app.config import Settings
from app.training import TrainingBootstrapError, bootstrap_training, ensure_training_ready


class FakeVanna:
    def __init__(self):
        self.documentation: list[str] = []
        self.ddl: list[str] = []
        self.question_sql: list[dict[str, str]] = []

    def get_training_data(self):
        rows = []
        for index, value in enumerate(self.documentation, start=1):
            rows.append(
                {
                    "id": f"{index}-doc",
                    "question": None,
                    "content": value,
                    "training_data_type": "documentation",
                }
            )
        for index, value in enumerate(self.ddl, start=1):
            rows.append(
                {
                    "id": f"{index}-ddl",
                    "question": None,
                    "content": value,
                    "training_data_type": "ddl",
                }
            )
        for index, value in enumerate(self.question_sql, start=1):
            rows.append(
                {
                    "id": f"{index}-sql",
                    "question": value["question"],
                    "content": value["sql"],
                    "training_data_type": "sql",
                }
            )
        return pd.DataFrame(rows)

    def get_training_plan_generic(self, _: pd.DataFrame) -> TrainingPlan:
        return TrainingPlan(
            [
                TrainingPlanItem(
                    item_type=TrainingPlanItem.ITEM_TYPE_IS,
                    item_group="public",
                    item_name="orders",
                    item_value="The orders table stores one row per purchase.",
                )
            ]
        )

    def add_documentation(self, value: str) -> None:
        self.documentation.append(value)

    def add_ddl(self, value: str) -> None:
        self.ddl.append(value)

    def add_question_sql(self, question: str, sql: str) -> None:
        self.question_sql.append({"question": question, "sql": sql})


class FakeDatabase:
    def fetch_information_schema_columns(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "table_catalog": "analytics",
                    "table_schema": "public",
                    "table_name": "orders",
                    "ordinal_position": 1,
                    "column_name": "id",
                    "data_type": "integer",
                    "is_nullable": "NO",
                    "column_default": None,
                    "udt_name": "int4",
                    "character_maximum_length": None,
                    "numeric_precision": 32,
                    "numeric_scale": 0,
                    "column_type": None,
                },
                {
                    "table_catalog": "analytics",
                    "table_schema": "public",
                    "table_name": "orders",
                    "ordinal_position": 2,
                    "column_name": "customer_name",
                    "data_type": "character varying",
                    "is_nullable": "YES",
                    "column_default": None,
                    "udt_name": "varchar",
                    "character_maximum_length": 255,
                    "numeric_precision": None,
                    "numeric_scale": None,
                    "column_type": None,
                },
            ]
        )

    def fetch_table_constraints(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "table_schema": "public",
                    "table_name": "orders",
                    "constraint_name": "orders_pkey",
                    "constraint_type": "PRIMARY KEY",
                    "column_name": "id",
                    "ordinal_position": 1,
                    "foreign_table_schema": None,
                    "foreign_table_name": None,
                    "foreign_column_name": None,
                }
            ]
        )


def build_settings(tmp_path, train_on_start: bool) -> Settings:
    training_dir = tmp_path / "training"
    training_dir.mkdir(parents=True, exist_ok=True)
    glossary_path = training_dir / "business_glossary.md"
    glossary_path.write_text("Orders represent completed purchases.\n", encoding="utf-8")
    example_pairs_path = training_dir / "example_question_sql.json"
    example_pairs_path.write_text(json.dumps([]), encoding="utf-8")

    return Settings(
        _env_file=None,
        db_name="analytics",
        db_user="analyst",
        db_password="secret",
        train_on_start=train_on_start,
        allow_bootstrap_sample_data=True,
        chroma_path=tmp_path / "chroma",
        training_data_dir=training_dir,
        business_glossary_path=glossary_path,
        example_pairs_path=example_pairs_path,
        bootstrap_state_path=training_dir / "bootstrap_state.json",
    )


def test_bootstrap_training_populates_local_training_state(tmp_path):
    settings = build_settings(tmp_path, train_on_start=True)
    vn = FakeVanna()
    db = FakeDatabase()

    summary = bootstrap_training(vn=vn, db=db, settings=settings)

    assert summary["training"]["total_entries"] > 0
    assert settings.bootstrap_state_path.exists()
    assert len(vn.ddl) >= 1
    assert len(vn.documentation) >= 1
    assert len(vn.question_sql) >= 1


def test_bootstrap_training_is_idempotent_for_same_inputs(tmp_path):
    settings = build_settings(tmp_path, train_on_start=True)
    vn = FakeVanna()
    db = FakeDatabase()

    bootstrap_training(vn=vn, db=db, settings=settings)
    first_counts = (len(vn.documentation), len(vn.ddl), len(vn.question_sql))
    bootstrap_training(vn=vn, db=db, settings=settings)
    second_counts = (len(vn.documentation), len(vn.ddl), len(vn.question_sql))

    assert first_counts == second_counts


def test_ensure_training_ready_raises_when_empty_and_train_on_start_is_disabled(tmp_path):
    settings = build_settings(tmp_path, train_on_start=False)
    vn = FakeVanna()
    db = FakeDatabase()

    with pytest.raises(TrainingBootstrapError):
        ensure_training_ready(vn=vn, db=db, settings=settings)
