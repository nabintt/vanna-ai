from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from app.config import Settings


class DatabaseConnectionError(RuntimeError):
    """Raised when the configured database cannot be reached."""


@dataclass(frozen=True)
class TableRef:
    schema: str
    table: str
    columns: tuple[str, ...]

    @property
    def display_name(self) -> str:
        return f"{self.schema}.{self.table}"


@dataclass
class SqlExecutionResult:
    sql: str
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int
    returns_rows: bool
    truncated: bool
    duration_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def quote_identifier(db_type: str, identifier: str) -> str:
    if db_type == "mysql":
        return f"`{identifier.replace('`', '``')}`"
    return f"\"{identifier.replace('\"', '\"\"')}\""


def qualify_table_name(db_type: str, schema: str, table: str) -> str:
    return f"{quote_identifier(db_type, schema)}.{quote_identifier(db_type, table)}"


def list_tables_from_columns(columns_df: pd.DataFrame) -> list[TableRef]:
    refs: list[TableRef] = []
    grouped = columns_df.sort_values(
        by=["table_schema", "table_name", "ordinal_position"],
    ).groupby(["table_schema", "table_name"], sort=False)

    for (schema, table), group in grouped:
        columns = tuple(group["column_name"].astype(str).tolist())
        refs.append(TableRef(schema=schema, table=table, columns=columns))

    return refs


def build_bootstrap_examples(db_type: str, columns_df: pd.DataFrame, table_limit: int = 3) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    for table_ref in list_tables_from_columns(columns_df)[:table_limit]:
        qualified_name = qualify_table_name(db_type, table_ref.schema, table_ref.table)
        examples.append(
            {
                "question": f"Show 10 rows from {table_ref.display_name}.",
                "sql": f"SELECT * FROM {qualified_name} LIMIT 10;",
            }
        )
        examples.append(
            {
                "question": f"How many rows are in {table_ref.display_name}?",
                "sql": f"SELECT COUNT(*) AS row_count FROM {qualified_name};",
            }
        )
        if table_ref.columns:
            selected = table_ref.columns[:3]
            selected_sql = ", ".join(quote_identifier(db_type, column) for column in selected)
            selected_text = ", ".join(selected)
            examples.append(
                {
                    "question": f"List the first 10 values for {selected_text} in {table_ref.display_name}.",
                    "sql": f"SELECT {selected_sql} FROM {qualified_name} LIMIT 10;",
                }
            )
    return examples


def _render_column_type(row: pd.Series) -> str:
    if pd.notna(row.get("column_type")) and str(row["column_type"]).strip():
        return str(row["column_type"])

    data_type = str(row["data_type"])
    char_length = row.get("character_maximum_length")
    precision = row.get("numeric_precision")
    scale = row.get("numeric_scale")
    udt_name = row.get("udt_name")

    if data_type in {"character varying", "varchar"} and pd.notna(char_length):
        return f"varchar({int(char_length)})"
    if data_type in {"character", "char"} and pd.notna(char_length):
        return f"char({int(char_length)})"
    if data_type in {"numeric", "decimal"} and pd.notna(precision):
        if pd.notna(scale):
            return f"{data_type}({int(precision)},{int(scale)})"
        return f"{data_type}({int(precision)})"
    if data_type == "ARRAY" and pd.notna(udt_name):
        return str(udt_name)
    return data_type


def build_ddl_statements(db_type: str, columns_df: pd.DataFrame, constraints_df: pd.DataFrame) -> list[str]:
    ddls: list[str] = []
    grouped_columns = columns_df.sort_values(
        by=["table_schema", "table_name", "ordinal_position"],
    ).groupby(["table_schema", "table_name"], sort=False)

    for (schema, table), group in grouped_columns:
        lines: list[str] = []
        table_constraints = constraints_df[
            (constraints_df["table_schema"] == schema)
            & (constraints_df["table_name"] == table)
        ].sort_values(by=["constraint_name", "ordinal_position"])

        for _, row in group.iterrows():
            column_name = quote_identifier(db_type, str(row["column_name"]))
            column_type = _render_column_type(row)
            nullable = " NOT NULL" if str(row["is_nullable"]).upper() == "NO" else ""
            default = ""
            if pd.notna(row.get("column_default")) and str(row["column_default"]).strip():
                default = f" DEFAULT {row['column_default']}"
            lines.append(f"    {column_name} {column_type}{default}{nullable}")

        primary_key_rows = table_constraints[table_constraints["constraint_type"] == "PRIMARY KEY"]
        if not primary_key_rows.empty:
            pk_columns = ", ".join(
                quote_identifier(db_type, str(column))
                for column in primary_key_rows["column_name"].tolist()
            )
            lines.append(f"    PRIMARY KEY ({pk_columns})")

        foreign_key_rows = table_constraints[table_constraints["constraint_type"] == "FOREIGN KEY"]
        for constraint_name, fk_group in foreign_key_rows.groupby("constraint_name", sort=False):
            fk_columns = ", ".join(
                quote_identifier(db_type, str(column))
                for column in fk_group["column_name"].tolist()
            )
            foreign_schema = str(fk_group.iloc[0]["foreign_table_schema"])
            foreign_table = str(fk_group.iloc[0]["foreign_table_name"])
            foreign_columns = ", ".join(
                quote_identifier(db_type, str(column))
                for column in fk_group["foreign_column_name"].tolist()
            )
            lines.append(
                "    CONSTRAINT "
                f"{quote_identifier(db_type, str(constraint_name))} FOREIGN KEY ({fk_columns}) "
                f"REFERENCES {qualify_table_name(db_type, foreign_schema, foreign_table)} ({foreign_columns})"
            )

        ddl = (
            f"CREATE TABLE {qualify_table_name(db_type, schema, table)} (\n"
            + ",\n".join(lines)
            + "\n);"
        )
        ddls.append(ddl)

    return ddls


class DatabaseClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.engine: Engine = create_engine(
            settings.sqlalchemy_url,
            future=True,
            pool_pre_ping=True,
        )

    def close(self) -> None:
        self.engine.dispose()

    def test_connection(self) -> None:
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
        except SQLAlchemyError as exc:
            raise DatabaseConnectionError(
                f"Unable to connect to {self.settings.database_target}: {exc}"
            ) from exc

    def execute_sql(self, sql: str, max_rows: int | None = None) -> SqlExecutionResult:
        sql = sql.strip()
        if not sql:
            raise ValueError("SQL cannot be empty.")

        fetch_limit = max_rows or self.settings.max_result_rows
        started = perf_counter()
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text(sql))
                duration_ms = round((perf_counter() - started) * 1000, 2)
                if result.returns_rows:
                    rows = [dict(row._mapping) for row in result.fetchmany(fetch_limit + 1)]
                    columns = list(result.keys())
                    truncated = len(rows) > fetch_limit
                    rows = rows[:fetch_limit]
                    return SqlExecutionResult(
                        sql=sql,
                        columns=columns,
                        rows=rows,
                        row_count=len(rows),
                        returns_rows=True,
                        truncated=truncated,
                        duration_ms=duration_ms,
                    )

                row_count = result.rowcount if result.rowcount is not None and result.rowcount >= 0 else 0
                return SqlExecutionResult(
                    sql=sql,
                    columns=[],
                    rows=[],
                    row_count=row_count,
                    returns_rows=False,
                    truncated=False,
                    duration_ms=duration_ms,
                )
        except SQLAlchemyError as exc:
            raise DatabaseConnectionError(f"SQL execution failed: {exc}") from exc

    def fetch_information_schema_columns(self) -> pd.DataFrame:
        query = self._postgres_information_schema_query()
        if self.settings.normalized_db_type == "mysql":
            query = self._mysql_information_schema_query()
        return pd.read_sql_query(text(query), self.engine)

    def fetch_table_constraints(self) -> pd.DataFrame:
        query = self._postgres_constraint_query()
        if self.settings.normalized_db_type == "mysql":
            query = self._mysql_constraint_query()
        return pd.read_sql_query(text(query), self.engine)

    @staticmethod
    def _postgres_information_schema_query() -> str:
        return """
        SELECT
            table_catalog,
            table_schema,
            table_name,
            ordinal_position,
            column_name,
            data_type,
            is_nullable,
            column_default,
            udt_name,
            character_maximum_length,
            numeric_precision,
            numeric_scale,
            NULL::text AS column_type
        FROM information_schema.columns
        WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
        ORDER BY table_schema, table_name, ordinal_position
        """

    @staticmethod
    def _mysql_information_schema_query() -> str:
        return """
        SELECT
            table_schema AS table_catalog,
            table_schema,
            table_name,
            ordinal_position,
            column_name,
            data_type,
            is_nullable,
            column_default,
            data_type AS udt_name,
            character_maximum_length,
            numeric_precision,
            numeric_scale,
            column_type
        FROM information_schema.columns
        WHERE table_schema = DATABASE()
        ORDER BY table_schema, table_name, ordinal_position
        """

    @staticmethod
    def _postgres_constraint_query() -> str:
        return """
        SELECT
            tc.table_schema,
            tc.table_name,
            tc.constraint_name,
            tc.constraint_type,
            kcu.column_name,
            kcu.ordinal_position,
            ccu.table_schema AS foreign_table_schema,
            ccu.table_name AS foreign_table_name,
            ccu.column_name AS foreign_column_name
        FROM information_schema.table_constraints tc
        LEFT JOIN information_schema.key_column_usage kcu
            ON tc.constraint_name = kcu.constraint_name
           AND tc.table_schema = kcu.table_schema
           AND tc.table_name = kcu.table_name
        LEFT JOIN information_schema.constraint_column_usage ccu
            ON tc.constraint_name = ccu.constraint_name
           AND tc.table_schema = ccu.table_schema
        WHERE tc.table_schema NOT IN ('information_schema', 'pg_catalog')
          AND tc.constraint_type IN ('PRIMARY KEY', 'FOREIGN KEY')
        ORDER BY tc.table_schema, tc.table_name, tc.constraint_name, kcu.ordinal_position
        """

    @staticmethod
    def _mysql_constraint_query() -> str:
        return """
        SELECT
            tc.table_schema,
            tc.table_name,
            tc.constraint_name,
            tc.constraint_type,
            kcu.column_name,
            kcu.ordinal_position,
            kcu.referenced_table_schema AS foreign_table_schema,
            kcu.referenced_table_name AS foreign_table_name,
            kcu.referenced_column_name AS foreign_column_name
        FROM information_schema.table_constraints tc
        LEFT JOIN information_schema.key_column_usage kcu
            ON tc.constraint_name = kcu.constraint_name
           AND tc.table_schema = kcu.table_schema
           AND tc.table_name = kcu.table_name
        WHERE tc.table_schema = DATABASE()
          AND tc.constraint_type IN ('PRIMARY KEY', 'FOREIGN KEY')
        ORDER BY tc.table_schema, tc.table_name, tc.constraint_name, kcu.ordinal_position
        """
