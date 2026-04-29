from __future__ import annotations

import asyncio
import json
import logging
import re
import traceback
from contextlib import suppress
from dataclasses import dataclass
from json import JSONDecodeError, JSONDecoder
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from vanna.capabilities.agent_memory import (
    AgentMemory,
    TextMemory,
    TextMemorySearchResult,
    ToolMemory,
    ToolMemorySearchResult,
)
from vanna.capabilities.sql_runner import RunSqlToolArgs
from vanna.components import (
    ComponentType,
    DataFrameComponent,
    NotificationComponent,
    SimpleTextComponent,
    UiComponent,
)
from vanna import Agent, AgentConfig
from vanna.core.enhancer import DefaultLlmContextEnhancer, LlmContextEnhancer
from vanna.core.filter import ConversationFilter
from vanna.core.llm import LlmMessage, LlmRequest, LlmResponse, LlmStreamChunk
from vanna.core.registry import ToolRegistry
from vanna.core.storage import Message
from vanna.core.system_prompt import SystemPromptBuilder
from vanna.core.tool import Tool, ToolCall, ToolContext, ToolResult, ToolSchema
from vanna.core.user import RequestContext, User
from vanna.core.user.resolver import UserResolver
from vanna.integrations.ollama import OllamaLlmService
from vanna.servers.base import ChatHandler, ChatRequest, ChatResponse
from starlette.concurrency import run_in_threadpool

from app.config import Settings
from app.db import DatabaseClient, DatabaseConnectionError

logger = logging.getLogger(__name__)

JSON_CODE_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
SQL_CODE_BLOCK_PATTERN = re.compile(r"```sql\s*(.*?)```", re.IGNORECASE | re.DOTALL)
ROWS_JSON_BLOCK_PATTERN = re.compile(r"Rows:\s*```json\s*(.*?)```", re.IGNORECASE | re.DOTALL)
ROW_COUNT_PATTERN = re.compile(r"Returned\s+(\d+)\s+row\(s\)", re.IGNORECASE)
NO_ROWS_PATTERN = re.compile(r"no rows were returned|no rows returned", re.IGNORECASE)
SEARCH_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")
DDL_TABLE_PATTERN = re.compile(
    r"CREATE\s+TABLE\s+(?:\"|`)?(?P<schema>[A-Za-z0-9_]+)(?:\"|`)?\.(?:\"|`)?(?P<table>[A-Za-z0-9_]+)(?:\"|`)?",
    re.IGNORECASE,
)
DDL_COLUMN_PATTERN = re.compile(
    r"^\s*(?:\"|`)?(?P<name>[A-Za-z0-9_]+)(?:\"|`)?\s+[A-Za-z]",
    re.MULTILINE,
)
READ_ONLY_SQL_PATTERN = re.compile(r"^\s*(select|with|show|describe|desc|explain)\b", re.IGNORECASE)
MUTATING_SQL_PATTERN = re.compile(
    r"\b(insert|update|delete|alter|drop|truncate|create|grant|revoke|merge|replace|upsert)\b",
    re.IGNORECASE,
)
DDL_SKIP_TOKENS = {"create", "table", "primary", "constraint", "foreign", "unique", "check"}
REQUEST_SCHEMA_CONTEXT_MARKER = "[Attached schema context for this request]"
REQUEST_SQL_FEEDBACK_MARKER = "[Recent SQL execution feedback for this conversation]"
FOLLOWUP_RUN_SQL_PATTERN = re.compile(
    r"\b(run|execute|use|try)\b.*\b(query|sql|statement|above|previous|that|this|it)\b",
    re.IGNORECASE,
)
CONTEXT_DEPENDENT_FOLLOWUP_PATTERN = re.compile(
    r"\b("
    r"above|previous|earlier|same|that|those|them|it|this query|this sql|that query|that sql|"
    r"add|include|exclude|remove|filter|sort|order|group|limit|change|modify|instead|continue|again|reuse"
    r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SchemaCatalogEntry:
    table_name: str
    columns: tuple[str, ...]
    ddl: str
    search_terms: frozenset[str]


@dataclass(frozen=True)
class RecentSqlAttempt:
    sql: str
    tables: tuple[str, ...]
    outcome: str
    detail: str


@dataclass(frozen=True)
class SuccessfulSqlResult:
    sql: str
    row_count: int
    columns: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]


class NoOpAgentMemory(AgentMemory):
    async def save_tool_usage(
        self,
        question: str,
        tool_name: str,
        args: dict[str, Any],
        context: ToolContext,
        success: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        return None

    async def save_text_memory(self, content: str, context: ToolContext) -> TextMemory:
        return TextMemory(memory_id=None, content=content, timestamp=None)

    async def search_similar_usage(
        self,
        question: str,
        context: ToolContext,
        *,
        limit: int = 10,
        similarity_threshold: float = 0.7,
        tool_name_filter: str | None = None,
    ) -> list[ToolMemorySearchResult]:
        return []

    async def search_text_memories(
        self,
        query: str,
        context: ToolContext,
        *,
        limit: int = 10,
        similarity_threshold: float = 0.7,
    ) -> list[TextMemorySearchResult]:
        return []

    async def get_recent_memories(
        self,
        context: ToolContext,
        limit: int = 10,
    ) -> list[ToolMemory]:
        return []

    async def get_recent_text_memories(
        self,
        context: ToolContext,
        limit: int = 10,
    ) -> list[TextMemory]:
        return []

    async def delete_by_id(self, context: ToolContext, memory_id: str) -> bool:
        return False

    async def delete_text_memory(self, context: ToolContext, memory_id: str) -> bool:
        return False

    async def clear_memories(
        self,
        context: ToolContext,
        tool_name: str | None = None,
        before_date: str | None = None,
    ) -> int:
        return 0


class SqlChatSystemPromptBuilder(SystemPromptBuilder):
    async def build_system_prompt(
        self,
        user: User,
        tools: list[ToolSchema],
    ) -> str:
        del user
        tool_names = ", ".join(tool.name for tool in tools) or "run_sql"
        return "\n".join(
            [
                "You are a careful SQL analyst assistant.",
                "Use only the real schema context and the available SQL tool.",
                "Treat each new standalone user question as a fresh request unless the user explicitly refers to a previous query, result, or instruction.",
                "Call `run_sql` only when you still need information to answer the user.",
                "When you make a tool call, the tool name must be exactly `run_sql` and the arguments must be `{\"sql\": \"...\"}`.",
                "If the user asks to run the previous or above query, reuse the latest SQL you already produced instead of inventing a new tool name.",
                "If the first successful SQL result already answers the question, stop calling tools and answer directly.",
                "Do not repeat an equivalent successful query.",
                "Do not return JSON unless you are making a tool call.",
                "The user can already see the query result table, so your final answer should summarize only the answer, key figures, and caveats.",
                f"Available tools: {tool_names}",
            ]
        )


class ReadOnlySqlTool(Tool[RunSqlToolArgs]):
    def __init__(self, db: DatabaseClient, max_rows: int):
        self.db = db
        self.max_rows = max_rows
        self._successful_sql_by_conversation: dict[str, str] = {}

    @property
    def name(self) -> str:
        return "run_sql"

    @property
    def description(self) -> str:
        return "Execute a single read-only SQL query against the configured database."

    def get_args_schema(self) -> type[RunSqlToolArgs]:
        return RunSqlToolArgs

    async def execute(self, context: ToolContext, args: RunSqlToolArgs) -> ToolResult:
        sql = args.sql.strip()
        if not is_read_only_sql(sql):
            return build_sql_tool_error(
                "Only single-statement read-only SQL is allowed here. Use SELECT, WITH, SHOW, DESCRIBE, or EXPLAIN."
            )

        sql_fingerprint = normalize_sql_for_comparison(sql)
        previous_sql = self._successful_sql_by_conversation.get(context.conversation_id)
        if previous_sql == sql_fingerprint:
            return ToolResult(
                success=True,
                result_for_llm=(
                    "This exact SQL already executed successfully in this conversation. "
                    "Do not run it again. Use the existing result and answer the user now."
                ),
                ui_component=None,
                metadata={"duplicate_sql": True, "sql": sql},
            )

        try:
            result = await run_in_threadpool(self.db.execute_sql, sql, self.max_rows)
        except (DatabaseConnectionError, ValueError) as exc:
            return build_sql_tool_error(str(exc))

        self._successful_sql_by_conversation[context.conversation_id] = sql_fingerprint

        description = (
            f"Returned {result.row_count} row(s)"
            + (" (truncated to the configured limit)." if result.truncated else ".")
            + f" Duration: {result.duration_ms} ms."
        )

        ui_component = UiComponent(
            rich_component=DataFrameComponent(
                rows=result.rows,
                columns=result.columns,
                title="Query Results",
                description=description,
                row_count=result.row_count,
                column_count=len(result.columns),
                max_rows_displayed=min(result.row_count or self.max_rows, 100),
            ),
            simple_component=SimpleTextComponent(text=description),
        )

        llm_lines = [
            "SQL executed successfully. Use this result to answer the user's question.",
            "Do not run another SQL query if this already answers the question.",
            description,
        ]
        if result.columns:
            llm_lines.append(f"Columns: {', '.join(result.columns)}")
        if result.rows:
            llm_lines.extend(
                [
                    "Rows:",
                    "```json",
                    json.dumps(result.rows[:10], default=str, indent=2),
                    "```",
                ]
            )
        else:
            llm_lines.append(
                "No rows were returned for this query. That only applies to this exact query."
            )

        return ToolResult(
            success=True,
            result_for_llm="\n".join(llm_lines),
            ui_component=ui_component,
            metadata={
                "sql": sql,
                "row_count": result.row_count,
                "columns": result.columns,
                "truncated": result.truncated,
                "duration_ms": result.duration_ms,
            },
        )


def is_read_only_sql(sql: str) -> bool:
    statement = sql.strip()
    if not statement:
        return False

    statement = statement.rstrip(";").strip()
    if ";" in statement:
        return False

    return bool(READ_ONLY_SQL_PATTERN.match(statement)) and not bool(
        MUTATING_SQL_PATTERN.search(statement)
    )


def build_sql_tool_error(message: str) -> ToolResult:
    return ToolResult(
        success=False,
        result_for_llm=message,
        ui_component=UiComponent(
            rich_component=NotificationComponent(
                type=ComponentType.NOTIFICATION,
                level="error",
                message=message,
            ),
            simple_component=SimpleTextComponent(text=message),
        ),
        error=message,
        metadata={"error_type": "sql_error"},
    )


class StandaloneQuestionConversationFilter(ConversationFilter):
    """Treat standalone user questions as fresh requests instead of sticky follow-ups."""

    def __init__(self, followup_window: int = 8):
        self.followup_window = followup_window

    async def filter_messages(self, messages: list[Message]) -> list[Message]:
        if len(messages) <= 1:
            return messages

        last_user_index = next(
            (index for index in range(len(messages) - 1, -1, -1) if messages[index].role == "user"),
            None,
        )
        if last_user_index is None:
            return messages

        last_user_message = (messages[last_user_index].content or "").strip()
        if is_context_dependent_followup_request(last_user_message):
            start_index = max(0, last_user_index - self.followup_window)
            return messages[start_index:]

        return [messages[last_user_index]]


class OllamaToolCallFallbackService(OllamaLlmService):
    """Parse tool-call JSON from model text when native tool calling is unavailable."""

    async def send_request(self, request: LlmRequest) -> LlmResponse:
        response = await super().send_request(request)
        return coerce_text_tool_calls(response, request.tools, request.messages)

    async def stream_request(
        self,
        request: LlmRequest,
    ) -> AsyncGenerator[LlmStreamChunk, None]:
        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        finish_reason: str | None = None
        metadata: dict[str, Any] = {}

        async for chunk in super().stream_request(request):
            if chunk.content:
                content_parts.append(chunk.content)
            if chunk.tool_calls:
                tool_calls.extend(chunk.tool_calls)
            if chunk.finish_reason:
                finish_reason = chunk.finish_reason
            if chunk.metadata:
                metadata.update(chunk.metadata)

        response = LlmResponse(
            content="".join(content_parts) or None,
            tool_calls=tool_calls or None,
            finish_reason=finish_reason,
            metadata=metadata,
        )
        normalized = coerce_text_tool_calls(response, request.tools, request.messages)
        yield LlmStreamChunk(
            content=normalized.content,
            tool_calls=normalized.tool_calls,
            finish_reason=normalized.finish_reason,
            metadata=normalized.metadata,
        )


def coerce_text_tool_calls(
    response: LlmResponse,
    tools: list[ToolSchema] | None,
    messages: list[LlmMessage] | None = None,
) -> LlmResponse:
    normalized = response
    if not normalized.tool_calls and normalized.content and tools:
        parsed_tool_calls = extract_text_tool_calls(normalized.content, tools)
        inferred_from_context = False
        if not parsed_tool_calls:
            parsed_tool_calls = infer_run_sql_tool_call_from_context(
                normalized.content,
                tools,
                messages or [],
            )
            inferred_from_context = bool(parsed_tool_calls)
        if parsed_tool_calls:
            metadata = dict(normalized.metadata)
            metadata["tool_call_fallback"] = "text_json"
            metadata["tool_call_fallback_count"] = len(parsed_tool_calls)
            if inferred_from_context:
                metadata["tool_call_fallback"] = "context_sql_reuse"

            normalized = LlmResponse(
                content=None,
                tool_calls=parsed_tool_calls,
                finish_reason=normalized.finish_reason or "tool_calls",
                usage=normalized.usage,
                metadata=metadata,
            )

    return suppress_duplicate_run_sql_tool_calls(normalized, messages or [])


def extract_text_tool_calls(
    content: str,
    tools: list[ToolSchema] | None,
) -> list[ToolCall]:
    allowed_tool_names = {tool.name for tool in tools or []}
    if not allowed_tool_names:
        return []

    candidates = [content]
    candidates.extend(
        match.group(1).strip()
        for match in JSON_CODE_BLOCK_PATTERN.finditer(content)
        if match.group(1).strip()
    )

    tool_calls: list[ToolCall] = []
    seen_fingerprints: set[str] = set()
    for candidate in candidates:
        for payload in decode_json_objects(candidate):
            name = payload.get("name")
            arguments = payload.get("arguments", {})
            if name not in allowed_tool_names:
                continue

            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except JSONDecodeError:
                    arguments = {"_raw": arguments}
            if not isinstance(arguments, dict):
                continue

            fingerprint = json.dumps(
                {"name": name, "arguments": arguments},
                sort_keys=True,
                default=str,
            )
            if fingerprint in seen_fingerprints:
                continue
            seen_fingerprints.add(fingerprint)

            tool_calls.append(
                ToolCall(
                    id=f"text-tool-call-{len(tool_calls) + 1}",
                    name=name,
                    arguments=arguments,
                )
            )

    return tool_calls


def infer_run_sql_tool_call_from_context(
    response_content: str,
    tools: list[ToolSchema] | None,
    messages: list[LlmMessage],
) -> list[ToolCall]:
    allowed_tool_names = {tool.name for tool in tools or []}
    if "run_sql" not in allowed_tool_names:
        return []

    last_user_message = next(
        (message.content or "" for message in reversed(messages) if message.role == "user"),
        "",
    ).strip()
    if not looks_like_followup_run_sql_request(last_user_message, response_content):
        return []

    sql = extract_latest_assistant_sql(messages)
    if not sql:
        return []

    return [
        ToolCall(
            id="context-tool-call-1",
            name="run_sql",
            arguments={"sql": sql},
        )
    ]


def suppress_duplicate_run_sql_tool_calls(
    response: LlmResponse,
    messages: list[LlmMessage],
) -> LlmResponse:
    if not response.tool_calls:
        return response

    last_success = extract_last_successful_run_sql_result(messages)
    if last_success is None:
        return response

    filtered_tool_calls: list[ToolCall] = []
    suppressed_duplicate = False
    for tool_call in response.tool_calls:
        if tool_call.name != "run_sql":
            filtered_tool_calls.append(tool_call)
            continue

        sql = str(tool_call.arguments.get("sql", "")).strip()
        if normalize_sql_for_comparison(sql) == normalize_sql_for_comparison(last_success.sql):
            suppressed_duplicate = True
            continue

        filtered_tool_calls.append(tool_call)

    if filtered_tool_calls:
        if not suppressed_duplicate:
            return response
        metadata = dict(response.metadata)
        metadata["suppressed_duplicate_tool_call"] = True
        return LlmResponse(
            content=response.content,
            tool_calls=filtered_tool_calls,
            finish_reason=response.finish_reason,
            usage=response.usage,
            metadata=metadata,
        )

    if not suppressed_duplicate:
        return response

    metadata = dict(response.metadata)
    metadata["suppressed_duplicate_tool_call"] = True
    return LlmResponse(
        content=build_duplicate_sql_answer(last_success),
        tool_calls=None,
        finish_reason="stop",
        usage=response.usage,
        metadata=metadata,
    )


def looks_like_followup_run_sql_request(last_user_message: str, response_content: str) -> bool:
    haystacks = [last_user_message, response_content]
    return any(FOLLOWUP_RUN_SQL_PATTERN.search(haystack or "") for haystack in haystacks)


def is_context_dependent_followup_request(message: str) -> bool:
    normalized = normalize_whitespace(message).lower()
    if not normalized:
        return False
    if looks_like_followup_run_sql_request(normalized, normalized):
        return True
    return bool(CONTEXT_DEPENDENT_FOLLOWUP_PATTERN.search(normalized))


def normalize_sql_for_comparison(sql: str) -> str:
    normalized = normalize_whitespace(sql).rstrip(";")
    return normalized.lower()


def extract_latest_assistant_sql(messages: list[LlmMessage]) -> str | None:
    for message in reversed(messages):
        if message.role != "assistant" or not message.content:
            continue
        for match in reversed(list(SQL_CODE_BLOCK_PATTERN.finditer(message.content))):
            sql = match.group(1).strip()
            if is_read_only_sql(sql):
                return sql
    return None


def extract_last_successful_run_sql_result(messages: list[LlmMessage]) -> SuccessfulSqlResult | None:
    tool_calls_by_id: dict[str, str] = {}
    for message in messages:
        if message.role != "assistant" or not message.tool_calls:
            continue
        for tool_call in message.tool_calls:
            if tool_call.name != "run_sql" or not tool_call.id:
                continue
            sql = str(tool_call.arguments.get("sql", "")).strip()
            if sql:
                tool_calls_by_id[tool_call.id] = sql

    for message in reversed(messages):
        if message.role != "tool" or not message.tool_call_id:
            continue
        sql = tool_calls_by_id.get(message.tool_call_id)
        if not sql:
            continue
        if not is_successful_run_sql_content(message.content or ""):
            continue

        content = message.content or ""
        return SuccessfulSqlResult(
            sql=sql,
            row_count=extract_row_count_from_tool_content(content),
            columns=tuple(extract_columns_from_tool_content(content)),
            rows=tuple(extract_rows_from_tool_content(content)),
        )

    return None


def is_successful_run_sql_content(content: str) -> bool:
    normalized = normalize_whitespace(content).lower()
    return "sql executed successfully." in normalized or "query executed successfully." in normalized


def extract_row_count_from_tool_content(content: str) -> int:
    match = ROW_COUNT_PATTERN.search(content)
    if match:
        return int(match.group(1))
    if NO_ROWS_PATTERN.search(content):
        return 0
    return 0


def extract_columns_from_tool_content(content: str) -> list[str]:
    for line in content.splitlines():
        if not line.lower().startswith("columns:"):
            continue
        raw_columns = line.split(":", 1)[1]
        return [column.strip() for column in raw_columns.split(",") if column.strip()]
    return []


def extract_rows_from_tool_content(content: str) -> list[dict[str, Any]]:
    match = ROWS_JSON_BLOCK_PATTERN.search(content)
    if not match:
        return []

    try:
        payload = json.loads(match.group(1).strip())
    except JSONDecodeError:
        return []

    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def build_duplicate_sql_answer(result: SuccessfulSqlResult) -> str:
    if result.row_count == 0:
        return "I already ran that exact SQL and it returned no rows, so I won't rerun it."

    if result.rows:
        first_row = result.rows[0]
        row_summary = ", ".join(f"`{key}` = {value}" for key, value in first_row.items())
        return (
            "I already ran that exact SQL successfully, so I won't rerun it. "
            f"The result is: {row_summary}."
        )

    if result.columns:
        columns_summary = ", ".join(f"`{column}`" for column in result.columns)
        return (
            "I already ran that exact SQL successfully, so I won't rerun it. "
            f"The result shown above contains {result.row_count} row(s) with columns {columns_summary}."
        )

    return "I already ran that exact SQL successfully, so I won't rerun it."


def decode_json_objects(candidate: str) -> list[dict[str, Any]]:
    decoder = JSONDecoder()
    index = 0
    payloads: list[dict[str, Any]] = []

    while index < len(candidate):
        object_start = candidate.find("{", index)
        if object_start == -1:
            break

        try:
            decoded, next_index = decoder.raw_decode(candidate, object_start)
        except JSONDecodeError:
            index = object_start + 1
            continue

        if isinstance(decoded, dict):
            payloads.append(decoded)
        elif isinstance(decoded, list):
            payloads.extend(item for item in decoded if isinstance(item, dict))
        index = next_index

    return payloads


class SchemaAwareLlmContextEnhancer(LlmContextEnhancer):
    """Inject relevant DDL, docs, and examples into the SQL-generation prompt."""

    def __init__(self, vn: Any, agent_memory: Any):
        self.vn = vn
        self.base_enhancer = DefaultLlmContextEnhancer(agent_memory)
        self.schema_catalog = build_schema_catalog(vn)

    async def enhance_system_prompt(
        self,
        system_prompt: str,
        user_message: str,
        user: User,
    ) -> str:
        prompt = await self.base_enhancer.enhance_system_prompt(system_prompt, user_message, user)
        context_bundle = build_schema_context_bundle(
            vn=self.vn,
            schema_catalog=self.schema_catalog,
            question=user_message,
        )

        prompt_parts = [
            prompt,
            "",
            "## SQL Generation Rules",
            "- Use only tables and columns that appear in the schema context or successful SQL examples below.",
            "- Do not invent columns such as `username`, `name`, or `skill` unless they are explicitly present in the provided schema.",
            "- If a SQL execution error mentions a missing table or column, treat it as a schema mismatch and correct the query before drawing conclusions.",
            "- Never conclude that a table is empty just because a previous SQL statement failed.",
            "- Treat recent SQL errors and empty-result attempts as negative evidence. Do not repeat the same query path unchanged.",
            "- If a successful SQL result already answers the question, stop and respond instead of running another query.",
            "- If the schema context is still insufficient, prefer a read-only metadata query against `information_schema` or relevant catalog views before guessing.",
        ]

        schema_section = format_schema_context(context_bundle["related_ddl"])
        if schema_section:
            prompt_parts.extend(["", "## Relevant Schema", schema_section])

        documentation_section = format_documentation_context(context_bundle["related_docs"])
        if documentation_section:
            prompt_parts.extend(["", "## Relevant Documentation", documentation_section])

        examples_section = format_question_sql_examples(context_bundle["similar_pairs"])
        if examples_section:
            prompt_parts.extend(["", "## Similar Successful SQL Patterns", examples_section])

        return "\n".join(prompt_parts)

    async def enhance_user_messages(
        self,
        messages: list[LlmMessage],
        user: User,
    ) -> list[LlmMessage]:
        messages = await self.base_enhancer.enhance_user_messages(messages, user)
        if not messages:
            return messages

        last_user_index = next(
            (index for index in range(len(messages) - 1, -1, -1) if messages[index].role == "user"),
            None,
        )
        if last_user_index is None:
            return messages

        original_message = messages[last_user_index]
        base_content = original_message.content or ""
        if REQUEST_SCHEMA_CONTEXT_MARKER in base_content:
            return messages

        context_bundle = build_schema_context_bundle(
            vn=self.vn,
            schema_catalog=self.schema_catalog,
            question=base_content,
        )
        request_context = format_request_schema_context(
            context_bundle,
            collect_recent_run_sql_attempts(messages),
        )
        if not request_context:
            return messages

        updated_messages = list(messages)
        updated_messages[last_user_index] = LlmMessage(
            role=original_message.role,
            content=f"{base_content.rstrip()}\n\n{request_context}",
            tool_calls=original_message.tool_calls,
            tool_call_id=original_message.tool_call_id,
        )
        return updated_messages


def build_schema_context_bundle(
    vn: Any,
    schema_catalog: list[SchemaCatalogEntry],
    question: str,
) -> dict[str, Any]:
    related_ddl = dedupe_text_items(safe_list(vn.get_related_ddl, question=question))
    if len(related_ddl) < 3:
        fallback_ddl = [entry.ddl for entry in search_schema_catalog(schema_catalog, question)]
        related_ddl = dedupe_text_items([*related_ddl, *fallback_ddl])

    similar_pairs = filter_question_sql_pairs_for_context(
        safe_list(vn.get_similar_question_sql, question=question),
        related_ddl=related_ddl,
        question=question,
    )

    return {
        "related_ddl": related_ddl,
        "related_docs": dedupe_text_items(safe_list(vn.get_related_documentation, question=question)),
        "similar_pairs": similar_pairs,
    }


def safe_list(method: Any, **kwargs: Any) -> list[Any]:
    try:
        result = method(**kwargs)
    except Exception:
        logger.warning("Failed to load Vanna schema context", exc_info=True)
        return []
    if not isinstance(result, list):
        return []
    return result


def dedupe_text_items(items: list[Any], limit: int = 5) -> list[str]:
    results: list[str] = []
    seen: set[str] = set()

    for item in items:
        if isinstance(item, dict):
            value = str(item.get("ddl") or item.get("documentation") or item.get("content") or "").strip()
        else:
            value = str(item).strip()
        if not value:
            continue

        fingerprint = " ".join(value.split())
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        results.append(value)
        if len(results) >= limit:
            break

    return results


def dedupe_question_sql_pairs(items: list[Any], limit: int = 4) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for item in items:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or "").strip()
        sql = str(item.get("sql") or "").strip()
        if not question or not sql:
            continue

        fingerprint = (" ".join(question.split()), " ".join(sql.split()))
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        results.append({"question": question, "sql": sql})
        if len(results) >= limit:
            break

    return results


def filter_question_sql_pairs_for_context(
    items: list[Any],
    related_ddl: list[str],
    question: str,
    limit: int = 4,
) -> list[dict[str, str]]:
    pairs = dedupe_question_sql_pairs(items, limit=max(limit * 3, limit))
    if not pairs:
        return []

    allowed_tables = {
        extract_table_name_from_ddl(ddl).lower()
        for ddl in related_ddl
        if extract_table_name_from_ddl(ddl) != "unknown_table"
    }
    question_tokens = tokenize_schema_text(question)
    scored_pairs: list[tuple[int, dict[str, str]]] = []

    for pair in pairs:
        sql_tables = {table.lower() for table in extract_table_names_from_sql(pair["sql"])}
        question_overlap = tokenize_schema_text(pair["question"]) & question_tokens
        score = len(question_overlap)

        if allowed_tables:
            table_overlap = sql_tables & allowed_tables
            if not table_overlap:
                continue
            score += 10 * len(table_overlap)

        scored_pairs.append((score, pair))

    scored_pairs.sort(key=lambda item: (-item[0], item[1]["question"], item[1]["sql"]))
    return [pair for _, pair in scored_pairs[:limit]]


def build_schema_catalog(vn: Any) -> list[SchemaCatalogEntry]:
    try:
        training_df = vn.get_training_data()
    except Exception:
        logger.warning("Failed to load training data for schema catalog", exc_info=True)
        return []

    if training_df is None or training_df.empty:
        return []

    catalog: list[SchemaCatalogEntry] = []
    ddl_rows = training_df.fillna("")
    ddl_rows = ddl_rows[ddl_rows["training_data_type"].astype(str).str.lower() == "ddl"]

    for ddl in ddl_rows["content"].astype(str).tolist():
        ddl = ddl.strip()
        if not ddl:
            continue

        table_name = extract_table_name_from_ddl(ddl)
        columns = extract_column_names_from_ddl(ddl)
        search_terms = tokenize_schema_text(table_name)
        for column in columns:
            search_terms.update(tokenize_schema_text(column))

        catalog.append(
            SchemaCatalogEntry(
                table_name=table_name,
                columns=tuple(columns),
                ddl=ddl,
                search_terms=frozenset(search_terms),
            )
        )

    return catalog


def extract_table_name_from_ddl(ddl: str) -> str:
    match = DDL_TABLE_PATTERN.search(ddl)
    if not match:
        return "unknown_table"
    return f"{match.group('schema')}.{match.group('table')}"


def extract_column_names_from_ddl(ddl: str) -> list[str]:
    column_names: list[str] = []
    for match in DDL_COLUMN_PATTERN.finditer(ddl):
        name = match.group("name")
        if name.lower() in DDL_SKIP_TOKENS:
            continue
        column_names.append(name)
    return column_names


def tokenize_schema_text(text: str) -> set[str]:
    tokens: set[str] = set()
    for raw_token in SEARCH_TOKEN_PATTERN.findall(text.lower()):
        if len(raw_token) < 2:
            continue
        tokens.add(raw_token)
        if "_" in raw_token:
            tokens.update(part for part in raw_token.split("_") if len(part) >= 2)
    return tokens


def search_schema_catalog(
    catalog: list[SchemaCatalogEntry],
    question: str,
    limit: int = 4,
) -> list[SchemaCatalogEntry]:
    if not catalog:
        return []

    question_lower = question.lower()
    question_tokens = tokenize_schema_text(question)
    scored_entries: list[tuple[int, SchemaCatalogEntry]] = []

    for entry in catalog:
        score = 0
        if entry.table_name != "unknown_table" and entry.table_name.lower() in question_lower:
            score += 10

        overlap = question_tokens & set(entry.search_terms)
        if overlap:
            score += len(overlap)

        if any(column.lower() in question_lower for column in entry.columns):
            score += 2

        if score > 0:
            scored_entries.append((score, entry))

    scored_entries.sort(key=lambda item: (-item[0], item[1].table_name))
    return [entry for _, entry in scored_entries[:limit]]


def truncate_text(value: str, limit: int) -> str:
    value = value.strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def normalize_whitespace(value: str) -> str:
    return " ".join(value.strip().split())


def format_schema_context(related_ddl: list[str]) -> str:
    if not related_ddl:
        return ""

    lines: list[str] = []
    for ddl in related_ddl[:4]:
        table_name = extract_table_name_from_ddl(ddl)
        columns = extract_column_names_from_ddl(ddl)
        if columns:
            column_preview = ", ".join(columns[:12])
            if len(columns) > 12:
                column_preview += ", ..."
            lines.append(f"- `{table_name}` columns: {column_preview}")
        lines.append(f"```sql\n{truncate_text(ddl, 1600)}\n```")
    return "\n".join(lines)


def format_documentation_context(related_docs: list[str]) -> str:
    if not related_docs:
        return ""
    return "\n".join(f"- {truncate_text(doc, 500)}" for doc in related_docs[:4])


def format_question_sql_examples(pairs: list[dict[str, str]]) -> str:
    if not pairs:
        return ""

    lines: list[str] = []
    for pair in pairs[:3]:
        lines.append(f"Question: {pair['question']}")
        lines.append(f"```sql\n{truncate_text(pair['sql'], 800)}\n```")
    return "\n".join(lines)


def format_request_schema_context(
    context_bundle: dict[str, Any],
    recent_sql_attempts: list[RecentSqlAttempt] | None = None,
) -> str:
    related_ddl = context_bundle.get("related_ddl", [])
    related_docs = context_bundle.get("related_docs", [])
    similar_pairs = context_bundle.get("similar_pairs", [])
    recent_sql_attempts = recent_sql_attempts or []

    parts = [REQUEST_SCHEMA_CONTEXT_MARKER]

    schema_section = format_schema_context(related_ddl)
    if schema_section:
        parts.extend(
            [
                "Use the following real database schema while answering this request.",
                "## Table Schema",
                schema_section,
            ]
        )

    documentation_section = format_documentation_context(related_docs)
    if documentation_section:
        parts.extend(["## Business Notes", documentation_section])

    examples_section = format_question_sql_examples(similar_pairs)
    if examples_section:
        parts.extend(["## Similar SQL Examples", examples_section])

    feedback_section = format_recent_sql_feedback(recent_sql_attempts)
    if feedback_section:
        parts.extend(["## Recent SQL Feedback", feedback_section])

    if len(parts) == 1:
        return ""

    return "\n".join(parts)


def collect_recent_run_sql_attempts(messages: list[LlmMessage], limit: int = 4) -> list[RecentSqlAttempt]:
    tool_calls_by_id: dict[str, str] = {}
    attempts: list[RecentSqlAttempt] = []
    seen: set[tuple[str, str]] = set()

    for message in messages:
        if message.role == "assistant" and message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.name != "run_sql" or not tool_call.id:
                    continue
                sql = str(tool_call.arguments.get("sql", "")).strip()
                if sql:
                    tool_calls_by_id[tool_call.id] = sql

    for message in messages:
        if message.role != "tool" or not message.tool_call_id:
            continue
        sql = tool_calls_by_id.get(message.tool_call_id)
        if not sql:
            continue

        outcome = classify_run_sql_outcome(message.content)
        if outcome is None:
            continue

        detail = normalize_whitespace(message.content or "")
        fingerprint = (normalize_whitespace(sql), outcome)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)

        attempts.append(
            RecentSqlAttempt(
                sql=sql,
                tables=tuple(extract_table_names_from_sql(sql)),
                outcome=outcome,
                detail=detail,
            )
        )

    return attempts[-limit:]


def classify_run_sql_outcome(content: str) -> str | None:
    normalized = " ".join((content or "").strip().lower().split())
    if not normalized:
        return None
    if "no rows returned" in normalized or "0 rows" in normalized:
        return "empty"
    if "sql executed successfully." in normalized or "query executed successfully." in normalized:
        return None
    if "results saved to file:" in normalized or "rows affected" in normalized:
        return None
    return "error"


def extract_table_names_from_sql(sql: str) -> list[str]:
    matches = re.findall(
        r"\b(?:from|join)\s+((?:\"[^\"]+\"|`[^`]+`|[A-Za-z0-9_]+)(?:\.(?:\"[^\"]+\"|`[^`]+`|[A-Za-z0-9_]+))?)",
        sql,
        flags=re.IGNORECASE,
    )
    table_names: list[str] = []
    seen: set[str] = set()
    for match in matches:
        normalized = match.replace('"', "").replace("`", "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        table_names.append(normalized)
    return table_names


def format_recent_sql_feedback(attempts: list[RecentSqlAttempt]) -> str:
    if not attempts:
        return ""

    lines = [REQUEST_SQL_FEEDBACK_MARKER]
    for attempt in attempts:
        tables_display = ", ".join(f"`{table}`" for table in attempt.tables) if attempt.tables else "unknown tables"
        sql_preview = truncate_text(attempt.sql, 320)
        if attempt.outcome == "empty":
            lines.append(
                f"- Empty result for SQL on {tables_display}: `{sql_preview}`. This only means that exact query returned no rows. Do not repeat the same join/filter unchanged, and do not assume those tables are empty."
            )
        else:
            lines.append(
                f"- Failed SQL on {tables_display}: `{sql_preview}`. Error: {truncate_text(attempt.detail, 220)}. Do not retry the same broken columns/tables unchanged."
            )
    return "\n".join(lines)


class LocalAdminUserResolver(UserResolver):
    """Treat local browser sessions as fully privileged local users."""

    async def resolve_user(self, request_context: RequestContext) -> User:
        email = (
            request_context.get_cookie("vanna_email")
            or request_context.get_header("x-vanna-user")
            or "local-admin@localhost"
        )
        username = email.split("@", 1)[0] if "@" in email else email
        return User(
            id=email,
            username=username,
            email=email,
            group_memberships=["user", "admin"],
            metadata={"remote_addr": request_context.remote_addr},
        )


def build_vanna_v2_chat_handler(vn: Any, db: DatabaseClient, settings: Settings) -> ChatHandler:
    agent_memory = NoOpAgentMemory()
    tool_registry = ToolRegistry()
    tool_registry.register_local_tool(
        ReadOnlySqlTool(db=db, max_rows=settings.max_result_rows),
        access_groups=["user", "admin"],
    )
    llm_service = OllamaToolCallFallbackService(
        model=settings.normalized_ollama_model,
        host=settings.ollama_host,
        timeout=settings.ollama_timeout,
        num_ctx=settings.ollama_num_ctx,
        temperature=0.0,
    )
    agent = Agent(
        llm_service=llm_service,
        tool_registry=tool_registry,
        agent_memory=agent_memory,
        user_resolver=LocalAdminUserResolver(),
        system_prompt_builder=SqlChatSystemPromptBuilder(),
        llm_context_enhancer=SchemaAwareLlmContextEnhancer(vn, None),
        conversation_filters=[StandaloneQuestionConversationFilter()],
        config=AgentConfig(
            max_tool_iterations=settings.chat_max_tool_iterations,
            stream_responses=True,
            include_thinking_indicators=False,
        ),
    )
    return ChatHandler(agent)


async def stream_with_keepalive(
    stream: AsyncGenerator[Any, None],
    keepalive_seconds: float,
) -> AsyncGenerator[Any | None, None]:
    if keepalive_seconds <= 0:
        async for item in stream:
            yield item
        return

    stream_iter = stream.__aiter__()
    pending = asyncio.create_task(stream_iter.__anext__())

    try:
        while True:
            done, _ = await asyncio.wait({pending}, timeout=keepalive_seconds)
            if pending not in done:
                yield None
                continue

            try:
                item = pending.result()
            except StopAsyncIteration:
                break

            yield item
            pending = asyncio.create_task(stream_iter.__anext__())
    finally:
        if not pending.done():
            pending.cancel()
            with suppress(asyncio.CancelledError):
                await pending


def build_vanna_v2_index_html(settings: Settings) -> str:
    title = html_escape(settings.vanna_ui_title)
    subtitle = html_escape(settings.vanna_ui_subtitle)
    cdn_url = html_escape(settings.vanna_ui_cdn_url)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script type="module" src="{cdn_url}"></script>
    <style>
        :root {{
            color-scheme: light;
            --bg: #f7f1e2;
            --panel: rgba(255, 255, 255, 0.88);
            --ink: #023d60;
            --accent: #15a8a8;
            --accent-2: #fe5d26;
            --grid: rgba(2, 61, 96, 0.08);
            --border: rgba(2, 61, 96, 0.18);
        }}

        * {{
            box-sizing: border-box;
        }}

        body {{
            margin: 0;
            min-height: 100vh;
            font-family: "Inter", "Segoe UI", sans-serif;
            color: var(--ink);
            background:
                radial-gradient(circle at top left, rgba(21, 168, 168, 0.20), transparent 32%),
                radial-gradient(circle at bottom right, rgba(254, 93, 38, 0.16), transparent 28%),
                linear-gradient(180deg, #fffdf7 0%, var(--bg) 100%);
        }}

        body::before {{
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            background-image:
                linear-gradient(var(--grid) 1px, transparent 1px),
                linear-gradient(90deg, var(--grid) 1px, transparent 1px);
            background-size: 28px 28px;
            opacity: 0.65;
        }}

        main {{
            position: relative;
            z-index: 1;
            width: min(1180px, calc(100vw - 32px));
            margin: 24px auto;
        }}

        .hero {{
            padding: 24px 28px;
            border: 1px solid var(--border);
            border-radius: 24px;
            background: var(--panel);
            backdrop-filter: blur(16px);
            box-shadow: 0 24px 60px rgba(2, 61, 96, 0.10);
        }}

        .eyebrow {{
            display: inline-block;
            margin-bottom: 10px;
            padding: 6px 10px;
            border-radius: 999px;
            background: rgba(21, 168, 168, 0.12);
            color: var(--accent);
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }}

        h1 {{
            margin: 0;
            font-size: clamp(30px, 5vw, 54px);
            line-height: 0.98;
            letter-spacing: -0.04em;
        }}

        p {{
            margin: 10px 0 0;
            max-width: 780px;
            font-size: 16px;
            line-height: 1.6;
            color: rgba(2, 61, 96, 0.82);
        }}

        .meta {{
            margin-top: 16px;
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            font-size: 13px;
        }}

        .meta span {{
            padding: 8px 12px;
            border-radius: 999px;
            background: rgba(2, 61, 96, 0.06);
        }}

        .chat-shell {{
            margin-top: 18px;
            height: min(76vh, 820px);
            border: 1px solid var(--border);
            border-radius: 24px;
            background: rgba(255, 255, 255, 0.94);
            overflow: hidden;
            box-shadow: 0 30px 80px rgba(2, 61, 96, 0.12);
        }}

        vanna-chat {{
            width: 100%;
            height: 100%;
            display: block;
        }}

        @media (max-width: 720px) {{
            main {{
                width: calc(100vw - 18px);
                margin: 10px auto 18px;
            }}

            .hero {{
                padding: 18px;
                border-radius: 18px;
            }}

            .chat-shell {{
                height: calc(100vh - 250px);
                min-height: 520px;
                border-radius: 18px;
            }}
        }}
    </style>
</head>
<body>
    <main>
        <section class="hero">
            <span class="eyebrow">Vanna 2 Chat</span>
            <h1>{title}</h1>
            <p>{subtitle}</p>
            <div class="meta">
                <span>UI: Vanna 2 web component</span>
                <span>LLM: Ollama at {html_escape(settings.ollama_host)}</span>
                <span>DB: {html_escape(settings.database_target)}</span>
            </div>
        </section>

        <section class="chat-shell">
            <vanna-chat
                sse-endpoint="/api/vanna/v2/chat_sse"
                ws-endpoint="/api/vanna/v2/chat_websocket"
                poll-endpoint="/api/vanna/v2/chat_poll">
            </vanna-chat>
        </section>
    </main>
</body>
</html>"""


def register_vanna_v2_routes(app: FastAPI, settings: Settings) -> None:
    @app.get("/", response_class=HTMLResponse)
    async def vanna_chat_home() -> str:
        return build_vanna_v2_index_html(settings)

    @app.post("/api/vanna/v2/chat_sse")
    async def chat_sse(
        chat_request: ChatRequest,
        http_request: Request,
    ) -> StreamingResponse:
        chat_handler = get_chat_handler(http_request.app)
        chat_request.request_context = build_request_context(http_request, chat_request.metadata)

        async def generate() -> AsyncGenerator[str, None]:
            try:
                async for chunk in stream_with_keepalive(
                    chat_handler.handle_stream(chat_request),
                    settings.sse_keepalive_seconds,
                ):
                    if chunk is None:
                        yield ": keep-alive\n\n"
                        continue
                    yield f"data: {chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as exc:
                traceback.print_exc()
                error_data = {
                    "type": "error",
                    "data": {"message": str(exc)},
                    "conversation_id": chat_request.conversation_id or "",
                    "request_id": chat_request.request_id or "",
                }
                yield f"data: {json.dumps(error_data)}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/vanna/v2/chat_poll")
    async def chat_poll(
        chat_request: ChatRequest,
        http_request: Request,
    ) -> ChatResponse:
        chat_handler = get_chat_handler(http_request.app)
        chat_request.request_context = build_request_context(http_request, chat_request.metadata)
        try:
            return await chat_handler.handle_poll(chat_request)
        except Exception as exc:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Chat failed: {exc}") from exc

    @app.websocket("/api/vanna/v2/chat_websocket")
    async def chat_websocket(websocket: WebSocket) -> None:
        await websocket.accept()
        chat_handler = get_chat_handler(websocket.app)

        try:
            while True:
                payload = await websocket.receive_json()
                metadata = payload.get("metadata", {})
                payload["request_context"] = RequestContext(
                    cookies=dict(websocket.cookies),
                    headers=dict(websocket.headers),
                    remote_addr=websocket.client.host if websocket.client else None,
                    query_params=dict(websocket.query_params),
                    metadata=metadata,
                )
                chat_request = ChatRequest(**payload)

                async for chunk in chat_handler.handle_stream(chat_request):
                    await websocket.send_json(chunk.model_dump())

                await websocket.send_json(
                    {
                        "type": "completion",
                        "data": {"status": "done"},
                        "conversation_id": chat_request.conversation_id or "",
                        "request_id": chat_request.request_id or "",
                    }
                )
        except WebSocketDisconnect:
            return
        except Exception as exc:
            traceback.print_exc()
            try:
                await websocket.send_json(
                    {
                        "type": "error",
                        "data": {"message": str(exc)},
                    }
                )
            finally:
                await websocket.close()


def get_chat_handler(app: FastAPI) -> ChatHandler:
    services = getattr(app.state, "services", None)
    if services is None or getattr(services, "vanna_v2_chat_handler", None) is None:
        raise RuntimeError("Vanna 2 chat handler is not ready yet.")
    return services.vanna_v2_chat_handler


def build_request_context(
    http_request: Request,
    metadata: dict[str, Any] | None,
) -> RequestContext:
    return RequestContext(
        cookies=dict(http_request.cookies),
        headers=dict(http_request.headers),
        remote_addr=http_request.client.host if http_request.client else None,
        query_params=dict(http_request.query_params),
        metadata=metadata or {},
    )


def html_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
