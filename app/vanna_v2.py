from __future__ import annotations

import json
import logging
import re
import traceback
from dataclasses import dataclass
from json import JSONDecodeError, JSONDecoder
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from vanna import Agent, AgentConfig
from vanna.core.enhancer import DefaultLlmContextEnhancer, LlmContextEnhancer
from vanna.core.llm import LlmMessage, LlmRequest, LlmResponse, LlmStreamChunk
from vanna.core.tool import ToolCall, ToolSchema
from vanna.core.user import RequestContext, User
from vanna.core.user.resolver import UserResolver
from vanna.integrations.ollama import OllamaLlmService
from vanna.legacy.adapter import LegacyVannaAdapter
from vanna.servers.base import ChatHandler, ChatRequest, ChatResponse

from app.config import Settings

logger = logging.getLogger(__name__)

JSON_CODE_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
SEARCH_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")
DDL_TABLE_PATTERN = re.compile(
    r"CREATE\s+TABLE\s+(?:\"|`)?(?P<schema>[A-Za-z0-9_]+)(?:\"|`)?\.(?:\"|`)?(?P<table>[A-Za-z0-9_]+)(?:\"|`)?",
    re.IGNORECASE,
)
DDL_COLUMN_PATTERN = re.compile(
    r"^\s*(?:\"|`)?(?P<name>[A-Za-z0-9_]+)(?:\"|`)?\s+[A-Za-z]",
    re.MULTILINE,
)
DDL_SKIP_TOKENS = {"create", "table", "primary", "constraint", "foreign", "unique", "check"}
REQUEST_SCHEMA_CONTEXT_MARKER = "[Attached schema context for this request]"


@dataclass(frozen=True)
class SchemaCatalogEntry:
    table_name: str
    columns: tuple[str, ...]
    ddl: str
    search_terms: frozenset[str]


class OllamaToolCallFallbackService(OllamaLlmService):
    """Parse tool-call JSON from model text when native tool calling is unavailable."""

    async def send_request(self, request: LlmRequest) -> LlmResponse:
        response = await super().send_request(request)
        return coerce_text_tool_calls(response, request.tools)

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
        normalized = coerce_text_tool_calls(response, request.tools)
        yield LlmStreamChunk(
            content=normalized.content,
            tool_calls=normalized.tool_calls,
            finish_reason=normalized.finish_reason,
            metadata=normalized.metadata,
        )


def coerce_text_tool_calls(
    response: LlmResponse,
    tools: list[ToolSchema] | None,
) -> LlmResponse:
    if response.tool_calls or not response.content or not tools:
        return response

    parsed_tool_calls = extract_text_tool_calls(response.content, tools)
    if not parsed_tool_calls:
        return response

    metadata = dict(response.metadata)
    metadata["tool_call_fallback"] = "text_json"
    metadata["tool_call_fallback_count"] = len(parsed_tool_calls)

    return LlmResponse(
        content=None,
        tool_calls=parsed_tool_calls,
        finish_reason=response.finish_reason or "tool_calls",
        usage=response.usage,
        metadata=metadata,
    )


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
        request_context = format_request_schema_context(context_bundle)
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

    return {
        "related_ddl": related_ddl,
        "related_docs": dedupe_text_items(safe_list(vn.get_related_documentation, question=question)),
        "similar_pairs": dedupe_question_sql_pairs(
            safe_list(vn.get_similar_question_sql, question=question)
        ),
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


def format_request_schema_context(context_bundle: dict[str, Any]) -> str:
    related_ddl = context_bundle.get("related_ddl", [])
    related_docs = context_bundle.get("related_docs", [])
    similar_pairs = context_bundle.get("similar_pairs", [])

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

    if len(parts) == 1:
        return ""

    return "\n".join(parts)


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


def build_vanna_v2_chat_handler(vn: Any, settings: Settings) -> ChatHandler:
    legacy_adapter = LegacyVannaAdapter(vn)
    llm_service = OllamaToolCallFallbackService(
        model=settings.normalized_ollama_model,
        host=settings.ollama_host,
        timeout=settings.ollama_timeout,
        num_ctx=settings.ollama_num_ctx,
        temperature=0.0,
    )
    agent = Agent(
        llm_service=llm_service,
        tool_registry=legacy_adapter,
        agent_memory=legacy_adapter,
        user_resolver=LocalAdminUserResolver(),
        llm_context_enhancer=SchemaAwareLlmContextEnhancer(vn, legacy_adapter),
        config=AgentConfig(
            stream_responses=True,
            include_thinking_indicators=False,
        ),
    )
    return ChatHandler(agent)


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
                async for chunk in chat_handler.handle_stream(chat_request):
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
