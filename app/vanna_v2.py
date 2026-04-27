from __future__ import annotations

import json
import traceback
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from vanna import Agent, AgentConfig
from vanna.core.user import RequestContext, User
from vanna.core.user.resolver import UserResolver
from vanna.integrations.ollama import OllamaLlmService
from vanna.legacy.adapter import LegacyVannaAdapter
from vanna.servers.base import ChatHandler, ChatRequest, ChatResponse

from app.config import Settings


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
    llm_service = OllamaLlmService(
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
