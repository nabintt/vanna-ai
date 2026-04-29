from __future__ import annotations

import asyncio

from app.db import build_statement_timeout_sql
from app.vanna_v2 import stream_with_keepalive


def test_build_statement_timeout_sql_for_postgres():
    assert build_statement_timeout_sql("postgres", 0) == "SET statement_timeout = 0"
    assert build_statement_timeout_sql("postgres", 45000) == "SET statement_timeout = 45000"


def test_build_statement_timeout_sql_ignores_other_dialects():
    assert build_statement_timeout_sql("mysql", 45000) is None


def test_stream_with_keepalive_emits_heartbeat_before_slow_result():
    async def slow_stream():
        await asyncio.sleep(0.03)
        yield "done"

    async def collect_items():
        items = []
        async for item in stream_with_keepalive(slow_stream(), 0.005):
            items.append(item)
        return items

    items = asyncio.run(collect_items())

    assert items[-1] == "done"
    assert None in items[:-1]


def test_stream_with_keepalive_can_be_disabled():
    async def quick_stream():
        yield "done"

    async def collect_items():
        items = []
        async for item in stream_with_keepalive(quick_stream(), 0):
            items.append(item)
        return items

    assert asyncio.run(collect_items()) == ["done"]
