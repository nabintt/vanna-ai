from __future__ import annotations

import asyncio
import pandas as pd
from vanna.core.llm import LlmResponse
from vanna.core.tool import ToolSchema
from vanna.core.user import User

from app.vanna_v2 import (
    SchemaAwareLlmContextEnhancer,
    build_schema_catalog,
    coerce_text_tool_calls,
    extract_text_tool_calls,
    search_schema_catalog,
)


def build_tool_schemas() -> list[ToolSchema]:
    return [
        ToolSchema(
            name="search_saved_correct_tool_uses",
            description="Search previously saved tool usage patterns.",
            parameters={"type": "object"},
            access_groups=["user"],
        ),
        ToolSchema(
            name="run_sql",
            description="Execute SQL.",
            parameters={"type": "object"},
            access_groups=["user"],
        ),
        ToolSchema(
            name="save_question_tool_args",
            description="Save a successful tool call.",
            parameters={"type": "object"},
            access_groups=["admin"],
        ),
    ]


def test_extract_text_tool_calls_from_markdown_json_blocks():
    content = """
To find the top players who bid 8 or more in their last game, I'll first search for any saved successful patterns related to this query.

```json
{
  "name": "search_saved_correct_tool_uses",
  "arguments": {
    "question": "give me top players who bid 8 or more in their last game"
  }
}
```

If there are no saved patterns or if the saved pattern does not match, I will proceed with executing an SQL query to fetch the relevant data.

```json
{
  "name": "run_sql",
  "arguments": {
    "sql": "SELECT username, MAX(highest_bid) AS last_highest_bid FROM top_skilled_player_games GROUP BY username HAVING MAX(highest_bid) >= 8 ORDER BY last_highest_bid DESC"
  }
}
```

After successfully executing the query and obtaining the results, I will save this pattern for future use.

```json
{
  "name": "save_question_tool_args",
  "arguments": {
    "question": "give me top players who bid 8 or more in their last game",
    "tool_name": "run_sql",
    "args": {
      "sql": "SELECT username, MAX(highest_bid) AS last_highest_bid FROM top_skilled_player_games GROUP BY username HAVING MAX(highest_bid) >= 8 ORDER BY last_highest_bid DESC"
    }
  }
}
```
"""

    tool_calls = extract_text_tool_calls(content, build_tool_schemas())

    assert [tool_call.name for tool_call in tool_calls] == [
        "search_saved_correct_tool_uses",
        "run_sql",
        "save_question_tool_args",
    ]
    assert tool_calls[0].arguments["question"] == (
        "give me top players who bid 8 or more in their last game"
    )
    assert "SELECT username" in tool_calls[1].arguments["sql"]
    assert tool_calls[2].arguments["tool_name"] == "run_sql"


def test_coerce_text_tool_calls_rewrites_response_into_tool_calls():
    response = LlmResponse(
        content="""
```json
{"name":"run_sql","arguments":{"sql":"SELECT 1"}}
```
""",
        tool_calls=None,
    )

    normalized = coerce_text_tool_calls(response, build_tool_schemas())

    assert normalized.content is None
    assert normalized.tool_calls is not None
    assert normalized.tool_calls[0].name == "run_sql"
    assert normalized.tool_calls[0].arguments == {"sql": "SELECT 1"}
    assert normalized.metadata["tool_call_fallback"] == "text_json"


def test_coerce_text_tool_calls_leaves_normal_text_alone():
    response = LlmResponse(content="Here is a plain-language answer.", tool_calls=None)

    normalized = coerce_text_tool_calls(response, build_tool_schemas())

    assert normalized.content == "Here is a plain-language answer."
    assert normalized.tool_calls is None


class FakeSchemaAwareVanna:
    def get_training_data(self):
        return pd.DataFrame(
            [
                {
                    "id": "ddl-1",
                    "question": None,
                    "content": (
                        'CREATE TABLE "public"."leaderboard_past" (\n'
                        '    "player_id" uuid NOT NULL,\n'
                        '    "skill" numeric,\n'
                        '    "rank_position" integer\n'
                        ");"
                    ),
                    "training_data_type": "ddl",
                },
                {
                    "id": "ddl-2",
                    "question": None,
                    "content": (
                        'CREATE TABLE "public"."player" (\n'
                        '    "id" uuid NOT NULL,\n'
                        '    "username" varchar(255)\n'
                        ");"
                    ),
                    "training_data_type": "ddl",
                },
            ]
        )

    def get_related_ddl(self, question: str):
        return []

    def get_related_documentation(self, question: str):
        return [
            "Use leaderboard tables carefully and only reference columns that actually exist in the table."
        ]

    def get_similar_question_sql(self, question: str):
        return [
            {
                "question": "Show leaderboard ranks with skill values.",
                "sql": 'SELECT "player_id", "skill", "rank_position" FROM "public"."leaderboard_past";',
            }
        ]


def test_schema_catalog_keyword_fallback_finds_relevant_table():
    vn = FakeSchemaAwareVanna()

    catalog = build_schema_catalog(vn)
    matches = search_schema_catalog(catalog, "show me leaderboard_past skill rankings")

    assert matches
    assert matches[0].table_name == "public.leaderboard_past"
    assert "skill" in matches[0].columns


def test_schema_aware_enhancer_injects_ddl_docs_and_examples():
    vn = FakeSchemaAwareVanna()
    enhancer = SchemaAwareLlmContextEnhancer(vn, agent_memory=None)
    user = User(
        id="u1",
        username="analyst",
        email="analyst@example.com",
        group_memberships=["user"],
    )

    prompt = asyncio.run(
        enhancer.enhance_system_prompt(
            "Base system prompt.",
            "show me leaderboard_past skill rankings",
            user,
        )
    )

    assert "## SQL Generation Rules" in prompt
    assert 'CREATE TABLE "public"."leaderboard_past"' in prompt
    assert "`public.leaderboard_past` columns: player_id, skill, rank_position" in prompt
    assert "Use leaderboard tables carefully" in prompt
    assert 'SELECT "player_id", "skill", "rank_position"' in prompt
