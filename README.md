# Local Vanna AI with Ollama

This repository is a fully local Vanna AI project built around:

- `vanna==0.7.9` for the classic training-plan + local vector-store workflow
- Ollama at `http://localhost:11434` for SQL generation
- PostgreSQL first, with MySQL-ready structure in the database layer
- ChromaDB persisted on disk in [`data/chroma`](./data/chroma)
- FastAPI for a minimal production-style API

The app refuses to start if the core prerequisites are not ready:

- Ollama must be reachable
- the configured database must be reachable
- training data must already exist, or `TRAIN_ON_START=true` must be enabled so bootstrap can run automatically

## Why this version?

`vanna` 2.x is a new agent/server architecture. This project intentionally pins `vanna==0.7.9`, which is the latest pre-2.0 release and still exposes the local `Ollama`, `ChromaDB_VectorStore`, `train`, `get_training_plan_generic`, and `get_training_data` APIs that match the requested bootstrap flow.

## Repository layout

```text
.
├── README.md
├── Makefile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── app
│   ├── __init__.py
│   ├── agent.py
│   ├── config.py
│   ├── db.py
│   ├── health.py
│   ├── logging_config.py
│   ├── server.py
│   └── training.py
├── scripts
│   ├── bootstrap_training.py
│   ├── check_ollama.py
│   └── run_server.py
├── data
│   ├── chroma
│   └── training
│       ├── business_glossary.md
│       └── example_question_sql.json
└── tests
    ├── test_config.py
    ├── test_health.py
    └── test_training_bootstrap.py
```

## Exact install commands

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Or use the Makefile:

```bash
make setup
```

## 1. Install and start Ollama

Install Ollama from the official docs for your platform.

Then pull at least the chat model:

```bash
ollama pull llama3.2
```

Optional: if you want embeddings to stay fully inside Ollama too, set `OLLAMA_EMBED_MODEL=embeddinggemma` in `.env` and pull it:

```bash
ollama pull embeddinggemma
```

Verify connectivity:

```bash
python scripts/check_ollama.py
```

## 2. Configure environment variables

Copy the example file and update it for your machine:

```bash
cp .env.example .env
```

Minimum required settings:

```dotenv
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2
DB_TYPE=postgres
DB_HOST=localhost
DB_PORT=5432
DB_NAME=vanna_local
DB_USER=vanna
DB_PASSWORD=vanna
TRAIN_ON_START=true
ALLOW_BOOTSTRAP_SAMPLE_DATA=true
```

Change the Ollama URL later by editing `OLLAMA_HOST` in [`.env.example`](./.env.example) or your local `.env`. That is the only place this project reads the Ollama base URL from.

## 3. Start PostgreSQL locally

If you already have Postgres running, point `.env` at it.

If not, use Docker Compose:

```bash
docker compose up -d postgres
```

The bundled Compose file starts:

- database: `vanna_local`
- user: `vanna`
- password: `vanna`
- host: `localhost`
- port: `5432`

## 4. Bootstrap training

Run the schema bootstrap once after the database is ready:

```bash
python scripts/bootstrap_training.py
```

What it does:

1. Connects to the configured database.
2. Reads `INFORMATION_SCHEMA` metadata.
3. Generates a Vanna training plan with `get_training_plan_generic`.
4. Adds plan-derived documentation.
5. Builds table DDL-like statements from schema metadata and constraints.
6. Loads business glossary text from [`data/training/business_glossary.md`](./data/training/business_glossary.md).
7. Loads example question/SQL pairs from [`data/training/example_question_sql.json`](./data/training/example_question_sql.json).
8. If that file is empty and `ALLOW_BOOTSTRAP_SAMPLE_DATA=true`, generates and saves starter examples automatically.
9. Verifies that local training data now exists in Chroma.

Bootstrap state is written to `data/training/bootstrap_state.json`.

## 5. Start the API

```bash
python scripts/run_server.py
```

Or:

```bash
make run
```

For development reload:

```bash
make dev
```

The server logs a startup summary with:

- Ollama host/model
- database target
- Chroma persistence path
- detected training entry count

## API endpoints

- `GET /health`
- `GET /ready`
- `POST /train`
- `POST /ask`
- `POST /generate_sql`
- `POST /run_sql`

### cURL examples

Health:

```bash
curl http://localhost:8000/health
```

Readiness:

```bash
curl http://localhost:8000/ready
```

Manual training/bootstrap:

```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{}'
```

Generate SQL only:

```bash
curl -X POST http://localhost:8000/generate_sql \
  -H "Content-Type: application/json" \
  -d '{"question":"How many rows are in public.orders?"}'
```

Ask and run SQL:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Show 10 rows from public.orders.","auto_train":true}'
```

Run raw SQL:

```bash
curl -X POST http://localhost:8000/run_sql \
  -H "Content-Type: application/json" \
  -d '{"sql":"SELECT * FROM public.orders LIMIT 5;","max_rows":5}'
```

## Developer workflow

```bash
make setup
make check-ollama
make train
make run
make test
```

## Architecture notes

- [`app/config.py`](./app/config.py): environment-driven settings and path management
- [`app/agent.py`](./app/agent.py): local Ollama checks, fail-fast model validation, and Vanna agent construction
- [`app/db.py`](./app/db.py): SQLAlchemy execution, information-schema reads, DDL rendering, and DB abstraction
- [`app/training.py`](./app/training.py): idempotent bootstrap, duplicate-safe training writes, startup training guard
- [`app/health.py`](./app/health.py): health and readiness diagnostics
- [`app/server.py`](./app/server.py): FastAPI endpoints and startup lifecycle

## Where training data lives

- Vector store: `data/chroma/`
- Glossary/documentation source: `data/training/business_glossary.md`
- Example question/SQL source: `data/training/example_question_sql.json`
- Last bootstrap summary: `data/training/bootstrap_state.json`

## First run checklist

1. Create a virtualenv and install dependencies.
2. Start Ollama and pull `llama3.2`.
3. Copy `.env.example` to `.env` and fill in your DB credentials.
4. Start Postgres locally if it is not already running.
5. Run `python scripts/check_ollama.py`.
6. Run `python scripts/bootstrap_training.py`.
7. Run `python scripts/run_server.py`.
8. Confirm `curl http://localhost:8000/ready` returns success.
