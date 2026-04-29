# Local Vanna AI with Ollama

This repo runs a local Vanna stack on top of:

- `vanna==2.0.2`
- Ollama for SQL generation
- Chroma for persisted training vectors
- FastAPI for both the API and the Vanna 2 chat UI
- PostgreSQL first, with MySQL-ready database wiring

The important design choice is that this project uses the new Vanna 2 chat/server surface while preserving the repo's existing local training bootstrap flow through `vanna.legacy.*`. That means you keep the same schema bootstrap, glossary docs, and question-to-SQL examples, but the browser UI is the Vanna 2 web component.

## What changed

- `make run` starts one FastAPI server.
- `GET /` serves the Vanna 2 chat UI.
- `GET /docs` serves FastAPI Swagger docs for the repo's API endpoints.
- Existing training data in `data/chroma/` is still used when present, but automatic training is off by default.

There is no separate Flask UI server anymore.

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
│   ├── training.py
│   └── vanna_v2.py
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
```

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Or:

```bash
make setup
```

## Configure

Copy the example env file:

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
TRAIN_ON_START=false
ALLOW_BOOTSTRAP_SAMPLE_DATA=true
```

Useful UI settings:

```dotenv
VANNA_UI_TITLE=Local Vanna AI
VANNA_UI_SUBTITLE=Ask questions in natural language using the Vanna 2 chat interface.
VANNA_UI_CDN_URL=https://img.vanna.ai/vanna-components.js
```

## Start Ollama

Install Ollama for your platform, then pull at least the chat model:

```bash
ollama pull llama3.2
```

Optional local embeddings:

```bash
ollama pull embeddinggemma
```

Then set:

```dotenv
OLLAMA_EMBED_MODEL=embeddinggemma
```

Verify Ollama:

```bash
make check-ollama
```

## Bootstrap training

Run bootstrap once after the database is reachable:

```bash
make train
```

Bootstrap does all of this:

1. Connects to the configured database.
2. Reads `information_schema`.
3. Builds a Vanna training plan from schema metadata.
4. Adds DDL-like schema statements.
5. Loads glossary documentation from `data/training/business_glossary.md`.
6. Loads reviewed question/SQL examples from `data/training/example_question_sql.json`.
7. Persists the resulting training data into local Chroma.

Training artifacts:

- Vector store: `data/chroma/`
- Glossary source: `data/training/business_glossary.md`
- Example pairs source: `data/training/example_question_sql.json`
- Bootstrap summary: `data/training/bootstrap_state.json`

## Run the app

Start the combined FastAPI + Vanna 2 server:

```bash
make run
```

For autoreload during development:

```bash
make dev
```

Then open:

- UI: [http://localhost:8000/](http://localhost:8000/)
- Swagger docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Readiness: [http://localhost:8000/ready](http://localhost:8000/ready)

## API endpoints

- `GET /health`
- `GET /ready`
- `POST /train`
- `POST /ask`
- `POST /generate_sql`
- `POST /run_sql`
- `POST /api/vanna/v2/chat_sse`
- `POST /api/vanna/v2/chat_poll`
- `WS /api/vanna/v2/chat_websocket`

### cURL examples

Health:

```bash
curl http://localhost:8000/health
```

Readiness:

```bash
curl http://localhost:8000/ready
```

Manual training:

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
  -d '{"question":"Show 10 rows from public.orders."}'
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

- `app/config.py`: environment-driven settings
- `app/agent.py`: local Ollama checks and legacy Vanna agent creation
- `app/training.py`: idempotent schema/docs/example bootstrap into Chroma
- `app/vanna_v2.py`: Vanna 2 chat handler and UI routes
- `app/server.py`: FastAPI lifecycle plus repo API endpoints

## First run checklist

1. Create the virtualenv and install dependencies.
2. Start Ollama and pull `llama3.2`.
3. Copy `.env.example` to `.env` and fill in your DB credentials.
4. Make sure your database is reachable.
5. Run `make check-ollama`.
6. Run `make train` only if you want to refresh the local Chroma training data.
7. Run `make run`.
8. Open `http://localhost:8000/`.
