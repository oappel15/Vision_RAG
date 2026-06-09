# Vision RAG

A Visual Retrieval-Augmented Generation (RAG) system that indexes PDF documents using the **ColQwen2** vision-language model and answers questions by retrieving and reasoning over page images.

## Architecture

| Service | Port | Role |
|---|---|---|
| `open-webui` | 3000 | Chat frontend |
| `pipelines` | 9099 | ColQwen2 CPU model + Qdrant search + background PDF indexing |
| `pdf-ingest` | 8082 | Upload/delete UI, triggers indexing |
| `qdrant` | 6333 | Vector DB (multi-vector MaxSim collection) |
| `image-server` | 8081 | Static file server for cached page images |
| `ollama` | 11434 | Local LLM runtime (optional) |

## How It Works

1. **Indexing** — PDFs placed in `my-pdfs/` are converted to page images, embedded with ColQwen2 (multi-vector per page), and stored in Qdrant. Indexing runs in a background thread so queries are never blocked.
2. **Retrieval** — Queries are embedded with ColQwen2 and matched against page vectors using MaxSim scoring.
3. **Answer generation** — Top-K page images are sent to an OpenRouter vision-language model (Qwen3-VL by default) which answers with inline `[REF:N]` citations linked to source pages.
4. **Adjacent expansion** — If a cited page references content on an adjacent page (e.g. a timing diagram, continued register table), the pipeline automatically fetches and includes that page in a supplementary pass.

## Quick Start

```bash
# 1. Set your OpenRouter API key
cp .env.example .env
# edit .env and set OPENROUTER_API_KEY=...

# 2. Start all services
docker compose up -d

# 3. Open the chat UI
open http://localhost:3000

# 4. Upload PDFs via the ingest UI
open http://localhost:8082
```

## Configuration

Key environment variables in `docker-compose.yml`:

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | — | Required. Your OpenRouter API key |
| `OPENROUTER_MODEL` | `qwen/qwen3-vl-30b-a3b-instruct` | VLM used for answer generation |
| `TARGET_KNOWLEDGE` | `my_docs` | Qdrant collection name |

Pipeline valves (tunable in Open WebUI):

| Valve | Default | Description |
|---|---|---|
| `TOP_K` | 8 | Number of pages retrieved per query |
| `ADJACENT_PAGES` | 1 | Pages to expand around cited pages |
| `SCORE_THRESHOLD` | 0.0 | Minimum retrieval score filter |
| `SHOW_SOURCE_PAGES` | true | Show source thumbnail table |

## Special Chat Commands

| Command | Effect |
|---|---|
| `status` | Show indexing progress and indexed files |

## Project Structure

```
pipelines/          # ColQwen2 pipeline (main RAG logic)
  colpali-pipeline.py
  requirements.txt
  cache/images/     # Cached page images served by image-server
pdf-ingest/         # FastAPI upload/delete sidecar
my-pdfs/            # Drop PDFs here to index them
docker-compose.yml
```
