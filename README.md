# PostgreSQL RAG Agent - Intelligent Knowledge Base Search

Agentic RAG system combining PostgreSQL + pgvector with Pydantic AI for intelligent document retrieval.

## Features

- **Hybrid Search**: Combines semantic vector search with full-text keyword search using a weighted blend
  - Vector similarity via pgvector, text relevance via `tsvector`
- **Multi-Format Ingestion**: PDF, Word, PowerPoint, Excel, HTML, Markdown, Audio transcription
- **Intelligent Chunking**: Docling HybridChunker preserves document structure and semantic boundaries
- **Conversational CLI**: Rich-based interface with real-time streaming and tool call visibility
- **Multiple LLM Support**: OpenAI, OpenRouter, Ollama, Gemini
- **Runs Locally**: Works with your local Postgres + pgvector (no Atlas required)
- **Multi-Format Ingestion**: PDF, Word, PowerPoint, Excel, HTML, Markdown, Audio transcription
- **Intelligent Chunking**: Docling HybridChunker preserves document structure and semantic boundaries
- **Conversational CLI**: Rich-based interface with real-time streaming and tool call visibility
- **Multiple LLM Support**: OpenAI, OpenRouter, Ollama, Gemini
- **Cost Effective**: Runs locally with Postgres; no cloud dependency

## Prerequisites

- Python 3.10+
- Local PostgreSQL with pgvector (Docker is fine)
- LLM provider API key (OpenAI, OpenRouter, Ollama via OpenAI API, etc.)
- Embedding provider API key (OpenAI or OpenRouter recommended)
- UV package manager

## Quick Start

### 1. Install UV Package Manager

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone and Setup Project

```bash
git clone https://github.com/coleam00/PostgreSQL-RAG-Agent.git
cd PostgreSQL-RAG-Agent

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # Unix/Mac
.venv\Scripts\activate     # Windows
uv sync
```

### 3. Start Postgres with pgvector (example docker-compose)

```yaml
services:
  db:
    image: ankane/pgvector
    environment:
      POSTGRES_USER: rag
      POSTGRES_PASSWORD: ragpass
      POSTGRES_DB: rag_db
    ports:
      - "5432:5432"
```

After the container is up, verify extensions inside psql:
```
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
```

### 4. Configure Environment Variables

```bash
# Copy the example file
cp .env.example .env
```

Edit `.env` with your credentials:
- **DATABASE_URL**: e.g., `postgresql://rag:ragpass@localhost:5432/rag_db`
- **LLM_API_KEY**: Your LLM provider API key (OpenRouter, OpenAI, etc.)
- **EMBEDDING_API_KEY**: Your API key for embeddings (such as OpenAI or OpenRouter)
- **EMBEDDING_MODEL / EMBEDDING_DIMENSION**: Ensure the dimension matches your pgvector column (default 1536)
- **TEXT_SEARCH_LANGUAGE**: Language for PostgreSQL text search (e.g., `english`, `dutch`)

### 5. Validate Configuration

```bash
uv run python -m src.test_config
```

You should see: `[OK] ALL CONFIGURATION CHECKS PASSED`

### 6. Run Ingestion Pipeline

```bash
# Add your documents to the documents/ folder
uv run python -m src.ingestion.ingest -d ./documents
```

This will:
- Process your documents (PDF, Word, PowerPoint, Excel, Markdown, etc.)
- Chunk them intelligently
- Generate embeddings
- Store everything in Postgres (`documents`, `chunks`); vector and FTS indexes are created automatically on first run.

### 7. Run the Agent

```bash
uv run python -m src.cli
```

Now you can ask questions and the agent will search your knowledge base!

## Project Structure

```
PostgreSQL-RAG-Agent/
├── src/                           # Postgres implementation
│   ├── settings.py               # Configuration management
│   ├── providers.py              # LLM/embedding providers
│   ├── dependencies.py           # Postgres connection & AgentDependencies
│   ├── test_config.py            # Configuration validation
│   ├── tools.py                  # Search tools (semantic, text, hybrid)
│   ├── agent.py                  # Pydantic AI agent with search tools
│   ├── cli.py                    # Rich-based conversational CLI
│   ├── prompts.py                # System prompts
│   └── ingestion/
│       ├── chunker.py            # Docling HybridChunker wrapper
│       ├── embedder.py           # Batch embedding generation
│       └── ingest.py             # Postgres ingestion pipeline
├── examples/                      # Legacy reference
├── documents/                     # Document folder (sample docs)
├── .agents/                       # Plans and analysis
└── pyproject.toml                # UV package configuration
```

## Technology Stack

- **Database**: PostgreSQL + pgvector + tsvector
- **Agent Framework**: Pydantic AI 0.1.0+
- **Document Processing**: Docling 2.14+ (PDF, Word, PowerPoint, Excel, Audio)
- **Async Driver**: asyncpg
- **CLI**: Rich 13.9+ (terminal formatting and streaming)
- **Package Manager**: UV 0.5.0+ (fast dependency management)

## Hybrid Search Implementation

Hybrid search blends pgvector cosine similarity with text ranking (`ts_rank_cd`). Adjust `default_text_weight` in `.env` to tune balance.

### Reciprocal Rank Fusion (RRF) note

The agent merges vector- and text-ranked results using RRF-style weighting so items that score well in both lists rise to the top. This preserves the strengths of semantic matching while keeping high-signal keyword hits.

## Usage Examples

### Interactive CLI

```bash
uv run python -m src.cli
```

**Example conversation:**
```
You: What is NeuralFlow AI's revenue goal for 2025?

  [Calling tool] search_knowledge_base
    Query: NeuralFlow AI's revenue goal for 2025
    Type: hybrid
    Results: 5
  [Search completed successfully]
