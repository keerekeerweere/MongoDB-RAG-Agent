"""Search tools for PostgreSQL + pgvector."""

from typing import Optional, List, Dict, Any
from pydantic_ai import RunContext
from pydantic import BaseModel, Field
import json
import math
import logging
from pathlib import Path

from src.dependencies import AgentDependencies

logger = logging.getLogger(__name__)

# Ensure a file logger exists for tooling errors
if not logger.handlers:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "agent.log")
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)


class SearchResult(BaseModel):
    """Model for search results."""

    chunk_id: str = Field(..., description="Chunk UUID")
    document_id: str = Field(..., description="Parent document UUID")
    content: str = Field(..., description="Chunk text content")
    similarity: float = Field(..., description="Relevance score (0-1)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    document_title: str = Field(..., description="Title from document lookup")
    document_source: str = Field(..., description="Source from document lookup")


def _embedding_to_vector_param(embedding: List[float]) -> str:
    # Postgres vector literal format: '[1,2,3]'
    return "[" + ",".join(map(str, embedding)) + "]"


async def semantic_search(
    ctx: RunContext[AgentDependencies],
    query: str,
    match_count: Optional[int] = None
) -> List[SearchResult]:
    """Pure vector similarity search using pgvector."""
    deps = ctx.deps

    try:
        if match_count is None:
            match_count = deps.settings.default_match_count
        match_count = min(match_count, deps.settings.max_match_count)

        query_embedding = await deps.get_embedding(query)
        embedding_literal = _embedding_to_vector_param(query_embedding)

        async with deps.db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    c.id AS chunk_id,
                    c.document_id,
                    c.content,
                    (1 - (c.embedding <-> $1::vector)) AS similarity,
                    c.metadata,
                    d.title AS document_title,
                    d.source AS document_source
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                ORDER BY c.embedding <-> $1::vector
                LIMIT $2;
                """,
                embedding_literal,
                match_count,
            )

        results = []
        for row in rows:
            metadata = row["metadata"] or {}
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except Exception:
                    metadata = {}
            results.append(
                SearchResult(
                    chunk_id=str(row["chunk_id"]),
                    document_id=str(row["document_id"]),
                    content=row["content"],
                    similarity=float(row["similarity"]),
                    metadata=metadata,
                    document_title=row["document_title"],
                    document_source=row["document_source"],
                )
            )
        return results
    except Exception as e:
        logger.exception("semantic_search_failed")
        return []


async def text_search(
    ctx: RunContext[AgentDependencies],
    query: str,
    match_count: Optional[int] = None
) -> List[SearchResult]:
    """Full-text search using tsvector ranking."""
    deps = ctx.deps
    lang = deps.settings.text_search_language

    try:
        if match_count is None:
            match_count = deps.settings.default_match_count
        match_count = min(match_count, deps.settings.max_match_count)

        async with deps.db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    c.id AS chunk_id,
                    c.document_id,
                    c.content,
                    ts_rank_cd(to_tsvector($3, c.content), plainto_tsquery($3, $1)) AS similarity,
                    c.metadata,
                    d.title AS document_title,
                    d.source AS document_source
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE to_tsvector($3, c.content) @@ plainto_tsquery($3, $1)
                ORDER BY similarity DESC
                LIMIT $2;
                """,
                query,
                match_count,
                lang,
            )

        results = []
        for row in rows:
            metadata = row["metadata"] or {}
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except Exception:
                    metadata = {}
            results.append(
                SearchResult(
                    chunk_id=str(row["chunk_id"]),
                    document_id=str(row["document_id"]),
                    content=row["content"],
                    similarity=float(row["similarity"]) if row["similarity"] is not None else 0.0,
                    metadata=metadata,
                    document_title=row["document_title"],
                    document_source=row["document_source"],
                )
            )
        return results
    except Exception:
        logger.exception("text_search_failed")
        return []


async def hybrid_search(
    ctx: RunContext[AgentDependencies],
    query: str,
    match_count: Optional[int] = None,
    text_weight: Optional[float] = None
) -> List[SearchResult]:
    """Hybrid search combining pgvector distance and full-text ranking."""
    deps = ctx.deps
    lang = deps.settings.text_search_language

    try:
        if match_count is None:
            match_count = deps.settings.default_match_count
        if text_weight is None:
            text_weight = deps.user_preferences.get("text_weight", deps.settings.default_text_weight)

        match_count = min(match_count, deps.settings.max_match_count)
        text_weight = max(0.0, min(1.0, text_weight))

        query_embedding = await deps.get_embedding(query)
        embedding_literal = _embedding_to_vector_param(query_embedding)

        async with deps.db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                WITH ranked AS (
                    SELECT
                        c.id AS chunk_id,
                        c.document_id,
                        c.content,
                        (1 - (c.embedding <-> $1::vector)) AS vector_similarity,
                        ts_rank_cd(to_tsvector($5::regconfig, c.content), plainto_tsquery($5::regconfig, $2)) AS text_similarity,
                        c.metadata,
                        d.title AS document_title,
                        d.source AS document_source
                    FROM chunks c
                    JOIN documents d ON d.id = c.document_id
                    ORDER BY c.embedding <-> $1::vector
                    LIMIT $3 * 4
                )
                SELECT
                    chunk_id,
                    document_id,
                    content,
                    COALESCE(vector_similarity, 0) AS vector_similarity,
                    COALESCE(text_similarity, 0) AS text_similarity,
                    (COALESCE(vector_similarity, 0) * (1 - $4)) + (COALESCE(text_similarity, 0) * $4) AS combined_score,
                    metadata,
                    document_title,
                    document_source
                FROM ranked
                ORDER BY combined_score DESC
                LIMIT $3;
                """,
                embedding_literal,
                query,
                match_count,
                text_weight,
                lang,
            )

        results = []
        for row in rows:
            metadata = row["metadata"] or {}
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except Exception:
                    metadata = {}
            combined = float(row["combined_score"]) if row["combined_score"] is not None else 0.0
            combined_norm = 1 / (1 + math.exp(-combined)) if combined != 0 else 0.0
            results.append(
                SearchResult(
                    chunk_id=str(row["chunk_id"]),
                    document_id=str(row["document_id"]),
                    content=row["content"],
                    similarity=combined_norm,
                    metadata=metadata,
                    document_title=row["document_title"],
                    document_source=row["document_source"],
                )
            )
        return results
    except Exception:
        logger.exception("hybrid_search_failed")
        return []
