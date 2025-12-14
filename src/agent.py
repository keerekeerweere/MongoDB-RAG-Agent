"""Main PostgreSQL RAG agent implementation with shared state."""

from pydantic_ai import Agent, RunContext
from pydantic import BaseModel
from typing import Optional, List

from pydantic_ai.ag_ui import StateDeps

from src.providers import get_llm_model
from src.dependencies import AgentDependencies
from src.tools import semantic_search, hybrid_search, text_search


class RAGState(BaseModel):
    """Shared state for the RAG agent."""

    last_prompt_id: Optional[str] = None
    last_search_type: Optional[str] = None
    last_tool_log: List[str] = []


# Create the RAG agent with AGUI support (system prompt set per run)
rag_agent = Agent(
    get_llm_model(),
    deps_type=StateDeps[RAGState],
)


@rag_agent.tool
async def search_knowledge_base(
    ctx: RunContext[StateDeps[RAGState]],
    query: str,
    match_count: Optional[int] = 10,
    search_type: Optional[str] = "hybrid"
) -> str:
    """
    Search the knowledge base for relevant information.
    """
    try:
        # Initialize database connection
        agent_deps = AgentDependencies()
        await agent_deps.initialize()

        # Create a context wrapper for the search tools
        class DepsWrapper:
            def __init__(self, deps):
                self.deps = deps

        deps_ctx = DepsWrapper(agent_deps)

        # Perform the search based on type
        if hasattr(ctx, "deps") and hasattr(ctx.deps, "state"):
            ctx.deps.state.last_tool_log = [
                f"search_knowledge_base search_type={search_type} match_count={match_count}"
            ]

        if search_type == "hybrid":
            results = await hybrid_search(
                ctx=deps_ctx,
                query=query,
                match_count=match_count
            )
        elif search_type == "semantic":
            results = await semantic_search(
                ctx=deps_ctx,
                query=query,
                match_count=match_count
            )
        else:
            results = await text_search(
                ctx=deps_ctx,
                query=query,
                match_count=match_count
            )

        # Record last search type for feedback logging
        if hasattr(ctx, "deps") and hasattr(ctx.deps, "state"):
            ctx.deps.state.last_search_type = search_type
            if not ctx.deps.state.last_tool_log:
                ctx.deps.state.last_tool_log = []
            ctx.deps.state.last_tool_log.append(
                f"results returned={len(results)}"
            )

        # Clean up
        await agent_deps.cleanup()

        # Format results as a simple string
        if not results:
            return "No relevant information found in the knowledge base."

        # Build a formatted response
        response_parts = [f"Found {len(results)} relevant documents:\n"]

        for i, result in enumerate(results, 1):
            response_parts.append(f"\n--- Document {i}: {result.document_title} (relevance: {result.similarity:.2f}) ---")
            response_parts.append(result.content)

        return "\n".join(response_parts)

    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"
