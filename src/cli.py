#!/usr/bin/env python3
"""Conversational CLI with real-time streaming and tool call visibility."""

import asyncio
from typing import List

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory

from pydantic_ai import Agent
from pydantic_ai.messages import PartDeltaEvent, PartStartEvent, TextPartDelta
from pydantic_ai.ag_ui import StateDeps
from dotenv import load_dotenv
import json
from datetime import datetime
from pathlib import Path
import secrets
from typing import Optional

# Import our agent and dependencies
from src.agent import rag_agent, RAGState
from src.settings import load_settings
from src.prompts import build_prompt

# Load environment variables
load_dotenv(override=True)

console = Console()
session_history = InMemoryHistory()
session = PromptSession(history=session_history)


def record_feedback(query: str, prompt_id: str, search_type: Optional[str], helpful: bool) -> None:
    """Append feedback to a JSONL log."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / "feedback.jsonl"
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "prompt_id": prompt_id,
        "search_type": search_type,
        "helpful": helpful,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


async def stream_agent_interaction(
    user_input: str,
    message_history: List,
    deps: StateDeps[RAGState],
    prompt_id: str,
    prompt_text: str
) -> tuple[str, List]:
    """
    Stream agent interaction with real-time tool call display.

    Args:
        user_input: The user's input text
        message_history: List of ModelRequest/ModelResponse objects for conversation context
        deps: StateDeps with RAG state

    Returns:
        Tuple of (streamed_text, updated_message_history)
    """
    try:
        return await _stream_agent(user_input, deps, message_history, prompt_id, prompt_text)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        return ("", [])


async def _stream_agent(
    user_input: str,
    deps: StateDeps[RAGState],
    message_history: List,
    prompt_id: str,
    prompt_text: str
) -> tuple[str, List]:
    """Stream the agent execution and return response."""

    response_text = ""

    # Stream the agent execution with message history
    async with rag_agent.iter(
        user_input,
        deps=deps,
        message_history=message_history,
        instructions=prompt_text
    ) as run:

        async for node in run:

            # Handle user prompt node
            if Agent.is_user_prompt_node(node):
                pass  # Clean start

            # Handle model request node - stream the thinking process
            elif Agent.is_model_request_node(node):
                # Show assistant prefix at the start
                console.print("[bold blue]Assistant:[/bold blue] ", end="")

                # Stream model request events for real-time text
                async with node.stream(run.ctx) as request_stream:
                    async for event in request_stream:
                        # Handle text part start events
                        if isinstance(event, PartStartEvent) and event.part.part_kind == 'text':
                            initial_text = event.part.content
                            if initial_text:
                                console.print(initial_text, end="")
                                response_text += initial_text

                        # Handle text delta events for streaming
                        elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                            delta_text = event.delta.content_delta
                            if delta_text:
                                console.print(delta_text, end="")
                                response_text += delta_text

                # New line after streaming completes
                console.print()

            # Handle tool calls
            elif Agent.is_call_tools_node(node):
                # Stream tool execution events
                async with node.stream(run.ctx) as tool_stream:
                    async for event in tool_stream:
                        event_type = type(event).__name__

                        if event_type == "FunctionToolCallEvent":
                            # Extract tool name from the event
                            tool_name = "Unknown Tool"
                            args = None

                            # Check if the part attribute contains the tool call
                            if hasattr(event, 'part'):
                                part = event.part

                                # Check for tool name
                                if hasattr(part, 'tool_name'):
                                    tool_name = part.tool_name
                                elif hasattr(part, 'function_name'):
                                    tool_name = part.function_name
                                elif hasattr(part, 'name'):
                                    tool_name = part.name

                                # Check for arguments
                                if hasattr(part, 'args'):
                                    args = part.args
                                elif hasattr(part, 'arguments'):
                                    args = part.arguments

                            console.print(f"  [cyan]Calling tool:[/cyan] [bold]{tool_name}[/bold]")

                            # Show search query if it's a search tool
                            if args and isinstance(args, dict):
                                if 'query' in args:
                                    console.print(f"    [dim]Query:[/dim] {args['query']}")
                                if 'search_type' in args:
                                    console.print(f"    [dim]Type:[/dim] {args['search_type']}")
                                if 'match_count' in args:
                                    console.print(f"    [dim]Results:[/dim] {args['match_count']}")
                            elif args:
                                args_str = str(args)
                                if len(args_str) > 100:
                                    args_str = args_str[:97] + "..."
                                console.print(f"    [dim]Args: {args_str}[/dim]")

                        elif event_type == "FunctionToolResultEvent":
                            console.print(f"  [green]Search completed successfully[/green]")

            # Handle end node
            elif Agent.is_end_node(node):
                pass

    # Get new messages from this run to add to history
    new_messages = run.result.new_messages()

    # Get final output
    final_output = run.result.output if hasattr(run.result, 'output') else str(run.result)
    response = response_text.strip() or final_output

    # Return both streamed text and new messages
    return (response, new_messages)


def display_welcome():
    """Display welcome message with configuration info."""
    settings = load_settings()

    welcome = Panel(
        "[bold blue]PostgreSQL RAG Agent[/bold blue]\n\n"
        "[green]Intelligent knowledge base search with pgvector hybrid search[/green]\n"
        f"[dim]LLM: {settings.llm_model}[/dim]\n\n"
        "[dim]Type 'exit' to quit, 'info' for system info, 'clear' to clear screen[/dim]",
        style="blue",
        padding=(1, 2)
    )
    console.print(welcome)
    console.print()


async def main():
    """Main conversation loop."""

    # Show welcome
    display_welcome()

    # Load settings once (shared)
    settings = load_settings()

    # Create the state that the agent will use
    state = RAGState()

    # Create StateDeps wrapper with the state
    deps = StateDeps[RAGState](state=state)

    console.print("[bold green]✓[/bold green] Search system initialized\n")

    # Initialize message history with proper Pydantic AI message objects
    message_history = []

    try:
        while True:
            try:
                # Get user input (async-friendly prompt_toolkit)
                user_input = (await session.prompt_async("[bold green]You> ")).strip()

                # Handle special commands
                if user_input.lower() in ['exit', 'quit', 'q']:
                    console.print("\n[yellow]👋 Goodbye![/yellow]")
                    break

                elif user_input.lower() == 'info':
                    console.print(Panel(
                        f"[cyan]LLM Provider:[/cyan] {settings.llm_provider}\n"
                        f"[cyan]LLM Model:[/cyan] {settings.llm_model}\n"
                        f"[cyan]Embedding Model:[/cyan] {settings.embedding_model}\n"
                        f"[cyan]Embedding Dimension:[/cyan] {settings.embedding_dimension}\n"
                        f"[cyan]Default Match Count:[/cyan] {settings.default_match_count}\n"
                        f"[cyan]Default Text Weight:[/cyan] {settings.default_text_weight}\n"
                        f"[cyan]Text Search Language:[/cyan] {settings.text_search_language}",
                        title="System Configuration",
                        border_style="magenta"
                    ))
                    continue

                elif user_input.lower() == 'clear':
                    console.clear()
                    display_welcome()
                    continue

                if not user_input:
                    continue

                # Pick a prompt variant for this turn
                prompt_id, prompt_text = build_prompt(
                    prompt_id=None,
                    language=settings.text_search_language
                )
                deps.state.last_prompt_id = prompt_id

                # Stream the interaction and get response
                response_text, new_messages = await stream_agent_interaction(
                    user_input,
                    message_history,
                    deps,
                    prompt_id,
                    prompt_text
                )

                # Add new messages to history (includes both user prompt and agent response)
                message_history.extend(new_messages)

                # Add spacing after response
                console.print()

                # Ask for feedback
                feedback = Prompt.ask("[dim]Helpful? (y/n, enter to skip)", default="").strip().lower()
                if feedback in ("y", "n"):
                    record_feedback(
                        query=user_input,
                        prompt_id=prompt_id,
                        search_type=deps.state.last_search_type,
                        helpful=feedback == "y",
                    )

            except KeyboardInterrupt:
                console.print("\n[yellow]Use 'exit' to quit[/yellow]")
                continue

            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                import traceback
                traceback.print_exc()
                continue

    finally:
        console.print("\n[dim]Goodbye![/dim]")


if __name__ == "__main__":
    asyncio.run(main())
