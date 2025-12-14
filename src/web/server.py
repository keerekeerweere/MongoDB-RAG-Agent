"""FastAPI backend for the PostgreSQL RAG Agent."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from pydantic_ai.ag_ui import StateDeps

from src.agent import rag_agent, RAGState
from src.prompts import build_prompt, list_prompts
from src.settings import load_settings

# Load environment variables
load_dotenv(override=True)

app = FastAPI(title="RAG Agent API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    prompt_id: Optional[str] = None
    language: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    prompt_id: str
    search_type: Optional[str] = None
    tool_log: Optional[list[str]] = None


class FeedbackRequest(BaseModel):
    query: str
    prompt_id: str
    search_type: Optional[str] = None
    helpful: bool


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/prompts")
async def prompts():
    return {"prompts": list_prompts()}


settings = load_settings()

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        prompt_id, prompt_text = build_prompt(
            prompt_id=req.prompt_id,
            language=req.language or settings.text_search_language,
        )

        state = RAGState(last_prompt_id=prompt_id)
        deps = StateDeps[RAGState](state=state)

        result = await rag_agent.run(
            req.message,
            deps=deps,
            instructions=prompt_text,
        )

        # search_type may be set by the tool
        search_type = getattr(deps.state, "last_search_type", None)
        tool_log = getattr(deps.state, "last_tool_log", None)

        return ChatResponse(
            answer=str(result.output),
            prompt_id=prompt_id,
            search_type=search_type,
            tool_log=tool_log,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _record_feedback(query: str, prompt_id: str, search_type: Optional[str], helpful: bool) -> None:
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


@app.post("/api/feedback")
async def feedback(req: FeedbackRequest):
    try:
        _record_feedback(req.query, req.prompt_id, req.search_type, req.helpful)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Convenience entrypoint for uvicorn: uv run uvicorn src.web.server:app --reload
