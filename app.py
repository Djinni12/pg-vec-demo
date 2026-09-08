"""FastAPI retrieval inspector and GST legal bot for the RAG pipeline."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.generators.answer_generator import (
    GenerationError,
    run_gst_answer_flow,
    stream_gst_answer_flow,
)
from src.retrieval_inspector import inspect_retrieval, load_models
from src.retrievers.legal_hybrid_retriever import DEFAULT_TOP_K
import json

load_dotenv(override=True)


DEFAULT_CHAT_TOP_K = 5


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=50)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(DEFAULT_CHAT_TOP_K, ge=1, le=50)
    model: str | None = None


class RateSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    limit: int = Field(5, ge=1, le=50)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models = load_models()
    yield


app = FastAPI(title="GST Bot and Retrieval Inspector", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_cache_control_header(request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/assets"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


static_dir = Path(__file__).parent / "web"
app.mount("/assets", StaticFiles(directory=static_dir), name="assets")


@app.get("/")
def index():
    return FileResponse(static_dir / "index.html")


@app.post("/search")
def search(request: SearchRequest):
    try:
        return inspect_retrieval(
            request.query,
            top_k=request.top_k,
            models=app.state.models,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/rates")
def rates(request: RateSearchRequest):
    try:
        from src.retrievers.rate_retriever import retrieve_rates
        return retrieve_rates(request.query, limit=request.limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/chat")
@app.post("/answer")
def chat(request: ChatRequest):
    try:
        return run_gst_answer_flow(
            request.query,
            top_k=request.top_k,
            models=app.state.models,
            openai_model=request.model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except GenerationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    def event_stream():
        try:
            for event in stream_gst_answer_flow(
                request.query,
                top_k=request.top_k,
                models=app.state.models,
                openai_model=request.model,
            ):
                yield f"data: {json.dumps(event)}\n\n"
        except ValueError as exc:
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"
        except GenerationError as exc:
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'error': 'Unexpected generation error'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

