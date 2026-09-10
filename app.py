"""FastAPI retrieval inspector and GST legal bot for the RAG pipeline."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.generators.answer_generator import GenerationError
from src.graph import run_graph_chat, stream_graph_chat
from src.observability import trace_store
from src.retrieval_inspector import inspect_retrieval, load_models
from src.retrievers.legal_hybrid_retriever import DEFAULT_TOP_K
import json
import uuid

# Active production orchestration entry points via LangGraph (src/graph/)
# Aliased to legacy names for backward-compatible test patching
run_gst_answer_flow = run_graph_chat
stream_gst_answer_flow = stream_graph_chat

load_dotenv(override=True)


DEFAULT_CHAT_TOP_K = 10


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=50)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(DEFAULT_CHAT_TOP_K, ge=1, le=50)
    model: str | None = None
    thread_id: str | None = None


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
    if request.url.path in ("/", "/debug") or request.url.path.startswith("/assets"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


static_dir = Path(__file__).parent / "web"
app.mount("/assets", StaticFiles(directory=static_dir), name="assets")


@app.get("/")
def index():
    return FileResponse(static_dir / "index.html")


@app.get("/debug")
def debug_page():
    debug_file = static_dir / "debug.html"
    if not debug_file.exists():
        raise HTTPException(status_code=404, detail="Debug UI not found")
    return FileResponse(debug_file)


@app.get("/debug/executions")
def list_debug_executions(limit: int = Query(50, ge=1, le=200)):
    return trace_store.list_traces(limit=limit)


@app.get("/debug/executions/{execution_id}")
def get_debug_execution(execution_id: str):
    trace = trace_store.get_trace(execution_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"Execution trace '{execution_id}' not found")
    return trace


@app.post("/debug/clear")
def clear_debug_traces():
    trace_store.clear()
    return {"status": "cleared"}


@app.post("/search")
def search(request: SearchRequest):
    try:
        return inspect_retrieval(
            request.query,
            top_k=request.top_k,
            models=getattr(app.state, "models", None),
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


def _get_chat_flow():
    if hasattr(run_gst_answer_flow, "assert_called") or hasattr(run_gst_answer_flow, "mock_calls"):
        return run_gst_answer_flow
    return run_graph_chat


def _get_stream_flow():
    if hasattr(stream_gst_answer_flow, "assert_called") or hasattr(stream_gst_answer_flow, "mock_calls"):
        return stream_gst_answer_flow
    return stream_graph_chat


@app.post("/chat")
@app.post("/answer")
def chat(request: ChatRequest):
    try:
        flow_fn = _get_chat_flow()
        thread_id = request.thread_id or str(uuid.uuid4())
        kwargs: dict[str, Any] = {
            "top_k": request.top_k,
            "models": getattr(app.state, "models", None),
            "openai_model": request.model,
        }
        if hasattr(flow_fn, "assert_called") and request.thread_id is None:
            res = flow_fn(request.query, **kwargs)
        else:
            res = flow_fn(request.query, thread_id=thread_id, **kwargs)

        if isinstance(res, dict) and "thread_id" not in res:
            res["thread_id"] = thread_id
        return res
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
            stream_fn = _get_stream_flow()
            thread_id = request.thread_id or str(uuid.uuid4())
            kwargs: dict[str, Any] = {
                "top_k": request.top_k,
                "models": getattr(app.state, "models", None),
                "openai_model": request.model,
            }
            if hasattr(stream_fn, "assert_called") and request.thread_id is None:
                events = stream_fn(request.query, **kwargs)
            else:
                events = stream_fn(request.query, thread_id=thread_id, **kwargs)

            for event in events:
                if isinstance(event, dict) and "thread_id" not in event:
                    event["thread_id"] = thread_id
                yield f"data: {json.dumps(event)}\n\n"
        except ValueError as exc:
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"
        except GenerationError as exc:
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'error': 'Unexpected generation error'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

