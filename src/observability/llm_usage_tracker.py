"""Centralized LLM token-usage and cost tracking for GST Bot.

Tracks per-request token usage, cached prompt tokens, output tokens,
latency, and estimated costs across all LLM stages (planner, synthesis,
grounded reasoning, etc.).

Ensures concurrency safety across requests via request-scoped storage
and ContextVars.
"""

from __future__ import annotations

from collections import deque
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import threading
from typing import Any, Optional


# -----------------------------------------------------------------------------
# Pricing Configuration (Centralized USD per 1,000,000 tokens)
# -----------------------------------------------------------------------------
MODEL_PRICING: dict[str, dict[str, float]] = {
    # GPT-4o-mini
    "gpt-4o-mini": {
        "input": 0.15,
        "cached_input": 0.075,
        "output": 0.60,
    },
    # GPT-4o
    "gpt-4o": {
        "input": 2.50,
        "cached_input": 1.25,
        "output": 10.00,
    },
    # GPT-4-turbo
    "gpt-4-turbo": {
        "input": 10.00,
        "cached_input": 5.00,
        "output": 30.00,
    },
    # GPT-3.5-turbo
    "gpt-3.5-turbo": {
        "input": 0.50,
        "cached_input": 0.50,
        "output": 1.50,
    },
    # o1-mini
    "o1-mini": {
        "input": 1.10,
        "cached_input": 0.55,
        "output": 4.40,
    },
    # o1-preview
    "o1-preview": {
        "input": 15.00,
        "cached_input": 7.50,
        "output": 60.00,
    },
    # o3-mini
    "o3-mini": {
        "input": 1.10,
        "cached_input": 0.55,
        "output": 4.40,
    },
    # Gemini models
    "gemini-3.1-flash-lite": {
        "input": 0.0375,
        "cached_input": 0.01,
        "output": 0.15,
    },
    "gemini-2.0-flash-lite": {
        "input": 0.0375,
        "cached_input": 0.01,
        "output": 0.15,
    },
    "gemini-2.0-flash": {
        "input": 0.075,
        "cached_input": 0.01875,
        "output": 0.30,
    },
    "gemini-1.5-flash": {
        "input": 0.075,
        "cached_input": 0.01875,
        "output": 0.30,
    },
    "gemini-1.5-pro": {
        "input": 1.25,
        "cached_input": 0.3125,
        "output": 5.00,
    },
}


def get_model_pricing(model_name: str) -> Optional[dict[str, float]]:
    """Retrieve pricing rates for a model, supporting prefix & version matching."""
    if not model_name or not isinstance(model_name, str):
        return None
    normalized = model_name.strip().lower()

    # 1. Exact match
    if normalized in MODEL_PRICING:
        return MODEL_PRICING[normalized]

    # 2. Match longest prefix (e.g. 'gpt-4o-mini-2024-07-18' matches 'gpt-4o-mini')
    matching_keys = [k for k in MODEL_PRICING if normalized.startswith(k)]
    if matching_keys:
        best_match = max(matching_keys, key=len)
        return MODEL_PRICING[best_match]

    return None


def calculate_cost(
    model: str,
    input_tokens: int,
    cached_input_tokens: int = 0,
    output_tokens: int = 0,
) -> Optional[float]:
    """Calculate estimated cost in USD based on uncached/cached inputs and outputs.

    Formula:
        uncached_input = max(0, input_tokens - cached_input_tokens)
        cost = (uncached_input * input_rate + cached_input_tokens * cached_rate + output_tokens * output_rate) / 1,000,000

    Returns None if model is not found in pricing dictionary.
    """
    pricing = get_model_pricing(model)
    if not pricing:
        return None

    inp_rate = pricing["input"]
    cached_rate = pricing.get("cached_input", inp_rate)
    out_rate = pricing["output"]

    cached_clean = max(0, int(cached_input_tokens or 0))
    total_inp = max(0, int(input_tokens or 0))
    uncached_inp = max(0, total_inp - cached_clean)
    out_clean = max(0, int(output_tokens or 0))

    cost = (
        (uncached_inp * inp_rate)
        + (cached_clean * cached_rate)
        + (out_clean * out_rate)
    ) / 1_000_000.0

    return round(cost, 6)


# -----------------------------------------------------------------------------
# Token Extraction Helper
# -----------------------------------------------------------------------------
def extract_usage_metadata(response: Any) -> dict[str, int]:
    """Safely extract input, cached, output, and total tokens from SDK response or dict.

    Supports:
    - OpenAI ChatCompletion object (response.usage)
    - LangChain / dict format (usage / usage_metadata)
    - None or objects without usage metadata
    """
    empty_result = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    if response is None:
        return empty_result

    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage") or response.get("usage_metadata")
    if usage is None and hasattr(response, "get"):
        usage = response.get("usage")

    if usage is None:
        # Check if response itself has prompt_tokens or similar directly
        if hasattr(response, "prompt_tokens"):
            usage = response
        elif isinstance(response, dict) and "prompt_tokens" in response:
            usage = response

    if usage is None:
        return empty_result

    # Handle dictionary usage
    if isinstance(usage, dict):
        inp = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        out = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        tot = int(usage.get("total_tokens") or (inp + out))

        cached = 0
        details = usage.get("prompt_tokens_details") or usage.get("input_token_details")
        if isinstance(details, dict):
            cached = int(details.get("cached_tokens") or 0)
        elif hasattr(details, "cached_tokens"):
            cached = int(getattr(details, "cached_tokens", 0) or 0)
        elif "cached_tokens" in usage:
            cached = int(usage.get("cached_tokens") or 0)

        return {
            "input_tokens": inp,
            "cached_input_tokens": cached,
            "output_tokens": out,
            "total_tokens": tot,
        }

    # Handle SDK object usage
    inp = int(getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0) or 0)
    out = int(getattr(usage, "completion_tokens", 0) or getattr(usage, "output_tokens", 0) or 0)
    tot = int(getattr(usage, "total_tokens", 0) or (inp + out))

    cached = 0
    details = getattr(usage, "prompt_tokens_details", None) or getattr(usage, "input_token_details", None)
    if details:
        if isinstance(details, dict):
            cached = int(details.get("cached_tokens", 0) or 0)
        else:
            cached = int(getattr(details, "cached_tokens", 0) or 0)
    elif hasattr(usage, "cached_tokens"):
        cached = int(getattr(usage, "cached_tokens", 0) or 0)

    return {
        "input_tokens": inp,
        "cached_input_tokens": cached,
        "output_tokens": out,
        "total_tokens": tot,
    }


# -----------------------------------------------------------------------------
# Request Context Variable for thread-safe request propagation
# -----------------------------------------------------------------------------
_current_request_id: ContextVar[Optional[str]] = ContextVar("current_request_id", default=None)


def set_current_request_id(request_id: Optional[str]) -> None:
    """Set the active request_id for the current task/thread context."""
    _current_request_id.set(request_id)


def get_current_request_id() -> Optional[str]:
    """Retrieve the active request_id from current context."""
    return _current_request_id.get()


# -----------------------------------------------------------------------------
# Call Record Data Class
# -----------------------------------------------------------------------------
@dataclass
class LLMCallRecord:
    request_id: str
    stage: str
    model: str
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: Optional[float] = None
    latency_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# -----------------------------------------------------------------------------
# Central Usage Tracker Store
# -----------------------------------------------------------------------------
class LLMUsageTracker:
    """Centralized, request-scoped tracker for token usage and estimated API cost."""

    def __init__(self, max_requests: int = 500) -> None:
        self.max_requests = max_requests
        self._calls: dict[str, list[LLMCallRecord]] = {}
        self._order: deque[str] = deque(maxlen=max_requests)
        self._lock = threading.Lock()

    def record_call(
        self,
        *,
        request_id: Optional[str] = None,
        stage: str,
        model: str,
        response: Any = None,
        input_tokens: Optional[int] = None,
        cached_input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        latency_ms: float = 0.0,
    ) -> LLMCallRecord:
        """Record an LLM call for a request with automatic token extraction and pricing."""
        req_id = request_id or get_current_request_id() or "unassigned"

        if response is not None:
            extracted = extract_usage_metadata(response)
            inp = input_tokens if input_tokens is not None else extracted["input_tokens"]
            cached = cached_input_tokens if cached_input_tokens is not None else extracted["cached_input_tokens"]
            out = output_tokens if output_tokens is not None else extracted["output_tokens"]
            tot = total_tokens if total_tokens is not None else extracted["total_tokens"]
        else:
            inp = input_tokens or 0
            cached = cached_input_tokens or 0
            out = output_tokens or 0
            tot = total_tokens if total_tokens is not None else (inp + out)

        cost = calculate_cost(
            model=model,
            input_tokens=inp,
            cached_input_tokens=cached,
            output_tokens=out,
        )

        record = LLMCallRecord(
            request_id=req_id,
            stage=stage,
            model=model,
            input_tokens=inp,
            cached_input_tokens=cached,
            output_tokens=out,
            total_tokens=tot,
            estimated_cost_usd=cost,
            latency_ms=round(float(latency_ms), 2),
        )

        with self._lock:
            if req_id not in self._calls:
                if len(self._order) >= self.max_requests:
                    oldest = self._order.popleft()
                    self._calls.pop(oldest, None)
                self._calls[req_id] = []
                self._order.append(req_id)
            self._calls[req_id].append(record)

        return record

    def get_request_usage(self, request_id: str) -> dict[str, Any]:
        """Aggregate all LLM calls recorded under request_id."""
        with self._lock:
            calls = list(self._calls.get(request_id, []))

        total_input = sum(c.input_tokens for c in calls)
        total_cached = sum(c.cached_input_tokens for c in calls)
        total_output = sum(c.output_tokens for c in calls)
        total_tokens = sum(c.total_tokens for c in calls)
        total_latency = sum(c.latency_ms for c in calls)

        known_costs = [c.estimated_cost_usd for c in calls if c.estimated_cost_usd is not None]
        has_unknown = any(c.estimated_cost_usd is None and c.total_tokens > 0 for c in calls)

        if calls and has_unknown and not known_costs:
            total_cost_usd = None
        elif known_costs:
            total_cost_usd = round(sum(known_costs), 6)
        else:
            total_cost_usd = 0.0 if calls else None

        stage_breakdown = [c.to_dict() for c in calls]

        return {
            "request_id": request_id,
            "total_calls": len(calls),
            "total_input_tokens": total_input,
            "total_cached_input_tokens": total_cached,
            "total_output_tokens": total_output,
            "total_tokens": total_tokens,
            "total_cost_usd": total_cost_usd,
            "total_latency_ms": round(total_latency, 2),
            "calls": stage_breakdown,
            "formatted_debug": format_llm_usage_debug(request_id, calls),
        }

    def clear(self) -> None:
        """Clear all stored calls."""
        with self._lock:
            self._calls.clear()
            self._order.clear()


def format_llm_usage_debug(request_id: str, calls: list[LLMCallRecord]) -> str:
    """Format human-readable developer/debug string for LLM usage and cost."""
    lines = [
        f"🪙 LLM Usage & Cost Breakdown [Request: {request_id}]",
        "-------------------------------------------------------",
    ]
    if not calls:
        lines.append("  No LLM calls recorded for this request (local/deterministic execution).")
        lines.append("-------------------------------------------------------")
        return "\n".join(lines)

    for c in calls:
        cost_str = f"${c.estimated_cost_usd:.6f}" if c.estimated_cost_usd is not None else "N/A (pricing unavailable)"
        lines.append(f"Stage: {c.stage} (model: {c.model})")
        lines.append(f"  Tokens: {c.input_tokens:,} input ({c.cached_input_tokens:,} cached) + {c.output_tokens:,} output = {c.total_tokens:,} total")
        lines.append(f"  Latency: {c.latency_ms:.1f} ms")
        lines.append(f"  Cost: {cost_str}")

    total_input = sum(c.input_tokens for c in calls)
    total_cached = sum(c.cached_input_tokens for c in calls)
    total_output = sum(c.output_tokens for c in calls)
    total_tokens = sum(c.total_tokens for c in calls)
    total_latency = sum(c.latency_ms for c in calls)
    known_costs = [c.estimated_cost_usd for c in calls if c.estimated_cost_usd is not None]
    total_cost_str = f"${sum(known_costs):.6f} USD" if known_costs else "N/A"

    lines.append("-------------------------------------------------------")
    lines.append(f"Total Tokens: {total_tokens:,} ({total_input:,} input [{total_cached:,} cached] + {total_output:,} output)")
    lines.append(f"Total LLM Latency: {total_latency:.1f} ms")
    lines.append(f"Total Estimated Cost: {total_cost_str}")
    return "\n".join(lines)


# Global singleton tracker instance
llm_usage_tracker = LLMUsageTracker(max_requests=500)


def record_llm_call(
    *,
    request_id: Optional[str] = None,
    stage: str,
    model: str,
    response: Any = None,
    input_tokens: Optional[int] = None,
    cached_input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    latency_ms: float = 0.0,
) -> LLMCallRecord:
    """Convenience helper to record an LLM call in the global usage tracker."""
    return llm_usage_tracker.record_call(
        request_id=request_id,
        stage=stage,
        model=model,
        response=response,
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        latency_ms=latency_ms,
    )


def get_request_usage(request_id: str) -> dict[str, Any]:
    """Convenience helper to get request usage breakdown from the global tracker."""
    return llm_usage_tracker.get_request_usage(request_id)
