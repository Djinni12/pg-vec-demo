"""LEGACY / DEPRECATED: Pre-LangGraph procedural orchestration pipeline.

This package contains historical orchestration modules retained strictly for reference.
It is NOT part of the active production runtime.
The active production orchestration layer is LangGraph in `src/graph/`.

DO NOT import or call this module in the active API, UI, or LangGraph runtime.
"""

from legacy.pre_langgraph_orchestration import (
    run_gst_answer_flow,
    stream_gst_answer_flow,
)

__all__ = [
    "run_gst_answer_flow",
    "stream_gst_answer_flow",
]
