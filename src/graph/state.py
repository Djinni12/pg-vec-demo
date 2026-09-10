"""LangGraph State definition for the GST Assistant pipeline.

Defines the shared state dictionary passed across all nodes in the graph.
"""

from __future__ import annotations

from typing import Annotated, Any, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class GSTGraphState(TypedDict, total=False):
    """Shared state dictionary passed across all nodes in the GST LangGraph.

    Attributes:
        messages: LangGraph-managed conversation message history.
        user_query: The incoming user query or question.
        needs_legal: Planner flag indicating statutory/legal RAG retrieval is needed.
        needs_rate: Planner flag indicating structured GST rate lookup is needed.
        needs_direct_reasoning: Planner flag indicating direct arithmetic or reasoning is needed.
        clean_queries: Cleaned/extracted subqueries for specialized tools (e.g. rate_query, legal_query).
        user_premises: Extracted user-supplied values (assumed rates, taxable amounts, discounts, ITC balances).
        legal_results: Retrieved legal document chunks (Sections, Rules, Forms) with full metadata.
        rate_results: Structured tariff items retrieved from the verified rate database (HSN, rates, notifs).
        reasoning_result: Text or calculation summary from the direct reasoning node.
        calculation_result: Structured deterministic calculation result dictionary.
        final_answer: The final grounded response generated for the user.
        sources: Citation source metadata preserved from legal retrieval and rate lookup.
        route: Resolved capability route string (e.g. 'legal', 'rate', 'mixed', 'direct').
        plan: Raw capability plan dictionary from the planner.
        timings_ms: Execution timing breakdown in milliseconds.
        model_used: Name of the LLM model used for synthesis.
        error: Optional error message if any node encountered an error.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    user_query: str
    needs_legal: bool
    needs_rate: bool
    needs_notification: bool
    needs_direct_reasoning: bool
    needs_calculation: bool
    needs_grounded_reasoning: bool
    clean_queries: dict[str, Optional[str]]
    user_premises: dict[str, Any]
    legal_results: list[dict[str, Any]]
    rate_results: list[dict[str, Any]]
    notification_results: list[dict[str, Any]]
    reasoning_result: Optional[str]
    calculation_inputs: Optional[dict[str, Any]]
    calculation_result: Optional[dict[str, Any]]
    final_answer: str
    sources: list[dict[str, Any]]
    route: Optional[str]
    plan: Optional[dict[str, Any]]
    timings_ms: Optional[dict[str, float]]
    model_used: Optional[str]
    error: Optional[str]
    execution_id: Optional[str]
    thread_id: Optional[str]

