"""Conditional routing logic for the GST LangGraph.

Evaluates multi-capability planner output stored in the shared state
and directs execution through distinct, dependency-ordered runtime stages:
- retrieval nodes: 'legal_retrieval', 'rate_lookup'
- grounded reasoning: 'grounded_reasoning'
- direct reasoning: 'direct_reasoning'
- deterministic calculation: 'calculation'
- final grounded synthesis: 'synthesis'
"""

from __future__ import annotations

from typing import Sequence

from src.graph.state import GSTGraphState


def route_capabilities(state: GSTGraphState) -> Sequence[str]:
    """Inspects state capability flags and determines initial tool nodes to execute.

    Supports multi-capability queries:
    - If needs_legal is True: includes 'legal_retrieval'
    - If needs_rate is True: includes 'rate_lookup'
    - If needs_notification is True and neither legal nor rate is required:
      includes 'notification_support'
    - If no retrievals are required:
      - If needs_calculation is True: routes to 'calculation'
      - If needs_direct_reasoning is True: routes to 'direct_reasoning'
      - Otherwise: routes directly to 'synthesis'
    - When retrievals are active, reasoning and calculation are deferred
      until retrieval evidence is populated in state.

    Returns:
        List of node names to execute initially.
    """
    targets: list[str] = []

    if state.get("needs_legal"):
        targets.append("legal_retrieval")

    if state.get("needs_rate"):
        targets.append("rate_lookup")

    if not targets and state.get("needs_notification"):
        targets.append("notification_support")

    if not targets:
        if state.get("needs_direct_reasoning"):
            return ["direct_reasoning"]
        if state.get("needs_calculation"):
            return ["calculation"]
        return ["synthesis"]

    return targets


def route_post_retrieval(state: GSTGraphState) -> str:
    """Conditional edge evaluating next stage after retrieval nodes complete.

    Ensures rate/legal-dependent calculations and reasoning wait for required retrieval results:
    - If needs_grounded_reasoning is True (or needs_direct_reasoning in dependent query): routes to 'grounded_reasoning'
    - Else if needs_calculation is True: routes to 'calculation'
    - Otherwise: routes directly to 'synthesis'.
    """
    if state.get("needs_grounded_reasoning") or state.get("needs_direct_reasoning"):
        return "grounded_reasoning"
    if state.get("needs_calculation"):
        return "calculation"
    return "synthesis"


def should_fallback_to_web_search(state: GSTGraphState) -> bool:
    """Evaluates whether local retrievals reported missing or insufficient evidence.

    Web search runs only as a fallback for retrieval queries (rate, legal, or notification)
    when local sources yield no results. Never runs for pure calculation or direct reasoning.
    Runs at most once per graph execution.
    """
    if state.get("web_search_attempted"):
        return False

    is_retrieval_query = (
        bool(state.get("needs_rate"))
        or bool(state.get("needs_legal"))
        or bool(state.get("needs_notification"))
    )
    if not is_retrieval_query:
        return False

    rate_missing = bool(state.get("needs_rate")) and not bool(state.get("rate_results"))
    legal_missing = (
        bool(state.get("needs_legal"))
        and not bool(state.get("legal_results"))
        and not bool(state.get("notification_results"))
    )
    notif_missing = (
        bool(state.get("needs_notification"))
        and not bool(state.get("notification_results"))
        and not bool(state.get("rate_results"))
        and not bool(state.get("legal_results"))
    )

    return rate_missing or legal_missing or notif_missing


def route_post_notification_support(state: GSTGraphState) -> str:
    """Conditional edge evaluating next stage after notification support completes.

    Triggers web_search only when local retrievals reported missing evidence.
    Otherwise delegates to standard post-retrieval routing.
    """
    if should_fallback_to_web_search(state):
        return "web_search"
    return route_post_retrieval(state)


def route_post_web_search(state: GSTGraphState) -> str:
    """Conditional edge after web_search fallback node completes.

    Directs to grounded_reasoning, calculation, or synthesis based on state requirements.
    """
    return route_post_retrieval(state)


def route_post_grounded_reasoning(state: GSTGraphState) -> str:
    """Conditional edge after grounded_reasoning node finishes.

    - If needs_calculation is True: routes to 'calculation'
    - Otherwise: routes directly to 'synthesis'.
    """
    if state.get("needs_calculation"):
        return "calculation"
    return "synthesis"


def route_post_direct_reasoning(state: GSTGraphState) -> str:
    """Conditional edge after direct_reasoning node finishes.

    - If needs_calculation is True: routes to 'calculation'
    - Otherwise: routes directly to 'synthesis'.
    """
    if state.get("needs_calculation"):
        return "calculation"
    return "synthesis"

