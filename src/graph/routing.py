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


# Notification support uses the same post-retrieval routing logic to evaluate next stage
route_post_notification_support = route_post_retrieval


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

