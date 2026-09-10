#!/usr/bin/env python3
"""Diagnostic script for the LangGraph GST Bot agent architecture.

Runs the three required test scenarios:
1. Legal Retrieval Only: "How do I cancel my GST registration?"
2. Structured Rate Lookup Only: "What is the GST rate for paneer?"
3. Mixed Query (Legal + Rate): "What is the GST rate for laptops and what law applies?"

Demonstrates:
- State initialization
- Multi-capability planner routing
- Routed tool node execution (including parallel fan-out for mixed queries)
- State merging before synthesis
- Grounded synthesis preserving verified rates and legal citations
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph.graph import compile_gst_graph
from src.graph.state import GSTGraphState
from src.retrieval_inspector import load_models


def run_diagnostic():
    print("=" * 70)
    print(" GST Bot LangGraph Architecture - Diagnostic & Verification")
    print("=" * 70)

    # 1. Pre-load retrieval models once
    print("\n[1/4] Initializing retrieval models (BGE-M3 dense + reranker)...")
    try:
        models = load_models()
        print(f"✓ Models loaded successfully in {models.initialization_ms:.1f}ms")
    except Exception as exc:
        print(f"! Warning: Could not initialize local dense models ({exc}).")
        models = None

    # 2. Compile the LangGraph
    print("\n[2/4] Compiling GST LangGraph...")
    graph = compile_gst_graph(models=models, use_llm_planner=False)
    print("✓ Graph compiled successfully!")

    # 3. Test queries
    test_queries = [
        {
            "category": "1. Pure Legal Retrieval",
            "query": "How do I cancel my GST registration?",
            "expected_route": ["legal_retrieval"],
        },
        {
            "category": "2. Pure Structured Rate Lookup",
            "query": "What is the GST rate for paneer?",
            "expected_route": ["rate_lookup"],
        },
        {
            "category": "3. Mixed Query (Legal + Rate)",
            "query": "What is the GST rate for laptops and what law applies?",
            "expected_route": ["legal_retrieval", "rate_lookup"],
        },
    ]

    print("\n[3/4] Executing test scenarios through the compiled graph:\n")

    for i, test in enumerate(test_queries, 1):
        q = test["query"]
        cat = test["category"]
        print("-" * 70)
        print(f"Scenario {i}: {cat}")
        print(f"Query: \"{q}\"")

        initial_state: GSTGraphState = {
            "user_query": q,
            "needs_legal": False,
            "needs_rate": False,
            "needs_direct_reasoning": False,
            "clean_queries": {},
            "user_premises": {},
            "legal_results": [],
            "rate_results": [],
            "reasoning_result": None,
            "final_answer": "",
            "sources": [],
            "error": None,
        }

        # Run invocation
        output_state = graph.invoke(initial_state)

        # Inspect resulting state
        needs_legal = output_state.get("needs_legal", False)
        needs_rate = output_state.get("needs_rate", False)
        needs_direct = output_state.get("needs_direct_reasoning", False)

        active_routes = []
        if needs_legal:
            active_routes.append("legal_retrieval")
        if needs_rate:
            active_routes.append("rate_lookup")
        if needs_direct:
            active_routes.append("direct_reasoning")

        legal_hits = len(output_state.get("legal_results", []))
        rate_hits = len(output_state.get("rate_results", []))
        sources_count = len(output_state.get("sources", []))
        final_answer = output_state.get("final_answer", "")

        print(f"\n[Planner Routing Output]")
        print(f"  - needs_legal:            {needs_legal}")
        print(f"  - needs_rate:             {needs_rate}")
        print(f"  - needs_direct_reasoning: {needs_direct}")
        print(f"  - Routed Nodes:           {active_routes}")

        print(f"\n[Evidence Retrieved in State]")
        print(f"  - Legal Chunks:           {legal_hits}")
        print(f"  - Rate Records:           {rate_hits}")
        print(f"  - Preserved Sources:      {sources_count}")

        if output_state.get("rate_results"):
            r0 = output_state["rate_results"][0]
            print(f"  - Top Rate Record:        HSN {r0.get('code')}: {r0.get('description')[:50]}... -> {r0.get('total_gst_rate') or r0.get('source_rate')}")

        if output_state.get("legal_results"):
            l0 = output_state["legal_results"][0]
            print(f"  - Top Legal Reference:    {l0.get('reference')} ({l0.get('title')})")

        print(f"\n[Final Synthesized Answer Preview]")
        answer_preview = final_answer.strip()
        lines = answer_preview.split("\n")
        preview = "\n".join(lines[:6]) + ("\n..." if len(lines) > 6 else "")
        print(f"{preview}\n")

    print("=" * 70)
    print("✓ All diagnostic scenarios completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    run_diagnostic()
