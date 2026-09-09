"""Structured Multi-Capability Planner for GST Assistant.

Analyzes user queries across multiple languages (English, Gujarati, Hindi)
and generates a structured capability plan without mutual-route exclusivity.

The planner decides WHAT capabilities and tools are required.
It does NOT invent tax rates, HSN codes, statutory text, or calculation answers.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """You are a GST Multi-Capability Query Planner.
Analyze the user query and determine which capabilities and tools are required to answer it.
Do NOT answer the query. Do NOT invent GST rates, HSN codes, legal provisions, or calculation answers.

Return ONLY a JSON object matching this schema:
{
  "needs_direct_reasoning": boolean,
  "needs_calculation": boolean,
  "needs_structured_rate_lookup": boolean,
  "needs_hsn_lookup": boolean,
  "needs_legal_retrieval": boolean,
  "needs_notification_retrieval": boolean,
  "needs_temporal_reasoning": boolean,
  "needs_comparison": boolean,
  "needs_exception_reasoning": boolean,
  "needs_clarification": boolean,
  "needs_grounded_synthesis": boolean,
  "clean_subqueries": {
    "rate_query": string or null,
    "legal_query": string or null,
    "notification_query": string or null
  },
  "user_premises": {
    "assumed_rate": float or null,
    "taxable_amount": float or null,
    "discount_pct": float or null,
    "itc_balances": object with float balances e.g. {"cgst": 2500.0, "sgst": 2500.0, "igst": 2500.0} or null,
    "supply_type": "interstate" or "intrastate" or null
  },
  "clarification_prompt": string or null,
  "reasoning": string
}

CAPABILITY RULES:
1. Multiple capabilities MUST be selected simultaneously if the query requires them. Never force queries into a single route.
2. Structured Rate & HSN:
   - needs_structured_rate_lookup = true if the query asks about GST rate, slab, cess, or tax liability on any product, service, or commodity.
   - needs_hsn_lookup = true if specifically asking for or supplying an HSN or SAC code. When needs_hsn_lookup is true, always also set needs_structured_rate_lookup = true.
   - clean_subqueries.rate_query MUST be the clean commodity/product name in English (e.g. "butter", "fresh milk", "motorcycles", "chocolate").
3. Legal Retrieval:
   - needs_legal_retrieval = true for statutory questions, procedural rules, registration, cancellation, revocation, appeals, legal definitions, or ITC utilization principles.
   - clean_subqueries.legal_query MUST be a neutral concept/topic search query describing the legal subject matter (e.g., "ITC utilization order for interstate supply and discharge of IGST liability using IGST, CGST and SGST input tax credits", "cancellation of GST registration procedure").
   - DO NOT inject or assume specific statutory Sections, Rules, or Forms (e.g. "Section 49", "Rule 88A") unless explicitly mentioned in the user's question. Preserve statutory references explicitly supplied by the user (e.g. "Section 29"), but let the legal retrieval layer discover unstated statutory provisions.
4. Notification Retrieval:
   - needs_notification_retrieval = true when specific notifications (e.g. "09/2025", "01/2017"), amending notifications, or rate notifications are mentioned or inquired about.
   - clean_subqueries.notification_query should specify the notification number or context.
5. Calculation & Direct Reasoning:
   - needs_calculation = true whenever mathematical computation is required (e.g. tax on ₹50,000, 10% discount + 5% GST, maximum taxable value given ITC balances).
   - needs_direct_reasoning = true whenever reasoning over user facts, hypothetical scenarios, or multi-step logic is needed.
6. Comparison & Exceptions:
   - needs_comparison = true when comparing tax treatment across goods, engine capacities, or slabs (e.g. "motorcycles below and above 350cc").
   - needs_exception_reasoning = true for exceptions, special conditions, conditional rates, or ITC restrictions.
7. Clarification:
8. Grounded Synthesis:
   - needs_grounded_synthesis = true for all standard queries requiring an answer synthesized from retrieved documents, rates, or facts (always true unless needs_clarification is true).
9. Multilingual Understanding:
   - Understand queries in Gujarati (e.g. "માખણ" -> butter, "તાજા દૂધ" -> fresh milk, "નોંધણી રદ" -> cancellation of registration, "કેટલો છે" -> how much is), Hindi, and English.
   - Translate clean_subqueries into standard English search terms for the downstream retrieval tools.
"""


class GSTPlan(BaseModel):
    """Structured plan containing required capabilities, extracted premises, and clean subqueries."""
    needs_direct_reasoning: bool = False
    needs_calculation: bool = False
    needs_structured_rate_lookup: bool = False
    needs_hsn_lookup: bool = False
    needs_legal_retrieval: bool = False
    needs_notification_retrieval: bool = False
    needs_temporal_reasoning: bool = False
    needs_comparison: bool = False
    needs_exception_reasoning: bool = False
    needs_clarification: bool = False
    needs_grounded_synthesis: bool = True
    clean_subqueries: dict[str, Optional[str]] = Field(
        default_factory=lambda: {
            "rate_query": None,
            "legal_query": None,
            "notification_query": None,
        }
    )
    user_premises: dict[str, Any] = Field(
        default_factory=lambda: {
            "assumed_rate": None,
            "taxable_amount": None,
            "discount_pct": None,
            "itc_balances": None,
            "supply_type": None,
        }
    )
    clarification_prompt: Optional[str] = None
    reasoning: Optional[str] = None


def get_default_planner_client() -> OpenAI | None:
    """Instantiate OpenAI client if API key is present."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def get_default_planner_model() -> str:
    """Return model configured for planner."""
    env_model = os.environ.get("OPENAI_MODEL")
    if env_model and env_model.strip():
        return env_model.strip()
    return "gemini-3.1-flash-lite"


def plan_capabilities_with_llm(
    query: str,
    client: OpenAI | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Generate structured multi-capability plan using LLM."""
    llm_client = client or get_default_planner_client()
    if not llm_client:
        raise ValueError("No OpenAI client or API key available for LLM planner.")

    model_name = model or get_default_planner_model()

    response = llm_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": query.strip()},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )

    content = response.choices[0].message.content or "{}"
    data = json.loads(content)

    # Normalize defaults: grounded synthesis is True for any standard answerable query
    if not data.get("needs_clarification", False):
        data["needs_grounded_synthesis"] = True
    else:
        data["needs_grounded_synthesis"] = False

    # If needs_hsn_lookup is True, ensure needs_structured_rate_lookup is True
    if data.get("needs_hsn_lookup"):
        data["needs_structured_rate_lookup"] = True

    # Validate against schema
    validated = GSTPlan(**data)
    return validated.model_dump()


def plan_capabilities_heuristic(query: str) -> dict[str, Any]:
    """Deterministic fallback planner using regex and domain rules."""
    from src.routers.query_router import plan_capabilities_heuristic as _heuristic
    return _heuristic(query)


def plan_capabilities(
    query: str,
    client: OpenAI | None = None,
    model: str | None = None,
    use_llm: bool | None = None,
) -> dict[str, Any]:
    """Entrypoint: Generate capability plan with LLM if enabled/available, fallback to heuristics."""
    should_use_llm = use_llm if use_llm is not None else (
        os.environ.get("USE_LLM_PLANNER", "false").lower() in ("true", "1")
    )
    if should_use_llm:
        try:
            return plan_capabilities_with_llm(query, client=client, model=model)
        except Exception as exc:
            logger.warning(f"LLM planner failed ({exc}), falling back to heuristic planner.")
            return plan_capabilities_heuristic(query)
    return plan_capabilities_heuristic(query)
