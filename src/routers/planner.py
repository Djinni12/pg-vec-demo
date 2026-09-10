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
from pydantic import BaseModel, ConfigDict, Field, field_validator
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
    "rate_is_user_assumed": boolean,
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
   - needs_structured_rate_lookup = true if the query asks about GST rate, slab, cess, or tax liability on any specific, identifiable product, service, or commodity.
   - Do NOT trigger rate lookup (needs_structured_rate_lookup = false) if the user merely states as a premise that an unnamed product is exempt (e.g. 'a product is exempt', 'Ek product actually GST exempt che') without identifying a specific commodity to look up. Rate searches on generic words ('product', 'item') return arbitrary irrelevant goods.
   - needs_hsn_lookup = true if specifically asking for or supplying an HSN or SAC code. When needs_hsn_lookup is true, always also set needs_structured_rate_lookup = true.
   - clean_subqueries.rate_query MUST be the clean commodity/product name in English (e.g. "butter", "fresh milk", "motorcycles", "chocolate").
3. Legal Retrieval:
   - needs_legal_retrieval = true for statutory questions, procedural rules, registration, cancellation, revocation, appeals, legal definitions, tax collected on exempt supplies / without authority, credit notes, or ITC utilization principles.
   - clean_subqueries.legal_query MUST be a focused search query describing the legal subject matter (e.g., "tax collected on exempt supply but not paid to Government Section 76 credit note Section 34", "ITC utilization order for interstate supply and discharge of IGST liability using IGST, CGST and SGST input tax credits", "cancellation of GST registration procedure").
   - When the question concerns tax mistakenly collected on exempt supplies, tax collected without authority, or refund/adjustment thereof: formulate clean_subqueries.legal_query around "tax collected on exempt supply but not paid to Government Section 76 credit note Section 34".
4. Notification Retrieval:
   - needs_notification_retrieval = true when specific notifications (e.g. "09/2025", "01/2017"), amending notifications, or rate notifications are mentioned or inquired about.
   - clean_subqueries.notification_query should specify the notification number or context.
5. Calculation & Direct Reasoning:
   - needs_calculation = true whenever mathematical computation is required (e.g. tax on ₹50,000, 10% discount + 5% GST, maximum taxable value given ITC balances).
   - needs_direct_reasoning = true whenever reasoning over user facts, hypothetical scenarios, or multi-step logic is needed.
   - user_premises.rate_is_user_assumed MUST be true ONLY if the user explicitly provided or assumed a hypothetical rate in their prompt (e.g. 'Assume GST is 12%', 'Suppose 18% GST'). Set to false if the rate was retrieved from official tariff records, derived from a commodity name, or carried over from a prior turn's statutory rate.
6. Comparison & Exceptions:
   - needs_comparison = true when comparing tax treatment across goods, engine capacities, or slabs (e.g. "motorcycles below and above 350cc").
   - needs_exception_reasoning = true for exceptions, special conditions, conditional rates, or ITC restrictions.
7. Clarification:
8. Grounded Synthesis:
   - needs_grounded_synthesis = true for all standard queries requiring an answer synthesized from retrieved documents, rates, or facts (always true unless needs_clarification is true).
9. Multilingual Understanding:
   - Understand queries in Gujarati (e.g. "માખણ" -> butter, "તાજા દૂધ" -> fresh milk, "નોંધણી રદ" -> cancellation of registration, "કેટલો છે" -> how much is, "ભૂલથી ટેક્સ લીધો" -> mistakenly collected tax), Hindi, and English.
   - Translate clean_subqueries into standard English search terms for the downstream retrieval tools.
10. LEGAL APPLICABILITY & FACTUAL SCOPE RULE:
    - Distinguish between similar but legally different scenarios:
      * Tax collected on an exempt supply (governed by Section 76 / Section 34 / Section 54) vs IGST paid instead of CGST/SGST (governed by Section 77 CGST Act / Section 19 IGST Act).
      * Never confuse tax collected on an exempt supply with an inter-State vs intra-State supply classification mismatch.
"""


def extract_json_payload(content: str) -> dict[str, Any]:
    """Extract and parse a JSON object from LLM response text.

    Safely handles:
    - Markdown code fences (```json ... ``` or ``` ... ```) commonly produced by Gemma
    - Leading or trailing conversational preambles
    - Plain JSON strings
    """
    if not content or not content.strip():
        return {}
    text = content.strip()
    text_unfenced = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text_unfenced = re.sub(r"\s*```$", "", text_unfenced)
    try:
        parsed = json.loads(text_unfenced.strip())
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    m = re.search(r"(\{[\s\S]*\})", text)
    if m:
        try:
            parsed = json.loads(m.group(1))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    return {}


class GSTPlan(BaseModel):
    """Structured plan containing required capabilities, extracted premises, and clean subqueries."""
    model_config = ConfigDict(extra="ignore")

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
            "rate_is_user_assumed": False,
            "taxable_amount": None,
            "discount_pct": None,
            "itc_balances": None,
            "supply_type": None,
        }
    )
    clarification_prompt: Optional[str] = None
    reasoning: Optional[str] = None

    @field_validator("clean_subqueries", "user_premises", mode="before")
    @classmethod
    def _ensure_dict(cls, v: Any) -> dict[str, Any]:
        if v is None or not isinstance(v, dict):
            return {}
        return v


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
    history: list[Any] | None = None,
) -> dict[str, Any]:
    """Generate structured multi-capability plan using LLM."""
    llm_client = client or get_default_planner_client()
    if not llm_client:
        raise ValueError("No OpenAI client or API key available for LLM planner.")

    model_name = model or get_default_planner_model()

    # Format recent history for resolving follow-ups, pronouns, and references
    hist_context = ""
    if history:
        lines = []
        for m in history[-6:]:
            if isinstance(m, dict):
                role = "User" if m.get("role") in ("human", "user") else "Assistant"
                c = (m.get("content") or "").strip()
            else:
                role = "User" if getattr(m, "type", "") in ("human", "user") else "Assistant"
                c = (getattr(m, "content", "") or "").strip()
            if c:
                if len(c) > 250:
                    c = c[:250] + "..."
                lines.append(f"{role}: {c}")
        if lines:
            hist_context = "PRIOR CONVERSATION CONTEXT:\n" + "\n".join(lines) + "\n\n"

    user_query_content = f"{hist_context}CURRENT USER QUERY:\n{query.strip()}" if hist_context else query.strip()

    # Gemma instruction format merges system instructions into user turn
    if "gemma" in model_name.lower():
        messages = [
            {"role": "user", "content": f"{PLANNER_SYSTEM_PROMPT}\n\n---\n\n{user_query_content}"}
        ]
    else:
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": user_query_content},
        ]

    response = llm_client.chat.completions.create(
        model=model_name,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.0,
    )

    content = response.choices[0].message.content or "{}"
    data = extract_json_payload(content)

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
    history: list[Any] | None = None,
) -> dict[str, Any]:
    """Entrypoint: Generate capability plan with LLM if enabled/available, fallback to heuristics."""
    should_use_llm = use_llm if use_llm is not None else (
        os.environ.get("USE_LLM_PLANNER", "true").lower() in ("true", "1")
    )
    if should_use_llm:
        try:
            return plan_capabilities_with_llm(query, client=client, model=model, history=history)
        except Exception as exc:
            logger.warning(f"LLM planner failed ({exc}), falling back to heuristic planner.")
            return plan_capabilities_heuristic(query)
    return plan_capabilities_heuristic(query)
