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
import time
from typing import Any, Literal, Optional

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from dotenv import load_dotenv

from src.observability.llm_usage_tracker import record_llm_call

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
  "needs_query_decomposition": boolean,
  "retrieval_subqueries": [
    {
      "type": "legal" or "rate",
      "query": string
    }
  ],
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
   - clean_subqueries.legal_query MUST be a focused search query describing the legal subject matter without injecting unmentioned sections/rules (e.g., "tax collected on exempt supply credit note procedure", "ITC utilization order for interstate supply and discharge of IGST liability using IGST, CGST and SGST input tax credits", "cancellation of GST registration procedure").
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
   - needs_clarification = true when the query is missing critical details needed to answer.
8. Grounded Synthesis:
   - needs_grounded_synthesis = true for all standard queries requiring an answer synthesized from retrieved documents, rates, or facts (always true unless needs_clarification is true).
9. Multilingual Understanding:
   - Understand queries in Gujarati (e.g. "માખણ" -> butter, "તાજા દૂધ" -> fresh milk, "નોંધણી રદ" -> cancellation of registration, "કેટલો છે" -> how much is, "ભૂલથી ટેક્સ લીધો" -> mistakenly collected tax), Hindi, and English.
   - Translate clean_subqueries and retrieval_subqueries into standard English search terms for the downstream retrieval tools.
10. LEGAL APPLICABILITY & FACTUAL SCOPE RULE:
    - Distinguish between similar but legally different scenarios:
      * Tax collected on an exempt supply (governed by Section 76 / Section 34 / Section 54) vs IGST paid instead of CGST/SGST (governed by Section 77 CGST Act / Section 19 IGST Act).
      * Never confuse tax collected on an exempt supply with an inter-State vs intra-State supply classification mismatch.

QUERY DECOMPOSITION RULES (needs_query_decomposition & retrieval_subqueries):
1. When to decompose:
   - Simple single-concept queries MUST NOT be decomposed:
     * If a query asks for a single provision, rule, section, rate, or calculation:
       set needs_query_decomposition = false, retrieval_subqueries = []
     * Examples of NO decomposition:
       - "What does Rule 88A say?" -> needs_query_decomposition = false, retrieval_subqueries = []
       - "What is the GST rate on butter?" -> needs_query_decomposition = false, retrieval_subqueries = []
       - "₹50,000 par 18% GST kitna hai?" -> needs_query_decomposition = false, retrieval_subqueries = []
       - "What does Section 29 say about cancellation?" -> needs_query_decomposition = false, retrieval_subqueries = []
   - Complex multi-concept queries MUST be decomposed:
     * When a single user query contains 2 or more independent legal or rate evidence needs that a single retrieval query would fail to cover adequately:
       set needs_query_decomposition = true
       generate 2 to 4 distinct standalone objects in retrieval_subqueries: [{"type": "legal" | "rate", "query": string}]
     * Example 1 (post-supply discount & liability adjustment):
       Query: "I gave a discount after invoicing. Can I reduce GST liability and how do I adjust it?"
       Decomposition: needs_query_decomposition = true
       retrieval_subqueries:
       [
         {"type": "legal", "query": "conditions for a post-supply discount to reduce taxable value"},
         {"type": "legal", "query": "procedure for adjusting tax liability after a post-supply discount"}
       ]
     * Example 2 (temporal / rate change across periods):
       Query: "What GST applied to tobacco before September 2025 and what applies now?"
       Decomposition: needs_query_decomposition = true
       retrieval_subqueries:
       [
         {"type": "rate", "query": "GST rate on tobacco before September 2025"},
         {"type": "rate", "query": "current GST rate on tobacco"}
       ]

2. Subquery Construction Constraints:
   - Distinct Evidence Needs: Each subquery must represent a DIFFERENT evidence need, not merely a paraphrase of another subquery.
   - Semantic Purpose Comparison: Before returning retrieval_subqueries, compare their semantic purpose.
     * When a query asks whether an action is permitted and how to adjust or comply (e.g. 'Can I do X? If not, how should it be adjusted?'):
       Subquery 1 should target the substantive statutory right, eligibility, restriction, or condition.
       Subquery 2 should target the procedural mechanism, determination formula, or reporting adjustment.
     * Bad (near paraphrases that retrieve the same chunks):
       1. "reversal of ITC for exempt supplies"
       2. "procedure for reversal of ITC for exempt supplies"
     * Good (distinct substantive vs procedural evidence needs):
       1. "eligibility or restriction of input tax credit when goods are used for taxable and exempt supplies"
       2. "method or procedure for determining and reversing input tax credit attributable to exempt supplies"
     * If two proposed subqueries would likely retrieve the same legal concept, combine or rewrite them so each targets a distinct question that must be established to answer the user.
   - Neutral & Concept-Based: Write concise, neutral search queries focused on the underlying statutory or tax concept.
   - Standalone: Each subquery must make complete sense independently.
   - Preserve Meaning & Facts: Preserve the exact facts, commodities, and conditions from the user's question.
   - DO NOT INVENT UNMENTIONED REFERENCES:
     NEVER inject Section numbers (e.g. Section 15(3)(b), Section 34), Rule numbers (e.g. Rule 53, Rule 88A), Form names, Notification numbers, rates, thresholds, dates, HSN codes, or legal conclusions NOT explicitly present in the user query.
     If the user did not say "Section 15(3)(b)", do NOT write "Section 15(3)(b)" in any retrieval subquery!
   - DO NOT split arithmetic: Never create retrieval subqueries for arithmetic or calculation steps.
   - NO Duplicates: Do not generate duplicate or overlapping subqueries.
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


class RetrievalSubquery(BaseModel):
    """A standalone retrieval subquery targeting a single distinct evidence need."""
    model_config = ConfigDict(extra="ignore")

    type: Literal["legal", "rate"] = Field(
        ...,
        description="Type of retrieval: 'legal' for statutory/procedural queries, 'rate' for commodity/rate queries."
    )
    query: str = Field(
        ...,
        description="Standalone, neutral, concept-based retrieval query without injected unmentioned sections/rules."
    )

    @field_validator("type", mode="before")
    @classmethod
    def _validate_type(cls, v: Any) -> str:
        s = str(v).strip().lower()
        if "rate" in s:
            return "rate"
        return "legal"

    @field_validator("query", mode="before")
    @classmethod
    def _validate_query(cls, v: Any) -> str:
        return str(v).strip() if v is not None else ""


def normalize_query_decomposition(
    needs_query_decomposition: bool,
    subqueries: list[Any] | None,
) -> tuple[bool, list[dict[str, str]]]:
    """Perform generic validation and normalization on retrieval subqueries:
    - allow 0-4 subqueries
    - allowed types: legal | rate
    - remove duplicates
    - reject empty queries
    - if fewer than 2 distinct retrieval evidence needs exist,
      needs_query_decomposition remains False
    - do not inject new legal concepts or references in Python
    """
    if not subqueries or not isinstance(subqueries, list):
        return False, []

    cleaned: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for item in subqueries:
        if isinstance(item, dict):
            q = str(item.get("query", "")).strip()
            raw_t = str(item.get("type", "legal")).strip().lower()
        elif hasattr(item, "query") and hasattr(item, "type"):
            q = str(item.query).strip()
            raw_t = str(item.type).strip().lower()
        else:
            continue

        # Reject empty queries
        if not q:
            continue

        # Allowed types: legal | rate
        t = "rate" if "rate" in raw_t else "legal"

        # Deduplication (case-insensitive query text + type)
        dedup_key = (t, q.lower())
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        cleaned.append({"type": t, "query": q})

        # Allow at most 4 subqueries
        if len(cleaned) == 4:
            break

    # If fewer than 2 distinct retrieval evidence needs exist, decomposition remains False
    if not needs_query_decomposition or len(cleaned) < 2:
        return False, []

    return True, cleaned


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
    needs_query_decomposition: bool = False
    retrieval_subqueries: list[RetrievalSubquery] = Field(default_factory=list)
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

    @field_validator("retrieval_subqueries", mode="before")
    @classmethod
    def _ensure_subqueries(cls, v: Any) -> list[Any]:
        if v is None or not isinstance(v, list):
            return []
        valid = []
        for item in v:
            if isinstance(item, dict):
                q = str(item.get("query", "")).strip()
                t = str(item.get("type", "legal")).strip().lower()
                if q:
                    valid.append({"type": "rate" if "rate" in t else "legal", "query": q})
            elif isinstance(item, RetrievalSubquery):
                if item.query.strip():
                    valid.append(item)
        return valid

    @model_validator(mode="after")
    def _validate_and_normalize_decomposition(self) -> GSTPlan:
        decomp, subqs = normalize_query_decomposition(
            self.needs_query_decomposition,
            self.retrieval_subqueries,
        )
        self.needs_query_decomposition = decomp
        self.retrieval_subqueries = [RetrievalSubquery(**sq) for sq in subqs]
        return self


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
    request_id: str | None = None,
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

    t_start = time.perf_counter()
    response = llm_client.chat.completions.create(
        model=model_name,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

    try:
        record_llm_call(
            request_id=request_id,
            stage="planner",
            model=model_name,
            response=response,
            latency_ms=latency_ms,
        )
    except Exception as track_err:
        logger.debug(f"Failed to record planner LLM usage: {track_err}")

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

    # Generic validation and normalization of query decomposition
    decomp, subqs = normalize_query_decomposition(
        bool(data.get("needs_query_decomposition", False)),
        data.get("retrieval_subqueries"),
    )
    data["needs_query_decomposition"] = decomp
    data["retrieval_subqueries"] = subqs

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
    request_id: str | None = None,
) -> dict[str, Any]:
    """Entrypoint: Generate capability plan with LLM if enabled/available, fallback to heuristics."""
    should_use_llm = use_llm if use_llm is not None else (
        os.environ.get("USE_LLM_PLANNER", "true").lower() in ("true", "1")
    )
    if should_use_llm:
        try:
            return plan_capabilities_with_llm(
                query,
                client=client,
                model=model,
                history=history,
                request_id=request_id,
            )
        except Exception as exc:
            logger.warning(f"LLM planner failed ({exc}), falling back to heuristic planner.")
            return plan_capabilities_heuristic(query)
    return plan_capabilities_heuristic(query)
