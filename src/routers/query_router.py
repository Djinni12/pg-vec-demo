"""Lightweight query router for GST assistant.

Routes incoming queries to:
- DIRECT: Arithmetic or reasoning using only values supplied by the user
- LEGAL: Legal provisions, rules, forms, and procedures (via legal RAG)
- RATE: Goods and services GST rates, HSN/SAC lookups, and cess (via structured SQL)
- MIXED: Queries requiring both structured rate information and legal/procedural context
"""

from __future__ import annotations

from enum import Enum
import re
from typing import Any, Optional


class RouteType(str, Enum):
    DIRECT = "direct"
    LEGAL = "legal"
    RATE = "rate"
    MIXED = "mixed"


# Regexes indicating rate / tariff intent
RATE_PATTERNS = [
    r"\b(?:hsn|sac)\b",
    r"\b(?:heading|chapter)\s*\d+\b",
    r"\b(?:gst|tax|cgst|sgst|igst|utgst)\s*(?:rate|rates|slab|slabs|percentage|percent|%|details|information)\b",
    r"(?:gst|cgst|sgst|igst|utgst)\s*દર",
    r"દર\s*(?:કેટલો|લાગુ|મુજબ)",
    r"\b(?:rate|rates|tax|taxes|slab|slabs)\s*(?:of|for|on)\b",
    r"\b(?:rate|rates|tax|taxes|slab|slabs)\s+(?:applies|apply|applicable)\b",
    r"\b(?:how\s+much\s+(?:gst|tax)|what\s+(?:is|are)\s+the\s+(?:gst|tax|rates?))\b",
    r"\b(?:compensation\s+cess|cess\s+rate|cess\s+on)\b",
    r"\b(?:tariff|schedule\s+[ivx]+)\b",
    r"\b(?:gst|tax)\s+on\s+[a-z0-9]",
    r"\b(?:applicable\s+gst|applicable\s+rate)\b",
    r"\b\d{1,2}(?:\.\d+)?%\s*(?:gst|tax|igst|cgst|sgst)?\b",
]

# Regexes indicating legal / procedural intent
LEGAL_PATTERNS = [
    r"\b(?:section|sec\.?|s\.)\s*\d+[a-zA-Z]?\b",
    r"\b(?:rule|rules|r\.)\s*\d+[a-zA-Z]?\b",
    r"\b(?:form\s+)?gst\s+[a-z]{3,4}-?\d{1,2}[a-z]?\b",
    r"\b(?:cancellation|cancel|cancelled|cancelling|revocation|revoke|revoked|suspension|suspend|suspended|deregister|deregistration)\b",
    r"\b(?:procedure|process|steps|how\s+to\s+apply|how\s+to\s+file|filing\s+process)\b",
    r"\b(?:proper\s+officer|show\s+cause|notice|scn|summons|adjudication)\b",
    r"\b(?:penalty|penalties|late\s+fee|interest|offence|offense|prosecution|arrest)\b",
    r"\b(?:appeal|appellate|tribunal|writ|revision)\b",
    r"\b(?:input\s+tax\s+credit|itc|reverse\s+charge|rcm)\b",
    r"\b(?:audit|scrutiny|assessment|search\s+and\s+seizure)\b",
    r"\b(?:time\s+limit|limitation|timeline|within\s+\d+\s+days)\b",
    r"\b(?:mandatory\s+registration|threshold\s+limit|aggregate\s+turnover|registration\s+procedure|register\s+under\s+gst)\b",
    r"\b(?:cgst\s+act|sgst\s+act|igst\s+act|cgst\s+rules)\b",
]

YEAR_EXCLUSIONS = {str(y) for y in range(1990, 2035)}

DIRECT_CALC_PATTERNS = [
    r"\bwhat\s+is\s+\d+(?:\.\d+)?\s*%\s+of\s+\d",
    r"\bcalculate\b",
    r"\bcompute\b",
    r"\bhow\s+much\b",
    r"\btaxable\s+(?:amount|value)\b",
    r"\btax\s+amount\b",
    r"\bgst\s+(?:amount|credits?|credit)\b",
    r"\b(?:at|charge|charged|charging)\s+\d+(?:\.\d+)?\s*%",
    r"(?:કિંમત|રકમ|ડિસ્કાઉન્ટ|બાકી|અંતિમ|કેટલી|ગણતરી)",
    r"(?:લગાવવામાં|લગાડવામાં|લગાવવામાં આવે|લગાડવામાં આવે)",
    r"(?:मूल्य|रकम|राशि|छूट|डिस्काउंट|शेष|बाकी|अंतिम|कितनी|गणना)",
    r"(?:लगाया|लगाया जाता|लगाया जाता है|लागू)",
]

EXTERNAL_KNOWLEDGE_PATTERNS = [
    r"\b(?:applicable|current|latest|official|prescribed)\s+(?:gst\s+)?rate\b",
    r"\b(?:rate|rates|tax|taxes|slab|slabs)\s+(?:applies|apply|applicable)\b",
    r"\b(?:rate|rates|tax|taxes|slab|slabs)\s+(?:of|for|on)\s+[a-z]",
    r"\b(?:using|use)\s+the\s+(?:applicable|document|documents|retrieved|official|prescribed)\b",
    r"\bfrom\s+(?:the\s+)?documents?\b",
    r"\b(?:hsn|sac|heading|chapter|schedule|notification|section|rule|form)\b",
    r"\b(?:threshold|eligibility|eligible|exemption|exempt|exception|condition|procedure|registration|itc|rcm|reverse\s+charge)\b",
    r"\bitems?\b",
    r"(?:લાગુ|કાયદા|કાનૂની|નિયમ|કલમ|ફોર્મ|દસ્તાવેજ|દસ્તાવેજો|એચએસએન|એસએસી|અધિસૂચના|મર્યાદા|છૂટ|પાત્રતા)",
]


def has_direct_calculation_intent(query: str) -> bool:
    """Check if query can be answered by arithmetic using only user-provided values."""
    lower = query.lower()
    has_number = re.search(r"\d", lower) is not None
    has_percent = re.search(r"\d+(?:\.\d+)?\s*%", lower) is not None
    if not (has_number and has_percent):
        return False
    if any(re.search(pattern, lower) for pattern in EXTERNAL_KNOWLEDGE_PATTERNS):
        return False
    return any(re.search(pattern, lower) for pattern in DIRECT_CALC_PATTERNS)


def has_rate_intent(query: str) -> bool:
    """Check if query indicates rate, HSN/SAC, tariff, or cess lookup intent."""
    lower = query.lower()
    for pattern in RATE_PATTERNS:
        if re.search(pattern, lower):
            return True

    # Check for standalone 4, 6, or 8 digit tariff codes (excluding years)
    for m in re.finditer(r"\b(\d{4,8})\b", query):
        val = m.group(1)
        if val not in YEAR_EXCLUSIONS:
            return True

    return False


def has_legal_intent(query: str) -> bool:
    """Check if query indicates legal provisions, rules, forms, or procedural intent."""
    lower = query.lower()
    for pattern in LEGAL_PATTERNS:
        if re.search(pattern, lower):
            return True

    return False


def classify_query(query: str) -> RouteType:
    """Classify user query into LEGAL, RATE, or MIXED route."""
    cleaned = query.strip()
    if not cleaned:
        return RouteType.LEGAL

    if has_direct_calculation_intent(cleaned):
        return RouteType.DIRECT

    rate_match = has_rate_intent(cleaned)
    legal_match = has_legal_intent(cleaned)

    if rate_match and legal_match:
        return RouteType.MIXED
    if rate_match:
        return RouteType.RATE
    if legal_match:
        return RouteType.LEGAL

    # Default fallback to LEGAL RAG
    return RouteType.LEGAL


def is_hsn_candidate(m_str: str, query: str) -> bool:
    """Determine if a 4-8 digit number looks like an HSN code rather than a monetary value."""
    start = query.find(m_str)
    prefix = query[max(0, start - 20):start].lower()
    if any(term in prefix for term in ["credit", "itc", "₹", "rs", "rupees", "amount", "balance", "value", "cgst", "sgst", "igst"]):
        return False
    return True


def plan_capabilities_heuristic(query: str) -> dict[str, Any]:
    """Analyze query and produce a structured multi-capability execution plan using heuristics.

    Allows multiple capabilities to be True simultaneously.
    Distinguishes user premises/assumptions from authoritative legal facts.
    Detects ambiguity requiring focused clarification.
    """
    cleaned = query.strip()
    if not cleaned:
        return {
            "needs_direct_reasoning": False,
            "needs_calculation": False,
            "needs_structured_rate_lookup": False,
            "needs_hsn_lookup": False,
            "needs_legal_retrieval": True,
            "needs_notification_retrieval": False,
            "needs_temporal_reasoning": False,
            "needs_comparison": False,
            "needs_exception_reasoning": False,
            "needs_clarification": False,
            "needs_grounded_synthesis": True,
            "user_premises": {
                "assumed_rate": None,
                "taxable_amount": None,
                "discount_pct": None,
                "itc_balances": {},
                "supply_type": None,
            },
            "clean_subqueries": {
                "rate_query": None,
                "legal_query": cleaned,
                "notification_query": None,
            },
            "clarification_prompt": None,
        }

    lower = cleaned.lower()

    # 1. User Premises Extraction
    assumed_rate = None
    m_assume = re.search(r"\b(?:assume|assuming|suppose|supposing|hypothetically)\b.*?\b(\d+(?:\.\d+)?)\s*%", lower)
    if m_assume:
        assumed_rate = float(m_assume.group(1))

    # Monetary taxable amounts (e.g. ₹50,000 or on 80,000)
    taxable_amount = None
    m_tax_val = re.search(r"(?:taxable\s+(?:value|amount)|on|of)\s*₹?\s*(\d+(?:,\d+)*(?:\.\d+)?)", lower)
    if not m_tax_val:
        m_tax_val = re.search(r"₹\s*(\d+(?:,\d+)*(?:\.\d+)?)", query)
    if m_tax_val:
        try:
            val_candidate = float(m_tax_val.group(1).replace(",", ""))
            val_start = query.find(m_tax_val.group(1))
            val_surrounding = query[max(0, val_start - 10):min(len(query), val_start + 25)].lower()
            if "credit" not in val_surrounding and "itc" not in val_surrounding:
                taxable_amount = val_candidate
        except ValueError:
            pass

    # Discounts (e.g. 10% discount)
    discount_pct = None
    m_disc = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:discount|off|छूट|ડિસ્કાઉન્ટ)", lower)
    if not m_disc:
        m_disc = re.search(r"(?:discount|छूट|ડિસ્કાઉન્ટ).*?(\d+(?:\.\d+)?)\s*%", lower)
    if m_disc:
        discount_pct = float(m_disc.group(1))

    # Input Tax Credit balances
    itc_balances: dict[str, float] = {}
    for c_type in ["cgst", "sgst", "igst"]:
        m_c = re.search(rf"\b{c_type}\s*(?:credit|itc)?\s*(?:of|is|:)?\s*₹?\s*(\d+(?:,\d+)*(?:\.\d+)?)", lower)
        if not m_c:
            m_c = re.search(rf"₹?\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:of\s*)?{c_type}", lower)
        if m_c:
            itc_balances[c_type] = float(m_c.group(1).replace(",", ""))

    # Supply Type (Inter-State vs Intra-State)
    supply_type = None
    if (
        "interstate" in lower
        or "inter-state" in lower
        or "inter state" in lower
        or "one state to another" in lower
        or "from one state" in lower
        or "to another state" in lower
        or "outside the state" in lower
        or "different state" in lower
        or "across state" in lower
    ):
        supply_type = "interstate"
    elif (
        "intrastate" in lower
        or "intra-state" in lower
        or "intra state" in lower
        or "within state" in lower
        or "within the state" in lower
        or "within the same state" in lower
        or "same state" in lower
    ):
        supply_type = "intrastate"

    # 2. Ambiguity & Clarification Detection
    # e.g., "I have ₹5,000 GST credit. How much can I sell?"
    has_credit_phrase = bool(re.search(r"\b(?:\d+|₹\s*\d+).*?\b(?:gst\s+)?credit\b", lower) or "gst credit" in lower)
    has_sell_phrase = bool(re.search(r"\bhow\s+much\s+(?:can\s+i|to)\s+sell\b|\bhow\s+much\s+(?:sales|turnover)\b", lower))
    has_rate_or_product = bool(
        (re.search(r"\b(?:at\s+\d+%|\d+%\s*gst|butter|milk|paneer|rate\s+for|rate\s+on)\b", lower)
         or "0405" in cleaned or "0402" in cleaned)
        and "gst credit" not in lower
    )
    needs_clarification = False
    clarification_prompt = None

    if has_credit_phrase and has_sell_phrase and not has_rate_or_product:
        needs_clarification = True
        clarification_prompt = (
            "To determine the taxable value you can supply against your GST credit, please clarify:\n"
            "1. What is the applicable GST rate or product/service being supplied (e.g. 5%, 12%, 18%, or 28%)?\n"
            "2. Is the supply Intra-State (within the state) or Inter-State (outside the state)?\n"
            "3. What is the breakdown of your credit balance between IGST, CGST, and SGST?"
        )

    # 3. Temporal & Notification Reasoning
    needs_temporal = bool(
        re.search(r"\b(?:before\s+and\s+after|amended\s+by|corrigendum|what\s+applies\s+(?:now|today)|earlier\s+rate|latest\s+notification|chronology|effective\s+date)\b", lower)
        or ("notification" in lower and ("amended" in lower or "corrigendum" in lower or "changed" in lower or "introduced" in lower))
    )
    needs_notification = bool(
        needs_temporal
        or re.search(r"\bnotification\s*(?:no\.?|number)?\s*\d{1,2}/\d{4}\b", lower)
    )

    # 4. Exception Reasoning
    needs_exception = bool(
        re.search(r"\b(?:normally\s+applies|does\s+exception\s+.*apply|any\s+exceptions?|is\s+there\s+an\s+exception|proviso\s+to|conditional\s+exemption|unless\s+otherwise)\b", lower)
    )

    # 5. Comparison
    needs_comparison = bool(re.search(r"\b(?:compare|comparison|difference\s+between|versus|vs\.?)\b", lower))

    # 6. Calculation Intent
    has_math_words = bool(
        re.search(r"\b(?:calculate|compute|how\s+much|taxable\s+(?:amount|value)|what\s+is\s+\d|final\s+amount)\b", lower)
        or re.search(r"(?:કિંમત|રકમ|ડિસ્કાઉન્ટ|બાકી|અંતિમ|કેટલી|ગણતરી)", query)
        or re.search(r"(?:मूल्य|रकम|राशि|छूट|डिस्काउंट|शेष|बाकी|अंतिम|कितनी|गणना)", query)
    )
    needs_calculation = bool(
        has_math_words
        or (taxable_amount is not None and ("gst" in lower or "tax" in lower))
        or bool(itc_balances and "what taxable value" in lower)
    )

    # 7. Pure Direct Reasoning
    has_external_search = bool(
        re.search(r"\b(?:what\s+is\s+the\s+(?:hsn|rate)|find\s+the\s+(?:applicable\s+)?rate|gst\s+rate\s+(?:on|for)|rate\s+of|hsn\s+code|sac\s+code)\b", lower)
    )

    is_pure_direct = False
    if assumed_rate is not None:
        is_pure_direct = True
    elif (
        not has_external_search
        and not needs_notification
        and not needs_exception
        and not needs_clarification
        and (re.search(r"\d+(?:\.\d+)?\s*%", query) or discount_pct is not None)
        and has_math_words
        and not itc_balances
    ):
        is_pure_direct = True

    # 8. Structured Rate & HSN Lookup
    needs_structured_rate = False
    needs_hsn = False
    rate_query = None

    if not is_pure_direct and not needs_clarification:
        if "butter" in lower:
            needs_structured_rate = True
            needs_hsn = True
            rate_query = "butter"
        elif not needs_temporal:
            valid_codes = [m.group(0) for m in re.finditer(r"\b\d{4,8}\b", query) if is_hsn_candidate(m.group(0), query)]
            has_rate_vocab = bool(re.search(r"\b(?:hsn|sac|rate|slab|tariff|gst\s+on)\b", lower))
            if valid_codes or has_rate_vocab:
                needs_structured_rate = True
                if "hsn" in lower or valid_codes:
                    needs_hsn = True
                rate_query = cleaned

    # 9. Legal Retrieval
    needs_legal = False
    legal_query = None
    if not is_pure_direct and not needs_clarification:
        if (
            bool(re.search(r"\b(?:section|rule|form|itc|credit|utilization|order\s+of\s+utilization|interstate|intrastate|cancellation|revocation|penalty|procedure|registration|send|parcel|supply|sell)\b", lower))
            or needs_exception
        ):
            needs_legal = True
            has_explicit_sec = bool(re.search(r"\b(?:section|rule|form)\s+\d+", lower))
            if has_explicit_sec:
                legal_query = cleaned
            elif itc_balances and supply_type == "interstate":
                legal_query = "ITC utilization order for interstate supply and discharge of IGST liability using IGST, CGST and SGST input tax credits"
            elif itc_balances and supply_type == "intrastate":
                legal_query = "ITC utilization order for intrastate supply using IGST, CGST and SGST input tax credits"
            elif itc_balances:
                legal_query = "order of utilization of input tax credit across IGST, CGST and SGST"
            else:
                legal_query = cleaned

    needs_direct_reasoning = bool(is_pure_direct or itc_balances or needs_calculation)
    needs_grounded_synthesis = not needs_clarification

    return {
        "needs_direct_reasoning": needs_direct_reasoning,
        "needs_calculation": needs_calculation,
        "needs_structured_rate_lookup": needs_structured_rate,
        "needs_hsn_lookup": needs_hsn,
        "needs_legal_retrieval": needs_legal,
        "needs_notification_retrieval": needs_notification,
        "needs_temporal_reasoning": needs_temporal,
        "needs_comparison": needs_comparison,
        "needs_exception_reasoning": needs_exception,
        "needs_clarification": needs_clarification,
        "needs_grounded_synthesis": needs_grounded_synthesis,
        "user_premises": {
            "assumed_rate": assumed_rate,
            "taxable_amount": taxable_amount,
            "discount_pct": discount_pct,
            "itc_balances": itc_balances,
            "supply_type": supply_type,
        },
        "clean_subqueries": {
            "rate_query": rate_query,
            "legal_query": legal_query,
            "notification_query": cleaned if needs_notification else None,
        },
        "clarification_prompt": clarification_prompt,
    }


# Expose multi-capability planner interface
from src.routers.planner import (
    GSTPlan,
    plan_capabilities,
    plan_capabilities_with_llm,
)
