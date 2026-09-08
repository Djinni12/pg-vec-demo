"""Lightweight query router for GST assistant.

Routes incoming queries to:
- LEGAL: Legal provisions, rules, forms, and procedures (via legal RAG)
- RATE: Goods and services GST rates, HSN/SAC lookups, and cess (via structured SQL)
- MIXED: Queries requiring both structured rate information and legal/procedural context
"""

from __future__ import annotations

from enum import Enum
import re


class RouteType(str, Enum):
    LEGAL = "legal"
    RATE = "rate"
    MIXED = "mixed"


# Regexes indicating rate / tariff intent
RATE_PATTERNS = [
    r"\b(?:hsn|sac)\b",
    r"\b(?:heading|chapter)\s*\d+\b",
    r"\b(?:gst|tax|cgst|sgst|igst|utgst)\s*(?:rate|rates|slab|slabs|percentage|percent|%)\b",
    r"\b(?:rate|rates|tax|taxes|slab|slabs)\s*(?:of|for|on)\b",
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
