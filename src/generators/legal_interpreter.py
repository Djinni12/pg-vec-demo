"""Structured Grounded Legal Interpreter.

Analyzes retrieved legal evidence (Acts, Rules, Forms) and produces a structured,
grounded interpretation for downstream calculation and synthesis.
Strictly requires every legal finding to be supported by retrieved evidence.
If retrieved evidence is insufficient, marks findings unresolved.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)


def extract_json_payload(content: str) -> dict[str, Any]:
    """Extract and parse JSON object from LLM response, handling markdown fences and preambles."""
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


INTERPRETER_SYSTEM_PROMPT = """You are a Grounded Legal Evidence Interpreter for GST.
Analyze the RETRIEVED LEGAL EVIDENCE and extract structured legal findings for downstream calculation.

STRICT GROUNDING RULES:
1. Grounding Requirement:
   - Every finding MUST be explicitly supported by the RETRIEVED LEGAL EVIDENCE.
   - If the retrieved evidence does not mention the rule or provision, set status = "unresolved" and explain in unresolved_reason.
   - Do NOT assume rules from memory that are not in the retrieved evidence.
2. Supply Type and ITC Utilization:
   - For interstate supplies (IGST liability): Check if Section 49 / Rule 88A is present in the context.
     * If Rule 88A is present: IGST credit must be used first to pay IGST. After that, CGST and SGST credit can both be used towards IGST liability in any order and proportion.
     * Set usable_credit_ledgers = ["igst", "cgst", "sgst"].
   - For intrastate supplies (CGST + SGST liability):
     * CGST liability can be paid using IGST credit and CGST credit (NOT SGST credit).
     * SGST liability can be paid using IGST credit and SGST credit (NOT CGST credit).
3. If no relevant legal provisions were retrieved:
   - Set status = "unresolved".
   - Set unresolved_reason = "No statutory provisions regarding credit utilization or classification were retrieved in context."
4. Legal Applicability & Factual Scope Rule:
   - Do not use a retrieved Section, Rule, Notification, Form, exception, refund mechanism, or procedure merely because it is related to the topic.
   - Before relying on legal evidence, verify that its factual scope matches the user's situation.
   - Distinguish between similar but legally different scenarios (e.g. tax collected on an exempt supply vs IGST paid instead of CGST/SGST). A provision governing one must not be applied to the other unless the retrieved text explicitly supports that application.

Return ONLY a JSON object matching this schema:
{
  "status": "resolved" | "unresolved",
  "supply_type": "interstate" | "intrastate" | null,
  "output_tax_type": string or null,
  "usable_credit_ledgers": ["igst", "cgst", "sgst"],
  "utilization_constraints": [string],
  "legal_evidence": [
    {
      "reference": string,
      "title": string,
      "excerpt": string
    }
  ],
  "unresolved_reason": string or null
}"""


class LegalEvidenceItem(BaseModel):
    model_config = ConfigDict(extra="ignore")
    reference: str = ""
    title: str = ""
    chunk_id: Optional[str] = None
    excerpt: str = ""


class StructuredLegalFindings(BaseModel):
    model_config = ConfigDict(extra="ignore")
    status: str = "unresolved"  # "resolved" | "unresolved"
    supply_type: Optional[str] = None  # "interstate" | "intrastate"
    output_tax_type: Optional[str] = None  # "IGST" | "CGST+SGST"
    usable_credit_ledgers: list[str] = Field(default_factory=list)
    usable_credit_balances: dict[str, float] = Field(default_factory=dict)
    total_usable_credit: Optional[float] = None
    utilization_constraints: list[str] = Field(default_factory=list)
    legal_evidence: list[LegalEvidenceItem] = Field(default_factory=list)
    unresolved_reason: Optional[str] = None

    @field_validator("usable_credit_ledgers", "utilization_constraints", "legal_evidence", mode="before")
    @classmethod
    def _ensure_list(cls, v: Any) -> list[Any]:
        if v is None or not isinstance(v, list):
            return []
        return v

    @field_validator("usable_credit_balances", mode="before")
    @classmethod
    def _ensure_dict(cls, v: Any) -> dict[str, Any]:
        if v is None or not isinstance(v, dict):
            return {}
        return v


def _find_rule88a_or_sec49_chunk(legal_chunks: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Check if retrieved legal chunks contain Rule 88A or Section 49 governing ITC order of utilization."""
    for c in legal_chunks:
        ref = str(c.get("reference") or "").lower()
        title = str(c.get("title") or "").lower()
        content = str(c.get("content") or "").lower()
        if "88a" in ref or "88a" in title or "rule 88a" in content:
            return c
        if "section 49" in ref or "section 49" in title or "section 49" in content:
            if "utilis" in content or "utiliz" in content:
                return c
    return None


def interpret_legal_findings_deterministic(
    legal_chunks: list[dict[str, Any]],
    user_premises: dict[str, Any],
) -> StructuredLegalFindings:
    """Deterministically extract grounded legal findings from retrieved chunks without LLM."""
    if not legal_chunks:
        return StructuredLegalFindings(
            status="unresolved",
            unresolved_reason="Legal knowledge-base gap: No legal context retrieved to establish credit utilization rules.",
        )

    matched_chunk = _find_rule88a_or_sec49_chunk(legal_chunks)
    if not matched_chunk:
        return StructuredLegalFindings(
            status="unresolved",
            unresolved_reason="Legal knowledge-base gap: Retrieved chunks do not contain Rule 88A or Section 49 governing ITC utilization order.",
        )

    supply_type = (user_premises.get("supply_type") or "interstate").lower()
    balances = user_premises.get("itc_balances") or {}

    ref = matched_chunk.get("reference") or "Rule 88A"
    title = matched_chunk.get("title") or "Order of utilization of input tax credit"
    content = matched_chunk.get("content") or ""
    chunk_id = matched_chunk.get("chunk_id")

    # Extract clean excerpt
    m_excerpt = re.search(r"(?:Input tax credit on account of integrated tax.*?utilised fully\.)", content, re.DOTALL | re.IGNORECASE)
    excerpt = m_excerpt.group(0) if m_excerpt else content[:250].strip()

    evidence = [
        LegalEvidenceItem(
            reference=ref,
            title=title,
            chunk_id=chunk_id,
            excerpt=excerpt,
        )
    ]

    if supply_type == "interstate":
        # Outward liability is IGST
        usable_ledgers = ["igst", "cgst", "sgst"]
        usable_balances = {k: float(v) for k, v in balances.items() if k.lower() in usable_ledgers and float(v) > 0}
        total_credit = round(sum(usable_balances.values()), 2)
        constraints = [
            "Input tax credit on account of integrated tax (IGST) shall first be utilised towards payment of integrated tax.",
            "After IGST credit is fully exhausted, input tax credit on account of CGST and SGST can be utilised towards payment of integrated tax in any order or proportion.",
        ]
        return StructuredLegalFindings(
            status="resolved",
            supply_type="interstate",
            output_tax_type="IGST",
            usable_credit_ledgers=usable_ledgers,
            usable_credit_balances=usable_balances,
            total_usable_credit=total_credit,
            utilization_constraints=constraints,
            legal_evidence=evidence,
        )
    else:
        # Intra-state supply: CGST + SGST liabilities
        usable_ledgers = ["cgst", "sgst", "igst"]
        usable_balances = {k: float(v) for k, v in balances.items() if k.lower() in usable_ledgers and float(v) > 0}
        total_credit = round(sum(usable_balances.values()), 2)
        constraints = [
            "Input tax credit on account of integrated tax must be exhausted first before utilizing CGST or SGST credit.",
            "Cross-utilization between CGST credit and SGST credit is not permitted.",
        ]
        return StructuredLegalFindings(
            status="resolved",
            supply_type="intrastate",
            output_tax_type="CGST+SGST",
            usable_credit_ledgers=usable_ledgers,
            usable_credit_balances=usable_balances,
            total_usable_credit=total_credit,
            utilization_constraints=constraints,
            legal_evidence=evidence,
        )


def interpret_legal_findings(
    legal_chunks: list[dict[str, Any]],
    user_premises: dict[str, Any],
    query: str,
    client: OpenAI | None = None,
    model: str | None = None,
    use_llm: bool = False,
) -> StructuredLegalFindings:
    """Extract grounded legal findings from retrieved chunks."""
    if not legal_chunks:
        return StructuredLegalFindings(
            status="unresolved",
            unresolved_reason="Legal knowledge-base gap: No legal documents retrieved.",
        )

    if not use_llm:
        return interpret_legal_findings_deterministic(legal_chunks, user_premises)

    # Use LLM with fallback
    try:
        from src.routers.planner import get_default_planner_client, get_default_planner_model
        llm_client = client or get_default_planner_client()
        if not llm_client:
            return interpret_legal_findings_deterministic(legal_chunks, user_premises)

        model_name = model or get_default_planner_model()
        formatted_chunks = [
            f"Reference: {c.get('reference')}\nTitle: {c.get('title')}\nContent: {c.get('content')}"
            for c in legal_chunks
        ]
        context_str = "\n\n---\n\n".join(formatted_chunks)

        b_str = ", ".join(f"{k.upper()}: {v}" for k, v in (user_premises.get("itc_balances") or {}).items())
        user_msg = (
            f"USER QUERY: {query}\n"
            f"SUPPLY TYPE: {user_premises.get('supply_type')}\n"
            f"USER CREDIT BALANCES: {b_str}\n\n"
            f"RETRIEVED LEGAL EVIDENCE:\n{context_str}"
        )

        if "gemma" in str(model_name).lower():
            messages = [
                {"role": "user", "content": f"{INTERPRETER_SYSTEM_PROMPT}\n\n---\n\n{user_msg}"}
            ]
        else:
            messages = [
                {"role": "system", "content": INTERPRETER_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]

        resp = llm_client.chat.completions.create(
            model=model_name,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        content = resp.choices[0].message.content or "{}"
        data = extract_json_payload(content)

        # Populate usable balances from user premises
        usable_ledgers = [k.lower() for k in data.get("usable_credit_ledgers", []) if isinstance(k, str)]
        user_b = user_premises.get("itc_balances") or {}
        usable_balances = {k: float(v) for k, v in user_b.items() if k.lower() in usable_ledgers and float(v) > 0}
        total_credit = round(sum(usable_balances.values()), 2) if usable_balances else None

        data["usable_credit_balances"] = usable_balances
        data["total_usable_credit"] = total_credit

        evidence_items = []
        for item in (data.get("legal_evidence") or []):
            if isinstance(item, dict):
                evidence_items.append(LegalEvidenceItem(**item))
        data["legal_evidence"] = evidence_items

        return StructuredLegalFindings(**data)
    except Exception as exc:
        logger.warning(f"LLM legal interpreter failed ({exc}), falling back to deterministic extraction.")
        return interpret_legal_findings_deterministic(legal_chunks, user_premises)
