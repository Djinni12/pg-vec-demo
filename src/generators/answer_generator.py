"""LLM answer generation module for the GST legal RAG pipeline."""

from __future__ import annotations

import os
import re
import time
from typing import Any

from dotenv import load_dotenv
import openai
from openai import OpenAI

from src.retrieval_inspector import DEFAULT_TOP_K, LoadedModels, inspect_retrieval
from src.retrievers.notification_retriever import retrieve_notifications
from src.retrievers.rate_retriever import retrieve_rates
from src.routers.query_router import RouteType, classify_query, plan_capabilities
from src.generators.legal_interpreter import (
    StructuredLegalFindings,
    interpret_legal_findings,
)
from src.tools.calculator import (
    CalculationInputs,
    CalculationResult,
    execute_calculator,
)

# Ensure environment variables from .env are loaded
load_dotenv(override=True)

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are a GST legal and tax information assistant.

Answer using ONLY the retrieved context.

Your answers may be read by taxpayers, accountants, tax professionals, lawyers, and other users.

LANGUAGE AND STYLE:
- Use clear, simple, professional English.
- Make the answer easy to understand without removing important legal details.
- Avoid unnecessarily complex legal wording.
- When a legal or technical term is necessary, use it but explain it in simple words.
- Do not replace precise legal terms when doing so would change their meaning.
- Keep relevant Sections, Rules, Forms, deadlines, conditions, exceptions, and requirements.
- Explain the practical meaning of a legal provision instead of merely repeating its wording.
- Prefer short sentences and well-organized headings.
- Use bullets or numbered steps when explaining a procedure.
- When answering tax rate questions, always structure your answer in three distinct parts:
  1. SIMPLE DIRECT ANSWER:
     - The very first sentence MUST answer the question directly in natural, human English as a normal person would.
     - NEVER start with database labels like "Item Description:", "Source Rate:", "Notification:", "According to the database...", or "The GST rate for ... is as follows:".
     - For EXEMPT products (rate_category: EXEMPTION or is_exempt: true):
       Start naturally: "[Product] is exempt from GST, so no GST is charged (0%)."
       Example: "Fresh milk is exempt from GST, so no GST is charged (0%)."
       Do NOT merely say "GST Rate: Exempt".
     - For NORMAL TAXABLE products (rate_category: CGST):
       Start directly with the total GST rate:
       "[Product] attracts [Total GST]% GST under HSN [Code]. For an intra-state supply, this consists of [CGST]% CGST and [SGST]% SGST."
       Example: "Butter attracts 5% GST under HSN 0405. For an intra-state supply, this consists of 2.5% CGST and 2.5% SGST."
     - For COMPENSATION CESS:
       Keep compensation cess distinct from base GST:
       If a base rate and compensation cess both apply:
       "[Product] attracts [Total GST]% GST under HSN [Code]. In addition, a compensation cess of [Cess] applies." (If cess is Nil: "In addition, Compensation Cess is Nil.")
       If the user specifically asked for compensation cess:
       Explain the cess directly (e.g., "The compensation cess on coal (HSN 2701) is Nil. In addition, coal attracts 18% base GST.").
       Never add compensation cess into the total GST rate, and never double cess.
     - For SPECIAL / CONDITIONAL rates (rate_category: SPECIAL or multiple condition-based rates):
       State that the GST rate depends on the applicable scheme/condition:
       "The GST rate for [Product] depends on the applicable scheme/ITC condition."
  2. OPTIONAL SHORT EXPLANATION:
     - Add a brief 1-2 sentence clarification if needed (e.g., explaining ITC conditions, distinction between loose vs packaged, or engine capacity thresholds).
  3. FULL RATE / SOURCE DETAILS:
     - Under "Details:" (or "Base GST Details:" and "Compensation Cess Details:" when both exist; or "Option 1:" and "Option 2:" when multiple condition-based options exist), list all retrieved source metadata cleanly:
       - Description: [Must ALWAYS show the full retrieved legal description verbatim. Never replace with a shortened product name]
       - HSN Code: [code]
       - GST Rate: [total GST, e.g. 5%, 18%, 40%, or Exempt (0%)]
       - CGST Rate: [CGST if applicable to taxable goods]
       - SGST Rate: [SGST if applicable to taxable goods]
       - IGST Rate: [IGST if applicable]
       - Source Rate: [exact retrieved source rate, e.g. 2.5%, Nil, 20%, 3%, 6%]
       - Compensation Cess: [when relevant, e.g. Nil, or specific cess rate]
       - Notification: [notification number]
       - Serial Number: [retrieved serial number]
       - Notification Date: [strictly labeled "Notification Date", never "Effective Date"]
       - Rate As On Date: [strictly labeled "Rate As On Date"]
       - Schedule: [retrieved schedule name]
       - Condition: [retrieved condition, or "None"]
       - Footnote: [if present]
       - Amendment Note: [if present]
     - Do NOT invent missing fields. Only show fields that apply to the record.
     - Never label Notification Date or Rate As On Date as Effective Date.
- Do not make the answer less detailed merely to make it easier to read.

GROUNDING:
- Use only information supported by the retrieved context.
- Do not invent legal requirements, Sections, Rules, Forms, dates, rates, or procedures.
- Final statutory GST rates, HSN codes, and legal provisions MUST be quoted strictly from the provided structured rate records or legal context. Never invent or estimate statutory tax rates or classifications.
- When the user asks for a calculation on a specific amount, or provides assumed numbers / ITC credit balances, perform the arithmetic accurately based on the retrieved rates or user balances, explaining the steps clearly.
- For ITC utilization and maximum supply value questions (e.g. how much value of goods can be supplied without paying cash tax):
  - Under Section 49(5) and Rule 88A, for an inter-state supply, the outward liability is IGST.
  - IGST liability must be offset first using IGST credit; once IGST credit is exhausted, CGST credit and SGST credit can both be utilized towards the remaining IGST liability in any order or proportion.
  - Therefore, all eligible credits (IGST + CGST + SGST) are available to cover the IGST liability.
  - The maximum taxable value that can be supplied without cash payment is: (Total Available Eligible ITC across all heads) / (Applicable IGST Rate).
  - Explicitly present the step-by-step credit utilization (exhausting IGST credit first, then applying CGST and SGST credits to clear the remaining IGST liability) confirming ₹0 cash payable.
- User-provided assumptions/numbers are hypothetical inputs for calculation; clearly state them as user assumptions and do not present them as verified statutory law.
- If no rate record matches the requested product or service, clearly state that the rate is not found in the available rate database.
- Clearly distinguish related concepts such as cancellation, suspension, and revocation of cancellation.
- Include only information relevant to the user's question.
- If the context is insufficient, clearly say so.
- Do not mention retrieval, embeddings, BM25, RRF, reranking, chunks, or internal system details.

REFERENCES:
- Mention relevant Sections, Rules, Forms, HSN/SAC codes, and notification references where supported.
- Use legal references to support the explanation, not as a substitute for explaining it."""


DIRECT_REASONING_SYSTEM_PROMPT = """You are a GST tax calculation assistant.

Answer directly using only the numbers, rates, discounts, and facts supplied in the user's question.

You may perform arithmetic and explain the formula step-by-step.
If the user provides an assumed or hypothetical GST rate (e.g., 'Assume GST is 12%'), use that rate directly to calculate the tax and total amounts. Clearly state that the calculation is based on the user's assumed rate and is not a verified statutory GST rate.

Do not look up, invent, estimate, or assume any GST law, statutory rate, threshold, exemption, eligibility rule, date, form, section, notification, or procedure.

If the user asks for an applicable statutory GST rate, legal rule, exception, threshold, eligibility condition, or any statutory fact not supplied in the question, say that document context is required."""


class GenerationError(Exception):
    """Raised when an error occurs during answer generation."""


def get_configured_model(model: str | None = None) -> str:
    """Return the configured OpenAI model name."""
    if model and model.strip():
        return model.strip()
    env_model = os.environ.get("OPENAI_MODEL")
    if env_model and env_model.strip():
        return env_model.strip()
    base_url = os.environ.get("OPENAI_BASE_URL", "")
    if "generativelanguage" in base_url:
        return "models/gemini-3.5-flash-lite"
    return DEFAULT_OPENAI_MODEL


def get_openai_client(
    api_key: str | None = None,
    base_url: str | None = None,
) -> OpenAI:
    """Create and return an OpenAI client using environment variables or explicit parameters."""
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY is not set. Please configure it in your environment or .env file.")

    url = base_url or os.environ.get("OPENAI_BASE_URL")
    kwargs: dict[str, Any] = {"api_key": key}
    if url:
        kwargs["base_url"] = url
    return OpenAI(**kwargs)


def format_chunk(chunk: dict[str, Any], index: int) -> str:
    """Format a single retrieved context chunk with required source metadata."""
    doc_type = chunk.get("document_type") or "unknown"
    ref = chunk.get("reference") or "No reference"
    title = chunk.get("title") or "Untitled"
    chunk_id = chunk.get("chunk_id") or f"chunk_{index}"
    content = chunk.get("content") or chunk.get("snippet") or ""

    return (
        f"--- Source {index} ---\n"
        f"Document Type: {doc_type}\n"
        f"Reference: {ref}\n"
        f"Title: {title}\n"
        f"Chunk ID: {chunk_id}\n"
        f"Content:\n"
        f"{content.strip()}"
    )


def format_context(chunks: list[dict[str, Any]]) -> str:
    """Format final reranked chunks into the retrieved context block."""
    if not chunks:
        return "No retrieved context available."
    return "\n\n".join(format_chunk(chunk, i) for i, chunk in enumerate(chunks, 1))


def format_rate_item(item: dict[str, Any], index: int) -> str:
    """Format a single structured GST rate result item with all key tariff fields."""
    item_type = (item.get("item_type") or "Goods").capitalize()
    code = item.get("code") or item.get("hsn_code") or "N/A"
    desc = item.get("description") or "No description"
    rate_cat = item.get("rate_category") or ""
    sec_heading = item.get("section_heading") or ""
    source_rate = item.get("source_rate") or item.get("gst_rate") or item.get("rate") or "Rate not specified"
    cgst_rate = item.get("cgst_rate")
    sgst_rate = item.get("sgst_rate")
    total_gst_rate = item.get("total_gst_rate")
    cess_rate = item.get("compensation_cess_rate") or item.get("compensation_cess")
    is_exempt = item.get("is_exempt") is True or str(item.get("is_exempt", "")).lower() == "true"
    schedule = item.get("schedule") or "Not specified"
    notif = item.get("notification_number") or item.get("notification_no") or ""
    serial = item.get("serial_no") or item.get("serial_number") or ""
    condition = item.get("condition") or "None"
    footnote = item.get("footnote") or ""
    amendment = item.get("amendment_note") or ""
    source_ref = item.get("source_reference") or item.get("source_file") or "GST rates2025.pdf"
    rate_as_on = item.get("rate_as_on_date")
    notif_date = item.get("notification_date")
    eff_date = item.get("effective_date")

    lines = [
        f"--- Rate Item {index} ({item_type}) ---",
        f"Tariff / HSN Code: {code}",
        f"Description: {desc}",
    ]
    if sec_heading:
        lines.append(f"Section Heading: {sec_heading}")
    if rate_cat:
        lines.append(f"Rate Category: {rate_cat}")

    # Specific rate presentation based on rate_category
    if rate_cat == "CGST":
        lines.append(f"Source Rate (CGST): {source_rate}")
        if total_gst_rate and cgst_rate and sgst_rate:
            lines.append(f"Derived CGST Rate: {cgst_rate}")
            lines.append(f"Derived SGST Rate: {sgst_rate}")
            lines.append(f"Total GST Rate: {total_gst_rate} (Intra-State: {cgst_rate} CGST + {sgst_rate} SGST; Inter-State: {total_gst_rate} IGST)")
        else:
            lines.append(f"CGST Rate: {source_rate}")
    elif rate_cat == "EXEMPTION" or is_exempt:
        lines.append(f"Source Rate: {source_rate}")
        lines.append("Exemption Status: Exempt (Total GST: 0%)")
    elif rate_cat == "COMPENSATION_CESS":
        lines.append(f"Compensation Cess Rate: {cess_rate or source_rate}")
        lines.append("Note: Additional Compensation Cess; does not replace base GST rate.")
    elif rate_cat in ("SPECIAL", "UNKNOWN"):
        lines.append(f"Source Rate: {source_rate}")
        lines.append("Note: Prescribed under special notification; no automatic SGST or total GST derivation.")
    else:
        rate_str = item.get("formatted_rate") or source_rate
        lines.append(f"GST Rate: {rate_str}")
        if item.get("cgst_rate_pct") is not None:
            cgst = f"{item['cgst_rate_pct']:g}%"
            sgst = f"{item['sgst_utgst_rate_pct']:g}%" if item.get("sgst_utgst_rate_pct") is not None else "N/A"
            igst = f"{item['igst_rate_pct']:g}%" if item.get("igst_rate_pct") is not None else "N/A"
            lines.append(f"CGST: {cgst} | SGST/UTGST: {sgst} | IGST: {igst}")

    lines.extend([
        f"Schedule: {schedule}",
        f"Serial Number: {serial}",
        f"Notification: {notif}",
        f"Condition: {condition}",
    ])
    if footnote:
        lines.append(f"Footnote: {footnote}")
    if amendment:
        lines.append(f"Amendment Note: {amendment}")

    if notif_date:
        lines.append(f"Notification Date: {notif_date}")
    if rate_as_on:
        lines.append(f"Rate As On Date: {rate_as_on}")
    if eff_date:
        lines.append(f"Effective Date: {eff_date}")

    lines.append(f"Source Reference: {source_ref}")
    return "\n".join(lines)


def format_rate_context(rates: list[dict[str, Any]]) -> str:
    """Format list of structured rate items into prompt context block."""
    if not rates:
        return "No matching rate entries found in the structured tariff database."
    return "\n\n".join(format_rate_item(r, i) for i, r in enumerate(rates, 1))


def extract_rate_pct(
    rate_results: list[dict[str, Any]] | None = None,
    user_premises: dict[str, Any] | None = None,
) -> float | None:
    """Extract applicable total tax rate percentage from user premises or rate results."""
    if user_premises and user_premises.get("assumed_rate") is not None:
        try:
            return float(user_premises["assumed_rate"])
        except (ValueError, TypeError):
            pass

    if not rate_results:
        return None

    r0 = rate_results[0]
    if r0.get("igst_rate_pct") is not None:
        try:
            val = float(r0["igst_rate_pct"])
            if val > 0:
                return val
        except (ValueError, TypeError):
            pass

    if r0.get("cgst_rate_pct") is not None:
        try:
            cgst = float(r0["cgst_rate_pct"])
            sgst = float(r0.get("sgst_utgst_rate_pct") or cgst)
            if cgst + sgst > 0:
                return cgst + sgst
        except (ValueError, TypeError):
            pass

    tot_str = str(r0.get("total_gst_rate") or r0.get("source_rate") or "")
    m = re.search(r"(\d+(?:\.\d+)?)%", tot_str)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass

    return None


def build_user_prompt(
    query: str,
    context: str | None = None,
    *,
    legal_context: str | None = None,
    rate_context: str | None = None,
    notification_context: str | None = None,
    user_premises: dict[str, Any] | None = None,
    legal_findings: StructuredLegalFindings | dict[str, Any] | None = None,
    calculation_result: CalculationResult | dict[str, Any] | None = None,
) -> str:
    """Build user prompt containing retrieved legal and/or rate context and question."""
    sections: list[str] = []
    if rate_context and rate_context.strip():
        sections.append(f"STRUCTURED GST RATE RECORDS\n\n{rate_context.strip()}")
    if legal_context and legal_context.strip():
        sections.append(f"RETRIEVED LEGAL CONTEXT\n\n{legal_context.strip()}")
    if notification_context and notification_context.strip():
        sections.append(f"RETRIEVED NOTIFICATION RECORDS\n\n{notification_context.strip()}")
    if not sections and context and context.strip():
        sections.append(context.strip())

    if user_premises:
        premise_lines = []
        if user_premises.get("assumed_rate") is not None:
            premise_lines.append(f"- User-Assumed Hypothetical GST Rate: {user_premises['assumed_rate']}% (Note: Use directly for calculation; state that it is user-assumed, not verified statutory law).")
        if user_premises.get("taxable_amount") is not None:
            premise_lines.append(f"- User Taxable / Base Amount: ₹{user_premises['taxable_amount']:,.2f}")
        if user_premises.get("discount_pct") is not None:
            premise_lines.append(f"- User Discount: {user_premises['discount_pct']}%")
        if user_premises.get("itc_balances"):
            b = user_premises["itc_balances"]
            balances_str = ", ".join(f"{k.upper()}: ₹{v:,.2f}" for k, v in b.items())
            total_itc = sum(b.values())
            premise_lines.append(f"- User Available ITC Balances: {balances_str}")
            premise_lines.append(f"- Total Combined Available ITC: ₹{total_itc:,.2f}")
        if user_premises.get("supply_type"):
            st = user_premises["supply_type"].capitalize()
            premise_lines.append(f"- Supply Type: {st} Supply")
        if premise_lines:
            sections.append("USER-PROVIDED PREMISES & INPUTS (HYPOTHETICAL / FACTUAL)\n\n" + "\n".join(premise_lines))

    if legal_findings:
        lf_dict = (
            legal_findings.model_dump()
            if isinstance(legal_findings, StructuredLegalFindings)
            else legal_findings
        )
        lf_status = lf_dict.get("status")
        lf_lines = [f"- Legal Determination Status: {lf_status}"]
        if lf_status == "resolved":
            if lf_dict.get("supply_type"):
                lf_lines.append(f"- Classified Supply Type: {str(lf_dict['supply_type']).capitalize()} Supply")
            if lf_dict.get("output_tax_type"):
                lf_lines.append(f"- Outward Tax Liability Type: {lf_dict['output_tax_type']}")
            if lf_dict.get("usable_credit_ledgers"):
                ledgers_str = ", ".join(k.upper() for k in lf_dict["usable_credit_ledgers"])
                lf_lines.append(f"- Usable Input Tax Credit Ledgers: {ledgers_str}")
            if lf_dict.get("usable_credit_balances"):
                b_str = ", ".join(f"{k.upper()}: ₹{v:,.2f}" for k, v in lf_dict["usable_credit_balances"].items())
                lf_lines.append(f"- Usable Credit Breakdown: {b_str}")
            if lf_dict.get("total_usable_credit") is not None:
                lf_lines.append(f"- Total Usable Input Tax Credit: ₹{lf_dict['total_usable_credit']:,.2f}")
            if lf_dict.get("utilization_constraints"):
                lf_lines.append("- Statutory Credit Utilization Order & Rules:")
                for c in lf_dict["utilization_constraints"]:
                    lf_lines.append(f"  * {c}")
            if lf_dict.get("legal_evidence"):
                lf_lines.append("- Retrieved Statutory Evidence / Grounding:")
                for ev in lf_dict["legal_evidence"]:
                    ref = ev.get("reference") or "Statutory Provision"
                    title = ev.get("title") or ""
                    excerpt = ev.get("excerpt") or ""
                    lf_lines.append(f"  * {ref} - {title}: \"{excerpt}\"")
        else:
            reason = lf_dict.get("unresolved_reason") or "Retrieved legal context does not establish the required statutory rule."
            lf_lines.append(f"- Knowledge-Base Gap Notice: {reason}")
            lf_lines.append(
                "  (CRITICAL INSTRUCTION: The retrieved legal context is insufficient to determine the statutory rule. "
                "You MUST explicitly report this knowledge-base gap to the user. Do NOT invent legal rules or assumptions.)"
            )
        sections.append("STRUCTURED LEGAL FINDINGS (ITC & STATUTORY DETERMINATION)\n\n" + "\n".join(lf_lines))

    if calculation_result:
        cr_dict = (
            calculation_result.model_dump()
            if isinstance(calculation_result, CalculationResult)
            else calculation_result
        )
        if cr_dict.get("status") == "success":
            res_val = cr_dict.get("result_value", 0.0) or 0.0
            cr_lines = [
                f"- Operation: {cr_dict.get('operation')}",
                f"- Status: Verified Success",
                f"- Verified Deterministic Formula: {cr_dict.get('formula')}",
                f"- Exact Computed Value: ₹{res_val:,.2f}",
            ]
            if cr_dict.get("steps"):
                cr_lines.append("- Verified Arithmetic Steps:")
                for s in cr_dict["steps"]:
                    cr_lines.append(f"  * {s}")
            cr_lines.append(
                f"(CRITICAL ARITHMETIC INSTRUCTION: The value ₹{res_val:,.2f} is deterministically computed by the system calculator. "
                f"You MUST use and quote this exact figure ₹{res_val:,.2f} in your answer. Do NOT perform independent calculations with different numbers.)"
            )
            sections.append("CALCULATOR ARITHMETIC RESULT (VERIFIED DETERMINISTIC ARITHMETIC)\n\n" + "\n".join(cr_lines))
        else:
            err = cr_dict.get("error_message") or "Calculation could not be completed."
            sections.append(f"CALCULATOR ARITHMETIC RESULT\n\n- Status: Error\n- Message: {err}")

    combined_context = "\n\n".join(sections) if sections else "No retrieved context available."
    return (
        f"RETRIEVED CONTEXT\n\n"
        f"{combined_context}\n\n"
        f"USER QUESTION\n\n"
        f"{query}\n\n"
        f"ANSWER"
    )


def build_direct_user_prompt(
    query: str,
    calculation_result: CalculationResult | dict[str, Any] | None = None,
) -> str:
    """Build prompt for direct arithmetic/reasoning with optional deterministic calculation result."""
    prompt_parts = []
    if calculation_result:
        cr_dict = (
            calculation_result.model_dump()
            if isinstance(calculation_result, CalculationResult)
            else calculation_result
        )
        if cr_dict.get("status") == "success":
            res_val = cr_dict.get("result_value", 0.0) or 0.0
            steps_str = "\n".join(f"  * {s}" for s in cr_dict.get("steps", []))
            prompt_parts.append(
                f"CALCULATOR ARITHMETIC RESULT (VERIFIED DETERMINISTIC ARITHMETIC)\n"
                f"- Operation: {cr_dict.get('operation')}\n"
                f"- Formula: {cr_dict.get('formula')}\n"
                f"- Exact Result: ₹{res_val:,.2f}\n"
                f"- Steps:\n{steps_str}\n"
                f"(CRITICAL: Quote and rely on this exact verified arithmetic result ₹{res_val:,.2f}.)\n"
            )
    prompt_parts.append(f"USER QUESTION\n\n{query}\n\nANSWER")
    return "\n\n".join(prompt_parts)


def build_full_prompt(
    query: str,
    chunks: list[dict[str, Any]] | None = None,
    rate_results: list[dict[str, Any]] | None = None,
    context: str | None = None,
) -> str:
    """Build the complete grounded prompt string."""
    legal_ctx = format_context(chunks) if chunks is not None else None
    rate_ctx = format_rate_context(rate_results) if rate_results is not None else None
    user_prompt = build_user_prompt(
        query,
        context=context,
        legal_context=legal_ctx,
        rate_context=rate_ctx,
    )
    return (
        f"SYSTEM PROMPT\n\n"
        f"{SYSTEM_PROMPT}\n\n"
        f"{user_prompt}"
    )


def build_direct_full_prompt(query: str) -> str:
    """Build the complete prompt string for direct arithmetic/reasoning."""
    return (
        f"SYSTEM PROMPT\n\n"
        f"{DIRECT_REASONING_SYSTEM_PROMPT}\n\n"
        f"{build_direct_user_prompt(query)}"
    )


def strip_reasoning(text: str) -> str:
    """Strip chain-of-thought or internal reasoning tags to prevent leaking thought tokens."""
    if not text:
        return ""
    # Strip closed thought/thinking tags
    cleaned = re.sub(r"<(thought|thinking)>[\s\S]*?</\1>", "", text, flags=re.IGNORECASE)
    # Strip any unclosed thought/thinking tag that starts the response
    cleaned = re.sub(r"^<(thought|thinking)>[\s\S]*$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def extract_sources(chunks: list[dict[str, Any]], start_rank: int = 1) -> list[dict[str, Any]]:
    """Extract source metadata items from final reranked chunks."""
    sources: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks, start_rank):
        content = chunk.get("content") or chunk.get("snippet") or ""
        sources.append(
            {
                "rank": chunk.get("rank", index),
                "document_type": chunk.get("document_type"),
                "reference": chunk.get("reference"),
                "title": chunk.get("title"),
                "chunk_id": chunk.get("chunk_id"),
                "content": content,
                "snippet": chunk.get("snippet") or _snippet(content),
                "reranker_score": chunk.get("reranker_score"),
            }
        )
    return sources


def extract_rate_sources(
    rate_results: list[dict[str, Any]], start_rank: int = 1
) -> list[dict[str, Any]]:
    """Extract source metadata items from structured rate results for unified citation display."""
    sources: list[dict[str, Any]] = []
    for index, item in enumerate(rate_results, start_rank):
        code = item.get("code") or item.get("hsn_code") or "N/A"
        item_type = item.get("item_type") or "goods"
        prefix = "HSN" if item_type == "goods" else "SAC"
        desc = item.get("description") or ""
        rate_cat = item.get("rate_category") or ""
        source_rate = item.get("source_rate") or item.get("gst_rate") or item.get("rate") or "Rate not specified"
        total_gst_rate = item.get("total_gst_rate")
        cgst_rate = item.get("cgst_rate")
        sgst_rate = item.get("sgst_rate")
        cess_rate = item.get("compensation_cess_rate") or item.get("compensation_cess")
        notif = item.get("notification_number") or item.get("notification_no") or ""
        sec_heading = item.get("section_heading") or ""

        if rate_cat == "CGST" and total_gst_rate:
            rate_label = f"Total GST: {total_gst_rate} (CGST: {cgst_rate}, SGST: {sgst_rate})"
            snippet_rate = f"Total GST: {total_gst_rate} (CGST: {cgst_rate} + SGST: {sgst_rate})"
        elif rate_cat == "EXEMPTION":
            rate_label = f"Exempt (0% GST, Source: {source_rate})"
            snippet_rate = "Exempt (0% GST)"
        elif rate_cat == "COMPENSATION_CESS":
            rate_label = f"Compensation Cess: {cess_rate or source_rate}"
            snippet_rate = f"Cess: {cess_rate or source_rate}"
        elif rate_cat in ("SPECIAL", "UNKNOWN"):
            rate_label = f"Special Rate: {source_rate}"
            snippet_rate = f"Special Rate: {source_rate}"
        else:
            rate_label = f"Applicable Rate: {source_rate}"
            snippet_rate = source_rate

        cond = f"\nCondition: {item['condition']}" if item.get("condition") else ""
        footnote = f"\nFootnote: {item['footnote']}" if item.get("footnote") else ""
        amendment = f"\nAmendment: {item['amendment_note']}" if item.get("amendment_note") else ""

        dates = []
        if item.get("notification_date"):
            dates.append(f"Notification Date: {item['notification_date']}")
        if item.get("rate_as_on_date"):
            dates.append(f"Rate As On Date: {item['rate_as_on_date']}")
        if item.get("effective_date"):
            dates.append(f"Effective Date: {item['effective_date']}")
        date_str = ("\n" + "\n".join(dates)) if dates else ""

        sec_str = f"\nSection Heading: {sec_heading}" if sec_heading else ""
        notif_str = f"\nNotification: {notif}" if notif else ""
        source_ref = item.get("source_reference") or item.get("source_file") or "GST rates2025.pdf"

        content = (
            f"Tariff Code: {prefix} {code}\n"
            f"Rate Details: {rate_label}"
            f"{sec_str}"
            f"{notif_str}"
            f"{cond}"
            f"{footnote}"
            f"{amendment}"
            f"{date_str}\n"
            f"Description: {desc}\n"
            f"Source Reference: {source_ref}"
        )
        sources.append(
            {
                "rank": index,
                "document_type": item_type,
                "reference": f"{prefix} {code}",
                "title": _snippet(desc, 80),
                "chunk_id": f"rate_{item_type}_{code}_{index}",
                "content": content,
                "snippet": f"{snippet_rate} · {_snippet(desc, 120)}",
                "reranker_score": item.get("score"),
            }
        )
    return sources


def extract_combined_sources(
    chunks: list[dict[str, Any]] | None = None,
    rate_results: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Extract and combine sources from structured rates and legal chunks."""
    sources: list[dict[str, Any]] = []
    if rate_results:
        sources.extend(extract_rate_sources(rate_results, start_rank=1))
    if chunks:
        sources.extend(extract_sources(chunks, start_rank=len(sources) + 1))
    return sources


def _snippet(content: str, max_length: int = 280) -> str:
    """Generate a readable snippet from content text."""
    if len(content) <= max_length:
        return content
    truncate_at = content.rfind(" ", 0, max_length)
    if truncate_at == -1:
        truncate_at = max_length
    return content[:truncate_at] + "..."


def generate_answer(
    query: str,
    chunks: list[dict[str, Any]] | None = None,
    *,
    rate_results: list[dict[str, Any]] | None = None,
    notification_chunks: list[dict[str, Any]] | None = None,
    user_premises: dict[str, Any] | None = None,
    legal_findings: StructuredLegalFindings | dict[str, Any] | None = None,
    calculation_result: CalculationResult | dict[str, Any] | None = None,
    model: str | None = None,
    client: OpenAI | None = None,
    temperature: float = 0.1,
    max_tokens: int = 1500,
    direct_reasoning: bool = False,
) -> dict[str, Any]:
    """Send grounded context to OpenAI and return the clean answer with timings and sources."""
    if not query.strip():
        raise ValueError("query cannot be empty")

    model_name = get_configured_model(model)
    client = client or get_openai_client()

    if direct_reasoning:
        system_prompt = DIRECT_REASONING_SYSTEM_PROMPT
        user_prompt = build_direct_user_prompt(query, calculation_result=calculation_result)
    else:
        system_prompt = SYSTEM_PROMPT
        legal_ctx = format_context(chunks or []) if chunks else ""
        rate_ctx = format_rate_context(rate_results or []) if rate_results else ""
        notif_ctx = format_context(notification_chunks or []) if notification_chunks else ""
        user_prompt = build_user_prompt(
            query,
            legal_context=legal_ctx,
            rate_context=rate_ctx,
            notification_context=notif_ctx,
            user_premises=user_premises,
            legal_findings=legal_findings,
            calculation_result=calculation_result,
        )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    gen_start = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except openai.AuthenticationError as exc:
        raise GenerationError(f"OpenAI authentication failed: {exc}") from exc
    except openai.RateLimitError as exc:
        raise GenerationError(f"OpenAI rate limit exceeded: {exc}") from exc
    except openai.APIConnectionError as exc:
        raise GenerationError(f"OpenAI connection error: {exc}") from exc
    except openai.OpenAIError as exc:
        raise GenerationError(f"OpenAI API error: {exc}") from exc
    except Exception as exc:
        raise GenerationError(f"Generation failed: {exc}") from exc

    gen_ms = round((time.perf_counter() - gen_start) * 1000, 3)

    choice = response.choices[0] if response.choices else None
    raw_content = choice.message.content if choice and choice.message else ""
    answer = strip_reasoning(raw_content or "")

    all_chunks = (chunks or []) + (notification_chunks or [])
    sources = extract_combined_sources(all_chunks, rate_results)

    return {
        "answer": answer,
        "sources": sources,
        "rate_results": rate_results or [],
        "generation_timing": gen_ms,
        "model_used": model_name,
    }


DEFAULT_ANSWER_TOP_K = 5


def run_gst_answer_flow(
    query: str,
    top_k: int = DEFAULT_ANSWER_TOP_K,
    *,
    models: LoadedModels,
    openai_model: str | None = None,
    openai_client: OpenAI | None = None,
    db_url: str | None = None,
    route_override: str | None = None,
) -> dict[str, Any]:
    """Execute capability-routed retrieval pipeline and generate a grounded answer."""
    plan = plan_capabilities(query)

    # 1. Clarification fast-path if query is ambiguous
    if plan.get("needs_clarification") and not route_override:
        clarification_text = plan.get("clarification_prompt") or "Please provide more details."
        obs = {
            "plan": plan,
            "tools_executed": [],
            "retrieved_evidence": {"rates_count": 0, "legal_chunks_count": 0, "notification_chunks_count": 0},
            "structured_legal_findings": None,
            "calculation_inputs": None,
            "calculation_result": None,
            "sources_used": [],
            "knowledge_base_gap": None,
        }
        return {
            "query": query,
            "route": "clarification",
            "answer": clarification_text,
            "rate_results": [],
            "sources": [],
            "sources_used": [],
            "retrieval_timing": 0.0,
            "generation_timing": 0.0,
            "total_timing": 0.0,
            "model_used": get_configured_model(openai_model),
            "model": get_configured_model(openai_model),
            "plan": plan,
            "tools_executed": [],
            "structured_legal_findings": None,
            "calculation_inputs": None,
            "calculation_result": None,
            "observability": obs,
            "timings_ms": {
                "retrieval": 0.0,
                "generation": 0.0,
                "total": 0.0,
                "rate_lookup": 0.0,
                "notification_lookup": 0.0,
                "dense": 0.0,
                "bm25": 0.0,
                "rrf": 0.0,
                "reranker": 0.0,
            },
            "retrieval_debug": {
                "results": [],
                "dense_results": [],
                "bm25_results": [],
                "hybrid_results": [],
                "metadata": {},
                "models": {},
                "config": {},
            },
        }

    # 2. Capability Tool Execution
    if route_override:
        route = RouteType(route_override)
        needs_rate = route in (RouteType.RATE, RouteType.MIXED)
        needs_legal = route in (RouteType.LEGAL, RouteType.MIXED)
        needs_notif = False
        is_direct = route == RouteType.DIRECT
        route_str = route.value
    else:
        needs_rate = plan.get("needs_structured_rate_lookup", False) or plan.get("needs_hsn_lookup", False)
        needs_legal = plan.get("needs_legal_retrieval", False)
        needs_notif = plan.get("needs_notification_retrieval", False)
        is_direct = (
            plan.get("needs_direct_reasoning", False)
            and not needs_rate
            and not needs_legal
            and not needs_notif
        )
        if is_direct:
            route_str = "direct"
        elif needs_rate and (needs_legal or needs_notif):
            route_str = "mixed"
        elif needs_rate:
            route_str = "rate"
        elif needs_legal or needs_notif:
            route_str = "legal"
        else:
            route_str = "legal"

    rate_results: list[dict[str, Any]] = []
    legal_chunks: list[dict[str, Any]] = []
    notif_chunks: list[dict[str, Any]] = []
    rate_timing = 0.0
    notif_timing = 0.0
    tools_executed: list[dict[str, Any]] = []

    retrieval_data: dict[str, Any] = {
        "results": [],
        "dense_results": [],
        "bm25_results": [],
        "hybrid_results": [],
        "metadata": {},
        "models": {},
        "config": {},
        "timings_ms": {"total": 0.0, "dense": 0.0, "bm25": 0.0, "rrf": 0.0, "reranker": 0.0},
    }

    if needs_rate:
        rate_start = time.perf_counter()
        rate_q = plan.get("clean_subqueries", {}).get("rate_query") or query
        rate_results = retrieve_rates(rate_q, db_url=db_url, limit=top_k)
        rate_timing = round((time.perf_counter() - rate_start) * 1000, 3)
        tools_executed.append({
            "tool": "retrieve_rates",
            "query": rate_q,
            "results_count": len(rate_results),
            "timing_ms": rate_timing,
        })

    if needs_legal:
        legal_q = plan.get("clean_subqueries", {}).get("legal_query") or query
        retrieval_data = inspect_retrieval(legal_q, top_k=top_k, models=models, db_url=db_url)
        legal_chunks = retrieval_data.get("results") or []
        legal_timing = retrieval_data.get("timings_ms", {}).get("total", 0.0)
        tools_executed.append({
            "tool": "inspect_retrieval",
            "query": legal_q,
            "results_count": len(legal_chunks),
            "timing_ms": legal_timing,
        })

    if needs_notif:
        notif_start = time.perf_counter()
        notif_q = plan.get("clean_subqueries", {}).get("notification_query") or query
        notif_chunks = retrieve_notifications(
            notif_q,
            top_k=top_k,
            db_url=db_url,
            model=models.embedding_model if models else None,
        )
        notif_timing = round((time.perf_counter() - notif_start) * 1000, 3)
        tools_executed.append({
            "tool": "retrieve_notifications",
            "query": notif_q,
            "results_count": len(notif_chunks),
            "timing_ms": notif_timing,
        })

    legal_timing = retrieval_data.get("timings_ms", {}).get("total", 0.0)
    retrieval_timing = round(rate_timing + legal_timing + notif_timing, 3)
    final_chunks = legal_chunks + notif_chunks

    # --- Stage 2: Structured Legal Findings ---
    legal_findings: StructuredLegalFindings | None = None
    user_premises = plan.get("user_premises") or {}
    has_itc_balances = bool(user_premises.get("itc_balances"))

    if needs_legal and (plan.get("needs_calculation") or has_itc_balances):
        interp_start = time.perf_counter()
        legal_findings = interpret_legal_findings(
            legal_chunks=legal_chunks,
            user_premises=user_premises,
            query=query,
        )
        interp_timing = round((time.perf_counter() - interp_start) * 1000, 3)
        tools_executed.append({
            "tool": "interpret_legal_findings",
            "status": legal_findings.status,
            "supply_type": legal_findings.supply_type,
            "usable_credit_ledgers": legal_findings.usable_credit_ledgers,
            "total_usable_credit": legal_findings.total_usable_credit,
            "unresolved_reason": legal_findings.unresolved_reason,
            "timing_ms": interp_timing,
        })

    # --- Stage 2: Deterministic Calculator Execution ---
    calc_inputs: CalculationInputs | None = None
    calc_result: CalculationResult | None = None

    if plan.get("needs_calculation"):
        calc_start = time.perf_counter()
        # Case 1: ITC Utilization question
        if has_itc_balances:
            if legal_findings is not None and legal_findings.status == "unresolved":
                tools_executed.append({
                    "tool": "execute_calculator",
                    "status": "skipped",
                    "reason": "unresolved_legal_findings",
                    "details": legal_findings.unresolved_reason,
                })
            elif legal_findings is not None and legal_findings.status == "resolved":
                tax_rate = extract_rate_pct(rate_results, user_premises)
                if tax_rate and tax_rate > 0 and legal_findings.total_usable_credit is not None:
                    calc_inputs = CalculationInputs(
                        operation="max_taxable_value_from_credit",
                        available_eligible_credit=legal_findings.total_usable_credit,
                        tax_rate_pct=tax_rate,
                        credit_breakdown=legal_findings.usable_credit_balances,
                    )
                    calc_result = execute_calculator(calc_inputs)
                    calc_timing = round((time.perf_counter() - calc_start) * 1000, 3)
                    tools_executed.append({
                        "tool": "execute_calculator",
                        "operation": calc_inputs.operation,
                        "status": calc_result.status,
                        "result_value": calc_result.result_value,
                        "formula": calc_result.formula,
                        "timing_ms": calc_timing,
                    })

        # Case 2: User Taxable / Base Amount with Discount or Direct
        elif user_premises.get("taxable_amount") is not None or user_premises.get("base_amount") is not None:
            base_amt = float(user_premises.get("taxable_amount") or user_premises.get("base_amount") or 0.0)
            disc_pct = float(user_premises.get("discount_pct") or 0.0)
            tax_rate = extract_rate_pct(rate_results, user_premises) or 5.0

            if disc_pct > 0:
                calc_inputs = CalculationInputs(
                    operation="discount_and_tax",
                    base_amount=base_amt,
                    discount_pct=disc_pct,
                    tax_rate_pct=tax_rate,
                )
            else:
                calc_inputs = CalculationInputs(
                    operation="tax_on_value",
                    taxable_value=base_amt,
                    tax_rate_pct=tax_rate,
                )
            calc_result = execute_calculator(calc_inputs)
            calc_timing = round((time.perf_counter() - calc_start) * 1000, 3)
            tools_executed.append({
                "tool": "execute_calculator",
                "operation": calc_inputs.operation,
                "status": calc_result.status,
                "result_value": calc_result.result_value,
                "formula": calc_result.formula,
                "timing_ms": calc_timing,
            })

    gen_data = generate_answer(
        query=query,
        chunks=legal_chunks,
        notification_chunks=notif_chunks,
        rate_results=rate_results,
        user_premises=plan.get("user_premises"),
        legal_findings=legal_findings,
        calculation_result=calc_result,
        model=openai_model,
        client=openai_client,
        direct_reasoning=is_direct,
    )

    generation_timing = gen_data["generation_timing"]
    total_timing = round(retrieval_timing + generation_timing, 3)

    observability = {
        "plan": plan,
        "tools_executed": tools_executed,
        "retrieved_evidence": {
            "rates_count": len(rate_results),
            "legal_chunks_count": len(legal_chunks),
            "notification_chunks_count": len(notif_chunks),
        },
        "structured_legal_findings": (
            legal_findings.model_dump() if legal_findings else None
        ),
        "calculation_inputs": (
            calc_inputs.model_dump() if calc_inputs else None
        ),
        "calculation_result": (
            calc_result.model_dump() if calc_result else None
        ),
        "sources_used": gen_data["sources"],
        "knowledge_base_gap": (
            legal_findings.unresolved_reason
            if (legal_findings and legal_findings.status == "unresolved")
            else None
        ),
    }

    return {
        "query": query,
        "route": route_str,
        "answer": gen_data["answer"],
        "rate_results": rate_results,
        "sources": gen_data["sources"],
        "sources_used": gen_data["sources"],
        "retrieval_timing": retrieval_timing,
        "generation_timing": generation_timing,
        "total_timing": total_timing,
        "model_used": gen_data["model_used"],
        "model": gen_data["model_used"],
        "plan": plan,
        "tools_executed": tools_executed,
        "structured_legal_findings": (
            legal_findings.model_dump() if legal_findings else None
        ),
        "calculation_inputs": (
            calc_inputs.model_dump() if calc_inputs else None
        ),
        "calculation_result": (
            calc_result.model_dump() if calc_result else None
        ),
        "observability": observability,
        "timings_ms": {
            "retrieval": retrieval_timing,
            "generation": generation_timing,
            "total": total_timing,
            "rate_lookup": rate_timing,
            "notification_lookup": notif_timing,
            "dense": retrieval_data["timings_ms"].get("dense", 0.0),
            "bm25": retrieval_data["timings_ms"].get("bm25", 0.0),
            "rrf": retrieval_data["timings_ms"].get("rrf", 0.0),
            "reranker": retrieval_data["timings_ms"].get("reranker", 0.0),
        },
        "retrieval_debug": {
            "results": final_chunks,
            "dense_results": retrieval_data.get("dense_results", []),
            "bm25_results": retrieval_data.get("bm25_results", []),
            "hybrid_results": retrieval_data.get("hybrid_results", []),
            "metadata": retrieval_data.get("metadata", {}),
            "models": retrieval_data.get("models", {}),
            "config": retrieval_data.get("config", {}),
        },
    }


class StreamingThoughtFilter:
    """Filters out <thought>...</thought> or <thinking>...</thinking> during live token streaming."""

    def __init__(self):
        self.in_thought = False
        self.buffer = ""

    def process(self, delta: str) -> str:
        if not delta:
            return ""
        self.buffer += delta
        output: list[str] = []

        while self.buffer:
            if self.in_thought:
                lower = self.buffer.lower()
                m_end = re.search(r"</(thought|thinking)>", lower)
                if m_end:
                    self.buffer = self.buffer[m_end.end():]
                    self.in_thought = False
                else:
                    m_partial = re.search(r"</[a-zA-Z]*$", lower)
                    if m_partial:
                        self.buffer = self.buffer[m_partial.start():]
                    else:
                        self.buffer = ""
                    break
            else:
                lower = self.buffer.lower()
                m_start = re.search(r"<(thought|thinking)>", lower)
                if m_start:
                    output.append(self.buffer[:m_start.start()])
                    self.buffer = self.buffer[m_start.end():]
                    self.in_thought = True
                else:
                    m_partial = re.search(r"<[a-zA-Z]*$", self.buffer)
                    if m_partial:
                        safe_idx = m_partial.start()
                        output.append(self.buffer[:safe_idx])
                        self.buffer = self.buffer[safe_idx:]
                        break
                    else:
                        output.append(self.buffer)
                        self.buffer = ""
                        break

        return "".join(output)

    def flush(self) -> str:
        if self.in_thought:
            return ""
        output = self.buffer
        self.buffer = ""
        return output


def stream_answer(
    query: str,
    chunks: list[dict[str, Any]] | None = None,
    *,
    rate_results: list[dict[str, Any]] | None = None,
    notification_chunks: list[dict[str, Any]] | None = None,
    user_premises: dict[str, Any] | None = None,
    legal_findings: StructuredLegalFindings | dict[str, Any] | None = None,
    calculation_result: CalculationResult | dict[str, Any] | None = None,
    model: str | None = None,
    client: OpenAI | None = None,
    temperature: float = 0.1,
    max_tokens: int = 1500,
    direct_reasoning: bool = False,
):
    """Stream response tokens from OpenAI, yielding (event_type, delta)."""
    if not query.strip():
        raise ValueError("query cannot be empty")

    model_name = get_configured_model(model)
    client = client or get_openai_client()

    if direct_reasoning:
        system_prompt = DIRECT_REASONING_SYSTEM_PROMPT
        user_prompt = build_direct_user_prompt(query, calculation_result=calculation_result)
    else:
        system_prompt = SYSTEM_PROMPT
        legal_ctx = format_context(chunks or []) if chunks else ""
        rate_ctx = format_rate_context(rate_results or []) if rate_results else ""
        notif_ctx = format_context(notification_chunks or []) if notification_chunks else ""
        user_prompt = build_user_prompt(
            query,
            legal_context=legal_ctx,
            rate_context=rate_ctx,
            notification_context=notif_ctx,
            user_premises=user_premises,
            legal_findings=legal_findings,
            calculation_result=calculation_result,
        )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        stream = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
    except openai.AuthenticationError as exc:
        raise GenerationError(f"OpenAI authentication failed: {exc}") from exc
    except openai.RateLimitError as exc:
        raise GenerationError(f"OpenAI rate limit exceeded: {exc}") from exc
    except openai.APIConnectionError as exc:
        raise GenerationError(f"OpenAI connection error: {exc}") from exc
    except openai.OpenAIError as exc:
        raise GenerationError(f"OpenAI API error: {exc}") from exc
    except Exception as exc:
        raise GenerationError(f"Generation failed: {exc}") from exc

    thought_filter = StreamingThoughtFilter()
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            raw_delta = chunk.choices[0].delta.content
            clean_delta = thought_filter.process(raw_delta)
            if clean_delta:
                yield ("token", clean_delta)

    trailing = thought_filter.flush()
    if trailing:
        yield ("token", trailing)


def stream_gst_answer_flow(
    query: str,
    top_k: int = DEFAULT_ANSWER_TOP_K,
    *,
    models: LoadedModels,
    openai_model: str | None = None,
    openai_client: OpenAI | None = None,
    db_url: str | None = None,
    route_override: str | None = None,
):
    """Stream retrieval and generation events for real-time live answers."""
    plan = plan_capabilities(query)
    model_name = get_configured_model(openai_model)

    if plan.get("needs_clarification") and not route_override:
        clarification_text = plan.get("clarification_prompt") or "Please provide more details."
        obs = {
            "plan": plan,
            "tools_executed": [],
            "retrieved_evidence": {"rates_count": 0, "legal_chunks_count": 0, "notification_chunks_count": 0},
            "structured_legal_findings": None,
            "calculation_inputs": None,
            "calculation_result": None,
            "sources_used": [],
            "knowledge_base_gap": None,
        }
        yield {
            "type": "meta",
            "query": query,
            "route": "clarification",
            "rate_results": [],
            "sources": [],
            "sources_used": [],
            "retrieval_timing": 0.0,
            "model_used": model_name,
            "model": model_name,
            "plan": plan,
            "tools_executed": [],
            "structured_legal_findings": None,
            "calculation_inputs": None,
            "calculation_result": None,
            "observability": obs,
            "retrieval_debug": {},
        }
        yield {
            "type": "token",
            "delta": clarification_text,
        }
        yield {
            "type": "done",
            "answer": clarification_text,
            "route": "clarification",
            "rate_results": [],
            "generation_timing": 0.0,
            "total_timing": 0.0,
            "plan": plan,
            "tools_executed": [],
            "structured_legal_findings": None,
            "calculation_inputs": None,
            "calculation_result": None,
            "observability": obs,
            "timings_ms": {"retrieval": 0.0, "generation": 0.0, "total": 0.0},
        }
        return

    if route_override:
        route = RouteType(route_override)
        needs_rate = route in (RouteType.RATE, RouteType.MIXED)
        needs_legal = route in (RouteType.LEGAL, RouteType.MIXED)
        needs_notif = False
        is_direct = route == RouteType.DIRECT
        route_str = route.value
    else:
        needs_rate = plan.get("needs_structured_rate_lookup", False) or plan.get("needs_hsn_lookup", False)
        needs_legal = plan.get("needs_legal_retrieval", False)
        needs_notif = plan.get("needs_notification_retrieval", False)
        is_direct = (
            plan.get("needs_direct_reasoning", False)
            and not needs_rate
            and not needs_legal
            and not needs_notif
        )
        if is_direct:
            route_str = "direct"
        elif needs_rate and (needs_legal or needs_notif):
            route_str = "mixed"
        elif needs_rate:
            route_str = "rate"
        elif needs_legal or needs_notif:
            route_str = "legal"
        else:
            route_str = "legal"

    rate_results: list[dict[str, Any]] = []
    legal_chunks: list[dict[str, Any]] = []
    notif_chunks: list[dict[str, Any]] = []
    rate_timing = 0.0
    notif_timing = 0.0
    tools_executed: list[dict[str, Any]] = []

    retrieval_data: dict[str, Any] = {
        "results": [],
        "dense_results": [],
        "bm25_results": [],
        "hybrid_results": [],
        "metadata": {},
        "models": {},
        "config": {},
        "timings_ms": {"total": 0.0, "dense": 0.0, "bm25": 0.0, "rrf": 0.0, "reranker": 0.0},
    }

    if needs_rate:
        rate_start = time.perf_counter()
        rate_q = plan.get("clean_subqueries", {}).get("rate_query") or query
        rate_results = retrieve_rates(rate_q, db_url=db_url, limit=top_k)
        rate_timing = round((time.perf_counter() - rate_start) * 1000, 3)
        tools_executed.append({
            "tool": "retrieve_rates",
            "query": rate_q,
            "results_count": len(rate_results),
            "timing_ms": rate_timing,
        })

    if needs_legal:
        legal_q = plan.get("clean_subqueries", {}).get("legal_query") or query
        retrieval_data = inspect_retrieval(legal_q, top_k=top_k, models=models, db_url=db_url)
        legal_chunks = retrieval_data.get("results") or []
        legal_timing = retrieval_data.get("timings_ms", {}).get("total", 0.0)
        tools_executed.append({
            "tool": "inspect_retrieval",
            "query": legal_q,
            "results_count": len(legal_chunks),
            "timing_ms": legal_timing,
        })

    if needs_notif:
        notif_start = time.perf_counter()
        notif_q = plan.get("clean_subqueries", {}).get("notification_query") or query
        notif_chunks = retrieve_notifications(
            notif_q,
            top_k=top_k,
            db_url=db_url,
            model=models.embedding_model if models else None,
        )
        notif_timing = round((time.perf_counter() - notif_start) * 1000, 3)
        tools_executed.append({
            "tool": "retrieve_notifications",
            "query": notif_q,
            "results_count": len(notif_chunks),
            "timing_ms": notif_timing,
        })

    legal_timing = retrieval_data.get("timings_ms", {}).get("total", 0.0)
    retrieval_timing = round(rate_timing + legal_timing + notif_timing, 3)

    final_chunks = legal_chunks + notif_chunks
    sources = extract_combined_sources(final_chunks, rate_results)

    # --- Stage 2: Structured Legal Findings ---
    legal_findings: StructuredLegalFindings | None = None
    user_premises = plan.get("user_premises") or {}
    has_itc_balances = bool(user_premises.get("itc_balances"))

    if needs_legal and (plan.get("needs_calculation") or has_itc_balances):
        interp_start = time.perf_counter()
        legal_findings = interpret_legal_findings(
            legal_chunks=legal_chunks,
            user_premises=user_premises,
            query=query,
        )
        interp_timing = round((time.perf_counter() - interp_start) * 1000, 3)
        tools_executed.append({
            "tool": "interpret_legal_findings",
            "status": legal_findings.status,
            "supply_type": legal_findings.supply_type,
            "usable_credit_ledgers": legal_findings.usable_credit_ledgers,
            "total_usable_credit": legal_findings.total_usable_credit,
            "unresolved_reason": legal_findings.unresolved_reason,
            "timing_ms": interp_timing,
        })

    # --- Stage 2: Deterministic Calculator Execution ---
    calc_inputs: CalculationInputs | None = None
    calc_result: CalculationResult | None = None

    if plan.get("needs_calculation"):
        calc_start = time.perf_counter()
        if has_itc_balances:
            if legal_findings is not None and legal_findings.status == "unresolved":
                tools_executed.append({
                    "tool": "execute_calculator",
                    "status": "skipped",
                    "reason": "unresolved_legal_findings",
                    "details": legal_findings.unresolved_reason,
                })
            elif legal_findings is not None and legal_findings.status == "resolved":
                tax_rate = extract_rate_pct(rate_results, user_premises)
                if tax_rate and tax_rate > 0 and legal_findings.total_usable_credit is not None:
                    calc_inputs = CalculationInputs(
                        operation="max_taxable_value_from_credit",
                        available_eligible_credit=legal_findings.total_usable_credit,
                        tax_rate_pct=tax_rate,
                        credit_breakdown=legal_findings.usable_credit_balances,
                    )
                    calc_result = execute_calculator(calc_inputs)
                    calc_timing = round((time.perf_counter() - calc_start) * 1000, 3)
                    tools_executed.append({
                        "tool": "execute_calculator",
                        "operation": calc_inputs.operation,
                        "status": calc_result.status,
                        "result_value": calc_result.result_value,
                        "formula": calc_result.formula,
                        "timing_ms": calc_timing,
                    })
        elif user_premises.get("taxable_amount") is not None or user_premises.get("base_amount") is not None:
            base_amt = float(user_premises.get("taxable_amount") or user_premises.get("base_amount") or 0.0)
            disc_pct = float(user_premises.get("discount_pct") or 0.0)
            tax_rate = extract_rate_pct(rate_results, user_premises) or 5.0

            if disc_pct > 0:
                calc_inputs = CalculationInputs(
                    operation="discount_and_tax",
                    base_amount=base_amt,
                    discount_pct=disc_pct,
                    tax_rate_pct=tax_rate,
                )
            else:
                calc_inputs = CalculationInputs(
                    operation="tax_on_value",
                    taxable_value=base_amt,
                    tax_rate_pct=tax_rate,
                )
            calc_result = execute_calculator(calc_inputs)
            calc_timing = round((time.perf_counter() - calc_start) * 1000, 3)
            tools_executed.append({
                "tool": "execute_calculator",
                "operation": calc_inputs.operation,
                "status": calc_result.status,
                "result_value": calc_result.result_value,
                "formula": calc_result.formula,
                "timing_ms": calc_timing,
            })

    observability = {
        "plan": plan,
        "tools_executed": tools_executed,
        "retrieved_evidence": {
            "rates_count": len(rate_results),
            "legal_chunks_count": len(legal_chunks),
            "notification_chunks_count": len(notif_chunks),
        },
        "structured_legal_findings": (
            legal_findings.model_dump() if legal_findings else None
        ),
        "calculation_inputs": (
            calc_inputs.model_dump() if calc_inputs else None
        ),
        "calculation_result": (
            calc_result.model_dump() if calc_result else None
        ),
        "sources_used": sources,
        "knowledge_base_gap": (
            legal_findings.unresolved_reason
            if (legal_findings and legal_findings.status == "unresolved")
            else None
        ),
    }

    yield {
        "type": "meta",
        "query": query,
        "route": route_str,
        "rate_results": rate_results,
        "sources": sources,
        "sources_used": sources,
        "retrieval_timing": retrieval_timing,
        "model_used": model_name,
        "model": model_name,
        "plan": plan,
        "tools_executed": tools_executed,
        "structured_legal_findings": (
            legal_findings.model_dump() if legal_findings else None
        ),
        "calculation_inputs": (
            calc_inputs.model_dump() if calc_inputs else None
        ),
        "calculation_result": (
            calc_result.model_dump() if calc_result else None
        ),
        "observability": observability,
        "retrieval_debug": {
            "results": final_chunks,
            "dense_results": retrieval_data.get("dense_results", []),
            "bm25_results": retrieval_data.get("bm25_results", []),
            "hybrid_results": retrieval_data.get("hybrid_results", []),
            "metadata": retrieval_data.get("metadata", {}),
            "models": retrieval_data.get("models", {}),
            "config": retrieval_data.get("config", {}),
        },
    }

    # 2. Generation streaming
    gen_start = time.perf_counter()
    full_answer_parts: list[str] = []

    for _, delta in stream_answer(
        query=query,
        chunks=legal_chunks,
        notification_chunks=notif_chunks,
        rate_results=rate_results,
        user_premises=plan.get("user_premises"),
        legal_findings=legal_findings,
        calculation_result=calc_result,
        model=openai_model,
        client=openai_client,
        direct_reasoning=is_direct,
    ):
        full_answer_parts.append(delta)
        yield {
            "type": "token",
            "delta": delta,
        }

    gen_ms = round((time.perf_counter() - gen_start) * 1000, 3)
    total_ms = round(retrieval_timing + gen_ms, 3)
    full_answer = "".join(full_answer_parts).strip()

    yield {
        "type": "done",
        "answer": full_answer,
        "route": route_str,
        "rate_results": rate_results,
        "generation_timing": gen_ms,
        "total_timing": total_ms,
        "plan": plan,
        "tools_executed": tools_executed,
        "structured_legal_findings": (
            legal_findings.model_dump() if legal_findings else None
        ),
        "calculation_inputs": (
            calc_inputs.model_dump() if calc_inputs else None
        ),
        "calculation_result": (
            calc_result.model_dump() if calc_result else None
        ),
        "observability": observability,
        "timings_ms": {
            "retrieval": retrieval_timing,
            "generation": gen_ms,
            "total": total_ms,
            "rate_lookup": rate_timing,
            "notification_lookup": notif_timing,
            "dense": retrieval_data["timings_ms"].get("dense", 0.0),
            "bm25": retrieval_data["timings_ms"].get("bm25", 0.0),
            "rrf": retrieval_data["timings_ms"].get("rrf", 0.0),
            "reranker": retrieval_data["timings_ms"].get("reranker", 0.0),
        },
    }
