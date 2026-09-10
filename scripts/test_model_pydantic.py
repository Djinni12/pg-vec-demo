#!/usr/bin/env python3
"""Diagnostic script to test whether an LLM (such as Gemma) returns valid
Pydantic-compliant responses and conforms to schema constraints without errors.

Usage:
    python3 scripts/test_model_pydantic.py
    python3 scripts/test_model_pydantic.py --model gemma-4-31b-it
    python3 scripts/test_model_pydantic.py --model gemini-3.1-flash-lite
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

# Load latest .env values
load_dotenv(override=True)

# Add project root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.routers.planner import (
    GSTPlan,
    PLANNER_SYSTEM_PROMPT,
    extract_json_payload,
)
from src.generators.legal_interpreter import (
    StructuredLegalFindings,
    INTERPRETER_SYSTEM_PROMPT,
)

# Terminal color formatting
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_banner(text: str) -> None:
    print(f"\n{BOLD}{CYAN}{'=' * 75}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 75}{RESET}")


def print_section(text: str) -> None:
    print(f"\n{BOLD}{YELLOW}--- {text} ---{RESET}")


def get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not api_key:
        print(f"{RED}Error: OPENAI_API_KEY is not set in .env or environment.{RESET}")
        sys.exit(1)
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def test_planner_query(
    client: OpenAI,
    model: str,
    query: str,
    description: str,
) -> dict[str, Any]:
    print_section(f"Planner Test: {description}")
    print(f"Query: {BOLD}\"{query}\"{RESET}")

    is_gemma = "gemma" in model.lower()
    if is_gemma:
        messages = [
            {"role": "user", "content": f"{PLANNER_SYSTEM_PROMPT}\n\n---\n\nUSER QUERY:\n{query.strip()}"}
        ]
    else:
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": query.strip()},
        ]

    start_t = time.perf_counter()
    raw_content = ""
    api_error = None
    direct_json_ok = False
    fenced_or_preamble = False
    pydantic_ok = False
    val_errors: list[str] = []
    parsed_dict: dict[str, Any] = {}

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        raw_content = response.choices[0].message.content or ""
    except Exception as exc:
        api_error = str(exc)

    elapsed_ms = round((time.perf_counter() - start_t) * 1000, 2)

    if api_error:
        print(f"  {RED}API Call Failed:{RESET} {api_error}")
        return {
            "name": description,
            "status": "FAIL_API",
            "elapsed_ms": elapsed_ms,
            "error": api_error,
        }

    print(f"  Latency: {elapsed_ms} ms")
    print(f"  Raw Content Preview (first 250 chars):\n    {CYAN}{raw_content[:250]!r}{RESET}")

    # Check 1: Direct JSON parsing without cleaning
    try:
        json.loads(raw_content.strip())
        direct_json_ok = True
    except Exception:
        direct_json_ok = False

    # Check 2: Preambles / Markdown code fences
    if "```" in raw_content or not raw_content.strip().startswith("{"):
        fenced_or_preamble = True

    # Check 3: Extract with payload sanitizer
    parsed_dict = extract_json_payload(raw_content)

    # Check 4: Pydantic Validation against GSTPlan
    if not parsed_dict:
        print(f"  {RED}JSON Extraction Failed:{RESET} Unable to extract JSON object from output.")
        return {
            "name": description,
            "status": "FAIL_JSON",
            "elapsed_ms": elapsed_ms,
            "raw_content": raw_content,
        }

    try:
        validated = GSTPlan(**parsed_dict)
        pydantic_ok = True
    except ValidationError as ve:
        pydantic_ok = False
        val_errors = [f"{err['loc']}: {err['msg']}" for err in ve.errors()]

    # Print diagnosis
    if direct_json_ok:
        print(f"  [1] Direct Raw JSON:         {GREEN}VALID (Strict clean JSON){RESET}")
    else:
        if fenced_or_preamble:
            print(f"  [1] Direct Raw JSON:         {YELLOW}WARNING (Enclosed in code fence or preamble text){RESET}")
        else:
            print(f"  [1] Direct Raw JSON:         {RED}INVALID JSON syntax{RESET}")

    if pydantic_ok:
        print(f"  [2] Pydantic Schema Check:   {GREEN}PASSED (Valid GSTPlan){RESET}")
        print(f"      - needs_structured_rate_lookup: {validated.needs_structured_rate_lookup}")
        print(f"      - rate_query: {validated.clean_subqueries.get('rate_query')}")
        print(f"      - needs_legal_retrieval: {validated.needs_legal_retrieval}")
        print(f"      - legal_query: {validated.clean_subqueries.get('legal_query')}")
        print(f"      - needs_calculation: {validated.needs_calculation}")
        print(f"      - needs_clarification: {validated.needs_clarification}")
    else:
        print(f"  [2] Pydantic Schema Check:   {RED}FAILED ({len(val_errors)} validation errors){RESET}")
        for err in val_errors[:5]:
            print(f"      * {RED}{err}{RESET}")

    status = "PASS" if pydantic_ok else "FAIL_PYDANTIC"
    return {
        "name": description,
        "status": status,
        "elapsed_ms": elapsed_ms,
        "direct_json": direct_json_ok,
        "fenced": fenced_or_preamble,
        "pydantic_ok": pydantic_ok,
        "errors": val_errors,
        "plan": parsed_dict if pydantic_ok else None,
    }


def test_legal_interpreter(
    client: OpenAI,
    model: str,
) -> dict[str, Any]:
    print_section("Legal Interpreter Schema Test (StructuredLegalFindings)")
    dummy_query = "What is the procedure for cancellation of GST registration?"
    dummy_context = (
        "[Section 29 of CGST Act]: The proper officer may cancel the registration of a person "
        "from such date, including any retrospective date, as he may deem fit."
    )
    user_msg = (
        f"USER QUERY: {dummy_query}\n\n"
        f"RETRIEVED LEGAL EVIDENCE:\n{dummy_context}"
    )

    is_gemma = "gemma" in model.lower()
    if is_gemma:
        messages = [
            {"role": "user", "content": f"{INTERPRETER_SYSTEM_PROMPT}\n\n---\n\n{user_msg}"}
        ]
    else:
        messages = [
            {"role": "system", "content": INTERPRETER_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

    start_t = time.perf_counter()
    raw_content = ""
    api_error = None
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        raw_content = resp.choices[0].message.content or ""
    except Exception as exc:
        api_error = str(exc)

    elapsed_ms = round((time.perf_counter() - start_t) * 1000, 2)
    if api_error:
        print(f"  {RED}API Call Failed:{RESET} {api_error}")
        return {"name": "Legal Interpreter", "status": "FAIL_API", "elapsed_ms": elapsed_ms, "error": api_error}

    parsed = extract_json_payload(raw_content)
    pydantic_ok = False
    val_errors = []
    try:
        findings = StructuredLegalFindings(**parsed)
        pydantic_ok = True
    except ValidationError as ve:
        val_errors = [f"{err['loc']}: {err['msg']}" for err in ve.errors()]

    if pydantic_ok:
        print(f"  Pydantic Schema Check: {GREEN}PASSED (Valid StructuredLegalFindings){RESET}")
        print(f"    - Status: {findings.status}")
        print(f"    - Supply Type: {findings.supply_type}")
        print(f"    - Evidence Count: {len(findings.legal_evidence)}")
    else:
        print(f"  Pydantic Schema Check: {RED}FAILED{RESET}")
        for err in val_errors[:5]:
            print(f"    * {RED}{err}{RESET}")

    return {
        "name": "Legal Interpreter",
        "status": "PASS" if pydantic_ok else "FAIL_PYDANTIC",
        "elapsed_ms": elapsed_ms,
        "pydantic_ok": pydantic_ok,
        "errors": val_errors,
    }


def main():
    parser = argparse.ArgumentParser(description="Test LLM structured Pydantic outputs")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to test (defaults to OPENAI_MODEL in .env)",
    )
    args = parser.parse_args()

    model = args.model or os.environ.get("OPENAI_MODEL") or "gemini-3.1-flash-lite"
    base_url = os.environ.get("OPENAI_BASE_URL", "default OpenAI URL")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    masked_key = f"{api_key[:6]}...{api_key[-4:]}" if len(api_key) > 10 else "(not set)"

    print_banner(f"LLM Pydantic Structured Output Test: {model}")
    print(f"Model Under Test: {BOLD}{model}{RESET}")
    print(f"Base URL:         {base_url}")
    print(f"API Key:          {masked_key}")

    client = get_client()

    queries = [
        ("What is the GST rate on butter?", "Standard English Rate Query"),
        ("मक्खन पर GST की दर कितनी है?", "Hindi Rate Query"),
        ("I am supplying goods worth ₹50,000 from Gujarat to Maharashtra with 18% IGST. What is tax?", "Calculation & Interstate Supply Query"),
        ("What does Section 29 say about cancellation?", "Statutory Section Lookup Query"),
        ("What GST rate applies to X?", "Ambiguous Item Query (Clarification)"),
    ]

    results = []
    for q, desc in queries:
        res = test_planner_query(client, model, q, desc)
        results.append(res)

    # Legal Interpreter Test
    res_legal = test_legal_interpreter(client, model)
    results.append(res_legal)

    # Final Scorecard
    print_banner("Test Summary Scorecard")
    total = len(results)
    passed = sum(1 for r in results if r.get("status") == "PASS")

    for r in results:
        status = r.get("status")
        if status == "PASS":
            tag = f"{GREEN}[PASS]{RESET}"
        elif status == "FAIL_API":
            tag = f"{RED}[FAIL: API ERROR]{RESET}"
        elif status == "FAIL_JSON":
            tag = f"{RED}[FAIL: INVALID JSON]{RESET}"
        else:
            tag = f"{RED}[FAIL: SCHEMA MISMATCH]{RESET}"
        print(f"  {tag:<28} {r.get('name')} ({r.get('elapsed_ms')} ms)")

    print(f"\n{BOLD}Result: {passed}/{total} tests passed.{RESET}")
    if passed == total:
        print(f"{GREEN}Model '{model}' successfully adheres to all Pydantic schemas and structured JSON formatting!{RESET}")
    else:
        print(f"{RED}Model '{model}' encountered failures with JSON/Pydantic serialization or schema conformity.{RESET}")


if __name__ == "__main__":
    main()
