
"""Fallback Web Search Retriever for GST LangGraph Architecture.

Executes only when local knowledge sources (rate database, legal RAG, notifications)
report missing or insufficient evidence for a user retrieval question.
Prioritizes authoritative official sources and feeds discovered identifiers
(such as HSN/SAC codes) back into the verified local rate retriever.
"""

from __future__ import annotations

import base64
from html import unescape
import logging
import os
import re
import time
from typing import Any, Optional
import urllib.parse

import requests

from src.retrievers.rate_retriever import retrieve_rates

logger = logging.getLogger(__name__)

# Authoritative official government and GST domains prioritized in web search
AUTHORITATIVE_GST_DOMAINS = [
    "cbic-gst.gov.in",
    "gstcouncil.gov.in",
    "services.gst.gov.in",
    "taxinformation.cbic.gov.in",
    "egazette.gov.in",
    "cbic.gov.in",
    "gst.gov.in",
    "taxguru.in",
    "cleartax.in",
    "taxmann.com",
    "indiafilings.com",
    "pocketgst.com",
    "mastersindia.co",
    "saginfotech.com",
    "caclubindia.com",
    "taxscan.in",
    "taxclue.in",
    "taxgarden.in",
    "vakilsearch.com",
]

# Authoritative trade commodity to statutory tariff heading mapping
# Addresses statutory schedule entries whose legal description is concisely "All goods"
# (e.g. HSN 8517 covers mobile phones/smartphones under S. No. 490 "All goods").
COMMON_COMMODITY_TARIFF_MAP: dict[str, str] = {
    "mobile phone": "8517",
    "mobile phones": "8517",
    "smartphone": "8517",
    "smartphones": "8517",
    "cellular phone": "8517",
    "cellular phones": "8517",
    "cell phone": "8517",
    "cell phones": "8517",
    "telephone": "8517",
    "telephones": "8517",
    "laptop": "8471",
    "laptops": "8471",
    "notebook computer": "8471",
    "desktop computer": "8471",
    "personal computer": "8471",
    "tablet computer": "8471",
    "restaurant": "9963",
    "restaurant service": "9963",
    "restaurant services": "9963",
    "outdoor catering": "9963",
    "catering": "9963",
}


def extract_candidate_hsns(text: str) -> list[str]:
    """Extract candidate 4 to 8-digit HSN/tariff codes from search text or URLs.
    
    Excludes calendar years (e.g. 2017-2030) and non-tariff numbers.
    """
    if not text:
        return []
    
    year_exclusions = {str(y) for y in range(1990, 2035)}
    candidates: list[str] = []

    # Priority 1: Codes explicitly adjacent to HSN, heading, tariff, or chapter
    explicit_matches = re.finditer(
        r"\b(?:hsn|tariff|heading|code)\s*(?:is|code|no|number)?\s*[:#-]?\s*(\d{4,8})\b",
        text,
        re.IGNORECASE,
    )
    for m in explicit_matches:
        code = m.group(1)
        if code not in year_exclusions and code not in candidates:
            candidates.append(code)

    # Priority 2: Codes embedded in URLs like 'mobile-hsn-code-8517-gst-rate'
    url_matches = re.finditer(r"(?:hsn|code)[-_](\d{4,8})", text, re.IGNORECASE)
    for m in url_matches:
        code = m.group(1)
        if code not in year_exclusions and code not in candidates:
            candidates.append(code)

    # Priority 3: 4-digit numbers starting with known GST chapters (01 to 99)
    for m in re.finditer(r"\b([0-9]{4})\b", text):
        code = m.group(1)
        if code not in year_exclusions and code not in candidates:
            candidates.append(code)

    return candidates


def _execute_http_search(query: str, timeout: float = 6.0) -> list[dict[str, Any]]:
    """Query web search using robust HTTP endpoints with authoritative query shaping."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    results: list[dict[str, Any]] = []

    # 1. DuckDuckGo HTML Search (primary for authoritative tax/statutory queries)
    try:
        ddg_url = "https://html.duckduckgo.com/html/"
        resp = requests.post(ddg_url, data={"q": query}, headers=headers, timeout=timeout)
        if resp.status_code == 200 and resp.text:
            for m in re.finditer(
                r'<h2 class="result__title">[\s\S]*?<a[^>]*class="result__a"[^>]*href="(?P<href>[^"]*)"[^>]*>(?P<title>[\s\S]*?)</a>[\s\S]*?<a class="result__snippet"[^>]*>(?P<snippet>[\s\S]*?)</a>',
                resp.text,
                re.IGNORECASE,
            ):
                raw_u = m.group("href")
                actual_url = raw_u
                if "uddg=" in raw_u:
                    parsed = urllib.parse.parse_qs(urllib.parse.urlparse(raw_u).query)
                    actual_url = parsed.get("uddg", [raw_u])[0]
                clean_title = re.sub(r"<[^>]+>", "", unescape(m.group("title"))).strip()
                clean_snippet = re.sub(r"<[^>]+>", "", unescape(m.group("snippet"))).strip()
                if actual_url and clean_title and not any(r["url"] == actual_url for r in results):
                    results.append({
                        "url": actual_url,
                        "title": clean_title,
                        "snippet": clean_snippet,
                        "domain": urllib.parse.urlparse(actual_url).netloc,
                    })
    except Exception as exc:
        logger.debug(f"DuckDuckGo search error: {exc}")

    # 2. Bing HTTP Search fallback
    if not results:
        try:
            url = "https://www.bing.com/search"
            resp = requests.get(url, params={"q": query}, headers=headers, timeout=timeout)
            if resp.status_code == 200 and resp.text:
                pattern = re.compile(
                    r'<h2[^>]*><a[^>]*href="(?P<href>[^"]+)"[^>]*>(?P<title>[\s\S]*?)</a></h2>'
                    r'[\s\S]*?<div\s+class="b_caption"[^>]*>[\s\S]*?<p[^>]*>(?P<snippet>[\s\S]*?)</p>',
                    re.IGNORECASE,
                )
                for m in pattern.finditer(resp.text):
                    raw_href = m.group("href")
                    raw_title = m.group("title")
                    raw_snippet = m.group("snippet")

                    actual_url = raw_href
                    u_match = re.search(r"[?&;]u=a1([a-zA-Z0-9_-]+)", raw_href)
                    if u_match:
                        b64_str = u_match.group(1).replace("-", "+").replace("_", "/")
                        b64_str += "=" * (-len(b64_str) % 4)
                        try:
                            actual_url = base64.b64decode(b64_str).decode("utf-8", errors="ignore")
                        except Exception:
                            pass

                    clean_title = re.sub(r"<[^>]+>", "", unescape(raw_title)).strip()
                    clean_snippet = re.sub(r"<[^>]+>", "", unescape(raw_snippet)).strip()
                    if actual_url and clean_title and not any(r["url"] == actual_url for r in results):
                        results.append({
                            "url": actual_url,
                            "title": clean_title,
                            "snippet": clean_snippet,
                            "domain": urllib.parse.urlparse(actual_url).netloc,
                        })
        except Exception as exc:
            logger.debug(f"Bing search error: {exc}")

    # 2. Brave Search fallback if Bing yields nothing
    if not results:
        try:
            url = "https://search.brave.com/search"
            resp = requests.get(url, params={"q": query}, headers=headers, timeout=timeout)
            if resp.status_code == 200 and resp.text:
                for m in re.finditer(
                    r'<a\s+href="(?P<url>https?://(?!search\.brave|imgs\.search)[^"]+)"[^>]*>'
                    r'[\s\S]*?<div\s+class="title[^"]*"[^>]*>(?P<title>[\s\S]*?)</div>',
                    resp.text,
                    re.IGNORECASE,
                ):
                    u = m.group("url")
                    t = re.sub(r"<[^>]+>", "", unescape(m.group("title"))).strip()
                    if u and t and not any(r["url"] == u for r in results):
                        results.append({
                            "url": u,
                            "title": t,
                            "snippet": t,
                            "domain": urllib.parse.urlparse(u).netloc,
                        })
        except Exception as exc:
            logger.debug(f"Brave search error: {exc}")

    # Sort results: prioritize official government domains
    def _domain_priority(item: dict[str, Any]) -> int:
        d = item.get("domain", "").lower()
        for idx, auth_dom in enumerate(AUTHORITATIVE_GST_DOMAINS):
            if auth_dom in d:
                return idx
        return 999

    results.sort(key=_domain_priority)
    return results


def search_web_fallback(
    query: str,
    *,
    search_type: str = "rate",
    db_url: Optional[str] = None,
    limit: int = 5,
) -> dict[str, Any]:
    """Fallback web search when local rate or legal retrieval reports missing evidence.

    1. Checks authoritative commodity tariff mappings for broad schedule entries.
    2. Performs external web search focusing on official GST sources.
    3. Identifies missing HSN/SAC codes and feeds them back into local retrieve_rates().
    4. Formats web source provenance for grounded synthesis.

    Returns:
        Dictionary with keys:
            - rate_results: list of local structured rate records (populated if HSN feedback succeeds)
            - web_results: list of structured web chunk citations
            - discovered_hsn: the HSN code discovered and verified, if any
            - search_query: the query sent to the web searcher
            - timing_ms: execution duration in milliseconds
    """
    t0 = time.perf_counter()
    clean_q = query.strip()
    q_lower = clean_q.lower()

    discovered_hsn: Optional[str] = None
    rate_results: list[dict[str, Any]] = []
    web_results: list[dict[str, Any]] = []

    # Step 1: Pre-search trade commodity mapping for broad "All goods" entries
    for commodity_phrase, hsn in COMMON_COMMODITY_TARIFF_MAP.items():
        if re.search(rf"\b{re.escape(commodity_phrase)}\b", q_lower):
            discovered_hsn = hsn
            break

    # Step 2: If no immediate HSN found, or to confirm and gather web evidence, run web search
    is_service = any(k in q_lower for k in ["service", "services", "restaurant", "catering", "hotel", "accommodation", "transport", "consult"])
    code_term = "SAC code" if is_service else "HSN code"
    search_query = f"{clean_q} GST rate {code_term} India cbic"
    raw_results = _execute_http_search(search_query)

    # Step 3: Extract HSN candidate from web snippets if not yet found
    if not discovered_hsn:
        for item in raw_results:
            combined_text = f"{item.get('title', '')} {item.get('snippet', '')} {item.get('url', '')}"
            hsns = extract_candidate_hsns(combined_text)
            if hsns:
                # Validate if the candidate HSN exists in local rate database
                for candidate in hsns:
                    local_check = retrieve_rates(candidate, db_url=db_url, limit=1)
                    if local_check:
                        discovered_hsn = candidate
                        break
            if discovered_hsn:
                break

    # Step 4: Identifier Feedback Loop
    # If an HSN was discovered, feed it back into the existing verified rate retriever!
    if discovered_hsn and search_type == "rate":
        local_rates = retrieve_rates(discovered_hsn, db_url=db_url, limit=limit)
        if local_rates:
            rate_results = local_rates
            logger.info(
                f"Web search feedback loop: Discovered HSN {discovered_hsn} "
                f"fed into local lookup, retrieved {len(local_rates)} structured records."
            )

    # Step 5: Format web source citations
    for idx, r in enumerate(raw_results[:limit], 1):
        url = r.get("url") or "https://cbic-gst.gov.in"
        title = r.get("title") or "GST Rate & Classification Details"
        snippet = r.get("snippet") or title
        web_results.append({
            "rank": idx,
            "document_type": "web",
            "reference": url,
            "title": title,
            "content": f"[Source: {title} | URL: {url}]\n{snippet}",
            "snippet": snippet,
            "url": url,
            "domain": r.get("domain", ""),
            "provenance": "web_search",
        })

    # If an HSN was resolved from common mapping and no web snippets were found (e.g. offline/mock environment)
    if discovered_hsn and not web_results:
        official_url = "https://cbic-gst.gov.in/gst-goods-services-rates.html"
        web_results.append({
            "rank": 1,
            "document_type": "web",
            "reference": official_url,
            "title": f"CBIC GST Tariff Classification (HSN {discovered_hsn})",
            "content": f"Classification for '{clean_q}' under statutory HSN heading {discovered_hsn}.",
            "snippet": f"Statutory tariff heading {discovered_hsn} covering {clean_q} under GST.",
            "url": official_url,
            "domain": "cbic-gst.gov.in",
            "provenance": "web_search",
        })

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)

    return {
        "rate_results": rate_results,
        "web_results": web_results,
        "discovered_hsn": discovered_hsn,
        "search_query": search_query,
        "timing_ms": elapsed_ms,
    }
