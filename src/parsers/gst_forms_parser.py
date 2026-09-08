"""Parse Hindi CGST Forms PDF into form-level JSON records."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover - older PyMuPDF import name
    import fitz

from transformers import AutoTokenizer

from ..chunkers.act_chunker import count_tokens

DEFAULT_FORMS_PDF = Path("data/form/CGST forms compiled 2017 hindi.pdf")
DEFAULT_FORMS_JSON = Path("data/form/gst_forms_forms.json")
TOKENIZER_NAME = "sentence-transformers/all-MiniLM-L6-v2"

FORM_WORD_RE = re.compile(r"(?:प्ररू[पऩ]|प्ररु[पऩ]|प्ररुऩ|\x1bjप|\x1bcप|jप|रूप|फार्म|FORM|Form)")
GST_RE = re.compile(r"जीएसट[ी(]|GST", re.IGNORECASE)
RULE_REF_RE = re.compile(r"[\[({]\s*[^\])}\n]*(?:नियम|ननमभ|धनमभ|<नयम|rule)[^\])}\n]*[\])}]?", re.IGNORECASE)
DASH_RE = re.compile(r"[\-–—]+")
SPACE_RE = re.compile(r"\s+")

# Hindi transliterations observed in the PDF plus the common English form families.
FAMILY_ALIASES = {
    "सीएभऩी": "CMP",
    "सीएमपी": "CMP",
    "आयईजी": "REG",
    "आरईजी": "REG",
    "आईट(सी": "ITC",
    "आईटीसी": "ITC",
    "ईएनआर": "ENR",
    "जीएसट(आर": "GSTR",
    "जीएसटीआर": "GSTR",
    "पीसीट(": "PCT",
    "पीसीटी": "PCT",
    "पीसट(": "PCT",
    "पीएमट(": "PMT",
    "पीएमटी": "PMT",
    "आरएफडी": "RFD",
    "आयएपडी": "RFD",
    "एएसएमट(": "ASMT",
    "एएसएमटी": "ASMT",
    "एडीट(": "ADT",
    "एडीटी": "ADT",
    "एआरए": "ARA",
    "एपीएल": "APL",
    "आईएनएस": "INS",
    "डीआयसी": "DRC",
    "डीआरसी": "DRC",
    "सीऩीडी": "CPD",
    "सीपीडी": "CPD",
    "CMP": "CMP",
    "REG": "REG",
    "ITC": "ITC",
    "ENR": "ENR",
    "GSTR": "GSTR",
    "PCT": "PCT",
    "PMT": "PMT",
    "RFD": "RFD",
    "ASMT": "ASMT",
    "ADT": "ADT",
    "ARA": "ARA",
    "APL": "APL",
    "INS": "INS",
    "DRC": "DRC",
    "CPD": "CPD",
}
FAMILY_RE = re.compile("|".join(re.escape(k) for k in sorted(FAMILY_ALIASES, key=len, reverse=True)), re.IGNORECASE)
NUMBER_RE = re.compile(r"\d+[A-Zए-ह]?", re.IGNORECASE)

NOISY_FAMILY_RE = re.compile(r"(एएसएम|एएसएमट|एडीट|एडीटी)")
NOISY_FAMILY_MAP = {"एएसएम": "ASMT", "एएसएमट": "ASMT", "एडीट": "ADT", "एडीटी": "ADT"}

REFERENCE_HINTS = (
    "देख", "भें", "में", "दिए", "दिया", "जार", "फाइल", "अपलोड", "अनुसार", "सायणी", "सारणी",
    "के अनुसार", "से", "का भाग", "क?", "की", "को", "लिया", "कमा", "किया",
)


def clean_text(text: str) -> str:
    lines = [SPACE_RE.sub(" ", line).strip() for line in text.splitlines()]
    compact = []
    previous_blank = False
    for line in lines:
        blank = not line
        if blank and previous_blank:
            continue
        compact.append(line)
        previous_blank = blank
    return "\n".join(compact).strip()


def normalize_form_number(raw: str) -> str:
    value = raw.upper().strip()
    value = value.replace("O", "0")
    return value.zfill(2) if value.isdigit() else value


def find_form_heading(line: str):
    normalized_line = SPACE_RE.sub(" ", line).strip()
    if not normalized_line:
        return None
    if not GST_RE.search(normalized_line):
        return None

    has_form_word = bool(FORM_WORD_RE.search(normalized_line))
    gstr_match = re.search(r"जीएसट[ी(]\s*आर\s*[-–—]\s*(\d+[A-Zए-ह]?)\b", normalized_line, re.IGNORECASE)
    if gstr_match and has_form_word:
        number = normalize_form_number(gstr_match.group(1))
        return {
            "form_number": f"GST GSTR-{number}",
            "form_family": "GSTR",
            "form_code": number,
            "heading": normalized_line,
        }

    noisy_family = NOISY_FAMILY_RE.search(normalized_line)
    if noisy_family and has_form_word:
        code_match = re.search(r"[-–—]\s*(\d+[A-Zए-ह]?)\b", normalized_line[noisy_family.end():], re.IGNORECASE)
        if code_match:
            family = NOISY_FAMILY_MAP[noisy_family.group(1)]
            number = normalize_form_number(code_match.group(1))
            return {
                "form_number": f"GST {family}-{number}",
                "form_family": family,
                "form_code": number,
                "heading": normalized_line,
            }

    family_match = FAMILY_RE.search(normalized_line)

    if family_match:
        tail = normalized_line[family_match.end() : family_match.end() + 90]
        number_match = None
        for candidate in re.finditer(r"(\d+[A-Zए-ह]?)\b", tail, re.IGNORECASE):
            between_tail = tail[: candidate.start()]
            if "[" in between_tail or "(" in between_tail:
                break
            if DASH_RE.search(between_tail) or FORM_WORD_RE.search(between_tail) or candidate.start() <= 3:
                number_match = candidate
                break
        if not number_match:
            return None
        absolute_number_end = family_match.end() + number_match.end()
        prefix = normalized_line[: family_match.start()]
        suffix = normalized_line[absolute_number_end:]
        if len(prefix) > 45:
            return None
        if not has_form_word:
            return None
        family = FAMILY_ALIASES.get(family_match.group(0), family_match.group(0).upper())
        number = normalize_form_number(number_match.group(1))
        stripped_suffix = suffix.strip()
        repeated_code_re = re.compile(rf"^(?:{re.escape(number.lstrip('0') or number)}|{re.escape(number)})\b\s*", re.IGNORECASE)
        while repeated_code_re.match(stripped_suffix):
            stripped_suffix = repeated_code_re.sub("", stripped_suffix, count=1).strip()
        if stripped_suffix and not stripped_suffix.startswith(("[", "(", "{")):
            if any(hint in stripped_suffix for hint in REFERENCE_HINTS) and len(stripped_suffix) > 12:
                return None
        return {
            "form_number": f"GST {family}-{number}",
            "form_family": family,
            "form_code": number,
            "heading": normalized_line,
        }

    # Some REG headings in this PDF are printed as plain GST-23 without the REG family.
    if not has_form_word:
        return None
    gst_match = GST_RE.search(normalized_line)
    number_match = re.search(r"[-–—]\s*(\d+[A-Zए-ह]?)\b", normalized_line[gst_match.end() :], re.IGNORECASE)
    if not number_match:
        return None
    prefix = normalized_line[: gst_match.start()]
    suffix = normalized_line[gst_match.end() + number_match.end() :]
    if len(prefix) > 35:
        return None
    if any(hint in suffix for hint in REFERENCE_HINTS) and len(suffix) > 12:
        return None
    number = normalize_form_number(number_match.group(1))
    return {
        "form_number": f"GST UNKNOWN-{number}",
        "form_family": None,
        "form_code": number,
        "heading": normalized_line,
        "family_missing": True,
    }


def extract_pages(pdf_path: Path):
    doc = fitz.open(pdf_path)
    for page_index, page in enumerate(doc, 1):
        yield page_index, page.get_text("text")


def find_boundaries(pdf_path: Path):
    boundaries = []
    char_offset = 0
    all_text_parts = []
    for page_number, page_text in extract_pages(pdf_path):
        all_text_parts.append(page_text)
        lines = page_text.splitlines(keepends=True)
        offsets = []
        local_offset = 0
        for line in lines:
            offsets.append(local_offset)
            local_offset += len(line)
        for index, line in enumerate(lines):
            windows = [line, " ".join(part.strip() for part in lines[index:index + 8])]
            for window in windows:
                heading = find_form_heading(window)
                if heading:
                    boundaries.append({**heading, "page_start": page_number, "offset": char_offset + offsets[index]})
                    break
        char_offset += len(page_text) + 1
    full_text = "\n".join(all_text_parts)
    return boundaries, full_text


def _numeric_code(value):
    match = re.match(r"\d+", str(value or ""))
    return int(match.group(0)) if match else None


def infer_missing_families(boundaries):
    for index, boundary in enumerate(boundaries):
        if not boundary.get("family_missing"):
            continue
        code = _numeric_code(boundary.get("form_code"))
        previous = boundaries[index - 1] if index else None
        following = boundaries[index + 1] if index + 1 < len(boundaries) else None
        if previous and previous.get("form_family") == "REG" and _numeric_code(previous.get("form_code")) == code - 1:
            boundary["form_family"] = "REG"
        elif following and following.get("form_family") == "REG" and _numeric_code(following.get("form_code")) == code + 1:
            boundary["form_family"] = "REG"
        else:
            boundary["drop_boundary"] = True
            continue
        boundary["form_number"] = f"GST {boundary['form_family']}-{boundary['form_code']}"
    return [boundary for boundary in boundaries if not boundary.get("drop_boundary")]


def dedupe_boundaries(boundaries: Iterable[dict]):
    deduped = []
    seen_nearby = set()
    for boundary in sorted(boundaries, key=lambda item: item["offset"]):
        key = (boundary["form_number"], boundary["page_start"])
        if key in seen_nearby:
            continue
        if deduped and boundary["offset"] - deduped[-1]["offset"] < 80 and boundary["form_number"] == deduped[-1]["form_number"]:
            continue
        seen_nearby.add(key)
        deduped.append(boundary)
    return infer_missing_families(deduped)


def add_stable_part_identifiers(forms):
    totals = {}
    seen = {}
    for form in forms:
        totals[form["form_number"]] = totals.get(form["form_number"], 0) + 1
    for form in forms:
        number = form["form_number"]
        seen[number] = seen.get(number, 0) + 1
        part_number = seen[number] if totals[number] > 1 else None
        slug = number.lower().replace(" ", "-")
        if part_number is not None:
            slug = f"{slug}-part-{part_number}-p{form['page_start']}"
        form["part_number"] = part_number
        form["form_uid"] = slug
    return forms


def extract_rule_references(content: str):
    references = []
    for match in RULE_REF_RE.finditer(content or ""):
        value = SPACE_RE.sub(" ", match.group(0)).strip()
        if value and value not in references:
            references.append(value)
    return references


def parse_forms(pdf_path: Path = DEFAULT_FORMS_PDF):
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    boundaries, full_text = find_boundaries(pdf_path)
    boundaries = dedupe_boundaries(boundaries)
    forms = []
    for index, boundary in enumerate(boundaries):
        start = boundary["offset"]
        end = boundaries[index + 1]["offset"] if index + 1 < len(boundaries) else len(full_text)
        content = clean_text(full_text[start:end])
        first_lines = [line.strip() for line in content.splitlines() if line.strip()]
        title = first_lines[2] if len(first_lines) > 2 and RULE_REF_RE.search(first_lines[1] if len(first_lines) > 1 else "") else (first_lines[1] if len(first_lines) > 1 else boundary["heading"])
        forms.append(
            {
                "form_number": boundary["form_number"],
                "form_family": boundary["form_family"],
                "form_code": boundary["form_code"],
                "form_title": title,
                "language": "hi",
                "rule_references": extract_rule_references(content[:1000]),
                "page_start": boundary["page_start"],
                "page_end": boundaries[index + 1]["page_start"] if index + 1 < len(boundaries) else None,
                "source_start_page": boundary["page_start"],
                "source_end_page": boundaries[index + 1]["page_start"] if index + 1 < len(boundaries) else None,
                "content": content,
                "token_count": count_tokens(tokenizer, content),
            }
        )
    return add_stable_part_identifiers(forms)
