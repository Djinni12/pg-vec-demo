"""Parse Chapter, Rule, and Sub-rule structure from the CGST Rules PDF."""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf

from gst_act_parser import normalize_text, subsection_marker_text


DEFAULT_RULES_PDF_PATH = Path("data/rules/gst_rules.pdf")

CHAPTER_HEADING_RE = re.compile(
    r"""
    (?im)
    ^\s*CHAPTER\s+(?P<number>[IVXLCDM]+)\s*\.?\s*$
    \n
    \s*(?P<title>(?:\d+\s*\[\s*)?[A-Z][A-Z ,&/()\-]+\]?|\d+\s*\[[A-Z][A-Z ,&/()\-]+\])\s*$
    |
    ^\s*CHAPTER\s+(?P<number_inline>[IVXLCDM]+)\s*\.?\s+(?P<title_inline>[A-Z][A-Z ,&/()\-]+)\s*$
    """,
    re.VERBOSE,
)

RULE_HEADING_RE = re.compile(
    r"""
    (?im)
    ^\s*
    (?P<prefix>(?:\*+\s*)?(?:\d+\s*)?(?:\[\s*)?\*?\s*)
    Rule
    \s*
    (?P<number>\d+\s*[A-Z]?)
    \s*
    \.
    \s*
    (?P<title>[^\n]*?(?:\n(?!\s*(?:\(|\*|\d+\.\s*(?:Substituted|Inserted|Omitted)|Rule\s+\d|CHAPTER\b))[^\n]*?){0,2})
    \s*
    (?:
        \.\s*-\s* |
        \]\s*-\s* |
        \]\s*$ |
        \*+\s*$ |
        \.\s*$ |
        (?=\n\s*(?:\*+\s*)?(?:\d+\s*)?\[?\s*\(\s*1[A-Z]?\s*\)) |
        (?=\n\s*[A-Z])
    )
    """,
    re.VERBOSE,
)

SUBRULE_HEADING_RE = re.compile(
    r"""
    (?m)
    ^\s*
    (?P<prefix>(?:\*+\s*)?(?:\d+\s*)?\[?\s*)?
    \(
    \s*
    (?P<number>\d+\s*[A-Z]?)
    \s*
    \)
    """,
    re.VERBOSE,
)

TOC_RULE_RE = re.compile(r"(?im)^\s*Rule\s+(?P<number>\d+\s*[A-Z]?)\s*$")

AMENDMENT_NOTE_RE = re.compile(
    r"(?im)^\s*\d+\.?\s*(?:Substituted|Inserted|Omitted|The words|In rule|In sub-rule|For the words|Against serial)\b"
)


ROMAN_ORDER = [
    "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII",
]


def normalize_rule_number(number):
    """Normalize rule numbers such as '88 C' to '88C'."""
    return re.sub(r"\s+", "", number).upper()


def extract_pdf_text(pdf_path=DEFAULT_RULES_PDF_PATH):
    """Extract plain text from every page using PyMuPDF."""
    with pymupdf.open(pdf_path) as document:
        return "\n".join(page.get_text() for page in document)


def clean_chapter_title(title):
    """Normalize chapter title text and remove simple amendment markers."""
    title = re.sub(r"^\d+\s*\[\s*", "", title.strip())
    title = title.replace("[", "").replace("]", "")
    title = re.sub(r"\s+", " ", title)
    return title.strip(" .-")


def clean_rule_title(title):
    """Normalize rule title text and remove common amendment markers."""
    if rule_status(title) == "omitted":
        return "Omitted"
    title = re.sub(r"-\s*\n\s*", "-", title)
    title = re.sub(r"^\d+\s*\[\s*", "", title.strip())
    title = title.replace("[", "").replace("]", "")
    title = title.replace("*", "")
    title = re.sub(r"\s+", " ", title)
    title = re.sub(r"^\d+\s+", "", title)
    return title.strip(" .-")


def rule_status(title):
    """Return omitted for rule headings whose title is omission text/marks."""
    if re.fullmatch(r"\s*\d*\s*\[?\s*\*{2,}\s*\]?\s*", title):
        return "omitted"
    simplified = re.sub(r"[\[\]\s.]", "", title).casefold()
    simplified = re.sub(r"^\d+", "", simplified)
    if simplified in {"omitted", "omitted****", "****"}:
        return "omitted"
    if re.fullmatch(r"\s*(?:\[\s*)?(?:Omitted)?\s*\*{2,}\s*\]?\s*", title, re.IGNORECASE):
        return "omitted"
    return "current"


def find_body_start(text):
    """Skip table-of-contents rule headings before the rules body."""
    match = re.search(r"(?im)^\s*Central Goods and Services Tax \(CGST\) Rules, 2017 Part A", text)
    if match:
        return match.start()
    chapter = CHAPTER_HEADING_RE.search(text)
    return chapter.start() if chapter else 0


def find_toc_rule_numbers(text):
    """Return rule numbers listed in the table of contents."""
    body_start = find_body_start(text)
    return list(dict.fromkeys(normalize_rule_number(match.group("number")) for match in TOC_RULE_RE.finditer(text[:body_start])))


def find_chapter_headings(text):
    """Return chapter headings with start positions."""
    body = text[find_body_start(text):]
    offset = len(text) - len(body)
    chapters = []
    for match in CHAPTER_HEADING_RE.finditer(body):
        number = match.group("number") or match.group("number_inline")
        title = match.group("title") or match.group("title_inline")
        chapters.append(
            {
                "chapter": f"CHAPTER {number.upper()}",
                "chapter_number": number.upper(),
                "chapter_title": clean_chapter_title(title),
                "start": offset + match.start(),
                "end": offset + match.end(),
            }
        )
    return chapters


def is_historical_rule_heading(text, match):
    """Detect old quoted rule headings inside amendment notes."""
    line_end = text.find("\n", match.start())
    line = text[match.start():line_end if line_end != -1 else len(text)]
    prefix = match.group("prefix")
    before = text[max(0, match.start() - 320):match.start()].casefold()
    has_amendment_marker = "*" in prefix or "[" in prefix or line.lstrip().startswith("[")
    if has_amendment_marker:
        return False
    return ("substituted" in before or "omitted" in before) and re.search(r"\bfor\s*[\n\"]*$", before)


def find_rule_headings(text):
    """Return current rule heading matches from the body, skipping TOC and historical quotes."""
    body_start = find_body_start(text)
    matches = [m for m in RULE_HEADING_RE.finditer(text, body_start) if not is_historical_rule_heading(text, m)]
    first_by_number = {}
    filtered = []
    for match in matches:
        number = normalize_rule_number(match.group("number"))
        if number in first_by_number:
            context = text[max(0, match.start() - 360):match.start()].casefold()
            if "substituted" in context or "omitted" in context:
                continue
            continue
        first_by_number[number] = match
        filtered.append(match)
    return filtered


def chapter_for_position(chapters, position):
    """Return the active chapter metadata for a rule start position."""
    active = None
    for chapter in chapters:
        if chapter["start"] <= position:
            active = chapter
        else:
            break
    return active or {"chapter": None, "chapter_title": None}


def strip_chapter_headings(content):
    """Remove standalone chapter headings from rule content."""
    return normalize_text(CHAPTER_HEADING_RE.sub("", content))


def subrule_search_content(content):
    """Use only the current rule body for sub-rule detection."""
    end = len(content)
    for pattern in (AMENDMENT_NOTE_RE,):
        match = pattern.search(content)
        if match:
            end = min(end, match.start())
    return content[:end]


def find_subrule_headings(content):
    """Return top-level numeric/alphanumeric sub-rule matches."""
    search_content = subrule_search_content(content)
    headings = []
    for match in SUBRULE_HEADING_RE.finditer(search_content):
        line_start = search_content.rfind("\n", 0, match.start()) + 1
        before_line = search_content[max(0, line_start - 120):line_start]
        before_since_heading = search_content[headings[-1].end() if headings else 0:match.start()]
        if re.search(r"sub\s*-\s*rules?\s*$", before_line, re.IGNORECASE):
            continue
        if re.search(r"sub\s*-\s*section\s*$", before_line, re.IGNORECASE):
            continue
        if re.search(r"sub\s*-?\s*sections?\s+\(\s*\d+[A-Z]?\s*\)(?:\s+and)?\s*$", before_line, re.IGNORECASE):
            continue
        if re.search(r"sub\s*-?\s*sections?.{0,80}(?:and|or)\s*$", before_line, re.IGNORECASE):
            continue
        if re.search(r"Explanation\s*\.?\s*-", before_since_heading, re.IGNORECASE):
            continue
        if is_inside_table_block(search_content, match.start()):
            continue
        if is_inside_illustration_block(search_content, match.start()):
            continue
        headings.append(match)
    return headings


def is_inside_table_block(content, position):
    """Return True when a marker appears inside a table region."""
    line_start = content.rfind("\n", 0, position) + 1
    line_end = content.find("\n", position)
    if line_end == -1:
        line_end = len(content)
    line = content[line_start:line_end]

    context_start = max(0, position - 700)
    context = content[context_start:position]
    table_start = max(context.rfind("TABLE"), context.rfind("Table"), context.rfind("table"))
    if table_start == -1:
        return False

    current = normalize_rule_number(re.search(r"\(\s*(\d+[A-Z]?)\s*\)", content[position:]).group(1))
    current_number = int(re.match(r"\d+", current).group(0))
    after_table = context[table_start:]
    if re.search(r"\b(?:Provided|Explanation|Illustration)\b", after_table):
        return False
    if re.search(r"\(\s*\d+[A-Z]?\s*\).{0,20}\(\s*\d+[A-Z]?\s*\)", line):
        return True
    window = content[max(0, position - 80):position + 100]
    table_marker_count = len(re.findall(r"\(\s*\d+[A-Z]?\s*\)", window))
    if (
        current_number <= 3
        and table_marker_count >= 3
        and re.search(r"\(\s*1\s*\)", window)
        and re.search(r"\(\s*2\s*\)", window)
    ):
        return True
    if re.search(r"\bSub\s*-\s*sections?\s+\(\s*\d+[A-Z]?\s*\)", line, re.IGNORECASE):
        return True
    if re.search(r"\b(?:Sl\.?\s*No|S\.?\s*No|column)\b", line, re.IGNORECASE):
        return True
    return False


def is_inside_illustration_block(content, position):
    """Return True when a marker appears inside an illustration/example region."""
    context_start = max(0, position - 500)
    context = content[context_start:position]
    illustration_start = max(context.rfind("Illustration"), context.rfind("Example"))
    if illustration_start == -1:
        return False

    after_illustration = context[illustration_start:]
    if re.search(r"\b(?:Provided|Explanation|TABLE|Table)\b", after_illustration):
        return False
    return True


def parse_subrules(content):
    """Return sub-rule dictionaries while keeping clauses/provisos/explanations inside."""
    headings = find_subrule_headings(content)
    search_content = subrule_search_content(content)
    subrules = []
    for index, match in enumerate(headings):
        text_start = match.end()
        text_end = headings[index + 1].start() if index + 1 < len(headings) else len(search_content)
        body = search_content[text_start:text_end].strip()
        subrules.append(
            {
                "subrule_number": normalize_rule_number(match.group("number")),
                "text": normalize_text(f"{subsection_marker_text(match)} {body}"),
            }
        )
    return subrules


def parse_rules(pdf_path=DEFAULT_RULES_PDF_PATH):
    """Return dictionaries with chapter, rule, content, and sub-rule structure."""
    text = normalize_text(extract_pdf_text(pdf_path))
    chapters = find_chapter_headings(text)
    headings = find_rule_headings(text)
    rules = []

    for index, match in enumerate(headings):
        content_start = match.end()
        content_end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        content = strip_chapter_headings(text[content_start:content_end])
        status = rule_status(match.group("title"))
        chapter = chapter_for_position(chapters, match.start())
        rules.append(
            {
                "rule_number": normalize_rule_number(match.group("number")),
                "rule_title": clean_rule_title(match.group("title")),
                "chapter": chapter.get("chapter"),
                "chapter_title": chapter.get("chapter_title"),
                "status": status,
                "content": content,
                "subrules": [] if status == "omitted" else parse_subrules(content),
            }
        )

    return rules


def rule_boundary_warnings(rule):
    """Return suspicious boundary notes for manual validation."""
    content = rule.get("content", "")
    warnings = []
    if CHAPTER_HEADING_RE.search(content):
        warnings.append("standalone chapter heading inside content")
    own_number = str(rule.get("rule_number", "")).upper()
    for match in RULE_HEADING_RE.finditer(content):
        ref = normalize_rule_number(match.group("number"))
        if ref != own_number:
            warnings.append(f"embedded Rule {ref} heading")
    subrules = rule.get("subrules") or []
    if subrules:
        first = find_subrule_headings(content)[0]
        before_first = content[:first.start()].strip()
        if before_first and before_first.casefold() != "p" and not re.fullmatch(r"[\W_]+", before_first):
            warnings.append(f"text appears before first sub-rule: {before_first[:80]}")
        seen = set()
        previous_numeric = None
        for subrule in subrules:
            number = subrule.get("subrule_number", "")
            if number in seen:
                warnings.append(f"duplicate sub-rule ({number})")
            seen.add(number)
            m = re.fullmatch(r"(\d+)([A-Z]?)", number)
            if m and not m.group(2):
                numeric = int(m.group(1))
                if previous_numeric is not None and numeric <= previous_numeric:
                    warnings.append(f"non-increasing sub-rule order near ({number})")
                previous_numeric = numeric
    return warnings
