"""Parse legal sections from the Central GST Act PDF."""

import re
from pathlib import Path

import pymupdf


DEFAULT_PDF_PATH = Path("data/acts/cental_gst_act_2017.pdf")

SECTION_HEADING_RE = re.compile(
    r"""
    ^\s*
    (?P<prefix>(?:\d+\s*)?(?:\[\s*)?\*?\s*)
    Section
    \s+
    (?P<number>\d+[A-Z]?)
    \s*
    \.
    \s*
    (?P<title>[^\n]*?(?:\n(?!\s*(?:\(|\*))[^\n]*?){0,2})
    \s*
    (?:
        \.\s*-\s* |
        \]\s*-\s* |
        \]\s*$ |
        \*\s*$ |
        \.\s*$ |
        (?=\n\s*\(1\))
    )
    """,
    re.IGNORECASE | re.MULTILINE | re.VERBOSE,
)

CHAPTER_HEADING_RE = re.compile(
    r"(?im)^\s*CHAPTER\s+[IVXLCDM]+\s*\.?\s+[A-Z][A-Z ,&-]+$"
)

SUBSECTION_HEADING_RE = re.compile(
    r"""
    (?m)
    ^\s*
    (?P<prefix>(?:\*+\s*)?(?:\d+\s*)?\[?\s*)?
    \(
    \s*
    (?P<number>\d+[A-Z]?)
    \s*
    \)
    """,
    re.VERBOSE,
)


def normalize_text(text):
    """Normalize PDF extraction artifacts while preserving section line breaks."""
    text = text.replace("\u200b", "")
    text = text.replace("\xa0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_text(pdf_path=DEFAULT_PDF_PATH):
    """Extract plain text from every page using PyMuPDF."""
    with pymupdf.open(pdf_path) as document:
        return "\n".join(page.get_text() for page in document)


def clean_heading_title(title):
    """Remove common footnote/bracket markers from a section title."""
    if section_status(title) == "omitted":
        return "Omitted"
    title = re.sub(r"-\s*\n\s*", "-", title)
    title = re.sub(r"^\d+\s*\[\s*", "", title.strip())
    title = title.replace("[", "").replace("]", "")
    title = title.replace("*", "")
    title = re.sub(r"\s+", " ", title)
    title = re.sub(r"^\d+\s+", "", title)
    return title.strip(" .-")


def section_status(title):
    """Return omitted for headings whose title is only amendment omission marks."""
    if re.fullmatch(r"\s*\d*\s*\[?\s*\*{2,}\s*\]?\s*", title):
        return "omitted"
    return "current"


def strip_trailing_chapter_heading(content):
    """Keep standalone chapter headings out of the previous section content."""
    return normalize_text(CHAPTER_HEADING_RE.sub("", content))


def is_historical_heading(text, match):
    """Detect old quoted headings inside substitution notes."""
    line = text[match.start():text.find("\n", match.start())]
    prefix = match.group("prefix")
    before = text[max(0, match.start() - 260):match.start()].casefold()
    has_amendment_marker = "*" in prefix or "[" in prefix or line.lstrip().startswith("[")
    if has_amendment_marker:
        return False
    return "substituted" in before and re.search(r"\bfor\s*$|\bfor\b", before)


def find_section_headings(text):
    """Return regex matches for legal section headings in extracted text."""
    headings = [match for match in SECTION_HEADING_RE.finditer(text) if not is_historical_heading(text, match)]
    first_by_number = {}
    filtered = []
    for match in headings:
        number = match.group("number").upper()
        if number in first_by_number:
            previous = first_by_number[number]
            previous_line = text[previous.start():text.find("\n", previous.start())]
            current_line = text[match.start():text.find("\n", match.start())]
            context = text[max(0, match.start() - 320):match.start()].casefold()
            if "substituted" in context or "omitted" in context:
                continue
            if "*" in current_line and "*" not in previous_line:
                filtered.remove(previous)
                first_by_number[number] = match
                filtered.append(match)
            continue
        first_by_number[number] = match
        filtered.append(match)
    return filtered


def find_chapter_headings(text):
    """Return standalone chapter headings parsed separately from section content."""
    return [
        {"chapter_heading": re.sub(r"\s+", " ", match.group(0)).strip(), "start": match.start()}
        for match in CHAPTER_HEADING_RE.finditer(text)
    ]


def subsection_marker_text(match):
    """Return the normalized marker text, without PDF amendment footnote wrappers."""
    return f"({match.group('number').upper()})"


def subsection_search_content(content):
    """Return the part of section content used for current subsection detection."""
    patterns = (
        r"(?im)^\s*SCHEDULE\s+[IVXLCDM]+\b",
        r"(?im)^\s*\*?\s*Enforced\s+w\.e\.f\.",
        r"(?im)^\s*\d+\.\s*(?:Substituted|Inserted|Omitted|The words|In clause|In sub-section)\b",
    )
    end = len(content)
    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            end = min(end, match.start())
    return content[:end]


def find_subsection_headings(content):
    """Return genuine top-level subsection matches inside one section."""
    headings = []
    search_content = subsection_search_content(content)
    for match in SUBSECTION_HEADING_RE.finditer(search_content):
        line_start = search_content.rfind("\n", 0, match.start()) + 1
        before_line = search_content[max(0, line_start - 120):line_start]
        before_since_heading = search_content[headings[-1].end() if headings else 0:match.start()]

        if re.search(r"sub\s*-\s*sections?\s*$", before_line, re.IGNORECASE):
            continue
        if re.search(r"sub\s*-\s*$", before_line, re.IGNORECASE):
            continue
        if re.search(r"Explanation\s*\.?\s*-", before_since_heading, re.IGNORECASE):
            continue

        headings.append(match)
    return headings


def parse_subsections(content):
    """Return subsection dictionaries while preserving lower-level clauses as text."""
    headings = find_subsection_headings(content)
    search_content = subsection_search_content(content)
    subsections = []

    for index, match in enumerate(headings):
        text_start = match.end()
        text_end = headings[index + 1].start() if index + 1 < len(headings) else len(search_content)
        body = search_content[text_start:text_end].strip()
        full_text = normalize_text(f"{subsection_marker_text(match)} {body}")
        subsections.append(
            {
                "subsection_number": match.group("number").upper(),
                "text": full_text,
            }
        )

    return subsections


def subsection_boundary_warnings(section):
    """Return suspicious subsection-boundary notes for manual parser inspection."""
    content = section.get("content", "")
    subsections = section.get("subsections", [])
    warnings = []

    if not subsections:
        return warnings

    first_marker = find_subsection_headings(content)[0]
    before_first = content[:first_marker.start()].strip()
    if (
        before_first
        and before_first.casefold() != "p"
        and not re.fullmatch(r"[\W_]+", before_first)
        and section.get("status") != "omitted"
    ):
        warnings.append(f"text appears before first subsection: {before_first[:80]}")

    seen = set()
    previous_numeric = None
    for subsection in subsections:
        number = subsection["subsection_number"]
        if number in seen:
            warnings.append(f"duplicate subsection ({number})")
        seen.add(number)

        match = re.fullmatch(r"(\d+)([A-Z]?)", number)
        if not match:
            warnings.append(f"malformed subsection number ({number})")
            continue

        numeric = int(match.group(1))
        suffix = match.group(2)
        if suffix:
            continue
        if previous_numeric is not None and numeric <= previous_numeric:
            warnings.append(f"non-increasing subsection order near ({number})")
        previous_numeric = numeric

    return warnings


def parse_sections(pdf_path=DEFAULT_PDF_PATH):
    """Return dictionaries with section_number, section_title, content, and subsections."""
    text = normalize_text(extract_pdf_text(pdf_path))
    headings = find_section_headings(text)
    sections = []

    for index, match in enumerate(headings):
        content_start = match.end()
        content_end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        content = strip_trailing_chapter_heading(text[content_start:content_end])
        status = section_status(match.group("title"))
        sections.append(
            {
                "section_number": match.group("number").upper(),
                "section_title": clean_heading_title(match.group("title")),
                "content": content,
                "subsections": [] if status == "omitted" else parse_subsections(content),
                "status": status,
            }
        )

    return sections
