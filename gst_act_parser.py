"""Parse legal sections from the Central GST Act PDF."""

import re
from pathlib import Path

import pymupdf


DEFAULT_PDF_PATH = Path("data/acts/cental_gst_act_2017.pdf")

SECTION_HEADING_RE = re.compile(
    r"""
    ^\s*
    (?:\d+\s*)?
    (?:\[\s*)?
    \*?
    \s*
    Section
    \s+
    (?P<number>\d+[A-Z]?)
    \s*
    \.
    \s*
    (?P<title>[^\n]*(?:\n(?!\s*\()\s*[^\n]*){0,2}?)
    \s*
    (?:
        \.\s*-\s* |
        \.-\s* |
        \.\s+ -\s* |
        \*\s*$
    )
    """,
    re.IGNORECASE | re.MULTILINE | re.VERBOSE,
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
    title = re.sub(r"-\s*\n\s*", "-", title)
    title = re.sub(r"^\d+\s*\[\s*", "", title.strip())
    title = title.replace("[", "").replace("]", "")
    title = title.replace("*", "")
    title = re.sub(r"\s+", " ", title)
    return title.strip(" .-")


def find_section_headings(text):
    """Return regex matches for legal section headings in extracted text."""
    return list(SECTION_HEADING_RE.finditer(text))


def parse_sections(pdf_path=DEFAULT_PDF_PATH):
    """Return dictionaries with section_number, section_title, and content."""
    text = normalize_text(extract_pdf_text(pdf_path))
    headings = find_section_headings(text)
    sections = []

    for index, match in enumerate(headings):
        content_start = match.end()
        content_end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        content = normalize_text(text[content_start:content_end])
        sections.append(
            {
                "section_number": match.group("number").upper(),
                "section_title": clean_heading_title(match.group("title")),
                "content": content,
            }
        )

    return sections
