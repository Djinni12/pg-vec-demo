"""Validate parsed CGST Act sections JSON before chunking."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re


DEFAULT_JSON_PATH = Path("data/acts/central_gst_act_2017_sections.json")
REQUIRED_FIELDS = {"section_number", "section_title", "content", "token_count"}
SECTION_REFERENCE_RE = re.compile(r"\bSection\s+(\d+[A-Z]?)\s*\.", re.IGNORECASE)
CHAPTER_RE = re.compile(r"\bCHAPTER\b", re.IGNORECASE)


def load_sections(path):
    with Path(path).open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("Expected the JSON root to be a list of sections")
    return data


def section_label(section, index):
    number = section.get("section_number", f"<missing #{index + 1}>")
    title = section.get("section_title", "")
    return f"Section {number} ({title})"


def validate(sections):
    failures = []
    warnings = []
    passes = []

    missing_fields = []
    empty_titles = []
    empty_content = []
    invalid_token_counts = []

    for index, section in enumerate(sections):
        if not isinstance(section, dict):
            failures.append(f"Entry {index + 1}: expected object, got {type(section).__name__}")
            continue

        missing = sorted(REQUIRED_FIELDS - set(section))
        if missing:
            missing_fields.append((index, missing))

        title = str(section.get("section_title", "")).strip()
        content = str(section.get("content", "")).strip()
        if not title:
            empty_titles.append(index)
        if not content:
            empty_content.append(index)

        token_count = section.get("token_count")
        if not isinstance(token_count, int):
            invalid_token_counts.append(index)

    if missing_fields:
        for index, missing in missing_fields:
            failures.append(f"{section_label(sections[index], index)} missing fields: {', '.join(missing)}")
    else:
        passes.append("Every entry has required fields")

    if empty_titles:
        for index in empty_titles:
            failures.append(f"{section_label(sections[index], index)} has an empty title")
    else:
        passes.append("No empty titles")

    if empty_content:
        for index in empty_content:
            failures.append(f"{section_label(sections[index], index)} has empty content")
    else:
        passes.append("No empty content")

    if invalid_token_counts:
        for index in invalid_token_counts:
            failures.append(f"{section_label(sections[index], index)} has non-integer token_count")
    else:
        passes.append("All token counts are integers")

    numbers = [section.get("section_number") for section in sections if isinstance(section, dict)]
    duplicate_numbers = sorted(number for number, count in Counter(numbers).items() if number and count > 1)
    if duplicate_numbers:
        for number in duplicate_numbers:
            positions = [str(index + 1) for index, section in enumerate(sections) if section.get("section_number") == number]
            warnings.append(f"Duplicate section number {number} at entries: {', '.join(positions)}")
    else:
        passes.append("No duplicate section numbers")

    suspicious_refs = []
    chapter_hits = []
    over_1000 = []
    over_2000 = []

    for index, section in enumerate(sections):
        if not isinstance(section, dict):
            continue
        own_number = str(section.get("section_number", "")).upper()
        content = str(section.get("content", ""))

        refs = []
        for match in SECTION_REFERENCE_RE.finditer(content):
            ref_number = match.group(1).upper()
            if ref_number != own_number:
                refs.append(ref_number)
        if refs:
            suspicious_refs.append((index, sorted(set(refs))))

        if CHAPTER_RE.search(content):
            chapter_hits.append(index)

        token_count = section.get("token_count")
        if isinstance(token_count, int):
            if token_count > 1000:
                over_1000.append(index)
            if token_count > 2000:
                over_2000.append(index)

    if suspicious_refs:
        for index, refs in suspicious_refs:
            warnings.append(f"{section_label(sections[index], index)} contains Section references with dots: {', '.join(refs)}")
    else:
        passes.append("No suspicious Section X. references inside content")

    if chapter_hits:
        for index in chapter_hits:
            warnings.append(f"{section_label(sections[index], index)} contains a CHAPTER heading")
    else:
        passes.append("No CHAPTER headings inside section content")

    if over_1000:
        warnings.append(f"Sections with token_count > 1000: {', '.join(sections[i]['section_number'] for i in over_1000)}")
    else:
        passes.append("No sections over 1000 tokens")

    if over_2000:
        warnings.append(f"Sections with token_count > 2000: {', '.join(sections[i]['section_number'] for i in over_2000)}")
    else:
        passes.append("No sections over 2000 tokens")

    return passes, warnings, failures


def print_sections(sections):
    print("SECTIONS IN ORDER")
    for section in sections:
        number = section.get("section_number", "<missing>")
        token_count = section.get("token_count", "<missing>")
        title = section.get("section_title", "<missing>")
        print(f"{number} | {token_count} | {title}")
    print()


def print_largest_sections(sections, limit=10):
    sortable = [
        section for section in sections
        if isinstance(section, dict) and isinstance(section.get("token_count"), int)
    ]
    largest = sorted(sortable, key=lambda section: section["token_count"], reverse=True)[:limit]
    print("10 LARGEST SECTIONS")
    for section in largest:
        print(f"{section['section_number']} | {section['token_count']} | {section['section_title']}")
    print()


def print_messages(title, messages):
    print(title)
    if messages:
        for message in messages:
            print(f"- {message}")
    else:
        print("- None")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_path", nargs="?", default=DEFAULT_JSON_PATH)
    args = parser.parse_args()

    try:
        sections = load_sections(args.json_path)
    except Exception as error:
        print(f"Failed to load JSON: {error}")
        print("SUMMARY")
        print("PASS: 0")
        print("WARNING: 0")
        print("FAIL: 1")
        raise SystemExit(1)

    passes, warnings, failures = validate(sections)

    print(f"JSON: {args.json_path}")
    print(f"Total entries: {len(sections)}")
    print()
    print_sections(sections)
    print_largest_sections(sections)
    print_messages("WARNINGS", warnings)
    print_messages("FAILURES", failures)
    print("SUMMARY")
    print(f"PASS: {len(passes)}")
    print(f"WARNING: {len(warnings)}")
    print(f"FAIL: {len(failures)}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
