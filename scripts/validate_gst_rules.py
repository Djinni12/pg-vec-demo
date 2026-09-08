"""Validate parsed CGST Rules JSON."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
from collections import Counter
import json

from gst_rules_parser import (
    CHAPTER_HEADING_RE,
    DEFAULT_RULES_PDF_PATH,
    RULE_HEADING_RE,
    find_toc_rule_numbers,
    normalize_text,
    extract_pdf_text,
    rule_boundary_warnings,
)

DEFAULT_JSON_PATH = Path("data/rules/gst_rules_rules.json")
REQUIRED_FIELDS = {"rule_number", "rule_title", "chapter", "chapter_title", "status", "content", "subrules", "token_count"}


def load_rules(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Expected JSON root to be a list of rules")
    return data


def label(rule):
    return f"Rule {rule.get('rule_number', '<missing>')} ({rule.get('rule_title', '')})"


def validate(rules, expected_numbers=None):
    warnings = []
    failures = []

    for index, rule in enumerate(rules, 1):
        if not isinstance(rule, dict):
            failures.append(f"Entry {index}: expected object")
            continue
        missing = sorted(REQUIRED_FIELDS - set(rule))
        if missing:
            failures.append(f"Entry {index}: missing fields: {', '.join(missing)}")
        if not str(rule.get("rule_number", "")).strip():
            failures.append(f"Entry {index}: empty rule_number")
        if not str(rule.get("rule_title", "")).strip():
            failures.append(f"{label(rule)} has empty title")
        if not str(rule.get("content", "")).strip():
            failures.append(f"{label(rule)} has empty content")
        if rule.get("status") not in {"current", "omitted"}:
            failures.append(f"{label(rule)} has invalid status {rule.get('status')!r}")
        if not isinstance(rule.get("subrules"), list):
            failures.append(f"{label(rule)} subrules must be a list")

        for note in rule_boundary_warnings(rule):
            warnings.append(f"{label(rule)}: {note}")

    current_numbers = [rule.get("rule_number") for rule in rules if rule.get("status") == "current"]
    duplicate_current = sorted(number for number, count in Counter(current_numbers).items() if number and count > 1)
    for number in duplicate_current:
        warnings.append(f"Duplicate current rule {number}")

    parsed_numbers = [rule.get("rule_number") for rule in rules]
    if expected_numbers:
        parsed_set = set(parsed_numbers)
        expected_set = set(expected_numbers)
        for number in expected_numbers:
            if number not in parsed_set:
                warnings.append(f"Missing rule from TOC: {number}")
        for number in parsed_numbers:
            if number not in expected_set:
                warnings.append(f"Unexpected rule not in TOC: {number}")

    omitted = [rule.get("rule_number") for rule in rules if rule.get("status") == "omitted"]
    if omitted:
        warnings.append(f"Omitted rules detected: {', '.join(omitted)}")

    embedded = []
    chapters = []
    for rule in rules:
        content = rule.get("content", "")
        own = str(rule.get("rule_number", "")).upper()
        for match in RULE_HEADING_RE.finditer(content):
            number = match.group("number").upper()
            if number != own:
                embedded.append((rule.get("rule_number"), number))
        if CHAPTER_HEADING_RE.search(content):
            chapters.append(rule.get("rule_number"))
    for owner, number in embedded:
        warnings.append(f"Rule {owner} contains embedded genuine Rule {number} heading")
    for number in chapters:
        warnings.append(f"Rule {number} contains standalone chapter heading")

    return warnings, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_path", nargs="?", default=DEFAULT_JSON_PATH)
    parser.add_argument("--pdf-path", default=DEFAULT_RULES_PDF_PATH)
    args = parser.parse_args()

    try:
        rules = load_rules(args.json_path)
        expected = find_toc_rule_numbers(normalize_text(extract_pdf_text(args.pdf_path)))
    except Exception as error:
        print(f"Failed to load validation input: {error}")
        raise SystemExit(1)

    warnings, failures = validate(rules, expected)
    print(f"JSON: {args.json_path}")
    print(f"Total rules: {len(rules)}")
    print(f"Rules with sub-rules: {sum(1 for rule in rules if rule.get('subrules'))}")
    print("RULES IN ORDER")
    for rule in rules:
        print(f"{rule.get('rule_number')} | {rule.get('status')} | {len(rule.get('subrules') or [])} | {rule.get('chapter')} | {rule.get('rule_title')}")
    print()
    print("WARNINGS")
    if warnings:
        for warning in warnings:
            print(f"- {warning}")
    else:
        print("- None")
    print()
    print("FAILURES")
    if failures:
        for failure in failures:
            print(f"- {failure}")
    else:
        print("- None")
    print()
    print("SUMMARY")
    print(f"WARNING: {len(warnings)}")
    print(f"FAIL: {len(failures)}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
