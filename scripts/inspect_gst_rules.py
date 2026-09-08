"""Inspect parsed CGST Rules before chunking."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import json

from transformers import AutoTokenizer

from gst_rules_parser import DEFAULT_RULES_PDF_PATH, parse_rules, rule_boundary_warnings

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def preview(text, length=300):
    return " ".join(text.split())[:length]


def token_count(tokenizer, text, words_per_batch=120):
    words = text.split()
    total = 0
    for start in range(0, len(words), words_per_batch):
        total += len(tokenizer.tokenize(" ".join(words[start:start + words_per_batch])))
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf_path", nargs="?", default=DEFAULT_RULES_PDF_PATH)
    parser.add_argument("--json-output", type=Path, default=Path("data/rules/gst_rules_rules.json"))
    args = parser.parse_args()

    rules = parse_rules(args.pdf_path)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    inspected = []

    print(f"PDF: {args.pdf_path}")
    print(f"Total rules found: {len(rules)}")
    print(f"Rules with sub-rules: {sum(1 for rule in rules if rule.get('subrules'))}")
    print()

    for rule in rules:
        count = token_count(tokenizer, rule["content"])
        inspected.append({**rule, "token_count": count, "preview": preview(rule["content"])})
        print(f"Rule {rule['rule_number']}: {rule['rule_title']}")
        print(f"Chapter: {rule.get('chapter')} | {rule.get('chapter_title')}")
        print(f"Status: {rule['status']} | Token count: {count} | Sub-rules: {len(rule.get('subrules', []))}")
        warnings = rule_boundary_warnings(rule)
        if warnings:
            print(f"Boundary warnings: {'; '.join(warnings)}")
        print(f"Preview: {preview(rule['content'])}")
        print()

    print("SUB-RULE COUNT PER RULE")
    for rule in rules:
        print(f"{rule['rule_number']} | {len(rule.get('subrules', []))} | {rule['rule_title']}")
    print()

    print("PARSED EXAMPLES")
    for rule in [rule for rule in rules if rule.get("subrules")][:5]:
        print(f"Rule {rule['rule_number']}: {rule['rule_title']}")
        for subrule in rule["subrules"][:3]:
            print(f"  {subrule['subrule_number']} | {preview(subrule['text'], 220)}")
        print()

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(inspected, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote JSON: {args.json_output}")


if __name__ == "__main__":
    main()
