"""Inspect parsed GST forms and optionally save JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gst_forms_parser import DEFAULT_FORMS_JSON, DEFAULT_FORMS_PDF, parse_forms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", nargs="?", type=Path, default=DEFAULT_FORMS_PDF)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_FORMS_JSON)
    parser.add_argument("--samples", type=int, default=12)
    args = parser.parse_args()

    forms = parse_forms(args.pdf)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(forms, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Forms found: {len(forms)}")
    print(f"Wrote JSON: {args.json_output}")
    for form in forms[: args.samples]:
        preview = " ".join(form["content"].split())[:250]
        print(f"{form['form_number']} | page {form['page_start']} | tokens {form['token_count']} | {form['form_title']}")
        print(preview)


if __name__ == "__main__":
    main()
