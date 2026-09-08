"""Parse GST Forms from Hindi CGST Forms PDF."""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf


DEFAULT_FORMS_PDF_PATH = Path("data/forms/cgst_forms_hindi.pdf")

# Form code patterns: GST CMP-01, GST REG-01, GSTR-7, etc.
FORM_CODE_RE = re.compile(
    r"""
    (?im)
    ^\s*
    (?P<form_code>
        (?:GST\s*)?(?:CMP|REG|REF|PRN|TRN|ITC|GSTR|PMTR|PMT|DRC|APL|REV|AMT|FL)\s*-?\s*\d+[A-Z]?
        |
        GSTR\s*-?\s*\d+[A-Z]?
    )
    \s*$
    """,
    re.VERBOSE,
)

# Alternative pattern for form codes embedded in titles
FORM_CODE_INLINE_RE = re.compile(
    r"""
    (?P<form_code>
        GST\s*(?:CMP|REG|REF|PRN|TRN|ITC|GSTR|PMTR|PMT|DRC|APL|REV|AMT|FL)\s*-?\s*\d+[A-Z]?
        |
        GSTR\s*-?\s*\d+[A-Z]?
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Pattern to detect form boundaries - new form starts with form code heading
FORM_BOUNDARY_RE = re.compile(
    r"""
    (?im)
    ^\s*
    (?:
        (?P<code>GST\s*(?:CMP|REG|REF|PRN|TRN|ITC|GSTR|PMTR|PMT|DRC|APL|REV|AMT|FL)\s*-?\s*\d+[A-Z]?)
        |
        (?P<code_short>GSTR\s*-?\s*\d+[A-Z]?)
    )
    \s*$
    """,
    re.VERBOSE,
)

# Pattern for Part/Section headings within forms
PART_HEADING_RE = re.compile(
    r"""
    (?im)
    ^\s*
    (?:Part|भाग)\s+
    (?P<number>[IVXLCDM]+|\d+|[अआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह]+)?
    \s*[:.\-]?\s*
    (?P<title>[A-Zअ-ह०-९][^\n]*?)?
    \s*$
    """,
    re.VERBOSE,
)

# Pattern for numbered fields/entries
FIELD_NUMBER_RE = re.compile(
    r"""
    (?m)
    ^\s*
    (?P<prefix>(?:\*+\s*)?)
    (?P<number>\d+[A-Z]?|[अ-ह०-९]+[अआइई]?)
    \s*[.\)]\s*
    """,
    re.VERBOSE,
)

# Pattern for table rows (detects tabular structure)
TABLE_ROW_RE = re.compile(
    r"""
    (?m)
    ^\s*
    (?:
        (?P<sl>\d+|[अ-ह०-९]+)\s*[.\)]
        |
        \|\s*.*\s*\|
        |
        \t.*\t
    )
    """,
    re.VERBOSE,
)

# Pattern for instructions section
INSTRUCTIONS_RE = re.compile(
    r"""
    (?im)
    ^\s*
    (?:Instructions?|निर्देश|अनुदेश)
    \s*[:.\-]?
    \s*$
    """,
    re.VERBOSE,
)

# Pattern for verification/declaration section
VERIFICATION_RE = re.compile(
    r"""
    (?im)
    ^\s*
    (?:Verification|सत्यापन|Declaration|घोषणा)
    \s*[:.\-]?
    \s*$
    """,
    re.VERBOSE,
)

# Pattern for attachments/documents required
ATTACHMENTS_RE = re.compile(
    r"""
    (?im)
    ^\s*
    (?:Attachments?|संलग्नक|Documents?|दस्तावेज़|List of documents)
    \s*[:.\-]?
    \s*$
    """,
    re.VERBOSE,
)

# Pattern for rule references
RULE_REFERENCE_RE = re.compile(
    r"""
    (?im)
    (?:Rule|नियम)\s+
    (?P<rule_number>\d+[A-Z]?)
    \s*(?:of|के)?
    \s*(?:CGST\s*Rules|सीजीएसटी\s*नियम)?
    \s*,?\s*
    (?P<year>2017)?
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Known form families
FORM_FAMILIES = {
    "CMP": "Composition Levy",
    "REG": "Registration",
    "REF": "Refund",
    "PRN": "Payment Reference Number",
    "TRN": "Temporary Reference Number",
    "ITC": "Input Tax Credit",
    "GSTR": "Return",
    "PMTR": "Payment",
    "PMT": "Payment",
    "DRC": "Demand and Recovery",
    "APL": "Appeal",
    "REV": "Revision",
    "AMT": "Amendment",
    "FL": "Filing",
}


def normalize_text(text):
    """Normalize PDF extraction artifacts while preserving Hindi text exactly."""
    text = text.replace("\u200b", "")
    text = text.replace("\xa0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_text_with_pages(pdf_path=DEFAULT_FORMS_PDF_PATH):
    """Extract plain text from every page with page metadata."""
    pages_data = []
    with pymupdf.open(pdf_path) as document:
        for page_num, page in enumerate(document, 1):
            text = page.get_text()
            pages_data.append({
                "page_number": page_num,
                "text": text,
            })
    return pages_data


def extract_form_code(text):
    """Extract form code from text."""
    # Try exact match first
    match = FORM_CODE_RE.search(text)
    if match:
        return normalize_form_code(match.group("form_code"))
    
    # Try inline match
    match = FORM_CODE_INLINE_RE.search(text)
    if match:
        return normalize_form_code(match.group("form_code"))
    
    return None


def normalize_form_code(code):
    """Normalize form code to standard format."""
    if not code:
        return None
    # Remove extra spaces and normalize
    code = re.sub(r"\s+", " ", code.strip())
    code = re.sub(r"\s*-\s*", "-", code)
    code = code.upper()
    # Ensure proper spacing: GST CMP-01
    if not code.startswith("GST ") and not code.startswith("GSTR"):
        code = "GST " + code
    return code


def get_form_family(form_code):
    """Determine form family from form code."""
    if not form_code:
        return None
    
    for prefix, family in FORM_FAMILIES.items():
        if prefix in form_code:
            return family
    return "Other"


def extract_rule_references(text):
    """Extract rule references from form text."""
    references = []
    for match in RULE_REFERENCE_RE.finditer(text):
        ref = {
            "rule_number": match.group("rule_number"),
            "year": match.group("year") or "2017",
        }
        references.append(ref)
    return references


def find_form_boundaries(pages_data):
    """Find form boundary positions across all pages."""
    boundaries = []
    
    for page_info in pages_data:
        page_num = page_info["page_number"]
        text = page_info["text"]
        
        for match in FORM_BOUNDARY_RE.finditer(text):
            code = match.group("code") or match.group("code_short")
            boundaries.append({
                "page_number": page_num,
                "position": match.start(),
                "form_code": normalize_form_code(code),
            })
    
    return boundaries


def detect_form_structure(text):
    """Detect structural elements within form text."""
    structure = {
        "parts": [],
        "instructions_start": None,
        "verification_start": None,
        "attachments_start": None,
        "table_regions": [],
        "field_numbers": [],
    }
    
    # Find part/section headings
    for match in PART_HEADING_RE.finditer(text):
        structure["parts"].append({
            "part_number": match.group("number"),
            "part_title": match.group("title"),
            "start": match.start(),
            "end": match.end(),
        })
    
    # Find instructions section
    match = INSTRUCTIONS_RE.search(text)
    if match:
        structure["instructions_start"] = match.start()
    
    # Find verification section
    match = VERIFICATION_RE.search(text)
    if match:
        structure["verification_start"] = match.start()
    
    # Find attachments section
    match = ATTACHMENTS_RE.search(text)
    if match:
        structure["attachments_start"] = match.start()
    
    # Detect table regions (simplified heuristic)
    lines = text.split("\n")
    in_table = False
    table_start = None
    for i, line in enumerate(lines):
        if TABLE_ROW_RE.match(line):
            if not in_table:
                in_table = True
                table_start = sum(len(l) + 1 for l in lines[:i])
        else:
            if in_table and table_start is not None:
                table_end = sum(len(l) + 1 for l in lines[:i])
                structure["table_regions"].append({
                    "start": table_start,
                    "end": table_end,
                })
                in_table = False
                table_start = None
    
    # Find field numbers
    for match in FIELD_NUMBER_RE.finditer(text):
        structure["field_numbers"].append({
            "number": match.group("number"),
            "start": match.start(),
        })
    
    return structure


def extract_form_title(text, form_code):
    """Extract form title from text near form code."""
    # Look for title after form code
    code_pos = text.find(form_code) if form_code else 0
    if code_pos == -1:
        code_pos = 0
    
    # Get text after form code (within 500 chars)
    title_region = text[code_pos:code_pos + 500]
    lines = title_region.split("\n")
    
    # First non-empty line after form code might be the title
    title_lines = []
    for line in lines[1:]:
        line = line.strip()
        if line and line != form_code:
            title_lines.append(line)
            if len(title_lines) >= 2:
                break
    
    return "\n".join(title_lines).strip() if title_lines else None


def extract_purpose(text):
    """Extract purpose/description from form text."""
    # Look for common purpose indicators
    purpose_patterns = [
        r"(?im)(?:Purpose|उद्देश्य|Application for|के लिए आवेदन)[:\s]+([^\n]+)",
        r"(?im)^([^\n]*(?:registration|composition|return|refund|दस्तावेज़)[^\n]*)$",
    ]
    
    for pattern in purpose_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip() if match.lastindex else match.group(0).strip()
    
    return None


def parse_form_content(text, form_code, page_start, page_end):
    """Parse a single form's content into structured blocks."""
    structure = detect_form_structure(text)
    
    # Extract sections based on detected boundaries
    sections = []
    
    # Main form content (before instructions)
    main_end = structure["instructions_start"] or structure["verification_start"] or len(text)
    main_content = text[:main_end].strip()
    
    if main_content:
        sections.append({
            "block_type": "main_form",
            "content_original": main_content,
            "start_offset": 0,
            "end_offset": len(main_content),
        })
    
    # Instructions section
    if structure["instructions_start"] is not None:
        instr_start = structure["instructions_start"]
        instr_end = structure["verification_start"] or structure["attachments_start"] or len(text)
        instructions = text[instr_start:instr_end].strip()
        if instructions:
            sections.append({
                "block_type": "instructions",
                "content_original": instructions,
                "start_offset": instr_start,
                "end_offset": instr_end,
            })
    
    # Verification section
    if structure["verification_start"] is not None:
        verif_start = structure["verification_start"]
        verif_end = structure["attachments_start"] or len(text)
        verification = text[verif_start:verif_end].strip()
        if verification:
            sections.append({
                "block_type": "verification",
                "content_original": verification,
                "start_offset": verif_start,
                "end_offset": verif_end,
            })
    
    # Attachments section
    if structure["attachments_start"] is not None:
        attachments = text[structure["attachments_start"]:].strip()
        if attachments:
            sections.append({
                "block_type": "attachments",
                "content_original": attachments,
                "start_offset": structure["attachments_start"],
                "end_offset": len(text),
            })
    
    return {
        "sections": sections,
        "structure": structure,
    }


def parse_forms(pdf_path=DEFAULT_FORMS_PDF_PATH):
    """Parse all GST forms from the Hindi PDF."""
    pages_data = extract_pdf_text_with_pages(pdf_path)
    
    # Combine all text for boundary detection
    full_text = "\n".join(page["text"] for page in pages_data)
    full_text = normalize_text(full_text)
    
    # Find form boundaries
    boundaries = find_form_boundaries(pages_data)
    
    # Sort boundaries by position
    boundaries.sort(key=lambda b: (b["page_number"], b["position"]))
    
    forms = []
    
    if not boundaries:
        # If no explicit boundaries found, treat entire document as one form
        # This shouldn't happen with valid forms PDF
        return forms
    
    # Create page offset mapping
    page_offsets = []
    current_offset = 0
    for page_info in pages_data:
        page_offsets.append({
            "page_number": page_info["page_number"],
            "offset": current_offset,
            "text_length": len(page_info["text"]),
        })
        current_offset += len(page_info["text"]) + 1  # +1 for newline
    
    # Parse each form
    for i, boundary in enumerate(boundaries):
        form_code = boundary["form_code"]
        start_page = boundary["page_number"]
        start_pos_in_page = boundary["position"]
        
        # Calculate absolute start position
        page_offset = next(
            (p["offset"] for p in page_offsets if p["page_number"] == start_page),
            0
        )
        abs_start = page_offset + start_pos_in_page
        
        # Determine end position (start of next form or end of document)
        if i + 1 < len(boundaries):
            next_boundary = boundaries[i + 1]
            end_page = next_boundary["page_number"]
            end_pos_in_page = next_boundary["position"]
            
            page_offset_end = next(
                (p["offset"] for p in page_offsets if p["page_number"] == end_page),
                0
            )
            abs_end = page_offset_end + end_pos_in_page
            
            end_pages = list(range(start_page, end_page + 1))
        else:
            abs_end = len(full_text)
            end_pages = list(range(start_page, len(page_offsets) + 1))
        
        # Extract form text
        form_text = full_text[abs_start:abs_end]
        
        # Extract metadata
        form_title = extract_form_title(form_text, form_code)
        purpose = extract_purpose(form_text)
        rule_refs = extract_rule_references(form_text)
        
        # Parse form content
        parsed_content = parse_form_content(form_text, form_code, abs_start, abs_end)
        
        # Build form record
        form_record = {
            "document_type": "form",
            "form_code": form_code,
            "form_family": get_form_family(form_code),
            "form_number": re.search(r"\d+[A-Z]?", form_code).group(0) if form_code else None,
            "language": "hi",
            "rule_references": rule_refs,
            "title_original": form_title,
            "purpose_original": purpose,
            "parts": parsed_content["structure"]["parts"],
            "instructions_original": None,
            "verification_original": None,
            "attachments_required": [],
            "content_original": form_text,
            "source_metadata": {
                "pdf_path": str(pdf_path),
                "start_page": start_page,
                "end_page": end_pages[-1] if end_pages else start_page,
                "pages": end_pages,
                "start_position": abs_start,
                "end_position": abs_end,
            },
            "blocks": parsed_content["sections"],
        }
        
        # Extract instructions and verification from blocks
        for block in parsed_content["sections"]:
            if block["block_type"] == "instructions":
                form_record["instructions_original"] = block["content_original"]
            elif block["block_type"] == "verification":
                form_record["verification_original"] = block["content_original"]
            elif block["block_type"] == "attachments":
                # Parse attachment list
                attachments_text = block["content_original"]
                form_record["attachments_required"] = [
                    line.strip() for line in attachments_text.split("\n")
                    if line.strip() and not INSTRUCTIONS_RE.match(line)
                ][1:]  # Skip the header line
        
        forms.append(form_record)
    
    return forms


def validate_forms(forms):
    """Validate parsed forms for common issues."""
    issues = {
        "missing_forms": [],
        "duplicate_forms": [],
        "malformed_codes": [],
        "empty_forms": [],
        "warnings": [],
    }
    
    seen_codes = set()
    expected_families = {"CMP", "REG", "REF", "GSTR", "DRC", "APL", "ITC"}
    found_families = set()
    
    for form in forms:
        form_code = form.get("form_code")
        
        # Check for empty forms
        content = form.get("content_original", "")
        if not content.strip():
            issues["empty_forms"].append(form_code or "UNKNOWN")
            continue
        
        # Check for malformed codes
        if not form_code or not re.match(r"GST\s*(?:CMP|REG|REF|PRN|TRN|ITC|GSTR|PMTR|PMT|DRC|APL|REV|AMT|FL)\s*-\s*\d+[A-Z]?", form_code, re.IGNORECASE):
            if form_code:
                issues["malformed_codes"].append(form_code)
        
        # Check for duplicates
        if form_code in seen_codes:
            issues["duplicate_forms"].append(form_code)
        seen_codes.add(form_code)
        
        # Track found families
        if form_code:
            for family in expected_families:
                if family in form_code:
                    found_families.add(family)
    
    # Check for missing expected forms (optional warning)
    missing_expected = expected_families - found_families
    if missing_expected:
        issues["missing_forms"] = list(missing_expected)
    
    return issues


def save_forms(forms, output_path=None):
    """Save parsed forms to JSON file."""
    import json
    
    if output_path is None:
        output_path = Path("data/forms/cgst_forms_parsed.json")
    
    with Path(output_path).open("w", encoding="utf-8") as f:
        json.dump(forms, f, ensure_ascii=False, indent=2)
    
    return output_path


if __name__ == "__main__":
    import sys
    
    pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FORMS_PDF_PATH
    
    if not pdf_path.exists():
        print(f"Error: PDF not found at {pdf_path}")
        print("Please upload the Hindi CGST Forms PDF to data/forms/")
        sys.exit(1)
    
    print(f"Parsing forms from {pdf_path}...")
    forms = parse_forms(pdf_path)
    print(f"Found {len(forms)} forms")
    
    # Validate
    issues = validate_forms(forms)
    print("\nValidation Results:")
    print(f"  Empty forms: {len(issues['empty_forms'])}")
    print(f"  Duplicate forms: {len(issues['duplicate_forms'])}")
    print(f"  Malformed codes: {len(issues['malformed_codes'])}")
    print(f"  Missing families: {issues['missing_forms']}")
    
    # Save
    output_path = save_forms(forms)
    print(f"\nSaved parsed forms to {output_path}")
