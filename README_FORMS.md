# GST Forms Parser and Chunker

This module provides parsing and chunking capabilities for Hindi CGST Forms PDFs.

## Files Created

- `gst_forms_parser.py` - Parser for extracting forms from PDF
- `forms_chunker.py` - Chunking logic for parsed forms
- `tests/test_gst_forms_parser.py` - Unit tests for parser
- `tests/test_forms_chunker.py` - Unit tests for chunker

## Usage

### 1. Parse Forms from PDF

```bash
python gst_forms_parser.py data/forms/cgst_forms_hindi.pdf
```

Output:
- `data/forms/cgst_forms_parsed.json` - Parsed forms with metadata

### 2. Chunk Parsed Forms

```bash
python forms_chunker.py data/forms/cgst_forms_parsed.json
```

Output:
- `data/forms/cgst_forms_chunks.json` - Form chunks ready for embedding
- `data/forms/cgst_forms_validation.json` - Validation report (PASS/FAIL)

## Features

### Parser (`gst_forms_parser.py`)

- Detects form codes: GST CMP-01, GST REG-01, GSTR-7, GST DRC-01, etc.
- Extracts metadata:
  - `document_type="form"`
  - `form_code`, `form_family`, `form_number`
  - `language="hi"` (preserved for Hindi content)
  - `rule_references` (extracted from text)
  - `title_original`, `purpose_original`
  - `parts/sections`
  - `instructions_original`, `verification_original`
  - `attachments_required`
  - `content_original` (Hindi text preserved exactly)
  - Source file/page metadata

### Chunker (`forms_chunker.py`)

Chunking strategy:
1. **Small forms** → Keep whole (single chunk)
2. **Large forms with parts** → Split by Part/logical section
3. **Large sections** → Group complete numbered fields
4. **Token-aware fallback** → Only when necessary

Each chunk preserves:
- `form_code`, `form_family`, `rule_references`
- `language="hi"`
- Section/part metadata
- Source pages

### Validation

Checks for:
- Missing forms
- Duplicate forms
- Malformed form codes
- Empty forms
- Content loss between parsed form and chunks
- Cross-form chunks
- Oversized chunks
- Broken table blocks

## Output Format

### Parsed Forms JSON Structure

```json
{
  "document_type": "form",
  "form_code": "GST CMP-01",
  "form_family": "Composition Levy",
  "form_number": "01",
  "language": "hi",
  "rule_references": [{"rule_number": "8", "year": "2017"}],
  "title_original": "...",
  "purpose_original": "...",
  "parts": [...],
  "instructions_original": "...",
  "verification_original": "...",
  "attachments_required": [],
  "content_original": "...",
  "source_metadata": {
    "pdf_path": "...",
    "start_page": 1,
    "end_page": 2,
    "pages": [1, 2]
  },
  "blocks": [...]
}
```

### Chunks JSON Structure

```json
{
  "chunk_id": "cgst-form-GST CMP-01-0001",
  "text": "Form Family\nForm GST CODE\n\n...content...",
  "token_count": 150,
  "metadata": {
    "document_type": "form",
    "form_code": "GST CMP-01",
    "form_family": "Composition Levy",
    "form_number": "01",
    "language": "hi",
    "rule_references": [...],
    "section_numbers": [],
    "part_number": null,
    "block_type": null,
    "chunk_strategy": "form",
    "source_pages": [1, 2],
    "start_page": 1,
    "end_page": 2
  }
}
```

### Validation Report

```json
{
  "missing_forms": [],
  "cross_form_chunks": [],
  "oversized_chunks": [],
  "broken_table_blocks": [],
  "content_loss": [],
  "empty_chunks": [],
  "PASS": true,
  "statistics": {
    "total_forms": 4,
    "total_chunks": 4,
    "chunks_per_form_avg": 1.0,
    "max_tokens_in_chunk": 117
  }
}
```

## Notes

- Hindi text is preserved exactly in `content_original` fields
- No translation is performed during parsing
- Ready for BGE-M3 multilingual embedding
- Tables, numbered fields, checkboxes are preserved as logical blocks
- Form boundaries are detected by form code patterns
