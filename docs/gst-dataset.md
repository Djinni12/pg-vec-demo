# Kaggle GST dataset and field mapping

Source: [Goods and Service Tax Rates Dataset by prasad22](https://www.kaggle.com/datasets/prasad22/goods-and-service-tax-rates-dataset). The archive was downloaded and inspected on 2026-09-07. These are dataset snapshot records; the files do not provide a structured effective-date or last-verified-date column.

## Files and columns

The archive contains exactly two CSVs. Both decode as Windows-1252 (`cp1252`); UTF-8 decoding fails. Headers and values contain extra spaces, newlines, and nonbreaking spaces. The column names below have whitespace normalized for readability. Exact original headers and cell contents are retained in each record's `metadata.raw_row`.

| File | Bytes | Data rows | Columns |
| --- | ---: | ---: | --- |
| `Goods.csv` | 285,072 | 1,850 | `Schedules`; `S. No.`; `Chapter / Heading / Sub-heading / Tariff item`; `Description of Goods`; `CGST Rate (%)`; `SGST / UTGST Rate (%)`; `IGST Rate (%)`; `Compensation Cess` |
| `Services.csv` | 126,370 | 232 | `S. No.`; `Chapter, Section or Heading`; `Description of Service`; `CGST Rate(%)`; `SGST/UTGST Rate(%)`; `IGST Rate(%)`; `Condition` |

Row numbers count CSV data records starting at 1, excluding the header; embedded newlines do not increment this record number.

| File | Imported | Omitted descriptions skipped | Blank descriptions skipped | Column-number rows skipped |
| --- | ---: | ---: | ---: | ---: |
| `Goods.csv` | 1,499 | 349 | 2 | 0 |
| `Services.csv` | 230 | 0 | 1 | 1 |
| Total | 1,729 | 349 | 3 | 1 |

The services column-number row contains `(1)`, `(2)`, `(3)`, etc., rather than an actual service. Category/header-like descriptions such as `All Services` remain searchable with unknown rates. Blank serial numbers are not forward-filled.

## Field decisions

| Source field | Storage | BM25 | Vector embedding | Exact lookup |
| --- | --- | --- | --- | --- |
| Goods/service description | Cleaned `description`, plus original in metadata | Included in `search_text` | Description only | No |
| Goods tariff classification / service chapter, section or heading | Cleaned `code`; conservative `exact_codes` array; original in metadata | Included in `search_text` | Excluded | Explicit codes only |
| Service `Condition` | Original in metadata | Included unless blank, `-`, or ellipsis placeholder | Excluded | No |
| Goods `Compensation Cess` | Original text in metadata | Included when present | Excluded | No |
| CGST, SGST/UTGST, IGST rates | Nullable numeric percentage columns; original and cleaned raw rate strings in metadata | Excluded as separate fields | Excluded | Available as metadata, not a code lookup key |
| Goods schedule and source serial number | Original in metadata | Excluded | Excluded | Not unique; not lookup keys |
| Filename, CSV record number, file SHA-256, dataset URL | Source columns plus provenance in metadata | Excluded | Excluded | `(source_file, source_row)` identifies the imported record |

Trigram fallback uses the cleaned `description`, preserving the existing typo-search behavior. Long conditions do not dilute its similarity score. Embedding descriptions alone avoids adding rate numbers and long shared conditions to the vector input. Long descriptions can still exceed the embedding model's token limit and be truncated by the model; chunking is intentionally outside this POC change.

## Rates are not uniform numbers

Despite the percentage labels in both files, goods cells encode rates as fractions: `0.025` CGST and `0.05` IGST become `2.5` and `5` in the numeric percentage columns. Service cells already use percentage units: `2.5` and `5` stay unchanged. Conversion uses Python `Decimal`, including small goods rates such as `0.00125` → `0.125` percent.

`Nil` becomes numeric zero. Blank cells, alternatives such as `5 or 12`, and instructions such as “Same rate … as … like goods” remain SQL `NULL` numerically, with their full original text retained. Missing rates are never assumed to be zero; cess is never forced into a numeric percentage because it includes per-unit and conditional amounts. Source rate inconsistencies are not corrected or recomputed. Search output prints normalized percentages when available, otherwise the source rate text or `unspecified`.

## Exact codes are not unique tax rules

Examples in the actual files include `0202, 0203`, `9405 91 00`, `5004 to 5006`, `0507 [Except 050790]`, `Any Chapter`, and `Heading 9954 or 9983 or 9987`.

The loader parses only complete, explicit code lists separated by commas, slashes, `or`, or `and`. It preserves leading zeros, removes internal tariff-code spacing, and accepts 2, 4, 6, or 8 digits. For services it removes `Chapter`/`Heading` prefixes and parenthesized category labels. It does not interpret sections, expand ranges, index excluded codes, infer parent/child matches, repair malformed expressions, or resolve “any chapter.” Ambiguous expressions retain the complete `code` text but get an empty `exact_codes` array.

In this snapshot, 1,643 retained records have parsed exact codes; 86 are searchable by text/vector only. Exact lookup for `0901` can return several rows, because coffee classifications have different descriptions and rates. The lookup is array membership, not a unique-code-to-rate mapping, and returns at most 20 entries by default. Full classification text, descriptions, and source conditions remain necessary to interpret a result.

## Schema and import behavior

`schema.sql` adds `gst_documents`; it does not alter or delete the previous `documents` table. Search and ingestion now exclusively target the GST table, so old medical records cannot appear in GST results.

| Column | Type / purpose |
| --- | --- |
| `source_file`, `source_row` | `TEXT`, `INTEGER`; composite primary key for repeat imports |
| `record_type` | `TEXT`; `goods` or `service` |
| `code` | `TEXT`; source classification expression, not a unique ID |
| `exact_codes` | `TEXT[]`; explicit codes, with GIN array index |
| `description` | `TEXT`; cleaned description for embedding, display, and fuzzy matching |
| `search_text` | `TEXT`; description + classification + condition/cess, indexed with Timescale BM25 |
| `cgst_rate_pct`, `sgst_utgst_rate_pct`, `igst_rate_pct` | Nullable `NUMERIC`; unambiguous percentage rates |
| `metadata` | `JSONB`; complete original row, raw rates, dataset URL, filename, record number, and source hash |
| `embedding` | `VECTOR(384)`; `all-MiniLM-L6-v2` description embedding |

Both CSVs are validated before any database write or model load. Embeddings are generated in batches of 32. One transaction upserts retained records and removes stale records from the two imported source files, including rows newly marked omitted. Reimporting a changed snapshot regenerates embeddings; no incremental model cache is added. The source filename/row identity is stable within a snapshot, not a permanent identifier across arbitrary file reordering. A failed import rolls back database changes.

The downloaded CSVs are in ignored `data/gst/csvs/`; they are not embedded into Python or committed as fixtures. `tests/test_pgvector.py` remains a small compatibility entry point to the GST loader.

## Snapshot fingerprints

```text
Goods.csv
356ce9b4776ef5a1ea67854624d678b7efb0f0ed7728284755ca6e5c313370f3

Services.csv
3d6f3428dbe9327caf598819ec25e034dd4dcbe24f574c9bf78d8a883fe8631c
```

Run `python ingest_gst.py --dry-run` to inspect counts and hashes for the files present locally. Counts above describe the inspected snapshot; future Kaggle downloads may differ.
