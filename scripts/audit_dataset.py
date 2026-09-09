import csv
import re
from collections import Counter, defaultdict
import pymupdf

PDF_PATH = "data/gst/GST rates2025.pdf"
CSV_PATH = "data/gst/gst_rates.csv"

doc = pymupdf.open(PDF_PATH)
print("PDF Total Pages:", len(doc))

with open(CSV_PATH, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f, escapechar="\\"))

print("CSV Total Rows:", len(rows))

# 1 & 2. PDF Sections and Notifications
pdf_headings = []
for pno in range(len(doc)):
    text = doc[pno].get_text()
    for line in text.split("\n"):
        line_clean = line.strip()
        if any(line_clean.startswith(p) for p in ["1.", "2.", "3.", "4.", "5."]) and any(k in line_clean.lower() for k in ["rates", "goods", "cess"]):
            pdf_headings.append((pno + 1, line_clean))

print("\n--- PDF Headings Discovered ---")
for pno, h in pdf_headings:
    print(f"Page {pno}: {h}")

# Check Notifications in CSV
notif_counts = Counter(r["notification_no"] for r in rows)
print("\n--- Notifications in CSV ---")
for notif, count in notif_counts.items():
    print(f"  {notif}: {count} rows")

# 3 & 4. Rate Category counts
cat_counts = Counter(r["rate_category"] for r in rows)
print("\n--- Category Counts ---")
for cat, count in cat_counts.items():
    print(f"  {cat}: {count} rows")

# 5. Exemption examples
exemption_rows = [r for r in rows if r["rate_category"] == "EXEMPTION"]
print(f"\n--- Exemption Rows Total: {len(exemption_rows)} ---")
for r in exemption_rows[:6]:
    print(f"HSN: {r['hsn_code']} | Rate: {r['source_rate']} | Notif: {r['notification_no']} | Heading: {r['section_heading']}")
    print(f"  Desc: {r['description'][:80]}")

# 6. Compensation Cess rows & HSN overlap
cess_rows = [r for r in rows if r["rate_category"] == "COMPENSATION_CESS"]
print(f"\n--- Compensation Cess Rows Total: {len(cess_rows)} ---")
for r in cess_rows[:6]:
    print(f"HSN: {r['hsn_code']} | Cess Rate: {r['source_rate']} | Page: {r['source_page']} | S.No: {r['serial_no']}")
    print(f"  Desc: {r['description'][:80]}")

# Cross-matching HSNs between CGST and Cess
cgst_hsn_map = defaultdict(list)
for r in rows:
    if r["rate_category"] == "CGST":
        # Extract individual codes
        for code in re.split(r"[,;\s]+", r["hsn_code"]):
            c_clean = re.sub(r"\D", "", code)
            if len(c_clean) >= 2:
                cgst_hsn_map[c_clean].append(r)
                if len(c_clean) > 4:
                    cgst_hsn_map[c_clean[:4]].append(r)
                if len(c_clean) > 2:
                    cgst_hsn_map[c_clean[:2]].append(r)

cess_hsn_map = defaultdict(list)
for r in cess_rows:
    for code in re.split(r"[,;\s]+", r["hsn_code"]):
        c_clean = re.sub(r"\D", "", code)
        if len(c_clean) >= 2:
            cess_hsn_map[c_clean].append(r)
            if len(c_clean) > 4:
                cess_hsn_map[c_clean[:4]].append(r)
            if len(c_clean) > 2:
                cess_hsn_map[c_clean[:2]].append(r)

# Find overlapping 4-digit chapters/headings or 2-digit chapters
overlapping_4digit = set()
for c in cess_hsn_map:
    if len(c) == 4 and c in cgst_hsn_map:
        overlapping_4digit.add(c)

print(f"\n--- Overlapping 4-digit HSNs between CGST and Cess ({len(overlapping_4digit)}) ---")
print(sorted(list(overlapping_4digit)))

# 7. SPECIAL / UNKNOWN rows
special_rows = [r for r in rows if r["rate_category"] == "SPECIAL"]
unknown_rows = [r for r in rows if r["rate_category"] == "UNKNOWN"]
print(f"\n--- SPECIAL Rows Total: {len(special_rows)} ---")
for r in special_rows:
    print(f"P.{r['source_page']} | S.No: {r['serial_no']} | HSN: {r['hsn_code']} | Notif: {r['notification_no']} | Sched: {r['schedule']} | Rate: '{r['source_rate']}'")
    print(f"  Desc: {r['description'][:80]}")

print(f"\n--- UNKNOWN Rows Total: {len(unknown_rows)} ---")

# 8. All distinct rate formats across all rows
rate_formats = Counter()
for r in rows:
    rate = r["source_rate"]
    if not rate or rate == "-":
        rate_formats["Empty / Dash (-)"] += 1
    elif re.match(r"^[\d\.]+\s*%$", rate):
        rate_formats["Standard Percentage (e.g. 2.5%, 9%, 14%)"] += 1
    elif rate.lower() == "nil":
        rate_formats["Nil"] += 1
    elif rate.upper() == "NIL":
        rate_formats["NIL (all-caps)"] += 1
    elif "per unit" in rate.lower():
        rate_formats["Rate per unit (e.g. 0.32R per unit)"] += 1
    elif "per thousand" in rate.lower():
        rate_formats["Specific / Ad valorem + specific (e.g. Rs. per thousand, % + Rs.)"] += 1
    else:
        rate_formats[f"Other non-standard: {rate}"] += 1

print("\n--- Rate Formats Found ---")
for fmt, count in rate_formats.items():
    print(f"  {fmt}: {count} rows")

# 9. Known cases
print("\n--- Known Cases Validation ---")
# Butter 0405
butter_cases = [r for r in rows if "0405" in r["hsn_code"] and "butter" in r["description"].lower()]
for b in butter_cases:
    print(f"Butter: HSN={b['hsn_code']}, CGST={b['cgst_rate']}, SGST={b['sgst_rate']}, Total={b['total_gst_rate']}, S_rate={b['source_rate']}, Notif={b['notification_no']}")

# Motorcycles 8711
moto_cases = [r for r in rows if "8711" in r["hsn_code"]]
print(f"\nFound {len(moto_cases)} rows with HSN 8711:")
for m in moto_cases:
    print(f"  Cat: {m['rate_category']:18} | Rate: {m['source_rate']:10} | CGST: {m['cgst_rate']:5} | SGST: {m['sgst_rate']:5} | Total: {m['total_gst_rate']:5} | Cess: {m['compensation_cess_rate']:5} | S.No: {m['serial_no']:5}")
    print(f"    Desc: {m['description'][:90]}")
