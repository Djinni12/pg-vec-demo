from src.retrievers.rate_retriever import load_rates_csv
from collections import Counter
import json

rows = load_rates_csv(force_reload=True)

print("Total rows:", len(rows))
sec_headings = Counter(r["section_heading"] for r in rows)
print("\n1. All section headings discovered:")
for k, v in sec_headings.items():
    print(f"   - {k} ({v} rows)")

sec_cat_map = {}
for r in rows:
    sec = r["section_heading"]
    cat = r["rate_category"]
    sec_cat_map.setdefault(sec, set()).add(cat)

print("\n2. Assigned rate_category for each section:")
for sec, scats in sec_cat_map.items():
    print(f"   - {sec} -> {sorted(list(scats))}")

cats = Counter(r["rate_category"] for r in rows)
print("\n3. Number of rows under each category:")
for k in ["CGST", "EXEMPTION", "COMPENSATION_CESS", "SPECIAL", "UNKNOWN"]:
    print(f"   - {k}: {cats.get(k, 0)}")

print(f"\n4. Number of CGST rows: {cats.get('CGST', 0)}")
print(f"5. Number of EXEMPTION rows: {cats.get('EXEMPTION', 0)}")
print(f"6. Number of COMPENSATION_CESS rows: {cats.get('COMPENSATION_CESS', 0)}")
print(f"7. Number of SPECIAL rows: {cats.get('SPECIAL', 0)}")
print(f"8. Number of UNKNOWN rows: {cats.get('UNKNOWN', 0)}")

unparseable = [r for r in rows if r["source_rate"] in ("-", "", "N/A")]
print(f"\n9. Rows whose source rate could not be parsed: {len(unparseable)}")
for u in unparseable:
    print(f"   - Row {u['_id']}: Notif={u['notification_no']}, Sched={u['schedule']}, S.No={u['serial_no']}, Source Rate='{u['source_rate']}'")

butter = next(r for r in rows if r["hsn_code"] == "0405")
print("\n10. Corrected butter record:")
print(json.dumps({
    "hsn_code": butter["hsn_code"],
    "description": butter["description"],
    "section_heading": butter["section_heading"],
    "rate_category": butter["rate_category"],
    "source_rate": butter["source_rate"],
    "cgst_rate": butter["cgst_rate"],
    "sgst_rate": butter["sgst_rate"],
    "total_gst_rate": butter["total_gst_rate"],
    "notification_no": butter["notification_no"],
    "notification_date": butter["notification_date"],
    "rate_as_on_date": butter["rate_as_on_date"],
    "source_page": butter["source_page"],
}, indent=2))

print("\n11. Five example normal CGST rows:")
examples_cgst = [
    rows[0],    # Live horses 2.5%
    rows[4],    # Yoghurt 2.5%
    rows[538],  # Computers 8471 9%
    rows[1178], # Motor vehicles 8703 14%
    rows[1197], # Gold 7108 1.5%
]
for ex in examples_cgst:
    print(f"   - HSN {ex['hsn_code']} ({ex['description'][:35]}...): source_rate={ex['source_rate']}, CGST={ex['cgst_rate']}, SGST={ex['sgst_rate']}, Total GST={ex['total_gst_rate']}")

print("\n12. Five special/exemption/cess examples showing NO incorrect doubling:")
examples_special = [
    rows[1218], # Exemption S.No 1 Live asses
    rows[1222], # Exemption S.No 5 Fresh milk
    rows[1562], # Special S.No 1 Fly ash bricks Notif 02/2022
    rows[1567], # Special S.No 1 Fly ash bricks Notif 14/2025
    rows[1572], # Cess S.No 1 Pan Masala
]
for ex in examples_special:
    cg = ex['cgst_rate'] or 'null'
    sg = ex['sgst_rate'] or 'null'
    tot = ex['total_gst_rate'] or 'null'
    cess = ex['compensation_cess_rate'] or 'null'
    print(f"   - [{ex['rate_category']}] HSN {ex['hsn_code']} ({ex['description'][:30]}...): source_rate={ex['source_rate']}, CGST={cg}, SGST={sg}, Total GST={tot}, Cess={cess}, is_exempt={ex['is_exempt']}")
