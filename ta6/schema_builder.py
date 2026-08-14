"""
Canonical TA6 schema builder  (Phase 1)
=======================================
Reads the editable 6th-edition TA6 (an AcroForm) and derives a single canonical
schema — the one source of truth the extractor AND the rules engine both read.

Output:
  ta6_schema.json  — list of fields: {field_id, type, page, section}
  prints a human-readable summary (sections, field types, counts).

Named text fields (Property Address, Seller 1, UPRN, ...) carry meaning directly.
Checkboxes are grouped by the section printed on their page. v2 will bind each
checkbox to its exact question via coordinates (the layout-aware step).
"""
import json, re, sys
from pathlib import Path
from pypdf import PdfReader


def page_sections(reader):
    """Detect the TA6 section header(s) printed on each page, e.g. '2. Disputes'."""
    out = {}
    for i, pg in enumerate(reader.pages):
        txt = pg.extract_text() or ""
        heads = re.findall(r"(?m)^\s*(\d{1,2})\.\s+([A-Z][A-Za-z ,/&-]{3,50})", txt)
        # keep plausible TA6 section titles
        titles = [f"{n}. {t.strip()}" for n, t in heads if int(n) <= 20 and len(t.strip()) > 3]
        out[i] = titles
    return out


def build(pdf_path: str, edition: str = "6th"):
    reader = PdfReader(pdf_path)
    sec_by_page = page_sections(reader)

    # map each widget to its page + rectangle by walking page annotations
    field_page, field_rect = {}, {}
    for pi, pg in enumerate(reader.pages):
        annots = pg.get("/Annots") or []
        for a in annots:
            try:
                obj = a.get_object()
            except Exception:
                continue
            name = obj.get("/T")
            parent = obj.get("/Parent")
            if name is None and parent is not None:
                name = parent.get_object().get("/T")
            if name is None:
                continue
            name = str(name)
            if name not in field_page:
                field_page[name] = pi
                r = obj.get("/Rect")
                if r:
                    field_rect[name] = [float(x) for x in r]

    fields = reader.get_fields() or {}
    FT = {"/Tx": "text", "/Btn": "checkbox", "/Ch": "choice", "/Sig": "signature"}

    schema = []
    for name, f in fields.items():
        pg = field_page.get(name)
        titles = sec_by_page.get(pg, []) if pg is not None else []
        # nearest section: the last header at/above this field on its page
        section = titles[-1] if titles else (f"page {pg+1}" if pg is not None else "unknown")
        schema.append({
            "field_id": name,
            "type": FT.get(str(f.get("/FT")), str(f.get("/FT"))),
            "page": (pg + 1) if pg is not None else None,
            "section": section,
            "edition": edition,
        })
    schema.sort(key=lambda d: (d["page"] or 99, d["field_id"]))
    return schema


def summarise(schema):
    from collections import Counter
    types = Counter(s["type"] for s in schema)
    print(f"Canonical schema — {len(schema)} fields")
    print(f"  by type: {dict(types)}")
    print(f"  named text fields (meaningful): {sum(1 for s in schema if s['type']=='text')}")
    print("\n  Sections detected (page → title):")
    seen = {}
    for s in schema:
        seen.setdefault(s["page"], s["section"])
    for pg in sorted(k for k in seen if k):
        print(f"    p{pg:<2} {seen[pg]}")
    print("\n  Example named fields:")
    for s in [s for s in schema if s["type"] == "text"][:10]:
        print(f"    - {s['field_id']:<28} (p{s['page']}, {s['section']})")


if __name__ == "__main__":
    import os
    _default = os.environ.get("TA6_TEMPLATE_PATH",
        str(Path(__file__).resolve().parents[2] / "TA 6 documents" / "EDITABLE TA6 - 6th Edition 0426.pdf"))
    pdf = sys.argv[1] if len(sys.argv) > 1 else _default
    schema = build(pdf, "6th")
    Path("ta6_schema.json").write_text(json.dumps(schema, indent=2))
    summarise(schema)
    print(f"\nWritten: ta6_schema.json ({len(schema)} fields)")
