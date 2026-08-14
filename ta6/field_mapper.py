"""
Field -> question mapper for the editable TA6 template.
=======================================================
The 6th-edition editable form names its answer boxes two ways: some carry a
real question number ('5.1 a y', '3.1 Text') but ONLY as the field's own
partial /T -- the fully-qualified (ancestor-joined) name is what's actually
unique; others are pure Acrobat auto-names ('Check Box103') with no semantic
meaning at all. To fill or read them meaningfully we (a) always key by the
FULLY QUALIFIED field name (see ta6.field_ids -- fixes a real collision bug:
the bare partial name "2" alone is reused by 7 unrelated questions across 6
different pages) and (b) recover, for every fillable field, the printed
question text next to it using field COORDINATES + the page's words.
Output: field_map.json  [{field_id, type, page, rect, label, section}].

This map is both (a) what lets us fill the real template meaningfully and
(b) the canonical schema with real question text — a methodology artifact.
"""
import sys, json
import pdfplumber
from pypdf import PdfReader
from ta6.field_ids import list_fields

FT_LABEL = {"/Tx": "text", "/Btn": "checkbox", "/Ch": "choice"}


def field_boxes(pdf_path):
    """field_id (FULLY QUALIFIED) -> (page_index, rect[x0,y0,x1,y1], type)."""
    r = PdfReader(pdf_path)
    out = []
    for f in list_fields(r):
        if f["page"] is None or f["rect"] is None:
            continue
        out.append((f["qname"], f["page"], f["rect"], FT_LABEL.get(f["ft"], "?")))
    return out


def nearest_label(field_rect, words, page_h):
    """Guess the question label for a field from nearby words.
    Prefer text on the same line to the LEFT; else the nearest line ABOVE."""
    x0, y0, x1, y1 = field_rect
    # convert field to top-left origin (pdfplumber)
    f_top, f_bot = page_h - y1, page_h - y0
    f_cy = (f_top + f_bot) / 2

    same_line = [w for w in words
                 if w["x1"] <= x0 + 2 and abs((w["top"] + w["bottom"]) / 2 - f_cy) < 7]
    if same_line:
        same_line.sort(key=lambda w: x0 - w["x1"])
        line_top = same_line[0]["top"]
        row = sorted([w for w in words if abs(w["top"] - line_top) < 4 and w["x1"] <= x1],
                     key=lambda w: w["x0"])
        return " ".join(w["text"] for w in row)[-90:]

    above = [w for w in words if w["bottom"] <= f_top + 1]
    if above:
        line_top = max(above, key=lambda w: w["bottom"])["top"]
        row = sorted([w for w in words if abs(w["top"] - line_top) < 4], key=lambda w: w["x0"])
        return " ".join(w["text"] for w in row)[-90:]
    return ""


def build(pdf_path):
    boxes = field_boxes(pdf_path)
    fields_by_page = {}
    for f in boxes:
        fields_by_page.setdefault(f[1], []).append(f)

    mapping = []
    with pdfplumber.open(pdf_path) as pdf:
        for pi, page in enumerate(pdf.pages):
            words = page.extract_words(use_text_flow=False)
            # section heading = the '<n>. Title' printed highest on the page
            import re
            heads = [w for w in words if re.match(r"^\d{1,2}\.$", w["text"])]
            section = ""
            for h in sorted(heads, key=lambda w: w["top"]):
                row = sorted([w for w in words if abs(w["top"] - h["top"]) < 4], key=lambda w: w["x0"])
                section = " ".join(w["text"] for w in row)[:40]
                break
            for name, _, rect, ftype in fields_by_page.get(pi, []):
                mapping.append({"field_id": name, "type": ftype, "page": pi + 1, "rect": rect,
                                "section": section, "label": nearest_label(rect, words, page.height)})
    return mapping


if __name__ == "__main__":
    import os
    from pathlib import Path
    _default = os.environ.get("TA6_TEMPLATE_PATH",
        str(Path(__file__).resolve().parents[2] / "TA 6 documents" / "EDITABLE TA6 - 6th Edition 0426.pdf"))
    pdf = sys.argv[1] if len(sys.argv) > 1 else _default
    m = build(pdf)
    json.dump(m, open("field_map.json", "w"), indent=2)
    named = [x for x in m if x["label"].strip()]
    print(f"Mapped {len(named)}/{len(m)} fields to a printed label.\n")
    print("Sample — fields that now carry meaning:")
    for x in m:
        lab = x["label"].strip()
        if lab and (x["type"] == "checkbox" or len(lab) > 15):
            print(f"  p{x['page']:<2} {x['type']:<9} {x['field_id']:<14} -> {lab!r}")
