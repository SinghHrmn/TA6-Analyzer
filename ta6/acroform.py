"""
AcroForm fill + read for the real editable TA6 template.
========================================================
- fill_template(): write a code-controlled ground-truth record into the REAL
  6th-edition template (checkboxes + text boxes) and save a filled PDF.
- read_acroform() / extract_record(): read a filled form's fields back into a
  structured record, grouping the 'Q# Yes/No/NK/Text' fields into per-question
  answers and attaching the printed question text from field_map.json.

This makes the 'digital fillable' route real (not a stub): fill -> read back ->
structured data, verified to match the ground truth by construction.
"""
import os, re, json
from pypdf import PdfReader, PdfWriter
from pypdf.generic import NameObject, TextStringObject, BooleanObject
from ta6.field_ids import list_fields

_MAP_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "field_map.json")


def _labels():
    try:
        return {x["field_id"]: x.get("label", "") for x in json.load(open(_MAP_PATH))}
    except Exception:
        return {}


def fill_template(template_path: str, values: dict, out_path: str):
    """values: {qualified_field_name: value}. For checkboxes use '/Yes' (tick) or
    '/Off'; for text fields use a string.

    IMPORTANT: keys must be the FULLY QUALIFIED field name (ta6.field_ids /
    data/field_map.json 'field_id'), not the bare partial /T. The template
    reuses bare partial names across unrelated questions (e.g. "2" alone is
    shared by 7 different questions on 6 different pages) -- filling by bare
    name silently cross-contaminates unrelated answers.
    """
    reader = PdfReader(template_path)
    writer = PdfWriter()
    writer.append(reader)
    unmatched = set(values)
    for f in list_fields(writer):
        if f["qname"] not in values:
            continue
        unmatched.discard(f["qname"])
        o = f["ref"].get_object()
        val = values[f["qname"]]
        if f["ft"] == "/Btn":
            st = NameObject(val if str(val).startswith("/") else "/" + str(val))
            o[NameObject("/V")] = st
            o[NameObject("/AS")] = st
        else:
            o[NameObject("/V")] = TextStringObject(str(val))
    # ask viewers to render appearances for the values we set
    try:
        acro = writer._root_object["/AcroForm"]
        acro[NameObject("/NeedAppearances")] = BooleanObject(True)
    except Exception:
        pass
    with open(out_path, "wb") as f:
        writer.write(f)
    return out_path, unmatched


def read_acroform(pdf_path: str) -> dict:
    """qualified_field_name -> value ('/Yes'|'/Off' for checkboxes, string for text)."""
    reader = PdfReader(pdf_path)
    vals = {}
    for f in list_fields(reader):
        o = f["ref"].get_object()
        v = o.get("/V")
        vals[f["qname"]] = str(v) if v is not None else None
    return vals


def extract_record(pdf_path: str) -> dict:
    """Read a filled TA6 into a structured record: header + per-question answers."""
    vals = read_acroform(pdf_path)
    lab = _labels()

    header = {k: vals.get(k) for k in ("Property Address", "Postcode", "Seller 1") if vals.get(k)}

    questions = {}
    for nm, v in vals.items():
        # `nm` is the FULLY QUALIFIED field name (e.g. "2.3 Yes", "5.4 Text"), not
        # the bare leaf. Matching on the bare leaf alone would re-introduce the
        # exact collision this module exists to avoid: "4 Text" and "5 Text" are
        # each reused by TWO unrelated questions on this template ("2.4"/"5.4" and
        # "2.5"/"5.5" respectively). Capturing everything up to the trailing
        # option token keeps the qualifying prefix, so "2.4 Text" and "5.4 Text"
        # are correctly treated as different questions.
        m = re.match(r"^(.+) (Yes|No|NK|Text|attached)$", nm) or \
            re.match(r"^(.+ [a-z]) (y|n)$", nm)
        if not m:
            continue
        q = m.group(1)
        opt = {"y": "Yes", "n": "No"}.get(m.group(2), m.group(2))
        questions.setdefault(q, {})[opt] = v

    def qsort(q):
        # q is a qualified prefix like "2.3" or "5.4 a" -- sort numerically on
        # the leading dotted number where possible, falling back to plain
        # string order for anything that doesn't parse (keeps this robust
        # rather than crashing on an unexpected field-naming shape).
        head = q.split()[0]
        try:
            return (0, tuple(int(p) for p in head.split(".")), q)
        except ValueError:
            return (1, (), q)

    records = []
    for q in sorted(questions, key=qsort):
        d = questions[q]
        answer = ("Yes" if d.get("Yes") == "/Yes" else
                  "No" if d.get("No") == "/Yes" else
                  "Not known" if d.get("NK") == "/Yes" else "blank")
        details = (d.get("Text") or "").strip()
        label = (lab.get(f"{q} Yes") or lab.get(f"{q} y") or lab.get(f"{q} Text")
                 or lab.get(f"{q} No") or lab.get(f"{q} n") or "")
        records.append({"q": q, "question": label[:70], "answer": answer, "details": details})
    return {"route": "acroform_digital", "header": header, "answers": records}


def check_missing_details(record: dict):
    """Generalised rule over ANY question: answered 'Yes' but details left blank."""
    flags = []
    for a in record["answers"]:
        if a["answer"] == "Yes" and not a["details"]:
            flags.append({"q": a["q"], "question": a["question"],
                          "issue": "missing_detail",
                          "enquiry": (f"Question {a['q']} (“{a['question']}”) was answered "
                                      "'Yes' but no details were provided. Please supply full details.")})
    return flags
