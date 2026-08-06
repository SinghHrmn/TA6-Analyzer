"""
Question-group discovery for the real TA6 6th-edition editable template.
==========================================================================
WHY THIS EXISTS
----------------
Of the 442 fillable widgets, only ~117 carry a field name that hints at the
question they belong to (e.g. "5.1 a y" / "3.1 Text"); the rest (~325
checkboxes) are pure Acrobat auto-names ("Check Box103") with zero semantic
content in the name itself. `scripts/generate_dataset.py` (v1) only filled
the ~13 questions it could safely pair by NAME REGEX -- which is why
generated forms were mostly blank: it never touched the other ~90% of the
form.

This module recovers the missing structure from GEOMETRY instead of names:
TA6's layout consistently prints each checkbox BEFORE its own caption word
("[ ] Yes  [ ] No", "[ ] Attached  [ ] To follow", "[ ] Ground water"), so
the caption immediately to a checkbox's right identifies its role, and
checkboxes at the same page height belong to the same question row.

Empirically verified pattern (see dev notes / eval_groups.py):
  - 2-box row, captions ["Yes", "No"]                 -> yes/no question
  - 3-box row, captions ["Yes", "No", "Not known"]     -> yes/no/not-known
  - 2-3 box row, captions ["Attached", "To follow", ...]-> evidence sub-flags
    for the preceding main question (attached / to follow / not applicable)
  - row of boxes with distinct non-generic captions    -> independent
    "tick all that apply" items (heating fuel, flood cause, connected
    service, etc.) -- each is its own boolean, not mutually exclusive
  - name-suffix groups already legible from the field name itself
    (e.g. "1 A Seller"/"1 A Neighbour"/"1 A Shared.../1 A NK..." for the
    2.1(a)-(d) boundary-responsibility grid, "5.1 a y"/"5.1 a n" for the
    5.1(a)-(i) alterations checklist) are grouped by that shared prefix
    instead -- geometry agrees but the name is more precise.

Each discovered group is a `Question`:
    kind: "yesno" | "yesno_nk" | "multiselect" | "radio" (name-suffix groups)
    prompt: best-effort question text (from field_map label)
    options: {option_label: field_id}   (field_id = fully qualified AcroForm name)
    detail_field: field_id of the nearest paired free-text/date box, or None
    attach_field: sub-group of Attached/To-follow/Not-applicable flags tied
                  to this question (dict option_label -> field_id), or {}
    page, section

This is a best-effort structural recovery, not a perfect parse: a handful of
rows with unusual layouts fall through to `orphan_checkboxes` /
`orphan_text` and are left unfilled by the generator rather than guessed at
-- which is the right trade-off (a form that's 95% correctly filled beats
one that's 100% filled with a chunk of misattributed answers).
"""
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Optional

import pdfplumber
from pypdf import PdfReader

from ta6.field_ids import list_fields
from ta6.field_mapper import nearest_label

ROW_TOL = 6.0          # px tolerance to cluster checkboxes into one row
DETAIL_MAX_GAP = 90.0  # px max vertical gap to pair a row with its detail box
NK_WORDS = {"not known", "not", "known"}


@dataclass
class Question:
    qid: str
    kind: str
    prompt: str
    page: int
    section: str
    options: Dict[str, str] = dc_field(default_factory=dict)
    detail_field: Optional[str] = None
    attach_field: Dict[str, str] = dc_field(default_factory=dict)


_NAME_SUFFIXES = [
    (r"^(?P<prefix>.+?) (?P<opt>Seller|Neighbour|Shared|NK)(?:_es_:date)?$",
     {"Seller": "Seller", "Neighbour": "Neighbour", "Shared": "Shared", "NK": "Not known"}),
    (r"^(?P<prefix>.+?) (?P<opt>y|n)(?:_es_:date)?$", {"y": "Yes", "n": "No"}),
    (r"^(?P<prefix>.+?)(?P<opt>y|n)$", {"y": "Yes", "n": "No"}),  # "1 fn" (no space)
    (r"^(?P<prefix>.+?) (?P<opt>Yes|No|NK)$", {"Yes": "Yes", "No": "No", "NK": "Not known"}),
    (r"^(?P<prefix>.+?) (?P<opt>att|tf|na)(?:_es_:date)?$",
     {"att": "Attached", "tf": "To follow", "na": "Not applicable"}),
]


_CANONICAL_OPTS = {
    "not known": "Not known", "known": "Not known",
    "not applicable": "Not applicable", "n/a": "Not applicable",
    "to follow": "To follow", "attached": "Attached",
    "yes": "Yes", "no": "No", "none": "None",
}


def _normalize_opt(caption: str) -> str:
    """Canonical casing for option labels, so the SAME string ("Not known")
    is produced regardless of which code path (or caption-merge boundary)
    discovered it -- a casing mismatch here silently breaks answer lookups
    (Question.options key vs. the generator's chosen answer string)."""
    key = re.sub(r"\s+", " ", (caption or "").strip().lower())
    return _CANONICAL_OPTS.get(key, caption.strip().title())


def _right_caption(rect, words, page_h):
    x0, y0, x1, y1 = rect
    top, bot = page_h - y1, page_h - y0
    cy = (top + bot) / 2
    cands = [w for w in words if w["x0"] >= x1 - 1 and abs((w["top"] + w["bottom"]) / 2 - cy) < 5]
    if not cands:
        return ""
    cands.sort(key=lambda w: w["x0"])
    out, prev_x1 = [cands[0]["text"]], cands[0]["x1"]
    for w in cands[1:]:
        if w["x0"] - prev_x1 < 8 and len(out) < 2:
            out.append(w["text"]); prev_x1 = w["x1"]
        else:
            break
    return " ".join(out)


def _cluster_rows_by_page(items, top_of):
    """Group items into rows: same page, consecutive `top` values within ROW_TOL."""
    by_page = defaultdict(list)
    for it in items:
        by_page[it["page"]].append(it)
    rows = []
    for page, its in by_page.items():
        its = sorted(its, key=top_of)
        cur, last = [], None
        for it in its:
            v = top_of(it)
            if last is not None and abs(v - last) > ROW_TOL:
                rows.append(cur); cur = []
            cur.append(it); last = v
        if cur:
            rows.append(cur)
    return rows


def _name_group_key(qname):
    """If the field's own leaf name encodes a recognisable option suffix,
    return (prefix, option_label); else None."""
    leaf = qname.split(".")[-1]
    for pat, opt_map in _NAME_SUFFIXES:
        m = re.match(pat, leaf)
        if m:
            opt = opt_map.get(m.group("opt"))
            if opt:
                return m.group("prefix").strip(), opt
    return None


def build(template_path: str) -> Dict:
    reader = PdfReader(template_path)
    fields = list_fields(reader)
    by_page = defaultdict(list)
    for f in fields:
        if f["page"] is not None:
            by_page[f["page"]].append(f)

    text_fields = {f["qname"]: f for f in fields if f["ft"] == "/Tx"}
    checkbox_fields = [f for f in fields if f["ft"] == "/Btn"]

    # ---- Pass 1: NAME-based groups (leaf name already encodes the option) ----
    name_groups_raw = defaultdict(dict)   # (page, prefix) -> {option: qname}
    for f in checkbox_fields:
        g = _name_group_key(f["qname"])
        if g:
            prefix, opt = g
            name_groups_raw[(f["page"], prefix)][opt] = f["qname"]

    # A single lonely match (e.g. "Role Seller" matching the Seller/Neighbour/../NK
    # pattern in isolation, with no sibling "X Neighbour"/"X Shared" on the same
    # page) is a false positive, not a real option-group -- fall back to geometry.
    name_groups = {k: v for k, v in name_groups_raw.items() if len(v) >= 2}
    named_qnames = {qn for opts in name_groups.values() for qn in opts.values()}

    geometry_checkboxes = [f for f in checkbox_fields if f["qname"] not in named_qnames]

    questions: List[Question] = []

    # section lookup from field_map.json (page -> section), 1-indexed pages there
    fm = json.load(open(_field_map_path()))
    sec_by_page = {}
    for x in fm:
        sec_by_page.setdefault(x["page"] - 1, x["section"])
    label_by_qname = {x["field_id"]: x["label"] for x in fm}

    with pdfplumber.open(template_path) as pdf:
        words_by_page = {pi: pdf.pages[pi].extract_words(use_text_flow=False) for pi in by_page}
        H_by_page = {pi: pdf.pages[pi].height for pi in by_page}

        # ---- name-based groups -> Question objects ----
        for (page, prefix), opts in name_groups.items():
            first_qname = sorted(opts.values())[0]
            f0 = next(x for x in checkbox_fields if x["qname"] == first_qname)
            prompt = nearest_label(f0["rect"], words_by_page[page], H_by_page[page])
            kind = "radio"
            qid = f"{page}:{prefix}"
            q = Question(qid=qid, kind=kind, prompt=prompt or label_by_qname.get(first_qname, ""),
                        page=page, section=sec_by_page.get(page, ""), options=opts)
            q.detail_field = _nearest_detail_field(f0["rect"], page, text_fields, prefix)
            questions.append(q)

        # ---- geometry-based rows for the unnamed checkboxes ----
        clean_rows = _cluster_rows_by_page(geometry_checkboxes, top_of=lambda f: f_top(f, H_by_page))

        orphan_checkboxes = []
        seen_qids = set()
        for row in clean_rows:
            page = row[0]["page"]
            words, H = words_by_page[page], H_by_page[page]
            row_sorted = sorted(row, key=lambda f: f["rect"][0])
            caps = [_right_caption(f["rect"], words, H).strip() for f in row_sorted]
            caps_l = [c.lower() for c in caps]

            if caps_l[:2] == ["yes", "no"] and len(caps_l) <= 3 and "known" not in caps_l[2:3]:
                # any 2- or 3-box row starting Yes/No is a single-select question,
                # whatever the (already-merged) third caption reads (Not known /
                # Not applicable / ...). "known" alone means the real 3rd caption
                # was "Not known" split across a caption-merge boundary -- handled
                # by the branch below instead.
                kind = "yesno" if len(caps_l) == 2 else "yesno_nk"
                opt_labels = ["Yes", "No"] if len(caps_l) == 2 else ["Yes", "No", _normalize_opt(caps[2])]
            elif caps_l[:2] == ["yes", "no"] and caps_l[2:3] == ["known"]:
                kind, opt_labels = "yesno_nk", ["Yes", "No", "Not known"]
            elif set(caps_l) <= {"attached", "to follow", "not applicable", "none"} and caps_l:
                kind, opt_labels = "attach", [_normalize_opt(c) for c in caps]
            else:
                kind, opt_labels = "multiselect", None  # each box its own item

            if kind == "attach":
                # tie to the most recently added main question on this page, above this row
                target = _find_owner_question(questions, page, row_sorted[0]["rect"][1])
                if target is not None:
                    for f, lab in zip(row_sorted, opt_labels):
                        target.attach_field[lab] = f["qname"]
                    continue
                # no owner found -> treat as its own tiny question
            if kind in ("yesno", "yesno_nk"):
                prompt = nearest_label(row_sorted[0]["rect"], words, H)
                qid = f"{page}:{round(row_sorted[0]['rect'][1])}"
                if qid in seen_qids:
                    continue
                seen_qids.add(qid)
                q = Question(qid=qid, kind=kind, prompt=prompt, page=page,
                            section=sec_by_page.get(page, ""),
                            options=dict(zip(opt_labels, [f["qname"] for f in row_sorted])))
                q.detail_field = _nearest_detail_field(row_sorted[0]["rect"], page, text_fields, None)
                questions.append(q)
            elif kind == "multiselect":
                for f in row_sorted:
                    cap = _right_caption(f["rect"], words, H).strip()
                    label = cap or nearest_label(f["rect"], words, H)
                    if not label:
                        orphan_checkboxes.append(f["qname"]); continue
                    qid = f"{page}:{f['qname']}"
                    q = Question(qid=qid, kind="multiselect", prompt=label, page=page,
                                section=sec_by_page.get(page, ""), options={"On": f["qname"]})
                    questions.append(q)
            elif kind == "attach":
                for f, lab in zip(row_sorted, opt_labels):
                    qid = f"{page}:{f['qname']}"
                    q = Question(qid=qid, kind="attach_standalone", prompt=lab, page=page,
                                section=sec_by_page.get(page, ""), options={lab: f["qname"]})
                    questions.append(q)

    used_qnames = set()
    for q in questions:
        used_qnames.update(q.options.values())
        used_qnames.update(q.attach_field.values())
        if q.detail_field:
            used_qnames.add(q.detail_field)
    all_qnames = {f["qname"] for f in fields}
    orphans = sorted(all_qnames - used_qnames)

    return {"questions": questions, "orphans": orphans, "n_fields": len(fields)}


def f_top(f, H_by_page):
    x0, y0, x1, y1 = f["rect"]
    return H_by_page[f["page"]] - y1


def _nearest_detail_field(rect, page, text_fields, prefix_hint):
    """Nearest text/date field below this checkbox row on the same page."""
    x0, y0, x1, y1 = rect
    best, best_gap = None, DETAIL_MAX_GAP
    for qname, f in text_fields.items():
        if f["page"] != page:
            continue
        tx0, ty0, tx1, ty1 = f["rect"]
        if ty1 > y0:  # not below
            continue
        gap = y0 - ty1
        if 0 <= gap < best_gap:
            best, best_gap = qname, gap
    return best


def _find_owner_question(questions, page, row_y):
    """Most recent yesno/yesno_nk/radio question on this page whose row sits
    above (higher y-in-PDF-space, i.e. earlier / above on the printed page)."""
    cands = [q for q in questions if q.page == page and q.kind in ("yesno", "yesno_nk", "radio")]
    if not cands:
        return None
    return cands[-1]


_FIELD_MAP_PATH = None


def _field_map_path():
    import os
    return os.path.join(os.path.dirname(__file__), "..", "data", "field_map.json")
