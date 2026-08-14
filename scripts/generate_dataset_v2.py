"""
Generate a realistic, near-fully-filled labelled TA6 dataset (v2).
====================================================================
Fixes the two root causes behind v1's mostly-empty output (see
ta6/field_ids.py and ta6/groups.py docstrings):

  1. Field identification bug: v1 (and the underlying acroform helpers)
     keyed fields by their bare partial /T name, which the real template
     reuses across unrelated questions (e.g. "2" alone names 7 different
     questions on 6 different pages) -- filling by that name silently
     cross-wrote unrelated fields. Fixed by ta6.field_ids (fully qualified
     names) and the rewritten ta6.acroform.

  2. Coverage bug: v1 only recognised ~13 of ~147 real questions (the ones
     whose leaf name happened to spell "N Yes"/"N No"). The other ~90% of
     the form -- almost all of it Acrobat auto-named checkboxes with no
     semantic name at all -- was never touched. Fixed by ta6.groups, which
     recovers question structure from geometry (row clustering + caption
     position) instead of relying on field names.

REALISM
-------
Real completed TA6 forms are ~90-99% filled with mostly low-risk / expected
answers and only a low, genuine rate of missing or blank detail -- not the
~30% blank / ~12%-fault noise v1 produced. ta6.content supplies per-question
Yes-probabilities tuned to that skew and a low background fault rate
(ta6.content.FAULT_PROB) so "faults" are rare and labelled, matching how
real solicitor review actually finds occasional problems, not systematic
gaps.

Usage:  python scripts/generate_dataset_v2.py --n 100 --seed 1 --out dataset_v2
"""
import os, sys, json, random, argparse, re
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ta6 import groups, content
from ta6.acroform import fill_template, read_acroform

# Relative to this file, with an env-var override -- was a hardcoded
# session-specific /sessions/... path, fixed 11 Aug 2026.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE = os.environ.get(
    "TA6_TEMPLATE_PATH",
    str(REPO_ROOT / "TA 6 documents" / "EDITABLE TA6 - 6th Edition 0426.pdf"))


def pick_radio(q, rng):
    opts = q.options
    keys = set(opts)
    if keys == {"Yes", "No"}:
        p_yes = content.yes_probability(q.prompt)
        return "Yes" if rng.random() < p_yes else "No"
    if {"Seller", "Neighbour", "Shared"} <= keys:
        weights = {"Seller": 0.35, "Neighbour": 0.30, "Shared": 0.25, "Not known": 0.10}
        choices = list(opts); w = [weights.get(c, 0.1) for c in choices]
        return rng.choices(choices, w)[0]
    if keys <= {"Attached", "To follow", "Not applicable", "None", "Not Applicable"}:
        weights = {"Attached": 0.65, "To follow": 0.25}
        choices = list(opts); w = [weights.get(c, 0.10) for c in choices]
        return rng.choices(choices, w)[0]
    return rng.choice(list(opts))


def fill_question(q, values, rng, gt_answers, faults, seller_name, label_by_id):
    if q.kind == "radio":
        choice = pick_radio(q, rng)
        for opt, qname in q.options.items():
            values[qname] = "/Yes" if opt == choice else "/Off"
        gt_answers[q.qid] = {"question": q.prompt, "kind": q.kind, "answer": choice}
        _maybe_detail(q, choice, values, rng, gt_answers, faults, seller_name, label_by_id)
        return

    if q.kind in ("yesno", "yesno_nk"):
        p_yes = content.yes_probability(q.prompt)
        r = rng.random()
        third_label = next((o for o in q.options if o not in ("Yes", "No")), None)
        if q.kind == "yesno_nk" and third_label and r < content.NK_PROB:
            # use the group's OWN third-option label (usually "Not known", but
            # e.g. 13.5(a) uses "No mortgage" instead) -- never hardcode the
            # string, or the chosen answer won't match any key in q.options
            # and the checkbox row silently ends up with nothing ticked.
            answer = third_label
        elif r < p_yes:
            answer = "Yes"
        else:
            answer = "No"
        # rare fault: leave the whole question unanswered
        if rng.random() < content.FAULT_PROB_UNANSWERED:
            for qname in q.options.values():
                values[qname] = "/Off"
            gt_answers[q.qid] = {"question": q.prompt, "kind": q.kind, "answer": "blank"}
            faults.append({"qid": q.qid, "question": q.prompt, "type": "unanswered"})
            return
        for opt, qname in q.options.items():
            values[qname] = "/Yes" if opt == answer else "/Off"
        gt_answers[q.qid] = {"question": q.prompt, "kind": q.kind, "answer": answer}
        _maybe_detail(q, answer, values, rng, gt_answers, faults, seller_name, label_by_id)
        return

    if q.kind == "multiselect":
        # independent boolean; low background tick rate unless it's an
        # obviously-common item (heating fuel / broadband-type words).
        p = content.yes_probability(q.prompt) if len(q.prompt) > 3 else 0.1
        on = rng.random() < max(p, 0.08)
        for qname in q.options.values():
            values[qname] = "/Yes" if on else "/Off"
        gt_answers[q.qid] = {"question": q.prompt, "kind": q.kind, "answer": "On" if on else "Off"}
        return

    # attach_standalone (orphan Attached/To-follow with no owning question found):
    # leave untouched -- safer than guessing an unrelated answer.


def _maybe_detail(q, answer, values, rng, gt_answers, faults, seller_name, label_by_id):
    if q.detail_field:
        detail_label = label_by_id.get(q.detail_field, "")
        # Most "detail" boxes are triggered by a Yes answer ("if yes, give
        # details"), but a few real TA6 questions ask the opposite (e.g.
        # 7.1 "If NO, who insures the property?") -- read the box's own
        # printed caption to get the trigger right instead of assuming Yes.
        trigger = "No" if re.match(r"^\s*if no\b", detail_label, re.I) else "Yes"
        if answer == trigger:
            if rng.random() < content.FAULT_PROB_MISSING_DETAIL:
                values[q.detail_field] = ""
                faults.append({"qid": q.qid, "question": q.prompt, "type": "missing_detail"})
            else:
                structured = content.structured_value(detail_label, rng, context=q.prompt)
                values[q.detail_field] = structured if structured is not None else \
                    content.detail_text(q.prompt, rng, seller_name)
    if answer == "Yes" and q.attach_field:
        choice = pick_radio_attach(q.attach_field, rng)
        for opt, qname in q.attach_field.items():
            values[qname] = "/Yes" if opt == choice else "/Off"


def pick_radio_attach(attach_field, rng):
    weights = {"Attached": 0.65, "To follow": 0.25}
    choices = list(attach_field); w = [weights.get(c, 0.10) for c in choices]
    return rng.choices(choices, w)[0]


ORPHAN_TEXT_FILL_PROB = 0.6


def fill_orphan_text(qname, label, values, rng):
    v = content.structured_value(label, rng)
    if v is None:
        return False
    values[qname] = v
    return True


def build_form(plan, rng, form_id):
    values = {}
    gt_answers, faults = {}, []
    addr = content.fake_address(rng)
    seller = content.fake_person(rng)
    solicitor = content.fake_person(rng)
    firm = rng.choice(content.SOLICITOR_FIRMS)

    header = {
        "Property Address": f"{addr['line1']}, {addr['town']}",
        "Postcode": addr["postcode"],
        "Seller 1": seller,
        "Property Date": content.fake_date(rng, 2024, 2026),
        "UPRN": str(rng.randint(10_000_000, 99_999_999)),
        "Solicitor": solicitor,
        "Solicitor Email": f"{solicitor.split()[0].lower()}.{solicitor.split()[1].lower()}@{firm.split()[0].lower()}.co.uk",
        "Quill Reference": f"CNV-{rng.randint(1000,9999)}/{rng.choice(['JS','AM','RK','LT'])}",
        "Role Seller": "/Yes", "Seller Executor": "/Off", "Seller Attorney": "/Off", "Seller Trustee": "/Off",
    }
    values.update(header)

    for q in plan["questions"]:
        fill_question(q, values, rng, gt_answers, faults, seller, plan["label_by_id"])

    filled_orphans = 0
    for qname, label in plan["orphan_text_labels"].items():
        if qname in values:
            continue
        if rng.random() < ORPHAN_TEXT_FILL_PROB and fill_orphan_text(qname, label, values, rng):
            filled_orphans += 1

    n_blank_questions = sum(1 for a in gt_answers.values() if a["answer"] == "blank")
    completeness = 1 - (n_blank_questions / max(1, len(gt_answers)))
    gt = {"form_id": form_id, "header": header, "answers": gt_answers, "faults": faults,
          "n_filled_orphan_extras": filled_orphans, "n_orphan_extras_available": len(plan["orphan_text_labels"]),
          "completeness": round(completeness, 4)}
    return gt, values


def load_plan(template):
    res = groups.build(template)
    fm = json.load(open(os.path.join(os.path.dirname(__file__), "..", "data", "field_map.json")))
    label_by_id = {x["field_id"]: x["label"] for x in fm}
    header_fields = {"Property Address", "Postcode", "Seller 1", "Seller 2", "Seller 3", "Seller 4",
                     "Property Date", "UPRN", "Solicitor", "Solicitor Email", "Quill Reference",
                     "Role Seller", "Seller Executor", "Seller Attorney", "Seller Trustee",
                     "Seller Company Country", "Seller Company Number", "Seller Director"}

    # The seller-role checkboxes ("Role Seller"/"Seller Executor"/"Seller
    # Attorney"/"Seller Trustee") are handled explicitly in build_form()'s
    # header (exactly one role ticked). The geometry grouper doesn't know
    # that and can independently rope some of those same field IDs into an
    # unrelated multiselect/radio group -- which would then silently
    # overwrite the header's correct single-role choice. Drop any discovered
    # question that touches a header-owned field before that can happen.
    questions = [q for q in res["questions"]
                if not (set(q.options.values()) | set(q.attach_field.values())) & header_fields]

    used = set()
    for q in questions:
        used.update(q.options.values()); used.update(q.attach_field.values())
        if q.detail_field:
            used.add(q.detail_field)
    orphan_text_labels = {x["field_id"]: x["label"] for x in fm
                          if x["type"] == "text" and x["field_id"] not in used
                          and x["field_id"] not in header_fields}
    all_ids = {x["field_id"] for x in fm}
    orphans = sorted(all_ids - used - header_fields)
    return {"questions": questions, "orphans": orphans, "n_fields": res["n_fields"],
            "orphan_text_labels": orphan_text_labels, "label_by_id": label_by_id}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="dataset_v2")
    ap.add_argument("--template", default=DEFAULT_TEMPLATE)
    ap.add_argument("--no-pdf", action="store_true", help="skip PDF rendering (ground truth JSON only, faster)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    print("Discovering question structure from the template (one-off)...")
    plan = load_plan(a.template)
    print(f"  {plan['n_fields']} fields -> {len(plan['questions'])} questions "
         f"({len(plan['orphans'])} structural orphans, {len(plan['orphan_text_labels'])} fillable as extras)")

    rng = random.Random(a.seed)
    manifest, n_fault_forms, n_faults_total = [], 0, 0
    completeness_vals, orphan_fill_vals = [], []
    for i in range(a.n):
        fid = f"form_{i+1:04d}"
        gt, values = build_form(plan, rng, fid)
        json.dump(gt, open(os.path.join(a.out, fid + ".json"), "w"), indent=2)
        if not a.no_pdf:
            out_pdf = os.path.join(a.out, fid + ".pdf")
            _, unmatched = fill_template(a.template, values, out_pdf)
            if unmatched:
                print(f"  WARNING {fid}: {len(unmatched)} keys not found in template: {list(unmatched)[:5]}")
        completeness_vals.append(gt["completeness"])
        orphan_fill_vals.append(gt["n_filled_orphan_extras"] / max(1, gt["n_orphan_extras_available"]))
        n_fault_forms += bool(gt["faults"])
        n_faults_total += len(gt["faults"])
        manifest.append({"id": fid, "n_faults": len(gt["faults"]), "faults": gt["faults"],
                         "completeness": gt["completeness"]})

    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=2)
    n_q = len(plan["questions"])
    print(f"\nGenerated {a.n} forms -> {a.out}/")
    print(f"  questions answered (non-blank)      : {sum(completeness_vals)/len(completeness_vals):.1%} avg, "
         f"of {n_q} discovered questions")
    print(f"  optional 'extra' text fields filled : {sum(orphan_fill_vals)/len(orphan_fill_vals):.1%} avg "
         f"(of {len(plan['orphan_text_labels'])} available)")
    print(f"  forms with >=1 labelled fault        : {n_fault_forms}/{a.n} ({n_fault_forms/a.n:.0%}); "
         f"{n_faults_total/a.n:.2f} faults/form avg")
    print(f"  structural coverage                 : {n_q} questions + {len(header_keys())} header fields "
         f"reach {plan['n_fields']-len(plan['orphans'])}/{plan['n_fields']} fields "
         f"({(plan['n_fields']-len(plan['orphans']))/plan['n_fields']:.0%}); "
         f"{len(plan['orphans'])} fields have no safe structural match and are left untouched.")
    print(f"  ground truth + manifest             : {a.out}/*.json, {a.out}/manifest.json")


def header_keys():
    return ["Property Address", "Postcode", "Seller 1", "Property Date", "UPRN",
           "Solicitor", "Solicitor Email", "Quill Reference", "Role Seller"]


if __name__ == "__main__":
    main()
