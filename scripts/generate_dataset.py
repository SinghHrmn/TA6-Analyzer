"""
Generate a DIVERSE labelled dataset by filling the REAL 6th-edition template.
=============================================================================
Discovers every reliably-nameable Yes/No question from the field map, and per
form assigns a realistic mix of states:
    No | Yes+details | Yes+blank-details (FAULT) | Not known | unanswered (FAULT)
Fills the real template, saves a filled PDF + per-form ground-truth JSON, and a
manifest. Labels are decided by code -> the set is labelled by construction.

Coverage note: the template names ~13 questions cleanly ('3 Yes', '1 a y', ...);
the other 275 checkboxes are generic ('Check Box46') and are out of scope for the
name-based filler (they need the coordinate map). Report this as a limitation.

Usage:  python scripts/generate_dataset.py --n 100 --seed 1 --out dataset
"""
import os, sys, re, json, random, argparse
from collections import Counter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ta6.acroform import fill_template

MAP = os.path.join(os.path.dirname(__file__), "..", "data", "field_map.json")

STREETS = ["Elm Close", "Hazel Grove", "Priory Walk", "Marlow Rise", "Kestrel Way", "Beckett Mews"]
TOWNS = [("Northgate", "NG"), ("Bramfield", "BR"), ("Westbourne", "WB"), ("Cavendish", "CV")]
FIRST = ["James", "Aisha", "Robert", "Priya", "Daniel", "Sofia", "Michael", "Grace", "Owen", "Leah"]
LAST = ["Whitmore", "Okafor", "Bennett", "Sharma", "Duncan", "Rossi", "Hartley", "Nguyen"]
DETAILS = ["Resolved amicably in 2021; see enclosed correspondence.",
           "Notice served by the local authority; copy attached.",
           "Minor works agreed with the neighbour; no dispute.",
           "Permission granted; certificate enclosed.",
           "Ongoing but not affecting the property's use."]

STATES = ["No", "Yes_ok", "NotKnown", "Yes_blank", "Unanswered"]
WEIGHTS = [0.62, 0.20, 0.06, 0.045, 0.075]      # Yes_blank & Unanswered are the faults
# ~12% of question-answers are faults -> plenty of positives for evaluation while
# keeping most answers clean. (Real-world fault prevalence is lower; noted as a limitation.)


def answerable_questions():
    """question_key -> {yes, no, nk, text, label} using the ACTUAL field names."""
    data = json.load(open(MAP))
    fields = [f["field_id"] for f in data]
    labels = {f["field_id"]: f.get("label", "") for f in data}
    groups = {}
    for n in fields:
        m = re.match(r"^(\d+) (Yes|No|NK)$", n)
        m2 = re.match(r"^(\d+ [a-z]) (y|n)$", n)
        if m:
            q, opt = m.group(1), m.group(2)
        elif m2:
            q, opt = m2.group(1), {"y": "Yes", "n": "No"}[m2.group(2)]
        else:
            continue
        groups.setdefault(q, {})[opt] = n
    out = {}
    for q, opts in groups.items():
        if "Yes" in opts and "No" in opts:
            tf = f"{q} Text" if f"{q} Text" in fields else None
            out[q] = {"yes": opts["Yes"], "no": opts["No"], "nk": opts.get("NK"),
                      "text": tf, "label": (labels.get(opts["Yes"]) or labels.get(opts["No"]) or "")[:60]}
    return out


def make_form(qs, rng):
    town, code = rng.choice(TOWNS)
    gt = {"header": {"Property Address": f"{rng.randint(1,180)} {rng.choice(STREETS)}, {town}",
                     "Postcode": f"{code}{rng.randint(1,9)} {rng.randint(1,9)}{rng.choice('ABDEHJLNPRSTWXYZ')}{rng.choice('ABDEHJLNPRSTWXYZ')}",
                     "Seller 1": f"{rng.choice(FIRST)} {rng.choice(LAST)}"},
          "answers": {}, "faults": []}
    fields = dict(gt["header"])
    for q, meta in qs.items():
        st = rng.choices(STATES, WEIGHTS)[0]
        if st == "NotKnown" and not meta["nk"]:
            st = "No"
        has_text = meta["text"] is not None
        answer, details, fault = "No", "", None
        if st == "No":
            answer = "No"
        elif st == "NotKnown":
            answer = "Not known"
        elif st == "Yes_ok":
            answer, details = "Yes", (rng.choice(DETAILS) if has_text else "")
        elif st == "Yes_blank":
            answer, fault = "Yes", ("missing_detail" if has_text else None)
        elif st == "Unanswered":
            answer, fault = "blank", "unanswered"

        gt["answers"][q] = {"answer": answer, "details": details, "question": meta["label"]}
        if fault:
            gt["faults"].append({"q": q, "type": fault, "question": meta["label"]})
        fields[meta["yes"]] = "/Yes" if answer == "Yes" else "/Off"
        fields[meta["no"]] = "/Yes" if answer == "No" else "/Off"
        if meta["nk"]:
            fields[meta["nk"]] = "/Yes" if answer == "Not known" else "/Off"
        if has_text and details:
            fields[meta["text"]] = details
    return gt, fields


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--template", default="/sessions/eloquent-sweet-rubin/mnt/uploads/EDITABLE TA6 - 6th Edition 0426.pdf")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)
    qs = answerable_questions()
    print(f"Reliably-nameable Yes/No questions used: {len(qs)}  -> {sorted(qs)}")

    states, ftypes, manifest, n_fault = Counter(), Counter(), [], 0
    for i in range(a.n):
        gt, fields = make_form(qs, rng)
        pid = f"form_{i+1:04d}"
        fill_template(a.template, fields, os.path.join(a.out, pid + ".pdf"))
        json.dump(gt, open(os.path.join(a.out, pid + ".json"), "w"), indent=2)
        for v in gt["answers"].values():
            key = v["answer"] if v["answer"] != "Yes" else ("Yes+details" if v["details"] else "Yes+blank")
            states[key] += 1
        for fl in gt["faults"]:
            ftypes[fl["type"]] += 1
        n_fault += bool(gt["faults"])
        manifest.append({"id": pid, "faults": gt["faults"]})
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=2)

    print(f"\nGenerated {a.n} filled real-template TA6 forms -> {a.out}/")
    print(f"  forms with >=1 fault : {n_fault}/{a.n}  ({n_fault/a.n:.0%})")
    print(f"  answer-state spread  : {dict(states)}")
    print(f"  fault types          : {dict(ftypes)}")
    print(f"  labels               : {a.out}/manifest.json (+ per-form .json)")


if __name__ == "__main__":
    main()
