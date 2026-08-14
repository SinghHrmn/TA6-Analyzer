"""
Demo: fill the REAL 6th-edition template with a controlled ground truth,
read the fields back into structured data, verify they match, and flag the fault.

FIXED 11 Aug 2026: this script previously wrote using bare field names
("3 Yes", "4 Text", ...), which fill_template() silently could not match
against the template's FULLY QUALIFIED field names -- nothing was ever
actually written, so every round-trip check failed (0/3) and no fault was
ever flagged. Ground-truth questions below are now keyed by their qualified
prefix ("2.3", "2.4", "2.5" -- Section 2, disputes/complaints-style Yes/No/
Text triples), matching what ta6.field_ids actually assigns on this template.

Usage: python scripts/fill_and_read.py "<EDITABLE TA6 6th edition.pdf>" [out.pdf]
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ta6.acroform import fill_template, extract_record, check_missing_details

# ---- ground truth we intend to write (labels are OURS, so evaluation is valid) ----
GROUND_TRUTH = {
    "header": {"Property Address": "170 Elm Close, Bramfield", "Postcode": "BR8 5XD",
               "Seller 1": "Daniel Bennett"},
    "answers": {
        "2.3": {"answer": "Yes", "details": ""},      # <-- injected fault: Yes but blank details
        "2.4": {"answer": "No",  "details": ""},
        "2.5": {"answer": "No",  "details": ""},
    },
}

def gt_to_fields(gt):
    v = dict(gt["header"])
    for q, a in gt["answers"].items():
        v[f"{q} Yes"] = "/Yes" if a["answer"] == "Yes" else "/Off"
        v[f"{q} No"]  = "/Yes" if a["answer"] == "No" else "/Off"
        if a["details"]:
            v[f"{q} Text"] = a["details"]
    return v


def main():
    tpl = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "filled_TA6_sample.pdf"

    fill_template(tpl, gt_to_fields(GROUND_TRUTH), out)
    print(f"Filled the real template -> {out}\n")

    rec = extract_record(out)                       # read the fields back
    print("Header read back:", rec["header"])
    print("\nPer-question answers read back from the filled form:")
    gtA = GROUND_TRUTH["answers"]
    ok = 0
    for a in rec["answers"]:
        q = str(a["q"])
        if q in gtA:
            match = (a["answer"] == gtA[q]["answer"] and a["details"] == gtA[q]["details"])
            ok += match
            print(f"  Q{q}: answer={a['answer']:<5} details={a['details']!r:<20} "
                  f"[ground truth {gtA[q]['answer']}] {'MATCH' if match else 'MISMATCH'}")

    print(f"\nRound-trip: {ok}/{len(gtA)} answers read back exactly as written.")
    print("\nDetection (generalised missing-details rule):")
    for fl in check_missing_details(rec):
        print(f"  [FLAG] Q{fl['q']} missing_detail — {fl['question']!r}")
        print(f"         ENQUIRY: {fl['enquiry']}")

    json.dump({"ground_truth": GROUND_TRUTH, "extracted": rec},
              open(os.path.splitext(out)[0] + "_ground_truth.json", "w"), indent=2)


if __name__ == "__main__":
    main()
