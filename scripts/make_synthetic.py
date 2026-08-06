"""
Synthetic TA6 generator (digital) + self-verification.
=======================================================
Produces realistic, readable TA6 forms in the REAL Law Society wording (so the
pipeline can extract them), each with a JSON ground-truth record and a labelled
injected fault. Then it re-extracts every form and checks the pipeline reads back
the SAME details it wrote — closing the loop:  generate -> PDF -> extract -> match.

Run:  python scripts/make_synthetic.py --n 6 --out synthetic_samples
All names/addresses are fabricated -> not personal data.
"""
import os, sys, json, random, argparse
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ta6.pipeline import extract_ta6, run_rule_checks

STREETS = ["Elm Close", "Hazel Grove", "Priory Walk", "Marlow Rise", "Kestrel Way", "Beckett Mews"]
TOWNS = [("Northgate", "NG"), ("Bramfield", "BR"), ("Westbourne", "WB"), ("Cavendish", "CV")]
FIRST = ["James", "Aisha", "Robert", "Priya", "Daniel", "Sofia", "Michael", "Grace"]
LAST = ["Whitmore", "Okafor", "Bennett", "Sharma", "Duncan", "Rossi", "Hartley"]
PTYPE = ["Terraced house", "Semi-detached house", "Detached house", "End-terrace house"]
WORKS = ["Single-storey rear extension completed in 2016.",
         "Loft conversion carried out before we purchased the property.",
         "Garage converted into a habitable room in 2018.",
         "Two-storey side extension built in 2015."]
EXPLAN = ["Permitted development rights applied; no consent was required.",
          "Building Regulations completion certificate is enclosed.",
          "The works were exempt from Building Regulations."]
GUARANTEES = ["NHBC / new-home warranty", "Damp-proofing guarantee",
              "Double-glazing (FENSA) certificate", "Central heating / boiler guarantee"]


def persona(rng):
    town, code = rng.choice(TOWNS)
    works_present = rng.random() < 0.6
    support_present = rng.random() < 0.5
    dispute = rng.random() < 0.35
    dispute_details_blank = dispute and rng.random() < 0.5   # a rule fault when True
    p = {
        "address": f"{rng.randint(1,180)} {rng.choice(STREETS)}, {town}",
        "postcode": f"{code}{rng.randint(1,9)} {rng.randint(1,9)}{rng.choice('ABDEHJLNPRSTWXYZ')}{rng.choice('ABDEHJLNPRSTWXYZ')}",
        "seller": f"{rng.choice(FIRST)} {rng.choice(LAST)}",
        "property_type": rng.choice(PTYPE),
        "tenure": rng.choice(["Freehold", "Freehold", "Leasehold"]),
        "works": rng.choice(WORKS) if works_present else "",
        "explanation": rng.choice(EXPLAN) if support_present else "",
        "disputes_answer": "Yes" if dispute else "No",
        "disputes_details": "" if dispute_details_blank else ("Boundary fence dispute, resolved 2021." if dispute else ""),
        "guarantees": rng.sample(GUARANTEES, k=rng.randint(0, 2)),
        "flooding": "No",
    }
    faults = []
    if p["works"] and not p["explanation"]:
        faults.append("works_without_support")
    if p["disputes_answer"] == "Yes" and not p["disputes_details"]:
        faults.append("missing_detail")
    p["injected_faults"] = faults
    return p


def render(p, path):
    c = canvas.Canvas(path, pagesize=A4); W, H = A4
    y = [H - 54]
    def L(t, dy=15, x=52, size=10, bold=False):
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        c.drawString(x, y[0], t); y[0] -= dy
    L("Law Society Property Information Form (TA6)  —  SYNTHETIC, research use only", 20, size=12, bold=True)
    L("1. Property and seller details", 16, bold=True)
    L(f"Address: {p['address']}   Postcode: {p['postcode']}")
    L(f"Seller: {p['seller']}    Tenure: {p['tenure']}    Property type: {p['property_type']}", 20)
    L("2. Disputes and complaints", 16, bold=True)
    L("2.1 Have there been any disputes or complaints regarding this property or a property nearby?")
    L(f"    {p['disputes_answer']}   If Yes, please give details:")
    L(f"        {p['disputes_details']}", 20, x=70)
    L("4. Alterations, planning and building control", 16, bold=True)
    L("4.1 (a) Building works (e.g. extension, loft or garage conversion). If Yes, please give")
    L("details including dates of all work undertaken:")
    L(f"    {p['works']}", 16, x=70)
    L("(b) Change of use   (c) Replacement windows since 1 April 2002", 16)
    L("4.2 If Yes to any of the questions in 4.1 and if the work was undertaken during ownership:")
    L("(a) please supply copies of the planning permissions, Building Regulations approvals; OR")
    L("(b) if none were required, please explain why these were not required - e.g. permitted")
    L("development rights applied or the work was exempt from Building Regulations:")
    L(f"    {p['explanation']}", 16, x=70)
    L("Further information about permitted development can be found at:")
    L("https://www.planningportal.co.uk/info/200126/applications", 18)
    L("4.3 Are any of the works disclosed in 4.1 above unfinished? If Yes, please give details.", 20)
    L("6. Guarantees and warranties", 16, bold=True)
    L(f"    {', '.join(p['guarantees']) or 'None declared'}", 20)
    L("8. Environmental matters", 16, bold=True)
    L(f"8.1 Has any part of the property ever flooded?   {p['flooding']}")
    c.showPage(); c.save()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="synthetic_samples")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)

    print(f"Generating {a.n} synthetic TA6 forms -> {a.out}/\n")
    print(f"{'form':<8}{'works?':<8}{'4.2?':<7}{'ground-truth faults':<34}{'extract matches?'}")
    ok = 0
    for i in range(a.n):
        p = persona(rng)
        pid = f"form_{i+1:02d}"
        render(p, os.path.join(a.out, pid + ".pdf"))
        json.dump(p, open(os.path.join(a.out, pid + ".json"), "w"), indent=2)

        # self-verify: re-extract and check the pipeline reads back the same state
        rec = extract_ta6(os.path.join(a.out, pid + ".pdf"))
        works_read = rec["alterations_made"]["answer"] == "Yes"
        support_read = rec["building_regs_completion_certificate"]["attachment_provided"]
        match = (works_read == bool(p["works"])) and (support_read == bool(p["explanation"]))
        ok += match
        print(f"{pid:<8}{('Yes' if p['works'] else 'No'):<8}{('Yes' if p['explanation'] else 'No'):<7}"
              f"{(', '.join(p['injected_faults']) or 'none'):<34}{'YES' if match else 'NO'}")

    print(f"\nExtraction round-trip agreement: {ok}/{a.n} forms read back correctly.")
    print(f"Open a sample: {a.out}/form_01.pdf  (details) vs {a.out}/form_01.json (ground truth).")


if __name__ == "__main__":
    main()
