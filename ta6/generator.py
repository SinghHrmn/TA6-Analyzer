"""
Synthetic TA6 Data Generator  ·  v0 (proof-of-concept)
=======================================================
Dissertation: AI-Assisted Document Analysis and Enquiry Generation for
UK Residential Conveyancing (H. Singh, MSc Applied AI, LSBU).

WHAT THIS DOES
--------------
Generates labelled synthetic seller-disclosure (TA6) records for training and
evaluating the pipeline WITHOUT needing real client data. For each record it produces:

  1. A structured TA6 record (the extraction ground truth).
  2. A matched supporting document (a planning extract) — the independent evidence.
  3. Zero or more injected inconsistencies, each with a ground-truth label
     (field, type, and whether a RULE or NLI/LLM method is required to catch it)
     -> this is the detection ground truth (precision/recall dataset).
  4. A model-answer solicitor enquiry for each injected issue -> generation gold standard.
  5. Rendered PDFs of the TA6 and the planning extract (to test OCR / layout extraction).

WHY IT IS DEFENSIBLE
--------------------
- Ground truth is known BY CONSTRUCTION (real forms would arrive unlabelled).
- Reproducible: fixed random seed.
- Realism is anchored in REAL public data sources at the marked TODOs
  (HM Land Registry Price Paid, the EPC register, planning.data.gov.uk).
- All names/addresses are fictitious -> no personal data, no GDPR exposure.

This is v0: a working skeleton to build on, not the final generator.
"""

import json
import random
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional

# ----------------------------------------------------------------------------
# Fictitious value pools.  In v1 these are replaced by real public data:
#   TODO(real-data): sample property (type, tenure, price, postcode) from
#                    HM Land Registry Price Paid Data.
#   TODO(real-data): pull energy features from the EPC Open Data register.
#   TODO(real-data): pull planning applications from planning.data.gov.uk.
# ----------------------------------------------------------------------------
STREETS = ["Elm Close", "Hazel Grove", "Priory Walk", "Marlow Rise", "Kestrel Way",
           "Weavers Lane", "Ashford Terrace", "Beckett Mews", "Sandpiper Drive"]
TOWNS = [("Northgate", "NG"), ("Bramfield", "BR"), ("Westbourne", "WB"),
         ("Cavendish", "CV"), ("Harlestone", "HR")]
PROPERTY_TYPES = ["Terraced house", "Semi-detached house", "Detached house",
                  "Flat / maisonette", "End-terrace house"]
HEATING = ["Gas central heating", "Electric storage heaters", "Air-source heat pump"]
GUARANTEE_POOL = ["NHBC / new-home warranty", "Damp-proofing guarantee",
                  "Double-glazing (FENSA) certificate", "Electrical work (NICEIC)",
                  "Roofing guarantee", "Central heating / boiler guarantee"]


def fake_address(rng: random.Random) -> Dict[str, str]:
    town, code = rng.choice(TOWNS)
    return {
        "line1": f"{rng.randint(1, 180)} {rng.choice(STREETS)}",
        "town": town,
        "postcode": f"{code}{rng.randint(1,9)} {rng.randint(1,9)}{rng.choice('ABDEFHJLNPQRSTUWXYZ')}{rng.choice('ABDEFHJLNPQRSTUWXYZ')}",
    }


@dataclass
class InjectedIssue:
    issue_id: str
    field: str                 # which TA6 field / doc pair the issue concerns
    issue_type: str            # missing_detail | missing_attachment | cross_doc_contradiction
    detection_method: str      # rule | nli   (which part of the pipeline should catch it)
    description: str           # human-readable explanation of the fault
    reference_enquiry: str     # gold-standard solicitor enquiry (generation target)


@dataclass
class TA6Record:
    record_id: str
    address: Dict[str, str]
    tenure: str
    property_type: str
    ta6: Dict                   # the seller's answers (extraction ground truth)
    planning_extract: Dict      # matched supporting document (independent evidence)
    injected_issues: List[InjectedIssue] = field(default_factory=list)


# ----------------------------------------------------------------------------
# Build a CONSISTENT baseline record (no faults yet).
# ----------------------------------------------------------------------------
def build_consistent_record(rid: str, rng: random.Random) -> TA6Record:
    addr = fake_address(rng)
    tenure = rng.choice(["Freehold", "Freehold", "Leasehold"])  # freehold weighted
    ptype = rng.choice(PROPERTY_TYPES)

    has_extension = rng.random() < 0.45
    ext_year = rng.randint(2009, 2022) if has_extension else None
    ext_desc = rng.choice(["Single-storey rear extension", "Loft conversion",
                           "Garage conversion", "Two-storey side extension"]) if has_extension else None

    ta6 = {
        # Section: Boundaries
        "boundaries_responsibility": rng.choice(["Left", "Right", "Rear", "Shared / not known"]),
        # Section: Disputes and complaints
        "disputes_or_complaints": {"answer": "No", "details": ""},
        # Section: Notices and proposals
        "notices_received": {"answer": "No", "details": ""},
        # Section: Alterations, planning and building control
        "alterations_made": {"answer": "Yes" if has_extension else "No",
                              "works": ext_desc or "None",
                              "year": ext_year},
        "building_regs_completion_certificate": {"answer": "Yes" if has_extension else "Not applicable",
                                                 "attachment_provided": True if has_extension else False},
        # Section: Guarantees and warranties
        "guarantees": rng.sample(GUARANTEE_POOL, k=rng.randint(0, 2)),
        # Section: Environmental / flooding
        "flooding": {"answer": "No", "details": ""},
        # Section: Services
        "heating_type": rng.choice(HEATING),
        "electrical_test_certificate": {"answer": rng.choice(["Yes", "No"]), "attachment_provided": True},
        # Section: Occupiers
        "other_occupiers_over_17": {"answer": "No", "names": ""},
    }

    # Matched supporting document: a planning extract that AGREES with the form.
    apps = []
    if has_extension:
        apps.append({
            "reference": f"{rng.randint(20,24):02d}/{rng.randint(1000,9999)}/FUL",
            "description": ext_desc,
            "decision": "Granted",
            "decision_year": ext_year,
        })
    planning_extract = {
        "source": "planning.data.gov.uk (synthetic placeholder)",
        "address_matched": f"{addr['line1']}, {addr['town']}",
        "applications": apps,
    }

    return TA6Record(record_id=rid, address=addr, tenure=tenure,
                     property_type=ptype, ta6=ta6, planning_extract=planning_extract)


# ----------------------------------------------------------------------------
# Mutation operators: inject a controlled, LABELLED fault.
# Each returns an InjectedIssue (or None if it does not apply to this record).
# ----------------------------------------------------------------------------
def mut_missing_detail(rec: TA6Record, rng: random.Random) -> Optional[InjectedIssue]:
    """RULE-catchable: set a Yes answer but blank its details."""
    rec.ta6["disputes_or_complaints"] = {"answer": "Yes", "details": ""}
    return InjectedIssue(
        issue_id="ISS-MISSING-DETAIL",
        field="disputes_or_complaints",
        issue_type="missing_detail",
        detection_method="rule",
        description="Seller answered 'Yes' to disputes/complaints but left the details box blank.",
        reference_enquiry=("The Property Information Form indicates that there have been disputes or "
                           "complaints relating to the property or a neighbouring property, but no details "
                           "have been provided. Please provide full details of the dispute(s), including "
                           "dates, the parties involved, and confirmation of whether the matter has been resolved."),
    )


def mut_missing_attachment(rec: TA6Record, rng: random.Random) -> Optional[InjectedIssue]:
    """RULE-catchable: certificate declared but not attached."""
    if rec.ta6["alterations_made"]["answer"] != "Yes":
        return None
    rec.ta6["building_regs_completion_certificate"] = {"answer": "Yes", "attachment_provided": False}
    return InjectedIssue(
        issue_id="ISS-MISSING-ATTACHMENT",
        field="building_regs_completion_certificate",
        issue_type="missing_attachment",
        detection_method="rule",
        description="Seller states a building regulations completion certificate exists but did not attach it.",
        reference_enquiry=("The seller has confirmed that a building regulations completion certificate is "
                           f"available for the {rec.ta6['alterations_made']['works'].lower()}, but a copy has not "
                           "been supplied. Please provide a copy of the building regulations completion "
                           "certificate for our review."),
    )


def mut_alteration_vs_planning(rec: TA6Record, rng: random.Random) -> Optional[InjectedIssue]:
    """NLI/LLM-catchable: form denies alterations, planning record shows one."""
    works = rng.choice(["Single-storey rear extension", "Loft conversion", "Two-storey side extension"])
    yr = rng.randint(2012, 2021)
    ref = f"{yr % 100:02d}/{rng.randint(1000,9999)}/FUL"
    # Form SAYS no alterations...
    rec.ta6["alterations_made"] = {"answer": "No", "works": "None", "year": None}
    rec.ta6["building_regs_completion_certificate"] = {"answer": "Not applicable", "attachment_provided": False}
    # ...but the planning record CONTRADICTS it.
    rec.planning_extract["applications"] = [{
        "reference": ref, "description": works, "decision": "Granted", "decision_year": yr,
    }]
    return InjectedIssue(
        issue_id="ISS-ALTERATION-VS-PLANNING",
        field="alterations_made | planning_extract.applications",
        issue_type="cross_doc_contradiction",
        detection_method="nli",
        description=(f"TA6 states no alterations were made, but the planning record shows permission "
                     f"({ref}) granted for a {works.lower()} in {yr}."),
        reference_enquiry=(f"The Property Information Form states that no alterations, extensions or other works "
                           f"have been carried out at the property. However, the local planning record shows that "
                           f"planning permission (ref {ref}) was granted for a {works.lower()} in {yr}. Please "
                           f"confirm whether these works were carried out and, if so, supply the planning permission "
                           f"and building regulations completion certificate for our review."),
    )


MUTATIONS = [mut_missing_detail, mut_missing_attachment, mut_alteration_vs_planning]


# ----------------------------------------------------------------------------
# Generate a dataset.
# ----------------------------------------------------------------------------
def generate(n: int, seed: int, fault_rate: float = 0.7) -> List[TA6Record]:
    rng = random.Random(seed)
    records = []
    for i in range(n):
        rid = f"TA6-{i+1:04d}"
        rec = build_consistent_record(rid, rng)
        if rng.random() < fault_rate:
            rng.shuffle(MUTATIONS)
            for mut in MUTATIONS:
                issue = mut(rec, rng)
                if issue:
                    rec.injected_issues.append(issue)
                    break  # one fault per record in v0
        records.append(rec)
    return records


# ----------------------------------------------------------------------------
# PDF rendering (so the extraction/OCR stage has real documents to read).
# ----------------------------------------------------------------------------
def render_pdfs(rec: TA6Record, outdir: Path):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.pdfgen import canvas
    except ImportError:
        return False

    def line(c, x, y, txt, size=10, bold=False):
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        c.drawString(x, y, txt)

    # --- TA6 PDF ---
    p = outdir / f"{rec.record_id}_TA6.pdf"
    c = canvas.Canvas(str(p), pagesize=A4)
    W, H = A4
    y = H - 20 * mm
    line(c, 20 * mm, y, "TA6 Property Information Form  (SYNTHETIC — research use only)", 13, True); y -= 6 * mm
    line(c, 20 * mm, y, f"Record {rec.record_id}   ·   Not a real property or person", 8); y -= 10 * mm
    a = rec.address
    line(c, 20 * mm, y, f"Property: {a['line1']}, {a['town']}, {a['postcode']}", 10, True); y -= 6 * mm
    line(c, 20 * mm, y, f"Tenure: {rec.tenure}    Type: {rec.property_type}", 10); y -= 10 * mm

    def qa(label, value):
        nonlocal y
        line(c, 22 * mm, y, label, 10, True); y -= 5 * mm
        for chunk in _wrap(value, 95):
            line(c, 26 * mm, y, chunk, 10); y -= 5 * mm
        y -= 2 * mm

    t = rec.ta6
    qa("2. Disputes and complaints",
       f"{t['disputes_or_complaints']['answer']}  —  {t['disputes_or_complaints']['details'] or '(no details given)'}")
    qa("4. Alterations, planning and building control",
       f"Alterations made: {t['alterations_made']['answer']}  ·  Works: {t['alterations_made']['works']}  ·  Year: {t['alterations_made']['year']}")
    qa("   Building regs completion certificate",
       f"{t['building_regs_completion_certificate']['answer']}  ·  Copy attached: {t['building_regs_completion_certificate']['attachment_provided']}")
    qa("5. Guarantees and warranties", ", ".join(t["guarantees"]) or "None declared")
    qa("6. Environmental / flooding", f"{t['flooding']['answer']}")
    qa("7. Services", f"Heating: {t['heating_type']}  ·  Electrical test cert: {t['electrical_test_certificate']['answer']}")
    qa("9. Occupiers (aged 17+)", f"{t['other_occupiers_over_17']['answer']}")
    c.showPage(); c.save()

    # --- Planning extract PDF (the supporting evidence) ---
    p2 = outdir / f"{rec.record_id}_planning.pdf"
    c = canvas.Canvas(str(p2), pagesize=A4)
    y = H - 20 * mm
    line(c, 20 * mm, y, "Local Planning Record — Extract  (SYNTHETIC)", 13, True); y -= 8 * mm
    line(c, 20 * mm, y, f"Address matched: {rec.planning_extract['address_matched']}", 10); y -= 8 * mm
    if not rec.planning_extract["applications"]:
        line(c, 22 * mm, y, "No planning applications on record.", 10)
    else:
        for app in rec.planning_extract["applications"]:
            line(c, 22 * mm, y, f"Ref {app['reference']}  ·  {app['decision']} {app['decision_year']}", 10, True); y -= 5 * mm
            line(c, 26 * mm, y, app["description"], 10); y -= 8 * mm
    c.showPage(); c.save()
    return True


def _wrap(text, width):
    words, lines, cur = str(text).split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines or [""]


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="synth_out")
    ap.add_argument("--pdf", action="store_true", help="also render PDFs")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    records = generate(args.n, args.seed)

    # Ground-truth JSON per record + a labels file + a summary.
    labels_path = outdir / "labels.jsonl"
    pdf_ok = None
    with open(labels_path, "w") as lf:
        for rec in records:
            (outdir / f"{rec.record_id}.json").write_text(json.dumps(asdict(rec), indent=2))
            for iss in rec.injected_issues:
                lf.write(json.dumps({"record_id": rec.record_id, **asdict(iss)}) + "\n")
            if args.pdf:
                pdf_ok = render_pdfs(rec, outdir)

    n_faults = sum(len(r.injected_issues) for r in records)
    by_method = {}
    for r in records:
        for iss in r.injected_issues:
            by_method[iss.detection_method] = by_method.get(iss.detection_method, 0) + 1

    print(f"Generated {len(records)} synthetic TA6 records  (seed={args.seed}) -> {outdir}/")
    print(f"  Records with an injected fault : {sum(1 for r in records if r.injected_issues)}")
    print(f"  Total labelled issues          : {n_faults}")
    print(f"  By detection method            : {by_method}")
    print(f"  Ground-truth labels            : {labels_path}")
    if args.pdf:
        print(f"  PDFs rendered                  : {'yes' if pdf_ok else 'reportlab not installed'}")
    print("\nEach issue carries: field, type, detection_method (rule|nli), and a reference_enquiry")
    print("=> extraction, detection AND generation ground truth, all by construction.")


if __name__ == "__main__":
    main()
