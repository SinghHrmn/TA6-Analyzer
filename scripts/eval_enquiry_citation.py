"""
Citation precision/recall for generated enquiries (dissertation audit Goal E1,
section 6.7), ALCE-style (Gao, Yen, Yih & Chen 2023, cited section 2.6): for
each generated enquiry, does it cite the correct source field/matter, and no
incorrect one.

Data: every issue the rule engine (run_rule_checks + run_structured_crosscheck)
actually detects across the 60 records in pipeline/synth_eval/ -- exhaustive
over what the current rule engine produces on that set (45 issues), not a
hand-picked subset.

Two things are measured:
  1. Field-citation recall/precision: does the enquiry name the field it
     should, and avoid naming a field it shouldn't. The alterations template
     legitimately asks for a completion certificate alongside the planning
     permission (a real conveyancing follow-up, not a wrong citation about an
     unrelated issue), so that keyword is excluded from the "wrong field"
     check for that one template -- checked empirically (see WHY_EXCLUDED
     below) that no record in this dataset has an independent
     missing_attachment issue that this could be confused with.
  2. Record-specific grounding (cross_doc_contradiction only, the only issue
     type whose template embeds record-specific data): does the enquiry cite
     THIS record's own planning reference number, and no other record's.

By default this scores the templated baseline (generate_enquiry(), fully
offline, no LLM needed). Pass --llm to additionally score the LLM-backed
generator (generate_enquiry_llm()) on the same 45 issues for a direct
template-vs-LLM comparison -- this requires a real backend
(dissertation audit Goals A1-A4 established the same constraint):

    TA6_NLI_BACKEND=ollama python scripts/eval_enquiry_citation.py --llm
    TA6_NLI_BACKEND=anthropic python scripts/eval_enquiry_citation.py --llm   # needs ANTHROPIC_API_KEY

Without a live backend, generate_enquiry_llm() falls back to the templated
baseline (by design -- ungrounded LLM output is rejected), so --llm run
without a real backend just reproduces the templated numbers; this script
prints a warning if that happens so the result isn't mistaken for a real
LLM-side score.
"""
import sys, os, json, glob, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ta6.pipeline import run_rule_checks, run_structured_crosscheck, generate_enquiry

FIELD_KEYWORDS = {
    "alterations_made | planning_extract": ["alteration", "planning permission", "works were carried out"],
    "disputes_or_complaints": ["dispute", "complaint"],
    "building_regs_completion_certificate": ["completion certificate", "building regulations"],
}
# The alterations template legitimately asks for a completion certificate as
# standard supporting evidence for disclosed works -- this is a bundled,
# expected follow-up request, not a false citation of an unrelated issue.
# WHY_EXCLUDED: verified separately that 0/60 records have both a
# cross_doc_contradiction and an independent missing_attachment issue, so
# this bundling never collides with a real, separately-flagged issue.
BUNDLE_EXCEPTIONS = {"alterations_made | planning_extract": ["building_regs_completion_certificate"]}


def score(records, use_llm=False):
    if use_llm:
        from ta6.pipeline import generate_enquiry_llm

    all_refs = []
    for rec in records.values():
        pe = rec.get("planning_extract")
        if pe:
            all_refs += [a["reference"] for a in pe.get("applications", []) if a.get("reference")]

    rows = []
    for rid, rec in records.items():
        ta6 = rec["ta6"]; planning = rec.get("planning_extract")
        issues = run_rule_checks(rid, ta6) + run_structured_crosscheck(rid, ta6, planning)
        for iss in issues:
            enq = generate_enquiry_llm(iss, ta6) if use_llm else generate_enquiry(iss, ta6)
            enq_l = enq.lower()
            true_field = iss.field
            hit_true = any(k in enq_l for k in FIELD_KEYWORDS.get(true_field, []))
            excluded = set(BUNDLE_EXCEPTIONS.get(true_field, []))
            false_hits = [f for f in FIELD_KEYWORDS if f != true_field and f not in excluded
                          and any(k in enq_l for k in FIELD_KEYWORDS[f])]
            row = {"record_id": rid, "issue_type": iss.issue_type, "field": true_field,
                   "recall_hit": hit_true, "precision_clean": len(false_hits) == 0}
            if iss.issue_type == "cross_doc_contradiction":
                own_refs = [a["reference"] for a in planning.get("applications", [])] if planning else []
                other_refs = [r for r in all_refs if r not in own_refs]
                row["ref_grounded"] = any(r in enq for r in own_refs) and not any(r in enq for r in other_refs)
            rows.append(row)
    return rows


def report(rows, label):
    n = len(rows)
    recall = sum(r["recall_hit"] for r in rows) / n
    precision = sum(r["precision_clean"] for r in rows) / n
    print(f"\n{label}  (N={n})")
    print(f"  field-citation recall    : {recall:.3f}")
    print(f"  field-citation precision : {precision:.3f}")
    cdc = [r for r in rows if r["issue_type"] == "cross_doc_contradiction"]
    if cdc:
        g = sum(r["ref_grounded"] for r in cdc) / len(cdc)
        print(f"  cross_doc_contradiction record-specific grounding: {g:.3f}  (n={len(cdc)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synth-dir", default=os.path.join(os.path.dirname(__file__), "..", "..", "pipeline", "synth_eval"))
    ap.add_argument("--llm", action="store_true", help="also score generate_enquiry_llm() (needs a real backend)")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.synth_dir, "*.json")))
    records = {json.load(open(fp))["record_id"]: json.load(open(fp)) for fp in files}
    print(f"{len(records)} synthetic records loaded from {a.synth_dir}")

    templated_rows = score(records, use_llm=False)
    report(templated_rows, "Templated baseline (generate_enquiry)")

    if a.llm:
        llm_rows = score(records, use_llm=True)
        report(llm_rows, "LLM-backed (generate_enquiry_llm)")
        templ_texts = {(r["record_id"], r["issue_type"]) for r in templated_rows}
        # crude fallback check: compare counts only (text identity checked in demo_enquiry_llm.py)
        if llm_rows == templated_rows:
            print("\n  WARNING: LLM-scored rows are identical to the templated baseline -- this "
                  "usually means TA6_NLI_BACKEND is unset or every call fell back to the templated "
                  "baseline (see generate_enquiry_llm()'s grounding check). Set TA6_NLI_BACKEND to "
                  "a real backend and re-run to get a genuine LLM-side score.")
