"""
Real-document evidence resolution.
===================================================================
Runs the pipeline on the real TA6, then — for every detected issue — searches
every real supporting document (Reports on Title, Local Authority search) for
candidate corroborating evidence, and attributes any hit to its filename.

IMPORTANT CAVEAT, stated up front rather than buried: the supporting documents
available for this project were NOT supplied as a matched set for the same
transaction as the real TA6 (see Chapter 3, "Real documents from the
collaborating firm" — they are from separate matters, redacted). This script
therefore does NOT claim to resolve the TA6's issue against genuine matching
evidence. What it demonstrates instead is (a) the resolution/attribution
mechanism working end-to-end on real, unstructured document text, and (b) a
concrete, real-data illustration of the false-match risk in keyword-based
evidence search — which is exactly why this uses transparent keyword search
with a human-confirmation caveat, rather than silently trusting a lexical
match to auto-resolve a query (that would be the same over-claiming problem
as the offline NLI stub).

Usage:  python scripts/run_real_with_evidence.py
"""
import os, re, sys, glob, json
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ta6.pipeline import (extract_ta6, run_rule_checks, generate_enquiry,
                          generate_enquiry_llm, _pdftext)

# Relative to this file, with an env-var override -- was a hardcoded
# session-specific /sessions/... path, fixed 11 Aug 2026 (see evaluate_v2.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
TA6_DIR = os.environ.get("TA6_DOCS_DIR", str(REPO_ROOT / "TA 6 documents"))
REAL_TA6 = os.path.join(TA6_DIR, "TA6 amended.pdf")
SUPPORTING_GLOB = ["Freehold Report on Title*.pdf", "Report on Title*.pdf",
                    "REPORT ON TITLE*.pdf", "Local Authority search*.pdf"]

# Evidence keyword sets per issue type. Deliberately simple and transparent
# (not a black-box embedding match) so a human can immediately see WHY
# something was flagged as candidate evidence, and judge for themselves
# whether it actually applies.
EVIDENCE_TERMS = {
    "works_without_support": ["building regulation", "completion certificate",
                               "planning permission", "consent", "loft conversion",
                               "loft", "extension", "conversion"],
}


def find_supporting_docs():
    seen, out = set(), []
    for pattern in SUPPORTING_GLOB:
        for p in glob.glob(os.path.join(TA6_DIR, pattern)):
            if os.path.basename(p) not in seen and os.path.basename(p) != os.path.basename(REAL_TA6):
                seen.add(os.path.basename(p))
                out.append(p)
    return sorted(out)


def search_evidence(issue, docs):
    """For one issue, search every supporting document's text for candidate
    evidence terms. Returns a list of (filename, matched_terms, snippet)."""
    terms = EVIDENCE_TERMS.get(issue.issue_type, [])
    hits = []
    for path in docs:
        text = _pdftext(path)
        low = text.lower()
        matched = [t for t in terms if t in low]
        if matched:
            # smallest useful snippet around the first match, for a human to check
            t0 = matched[0]
            m = re.search(r"[^.\n]{0,80}" + re.escape(t0) + r"[^.\n]{0,80}", text, re.I)
            snippet = m.group(0).strip() if m else ""
            hits.append({"filename": os.path.basename(path), "matched_terms": matched,
                         "snippet": snippet})
    return hits


def main():
    print("=" * 74)
    print("REAL-DOCUMENT PIPELINE RUN WITH EVIDENCE RESOLUTION")
    print("=" * 74)
    print(f"TA6 input       : {os.path.basename(REAL_TA6)}")
    docs = find_supporting_docs()
    print(f"Supporting docs : {len(docs)} found")
    for d in docs:
        print(f"  - {os.path.basename(d)}")
    print("\nCAVEAT: these supporting documents are from separate, unmatched matters "
          "(see Chapter 3) — this run tests the resolution MECHANISM on real document "
          "text, not genuine same-transaction corroboration. Any 'evidence found' below "
          "requires human confirmation, not automatic trust.\n")

    rec = extract_ta6(REAL_TA6)
    issues = run_rule_checks("REAL-TA6", rec)

    if not issues:
        print("No issues detected — nothing to resolve.")
        return

    results = []
    for iss in issues:
        print("-" * 74)
        print(f"ISSUE: {iss.issue_type}  ({iss.detection_method})")
        print(f"  {iss.description}")
        hits = search_evidence(iss, docs)
        if hits:
            print(f"  Candidate evidence found in {len(hits)} document(s):")
            for h in hits:
                print(f"    [{h['filename']}] matched: {h['matched_terms']}")
                print(f"       \"...{h['snippet']}...\"")
            status = "candidate_evidence_found_unconfirmed"
        else:
            print(f"  No candidate evidence found in any of the {len(docs)} supporting document(s).")
            status = "unresolved_no_evidence"

        templated = generate_enquiry(iss, rec)
        backend = os.getenv("TA6_NLI_BACKEND")
        enquiry = generate_enquiry_llm(iss, rec) if backend else templated
        source = "templated" if enquiry == templated else "LLM-drafted"
        if hits:
            filenames = ", ".join(sorted({h["filename"] for h in hits}))
            enquiry += (f" (Note for reviewing solicitor: possible related mentions were found in "
                        f"{filenames} during an automated keyword search; these documents are not "
                        f"confirmed to relate to this transaction and the match has not been verified "
                        f"— please check before treating this as resolved.)")

        print(f"\n  ENQUIRY (status: {status}, source: {source}):")
        print(f"  {enquiry}")

        results.append({"issue_type": iss.issue_type, "description": iss.description,
                        "status": status, "evidence": hits, "enquiry": enquiry,
                        "enquiry_source": source})

    out_path = os.path.join(os.path.dirname(__file__), "..", "real_evidence_run.json")
    json.dump({"ta6": os.path.basename(REAL_TA6), "supporting_docs": [os.path.basename(d) for d in docs],
              "results": results}, open(out_path, "w"), indent=2)
    print("\n" + "=" * 74)
    print(f"Full run log written to {out_path}")


if __name__ == "__main__":
    main()
