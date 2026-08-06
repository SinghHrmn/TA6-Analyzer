"""
Full detection demo: rules (missing fields) + NLI (cross-document contradictions)
running together as one pass, each with a generated enquiry.
Backend for the NLI part follows TA6_NLI_BACKEND (stub offline; ollama/anthropic if set).
"""
from ta6.pipeline import run_rule_checks, detect_free_text_contradiction, generate_enquiry

# a TA6 record: one within-form rule fault + a claim to check across documents
ta6 = {
    "disputes_or_complaints": {"answer": "Yes", "details": ""},            # rule: yes-but-blank
    "alterations_made": {"answer": "No", "works": ""},
    "building_regs_completion_certificate": {"answer": "Not applicable", "attachment_provided": False},
}
claims = ["No alterations, extensions or other works have been carried out at the property."]
supporting = {
    "Local planning record": "Application 19/0421/HH — erection of a single-storey rear extension — Granted, 2019.",
    "EPC": "Walls: solid brick as built. Roof: pitched, insulated. A rear extension is present.",
    "Local Authority search": "No adverse entries. Building control completion certificate on file.",
}

rid = "DEMO-1"
issues = run_rule_checks(rid, ta6) + detect_free_text_contradiction(rid, claims, supporting)

print(f"{len(issues)} issue(s) detected on one form (rules + cross-document NLI):\n")
for i in issues:
    print(f"[{i.issue_type}]  via {i.detection_method}")
    print(f"   {i.description}")
    enq = i.enquiry or generate_enquiry(i, ta6)
    print(f"   ENQUIRY: {enq}\n")
