"""
Demo / verification for generate_enquiry_llm() (dissertation audit Goal A3).
=====================================================================================
Runs the newly-wired LLM enquiry generator against one real detected issue (from the
genuine TA6, section 6.5) and prints both the LLM-generated enquiry and the templated
baseline side by side, so the difference -- and whether the LLM path actually fired,
rather than silently falling back -- is visible at a glance.

Usage:
    TA6_NLI_BACKEND=ollama python scripts/demo_enquiry_llm.py
    TA6_NLI_BACKEND=anthropic python scripts/demo_enquiry_llm.py   # needs ANTHROPIC_API_KEY
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ta6.pipeline import Issue, generate_enquiry, generate_enquiry_llm
from ta6 import nli

# The real, code-verified issue from the genuine TA6 (section 6.5 / Table 6.6).
issue = Issue(
    record_id="REAL-TA6",
    field="alterations_made",
    issue_type="works_without_support",
    detection_method="rule",
    description="Alterations are described ('Loft conversion carried out before I occupied "
                "the property.') but no completion certificate or exemption explanation is provided.",
)
ta6 = {"alterations_made": {"answer": "Yes",
                            "works": "Loft conversion carried out before I occupied the property."}}

backend = nli._resolve(None)
print(f"Backend: {backend}\n")

templated = generate_enquiry(issue, ta6)
print("Templated baseline (generate_enquiry):")
print(f"  {templated}\n")

llm_result = generate_enquiry_llm(issue, ta6)
fell_back = (llm_result == templated)
print("LLM-backed (generate_enquiry_llm):")
print(f"  {llm_result}\n")
print(f"Fell back to the templated baseline: {fell_back}")
if fell_back and backend in ("ollama", "anthropic"):
    print("  (a real backend is set but the result matches the template exactly -- this usually "
          "means the model's JSON response was malformed or failed the grounding check; add a "
          "print(raw) in generate_enquiry_llm() to inspect the model's actual output if this "
          "happens on a real backend.)")
