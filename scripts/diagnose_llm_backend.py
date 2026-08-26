"""
Diagnostic for "TA6_NLI_BACKEND=ollama ... --llm isn't generating anything
useful" -- isolates each step so we can see WHERE it's failing, rather than
just seeing eval_enquiry_citation.py's all-rows-identical-to-baseline result.

Does not touch any pipeline/dissertation code. Safe to run any number of
times; changes nothing.

Usage:
    TA6_NLI_BACKEND=ollama python scripts/diagnose_llm_backend.py
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

print(f"TA6_NLI_BACKEND env var = {os.getenv('TA6_NLI_BACKEND')!r}")
print(f"OLLAMA_MODEL env var    = {os.getenv('OLLAMA_MODEL')!r}  (code default: qwen2.5:7b)")
print()

# ---- Step 1: is the resolved backend what we expect? ----
from ta6 import nli
resolved = nli._resolve(None)
print(f"Step 1 -- backend resolution: nli._resolve(None) = {resolved!r}")
if resolved != "ollama":
    print("  !! This is not 'ollama'. If you set TA6_NLI_BACKEND=ollama and still see this,")
    print("     the env var isn't reaching this process -- check you're exporting it in the")
    print("     same shell/command, not a different terminal tab.")
    sys.exit(1)
print("  OK -- backend correctly resolves to ollama.\n")

# ---- Step 2: can we reach Ollama at all, with a trivial prompt? ----
print("Step 2 -- raw call_model() with a trivial prompt (tests connectivity + JSON mode):")
try:
    raw = nli.call_model('Reply with ONLY this JSON object: {"ok": true}', backend="ollama", max_tokens=50)
    print(f"  raw response: {raw!r}")
except Exception as e:
    print(f"  !! call_model raised: {type(e).__name__}: {e}")
    print("     Ollama is probably not running, or OLLAMA_MODEL isn't pulled.")
    print("     Check: `ollama list` should show qwen2.5:7b. `ollama serve` should be running")
    print("     (it usually auto-starts after `ollama run <model>` once).")
    sys.exit(1)
print()

# ---- Step 3: the actual generate_enquiry_llm() call, with full visibility ----
print("Step 3 -- generate_enquiry_llm() on one real issue, showing every intermediate value:")
from ta6.pipeline import Issue, generate_enquiry_llm, _ENQUIRY_PROMPT

issue = Issue(
    record_id="diag-test",
    field="building_regs_completion_certificate",
    issue_type="missing_attachment",
    detection_method="rule",
    description="A building regulations completion certificate is stated to exist but was not attached.",
)

prompt = _ENQUIRY_PROMPT.format(issue_type=issue.issue_type, field=issue.field,
                                 description=issue.description, context_block="")
print("  --- exact prompt sent to the model ---")
print(prompt)
print("  --- end prompt ---\n")

try:
    raw = nli.call_model(prompt, backend="ollama", max_tokens=400)
except Exception as e:
    print(f"  !! call_model raised: {type(e).__name__}: {e}")
    sys.exit(1)

print(f"  raw model response:\n  {raw!r}\n")

d = nli._parse_json_object(raw, {})
enquiry = str(d.get("enquiry", "")).strip()
cites = str(d.get("cites_field", "")).strip()
print(f"  parsed enquiry     : {enquiry!r}")
print(f"  parsed cites_field : {cites!r}")

grounded = bool(enquiry) and bool(cites) and (
    cites.lower() in issue.field.lower() or issue.field.lower() in cites.lower()
    or cites.lower() in issue.description.lower())
print(f"  grounding check    : {grounded}")
if not grounded:
    print()
    print("  !! THIS is very likely the actual failure. The grounding check requires")
    print(f"     cites_field ({cites!r}) to literally be a substring of the raw internal")
    print(f"     field name ({issue.field!r}) or vice versa, or of the description. A model")
    print("     that answers in natural language ('the building regulations certificate')")
    print("     instead of copying the raw snake_case identifier will fail this check even")
    print("     when its answer is genuinely correct and well-grounded -- and")
    print("     generate_enquiry_llm() silently falls back to the templated baseline in that")
    print("     case, which is exactly 'looks identical to the templated baseline, not useful'.")
else:
    print("\n  Grounding passed -- generate_enquiry_llm() should be returning real LLM output.")
    print("  If eval_enquiry_citation.py --llm still looks identical to the baseline, the issue")
    print("  is elsewhere (e.g. a different field/issue type behaving differently) -- re-run this")
    print("  script with a few of the other real field names from ta6/pipeline.py's")
    print("  YESNO_DETAIL_FIELDS / FIELD_KEYWORDS to check across issue types.")

print()
final = generate_enquiry_llm(issue, {})
print(f"Step 4 -- what generate_enquiry_llm() actually returns: {final!r}")
from ta6.pipeline import generate_enquiry
baseline = generate_enquiry(issue, {})
print(f"          templated baseline for comparison         : {baseline!r}")
print(f"          identical to baseline?                     : {final == baseline}")
