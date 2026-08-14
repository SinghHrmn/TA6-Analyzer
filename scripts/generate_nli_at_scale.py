"""
Cross-document NLI at scale, matched against the realistic v2 dataset (dissertation
audit Goal A4).
=====================================================================================
Every item in data/nli_eval_set.json is hand-authored or hand-picked-and-labelled --
that is what makes it trustworthy, and also why it tops out at 62 items. Section 6.6.2
currently says evaluating cross-document NLI at scale on the realistic (v2, 143-question,
101-form) dataset is impossible, because that dataset was built for the single-document
missing_detail/unanswered rule check (evaluate_v2.py), not for claim-vs-supporting-document
pairs.

This script closes that gap for ONE question family: "(i) other building works or changes
to the property" (qid 6:1 i on the real template, section 5 "Alterations") -- the same
question the dissertation's real-TA6 worked example (Table 6.6, works_without_support) is
about, so this is a scaled-up version of an already-verified issue type, not a new one.

For each generated form it:
  1. reads the form's OWN answer to that question (Yes+detail / Yes+missing-detail-fault /
     No / blank) straight from generate_dataset_v2.build_form()'s ground truth -- the exact
     same generator evaluate_v2.py already uses for the dissertation's headline 101-form
     result, so this is matched infrastructure, not a new synthetic pipeline;
  2. derives a claim + a supporting-document scenario (confirm / contradict / contradict
     +works-not-declared / silent) from that answer, via a fixed, documented rule (see
     _derive_case below) -- there is no hand-labelling here;
  3. calls scripts.generate_supporting_docs.compose_texts() (PDF-free path, added in this
     same audit item) to get matched document TEXT for that scenario, keyed by the SAME
     rule/scenario semantics the existing CLI tool (generate_supporting_docs.py) already
     documents and the dissertation already describes;
  4. writes one NLI item per (form, document) pair, in the same schema eval_nli.py reads,
     to a SEPARATE file (not merged into nli_eval_set.json).

IMPORTANT CAVEAT -- read this before citing a number from this script anywhere:
Gold labels here are MECHANICALLY DERIVED from the same rule that generated the document
(step 2 and step 3 share one scenario value) -- they are not an independent human
judgement. This checks something real (does the backend recover a label that is a
deterministic, known function of the input, at a scale the hand-labelled set can't reach)
but it is a WEAKER form of evidence than data/nli_eval_set.json's hand-labelled items,
and must be reported as a separate, clearly-caveated number -- never blended into the
main NLI result or presented as if it were hand-labelled.

Usage:
    python scripts/generate_nli_at_scale.py --n 60 --seed 2026 --out data/nli_eval_set_at_scale.json
    TA6_NLI_BACKEND=ollama python scripts/eval_nli.py --set data/nli_eval_set_at_scale.json --repeats 3
"""
import os, sys, json, argparse, re, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.generate_dataset_v2 import load_plan, build_form
from scripts.generate_supporting_docs import compose_texts
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE = os.environ.get(
    "TA6_TEMPLATE_PATH",
    str(REPO_ROOT / "TA 6 documents" / "EDITABLE TA6 - 6th Edition 0426.pdf"))

# The question this script targets. Matched by exact prompt text (not a
# hardcoded qid) so it fails loudly, rather than silently scoring the wrong
# question, if ta6.groups' geometric recovery ever re-numbers questions.
TARGET_PROMPT = "(i) other building works or changes to the property Yes"

_WORKS_LABELS = [
    (r"\bloft\b", "loft conversion"),
    (r"\bextension\b", "rear extension"),
    (r"\bconservatory\b", "conservatory"),
    (r"\binsulation\b", "insulation works"),
    (r"\bgarage\b", "garage conversion"),
    (r"\binternal wall", "internal wall removal"),
    (r"\bchimney\b", "chimney breast removal"),
]
_UNDECLARED_CANDIDATES = ["loft conversion", "rear extension", "garage conversion", "conservatory"]


def _short_works_label(detail_text):
    t = (detail_text or "").lower()
    for pat, label in _WORKS_LABELS:
        if re.search(pat, t):
            return label
    return "building works"


def _works_year(detail_text, rng):
    m = re.search(r"\b(19|20)\d{2}\b", detail_text or "")
    return int(m.group(0)) if m else rng.randint(2008, 2023)


def _derive_case(answer, has_missing_detail_fault, detail_text, rng):
    """The ONE rule this whole script's gold labels rest on. Four cases:

      Yes + detail given        -> scenario=confirm            -> gold=consistent
        (declared works, and the supporting documents show matching consent --
        the documents corroborate the claim)
      Yes + missing_detail fault-> scenario=contradict          -> gold=contradict
        (declared works, but the seller gave no detail -- documents show the
        works exist with NO matching consent on record; this is the scaled-up
        version of the real works_without_support case in Table 6.6)
      No / blank, ~40% of these -> scenario=contradict+undeclared -> gold=contradict
        (seller declared no works, but the supporting documents show works
        DID happen -- the reverse-direction undisclosed-works case)
      No / blank, other ~60%    -> scenario=silent               -> gold=consistent
        (seller declared no works, documents don't mention any works either --
        silence doesn't contradict a "no works" claim)

    Returns (scenario, works_not_declared, works_label, works_year, gold_label).
    """
    if answer == "Yes" and not has_missing_detail_fault and detail_text:
        return "confirm", False, _short_works_label(detail_text), _works_year(detail_text, rng), "consistent"
    if answer == "Yes" and has_missing_detail_fault:
        return "contradict", False, _short_works_label(detail_text), _works_year(detail_text, rng), "contradict"
    # No / blank
    if rng.random() < 0.4:
        label = rng.choice(_UNDECLARED_CANDIDATES)
        return "contradict", True, label, rng.randint(2008, 2023), "contradict"
    return "silent", False, "building works", rng.randint(2008, 2023), "consistent"


def _claim_text(answer, has_missing_detail_fault, detail_text):
    if answer == "Yes" and not has_missing_detail_fault and detail_text:
        return (f"The seller discloses that other building works or changes have been "
                f"carried out at the property. Details given: \"{detail_text}\"")
    if answer == "Yes" and has_missing_detail_fault:
        return ("The seller discloses that other building works or changes have been carried "
                "out at the property, but no further detail of the work or any consent has "
                "been given.")
    return ("The seller discloses that no building works or changes have been carried out "
            "at the property.")


def generate(n, seed, template, doc_names, out_path):
    plan = load_plan(template)
    target = next((q for q in plan["questions"] if q.prompt == TARGET_PROMPT), None)
    if target is None:
        raise SystemExit(
            f"Could not find the target question (prompt={TARGET_PROMPT!r}) in the recovered "
            f"question structure -- ta6.groups' geometric mapping may have changed. Run "
            f"scripts/generate_dataset_v2.py's load_plan() and inspect plan['questions'] by hand "
            f"before re-running this script; do not silently fall back to a different question.")
    qid = target.qid
    detail_field = target.detail_field

    rng = random.Random(seed)
    items = []
    scenario_counts, gold_counts = {}, {}
    n_forms_used = 0

    for i in range(n):
        fid = f"scaleform_{i+1:04d}"
        gt, values = build_form(plan, rng, fid)
        ans = gt["answers"].get(qid, {}).get("answer", "blank")
        has_fault = any(f["qid"] == qid and f["type"] == "missing_detail" for f in gt["faults"])
        detail_text = (values.get(detail_field, "") or "").strip() if detail_field else ""

        scenario, undeclared, works_label, works_year, gold = _derive_case(ans, has_fault, detail_text, rng)
        claim = _claim_text(ans, has_fault, detail_text)
        address = f"{gt['header']['Property Address']}, {gt['header']['Postcode']}"
        doc_seed = rng.randint(1, 10_000_000)

        texts = compose_texts(address, works_label, works_year, scenario, undeclared, doc_seed,
                              doc_names=doc_names)
        for doc_name, doc_text in texts.items():
            item_id = f"as{i+1:04d}_{doc_name.split()[0].lower()}"
            items.append({
                "id": item_id,
                "claim": claim,
                "document_name": doc_name,
                "document_text": doc_text,
                "gold_label": gold,
                "claim_source": "generated_at_scale:v2_form",
                "document_source": f"generated_at_scale:{scenario}" + ("+undeclared" if undeclared else ""),
                "form_id": fid,
                "form_answer": ans,
                "form_had_missing_detail_fault": has_fault,
                "scenario": scenario,
            })
            scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
            gold_counts[gold] = gold_counts.get(gold, 0) + 1
        n_forms_used += 1

    out = {
        "description": ("Cross-document NLI items generated AT SCALE from the realistic v2 "
                        "dataset generator (dissertation audit Goal A4), targeting the "
                        "'(i) other building works or changes to the property' question "
                        "(section 5, Alterations) -- the same question family as the real-TA6 "
                        "works_without_support case in Table 6.6."),
        "caveat": ("Gold labels are MECHANICALLY DERIVED from the same rule that generated the "
                  "matched document (see _derive_case in generate_nli_at_scale.py) -- they are "
                  "NOT independently hand-labelled. Report this separately from "
                  "data/nli_eval_set.json's hand-labelled result; do not average the two together "
                  "or present this as hand-labelled evidence."),
        "target_question": {"qid": qid, "prompt": target.prompt, "detail_field": detail_field},
        "n_forms": n_forms_used,
        "n_items": len(items),
        "doc_names": doc_names,
        "scenario_counts": scenario_counts,
        "gold_label_counts": gold_counts,
        "generation_seed": seed,
        "items": items,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=60, help="number of forms to generate")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--template", default=DEFAULT_TEMPLATE)
    ap.add_argument("--docs", default="Local Authority search,Report on Title",
                    help="comma-separated document names to match per form (from: "
                         "'Local Authority search', 'Report on Title', 'EPC', 'Planning extract')")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data",
                                                   "nli_eval_set_at_scale.json"))
    a = ap.parse_args()
    doc_names = [d.strip() for d in a.docs.split(",") if d.strip()]

    out = generate(a.n, a.seed, a.template, doc_names, a.out)
    print(f"Target question: {out['target_question']['prompt']!r} (qid={out['target_question']['qid']})")
    print(f"Generated {out['n_items']} items from {out['n_forms']} forms "
          f"({len(doc_names)} matched document(s) per form) -> {a.out}")
    print(f"Scenario counts: {out['scenario_counts']}")
    print(f"Gold label counts: {out['gold_label_counts']}")
    print("\nCAVEAT:", out["caveat"])
    print(f"\nScore it:  TA6_NLI_BACKEND=ollama python scripts/eval_nli.py --set {a.out} --repeats 3")


if __name__ == "__main__":
    main()
