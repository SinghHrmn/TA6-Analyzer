"""
Full-loop detection evaluation on the REALISTIC dataset (System B).
=====================================================================
Unlike evaluate.py (which scores detection against the simple hand-designed
schema in pipeline/synth_eval), this script exercises the ACTUAL real-template
pipeline end to end:

    generate a form's ground truth (ta6.groups + ta6.content, as in
    generate_dataset_v2.py)
      -> FILL the real 442-field Law Society template with it (ta6.acroform.fill_template)
      -> WRITE a real PDF
      -> READ the PDF back (ta6.acroform.read_acroform) -- this is what a
         genuine "digital extraction" route would see, not the ground truth
      -> reconstruct each question's answer/detail from the fields actually
         read back (NOT from the generator's own gt_answers -- that would be
         circular)
      -> run a GENERALISED detection rule (missing_detail / unanswered) over
         every yes/no question the geometric mapper discovered, not just the
         3 hardcoded field names in ta6.pipeline.run_rule_checks
      -> score detected issues against the manifest fault labels

This is the evaluation ta6_analyser/scripts/evaluate.py could not run, because
it expects the pipeline/synth_eval schema (TA6-*.json, hand-designed fields),
not this one (form_*.json, 143 geometrically-discovered questions per form).

Usage:  python scripts/evaluate_v2.py --n 101 --seed 1 --template "<path>"
"""
import os, sys, argparse, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random
from pathlib import Path
from ta6 import content
from ta6.acroform import fill_template, read_acroform
from scripts.generate_dataset_v2 import load_plan, build_form

# Resolved relative to this file, with an env-var override, instead of a
# session-specific sandbox path -- three earlier scripts in this repo had a
# hardcoded /sessions/<name>/... path baked in, which only ever worked on the
# one sandbox that wrote them. Fixed 11 Aug 2026.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE = os.environ.get(
    "TA6_TEMPLATE_PATH",
    str(REPO_ROOT / "TA 6 documents" / "EDITABLE TA6 - 6th Edition 0426.pdf"))


def reconstruct_and_detect(form_id, plan, filled_vals):
    """Read the ACTUAL filled-PDF field values back (not the generator's own
    ground truth) and run a generalised rule check over every question the
    geometric mapper found. Returns a list of (qid, issue_type) detections."""
    detected = []
    for q in plan["questions"]:
        if q.kind not in ("yesno", "yesno_nk", "radio"):
            continue  # unanswered/missing-detail faults only injected on these kinds

        # which option (if any) actually reads "/Yes" in the filled-and-reread PDF?
        chosen = None
        for opt, qname in q.options.items():
            if filled_vals.get(qname) == "/Yes":
                chosen = opt
                break

        if q.kind in ("yesno", "yesno_nk") and chosen is None:
            detected.append((q.qid, "unanswered"))
            continue  # a blank question has no detail to check

        if q.detail_field:
            detail_label = plan["label_by_id"].get(q.detail_field, "")
            trigger = "No" if detail_label.strip().lower().startswith("if no") else "Yes"
            if chosen == trigger:
                detail_val = (filled_vals.get(q.detail_field) or "").strip()
                if not detail_val:
                    detected.append((q.qid, "missing_detail"))
    return detected


def run_eval_v2(n=101, seed=1, template=None, workdir=None, quiet=False):
    """Core full-loop evaluation logic, callable from the CLI (main, below) or
    directly from the results notebook -- both paths run this exact same
    code. Returns a results dict (same shape as evaluate.py's run_eval)."""
    template = template or DEFAULT_TEMPLATE
    if not quiet:
        print("Discovering question structure from the real template (one-off)...")
    plan = load_plan(template)
    if not quiet:
        print(f"  {plan['n_fields']} fields -> {len(plan['questions'])} questions "
              f"({len(plan['orphans'])} structural orphans)\n")

    workdir = workdir or tempfile.mkdtemp(prefix="ta6_eval_v2_")
    rng = random.Random(seed)

    tp = fp = fn = 0
    per_type = {}
    n_forms_with_fault = 0
    n_pdf_write_issues = 0

    for i in range(n):
        fid = f"form_{i+1:04d}"
        gt, values = build_form(plan, rng, fid)
        gold = {(f["qid"], f["type"]) for f in gt["faults"]}
        if gold:
            n_forms_with_fault += 1

        out_pdf = os.path.join(workdir, fid + ".pdf")
        _, unmatched = fill_template(template, values, out_pdf)
        if unmatched:
            n_pdf_write_issues += 1

        filled_vals = read_acroform(out_pdf)
        pred = set(reconstruct_and_detect(fid, plan, filled_vals))

        for t in {g[1] for g in gold} | {p[1] for p in pred}:
            per_type.setdefault(t, [0, 0, 0])
        for x in gold & pred:
            tp += 1; per_type[x[1]][0] += 1
        for x in pred - gold:
            fp += 1; per_type[x[1]][1] += 1
        for x in gold - pred:
            fn += 1; per_type[x[1]][2] += 1

        if not quiet and (i + 1) % 20 == 0:
            print(f"  ...{i+1}/{n} forms processed")

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    results = {"n_forms": n, "n_forms_with_fault": n_forms_with_fault,
               "n_pdf_write_issues": n_pdf_write_issues,
               "tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1,
               "n_fields": plan["n_fields"], "n_questions": len(plan["questions"]),
               "n_orphans": len(plan["orphans"]), "workdir": workdir,
               "per_type": {t: {"tp": a, "fp": b, "fn": c,
                                 "precision": (a/(a+b) if a+b else 0.0),
                                 "recall": (a/(a+c) if a+c else 0.0)}
                            for t, (a, b, c) in per_type.items()}}

    if not quiet:
        print("\n" + "=" * 70)
        print(f"FULL-LOOP DETECTION EVALUATION  ·  System B (realistic, real-template)")
        print(f"  {n} forms  ({n_forms_with_fault} with >=1 labelled fault)")
        print(f"  pipeline: generate -> fill REAL 442-field PDF -> read AcroForm back")
        print(f"            -> generalised rule check -> score vs manifest ground truth")
        print("=" * 70)
        print(f"  True positives : {tp}")
        print(f"  False positives: {fp}")
        print(f"  False negatives: {fn}")
        print(f"  PDF fields the filler could not match: {n_pdf_write_issues} form(s) affected")
        print("-" * 70)
        print(f"  Precision : {prec:.3f}")
        print(f"  Recall    : {rec:.3f}")
        print(f"  F1        : {f1:.3f}")
        print("-" * 70)
        print("  Per fault type (tp / fp / fn):")
        for t, (tpx, fpx, fnx) in sorted(per_type.items()):
            p = tpx / (tpx + fpx) if tpx + fpx else 0
            r = tpx / (tpx + fnx) if tpx + fnx else 0
            print(f"    {t:<16} tp={tpx:<4} fp={fpx:<4} fn={fnx:<4}  P={p:.2f} R={r:.2f}")
        print("=" * 70)
        print(f"Working files: {workdir}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=101)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--template", default=DEFAULT_TEMPLATE)
    ap.add_argument("--workdir", default=None)
    a = ap.parse_args()
    run_eval_v2(n=a.n, seed=a.seed, template=a.template, workdir=a.workdir)


if __name__ == "__main__":
    main()
