"""
Dependency-aware re-scoring on top of evaluate_v2's full-loop evaluation.
==========================================================================
Does NOT modify evaluate_v2.py or ta6/groups.py. Reuses their exact
generation/fill/read pipeline, then applies ta6.dependencies' skip-gate
suppression to the "unanswered" detections before scoring -- so the
comparison against evaluate_v2's own (frozen, already-reported) numbers
tells us directly whether the skip-logic gap actually changes anything
for the synthetic evaluation.

Usage: python scripts/evaluate_v3_dependency_aware.py --n 101 --seed 1
"""
import os, sys, argparse, tempfile, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.generate_dataset_v2 import load_plan, build_form
from scripts.evaluate_v2 import DEFAULT_TEMPLATE
from ta6.acroform import fill_template, read_acroform
from ta6.dependencies import is_legitimately_skipped, build_gate_index


def reconstruct_answers_and_detect(form_id, plan, filled_vals):
    """Same logic as evaluate_v2.reconstruct_and_detect, but ALSO returns the
    per-question chosen answer (needed to evaluate skip-gates), and applies
    suppression before returning the detected issue list."""
    gate_index = build_gate_index()
    answers = {}
    raw_detected = []   # (qid, issue_type) before suppression

    for q in plan["questions"]:
        if q.kind not in ("yesno", "yesno_nk", "radio"):
            continue
        chosen = None
        for opt, qname in q.options.items():
            if filled_vals.get(qname) == "/Yes":
                chosen = opt
                break
        answers[q.qid] = chosen

        if q.kind in ("yesno", "yesno_nk") and chosen is None:
            raw_detected.append((q.qid, "unanswered"))
            continue
        if q.detail_field:
            detail_label = plan["label_by_id"].get(q.detail_field, "")
            trigger = "No" if detail_label.strip().lower().startswith("if no") else "Yes"
            if chosen == trigger:
                detail_val = (filled_vals.get(q.detail_field) or "").strip()
                if not detail_val:
                    raw_detected.append((q.qid, "missing_detail"))

    suppressed = []
    detected = []
    for qid, itype in raw_detected:
        if itype == "unanswered" and is_legitimately_skipped(qid, answers, gate_index=gate_index):
            suppressed.append((qid, itype))
            continue
        detected.append((qid, itype))
    return detected, suppressed


def run(n=101, seed=1, template=None, quiet=False):
    template = template or DEFAULT_TEMPLATE
    plan = load_plan(template)
    workdir = tempfile.mkdtemp(prefix="ta6_eval_v3_")
    rng = random.Random(seed)

    tp = fp = fn = 0
    total_suppressed = 0
    suppressed_examples = {}

    for i in range(n):
        fid = f"form_{i+1:04d}"
        gt, values = build_form(plan, rng, fid)
        gold = {(f["qid"], f["type"]) for f in gt["faults"]}
        out_pdf = os.path.join(workdir, fid + ".pdf")
        fill_template(template, values, out_pdf)
        filled_vals = read_acroform(out_pdf)
        detected, suppressed = reconstruct_answers_and_detect(fid, plan, filled_vals)
        pred = set(detected)

        for qid, itype in suppressed:
            total_suppressed += 1
            suppressed_examples.setdefault(qid, 0)
            suppressed_examples[qid] += 1

        for x in gold & pred:
            tp += 1
        for x in pred - gold:
            fp += 1
        for x in gold - pred:
            fn += 1

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    if not quiet:
        print(f"n={n} seed={seed}")
        print(f"TP={tp} FP={fp} FN={fn}  Precision={prec:.3f} Recall={rec:.3f} F1={f1:.3f}")
        print(f"Total 'unanswered' flags suppressed by skip-gate logic: {total_suppressed}")
        print(f"Suppressed, by qid: {suppressed_examples}")
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1,
            "total_suppressed": total_suppressed, "suppressed_by_qid": suppressed_examples}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=101)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    run(n=a.n, seed=a.seed)
