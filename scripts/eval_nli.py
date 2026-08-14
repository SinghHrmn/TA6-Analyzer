"""
Evaluate cross-document CONTRADICTION DETECTION (the dissertation's core claim).
================================================================================
Scores the NLI stage against a hand-labelled set of (TA6 claim, supporting
document, gold_label) triples. Contradiction detection is scored as a binary
task: 'contradict' vs not (consistent/neutral collapsed), because a missed
contradiction (false negative) is the costly error in conveyancing.

Backends (set TA6_NLI_BACKEND):  stub (offline baseline) | ollama | anthropic.

Usage:
    python scripts/eval_nli.py                  # single run, full set
    python scripts/eval_nli.py --repeats 3       # 3 independent runs, mean +/- range
    python scripts/eval_nli.py --split dev       # dev split only (for any exploration/tuning)
    python scripts/eval_nli.py --split test      # test split only (final numbers ONLY, once)

12 Aug 2026: added --repeats. A real LLM backend is non-deterministic
(sampling temperature > 0 by default), so a single run is exactly the kind
of fragility this project already root-caused for OCR (Chapter 6, section
6.4) -- reporting one number here would be an inconsistent standard. This
runs the full set N times and aggregates.

12 Aug 2026: added --split, once data/nli_eval_set.json grew to 62 items with
a fixed dev/test split (dissertation audit Goal A2). Discipline: any prompt
or backend exploration should score against --split dev only; --split test
is scored once, at the end, and that number is the one reported. Omitting
--split scores the full set (both splits combined), which is what the
existing Table 6.4 numbers already reported before the split existed --
kept as the default so old invocations don't silently change behaviour.

12 Aug 2026: added --set, so this same scoring logic (run_once/print_run) can
also score data/nli_eval_set_at_scale.json (dissertation audit Goal A4,
scripts/generate_nli_at_scale.py) without a second copy of the metric code.
That file's gold labels are mechanically derived, not hand-labelled -- see
its own "caveat" field -- so report numbers from it as a distinct, clearly
caveated result, never blended into this file's hand-labelled numbers.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ta6 import nli

SET = os.path.join(os.path.dirname(__file__), "..", "data", "nli_eval_set.json")


def run_once(data, backend):
    tp = fp = fn = tn = 0
    errors = []
    for it in data:
        found = nli.detect_contradictions([it["claim"]],
                                          {it["document_name"]: it["document_text"]},
                                          backend=backend)
        pred_contradict = len(found) > 0
        gold_contradict = it["gold_label"] == "contradict"
        if gold_contradict and pred_contradict: tp += 1
        elif gold_contradict and not pred_contradict: fn += 1; errors.append((it["id"], "MISSED", it["gold_label"]))
        elif not gold_contradict and pred_contradict: fp += 1; errors.append((it["id"], "FALSE-FLAG", it["gold_label"]))
        else: tn += 1

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / len(data)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec,
            "f1": f1, "accuracy": acc, "errors": errors}


def print_run(i, n_runs, backend, data, r, split_label=""):
    print("=" * 60)
    label = f"CONTRADICTION DETECTION  ·  backend = {backend}"
    if split_label:
        label += f"  ·  split = {split_label}"
    label += f"  ·  run {i}/{n_runs}" if n_runs > 1 else ""
    print(label)
    print(f"  {len(data)} items ({sum(1 for d in data if d['gold_label']=='contradict')} contradictions, "
          f"{sum(1 for d in data if d['gold_label']!='contradict')} not)")
    print("=" * 60)
    print(f"  TP={r['tp']}  FP={r['fp']}  FN={r['fn']}  TN={r['tn']}")
    print(f"  Precision {r['precision']:.2f}   Recall {r['recall']:.2f}   F1 {r['f1']:.2f}   Accuracy {r['accuracy']:.2f}")
    if r["errors"]:
        print("  Errors:", ", ".join(f"{i_}:{k}" for i_, k, _ in r["errors"]))
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, default=1,
                    help="run the full set this many times and report mean +/- range "
                         "(recommended: 3, for any non-deterministic backend)")
    ap.add_argument("--split", choices=["dev", "test"], default=None,
                    help="score only the dev or test split (data/nli_eval_set.json 'split' key). "
                         "Omit to score the full set (both splits combined).")
    ap.add_argument("--set", dest="set_path", default=SET,
                    help="path to an alternate eval-set JSON with the same {'items': [...]} "
                         "schema, e.g. data/nli_eval_set_at_scale.json (Goal A4). Defaults to "
                         "the hand-labelled data/nli_eval_set.json.")
    a = ap.parse_args()

    full_doc = json.load(open(a.set_path))
    full = full_doc["items"]
    if "caveat" in full_doc:
        print(f"NOTE ({a.set_path}): {full_doc['caveat']}\n")
    data = [it for it in full if a.split is None or it.get("split") == a.split]
    if a.split and not data:
        print(f"No items found with split={a.split!r} -- has data/nli_eval_set.json been "
              f"regenerated with a 'split' key? (Goal A2)")
        return
    backend = nli._resolve(None)

    runs = []
    for i in range(1, a.repeats + 1):
        r = run_once(data, backend)
        print_run(i, a.repeats, backend, data, r, split_label=(a.split or "all"))
        runs.append(r)

    if a.repeats > 1:
        import statistics as st
        print("\n" + "#" * 60)
        print(f"AGGREGATE over {a.repeats} runs  ·  backend = {backend}")
        print("#" * 60)
        for metric in ("precision", "recall", "f1", "accuracy"):
            vals = [r[metric] for r in runs]
            mean = st.mean(vals)
            print(f"  {metric.capitalize():<10} mean={mean:.3f}  range=[{min(vals):.3f}, {max(vals):.3f}]"
                  f"  values={[round(v,3) for v in vals]}")
        all_error_ids = [set(i_ for i_, _, _ in r["errors"]) for r in runs]
        stable_errors = set.intersection(*all_error_ids) if all_error_ids else set()
        unstable_errors = set.union(*all_error_ids) - stable_errors if all_error_ids else set()
        print(f"\n  Errors in EVERY run (stable failure mode): {sorted(stable_errors) or 'none'}")
        print(f"  Errors in SOME runs only (sampling noise) : {sorted(unstable_errors) or 'none'}")
        print("#" * 60)

    print("\nNote: 'stub' is the offline lexical BASELINE, not the contribution.")
    print("Re-run with TA6_NLI_BACKEND=ollama --repeats 3 for the model result (on your machine).")


if __name__ == "__main__":
    main()
