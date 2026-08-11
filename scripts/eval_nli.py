"""
Evaluate cross-document CONTRADICTION DETECTION (the dissertation's core claim).
================================================================================
Scores the NLI stage against a hand-labelled set of (TA6 claim, supporting
document, gold_label) triples. Contradiction detection is scored as a binary
task: 'contradict' vs not (consistent/neutral collapsed), because a missed
contradiction (false negative) is the costly error in conveyancing.

Backends (set TA6_NLI_BACKEND):  stub (offline baseline) | ollama | anthropic.

Usage:  python scripts/eval_nli.py            # scores whichever backend is set
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ta6 import nli

SET = os.path.join(os.path.dirname(__file__), "..", "data", "nli_eval_set.json")


def main():
    data = json.load(open(SET))["items"]
    backend = nli._resolve(None)
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

    print("=" * 60)
    print(f"CONTRADICTION DETECTION  ·  backend = {backend}")
    print(f"  {len(data)} items ({sum(1 for d in data if d['gold_label']=='contradict')} contradictions, "
          f"{sum(1 for d in data if d['gold_label']!='contradict')} not)")
    print("=" * 60)
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Precision {prec:.2f}   Recall {rec:.2f}   F1 {f1:.2f}   Accuracy {acc:.2f}")
    if errors:
        print("  Errors:", ", ".join(f"{i}:{k}" for i, k, _ in errors))
    print("=" * 60)
    print("Note: 'stub' is the offline lexical BASELINE, not the contribution.")
    print("Re-run with TA6_NLI_BACKEND=ollama for the model result (on your machine).")


if __name__ == "__main__":
    main()
