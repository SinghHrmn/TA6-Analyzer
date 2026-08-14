"""
Cohen's kappa for the NLI double-labelling exercise (dissertation audit Goal A2).
=====================================================================================
Reads a second labeller's completed data/double_label_BLIND.csv (the "your_label"
column filled in) and compares it against Harman's original gold labels in
data/nli_eval_set.json, item by item, matched on id. Reports raw agreement,
Cohen's kappa, and every disagreement (so a genuine, hard-to-label item can be
told apart from a labelling mistake).

No sklearn dependency -- kappa is computed directly from its definition so this
runs anywhere the rest of the project already runs.

Usage:
    python scripts/compute_kappa.py data/double_label_BLIND.csv
"""
import sys, csv, json
from pathlib import Path
from collections import Counter

SET = Path(__file__).resolve().parents[1] / "data" / "nli_eval_set.json"


def cohens_kappa(labels_a, labels_b, categories):
    n = len(labels_a)
    assert n == len(labels_b) and n > 0
    po = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    count_a = Counter(labels_a)
    count_b = Counter(labels_b)
    pe = sum((count_a[c] / n) * (count_b[c] / n) for c in categories)
    kappa = (po - pe) / (1 - pe) if pe != 1 else 1.0
    return po, pe, kappa


def interpret(k):
    # Landis & Koch (1977) benchmarks -- standard reference for reporting kappa.
    if k < 0: return "poor (worse than chance)"
    if k < 0.20: return "slight"
    if k < 0.40: return "fair"
    if k < 0.60: return "moderate"
    if k < 0.80: return "substantial"
    return "almost perfect"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    blind_path = Path(sys.argv[1])

    gold_by_id = {it["id"]: it["gold_label"] for it in json.loads(SET.read_text())["items"]}

    rows = list(csv.DictReader(open(blind_path)))
    label_col = [c for c in rows[0].keys() if "your_label" in c][0]

    pairs = []
    missing = []
    for row in rows:
        rid = row["id"]
        second = row[label_col].strip().lower()
        if not second:
            missing.append(rid); continue
        if second not in ("contradict", "consistent", "neutral"):
            print(f"WARNING: {rid} has an unrecognised label {second!r} -- skipping. "
                  f"Expected exactly one of: contradict, consistent, neutral.")
            continue
        if rid not in gold_by_id:
            print(f"WARNING: {rid} not found in {SET} -- skipping.")
            continue
        pairs.append((rid, gold_by_id[rid], second))

    if missing:
        print(f"{len(missing)} item(s) not yet labelled (blank your_label): {missing}")
    if len(pairs) < 10:
        print(f"\nOnly {len(pairs)} labelled pairs found -- fill in more of "
              f"{blind_path.name} before reporting kappa (aim for the full sheet, "
              f"at least 20 as required by Goal A2).")
        if not pairs:
            return

    gold = [g for _, g, _ in pairs]
    second = [s for _, _, s in pairs]
    categories = sorted(set(gold) | set(second))
    po, pe, kappa = cohens_kappa(gold, second, categories)

    print("=" * 64)
    print(f"INTER-ANNOTATOR AGREEMENT  ·  {len(pairs)} double-labelled items")
    print("=" * 64)
    print(f"  Raw agreement (po)   : {po:.3f}  ({sum(1 for g,s in zip(gold,second) if g==s)}/{len(pairs)})")
    print(f"  Expected agreement (pe): {pe:.3f}")
    print(f"  Cohen's kappa         : {kappa:.3f}  ({interpret(kappa)}, Landis & Koch 1977)")
    print("-" * 64)

    disagreements = [(rid, g, s) for rid, g, s in pairs if g != s]
    if disagreements:
        print(f"  Disagreements ({len(disagreements)}):")
        for rid, g, s in disagreements:
            print(f"    {rid:<6} Harman={g:<11} second labeller={s}")
    else:
        print("  No disagreements.")
    print("=" * 64)
    print("Report this kappa value (and n) in section 6.6 as the inter-annotator")
    print("agreement figure for the NLI evaluation set.")


if __name__ == "__main__":
    main()
