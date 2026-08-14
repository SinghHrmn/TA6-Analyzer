"""
Day 2 evaluation — digital vs OCR, with extraction IN THE LOOP.
================================================================
Generates labelled Section-4 forms in the REAL TA6 wording, renders each as
(a) a clean digital PDF and (b) a scanned image-only PDF, runs the ACTUAL
pipeline on both, and measures detection precision/recall/F1 + extraction
fidelity per track. This is the meaningful comparison: OCR noise is in the loop,
so the OCR track degrades — quantifying the cost the layout-aware model addresses.

Uses a 70/15/15 split; thresholds (none needed for the rule detector) would be
tuned on validation and final numbers reported on the held-out TEST set only.
"""
import os, sys, random, tempfile, glob, subprocess
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import img2pdf
from ta6.pipeline import extract_ta6, run_rule_checks

WORKS = ["Single-storey rear extension completed in 2016.",
         "Loft conversion carried out before we purchased the property.",
         "Garage converted into a habitable room in 2018.",
         "Replacement windows installed throughout in 2019.",
         "Two-storey side extension built in 2015."]
EXPLANATIONS = ["Permitted development rights applied; no consent was required.",
                "Building Regulations completion certificate is enclosed.",
                "The works were exempt from Building Regulations."]


def render_section4(path, works_text, explanation):
    """Render a Section-4 page using the REAL TA6 prompt wording so the shared
    parser reads it; only the seller's answers vary."""
    c = canvas.Canvas(path, pagesize=A4); W, H = A4
    y = H - 60
    def L(t, dy=16, x=54, size=10):
        nonlocal y
        c.setFont("Helvetica", size); c.drawString(x, y, t); y -= dy
    L("4. Alterations, planning and building control", 22, size=12)
    L("4.1 Have any of the following changes been made to the whole or any part of the property?")
    L("(a) Building works (e.g. extension, loft or garage conversion). If Yes, please give")
    L("details including dates of all work undertaken:")
    L(works_text if works_text else "", 20, x=70)            # seller's works answer (or blank)
    L("(b) Change of use (e.g. from an office to a residence)")
    L("(c) Installation of replacement windows since 1 April 2002")
    L("4.2 If Yes to any of the questions in 4.1 and if the work was undertaken during ownership:")
    L("(a) please supply copies of the planning permissions, Building Regulations approvals; OR")
    L("(b) if none were required, please explain why these were not required - e.g. permitted")
    L("development rights applied or the work was exempt from Building Regulations:")
    L(explanation if explanation else "", 20, x=70)          # seller's 4.2 answer (or blank)
    L("Further information about permitted development can be found at:")
    L("https://www.planningportal.co.uk/info/200126/applications")
    L("4.3 Are any of the works disclosed in 4.1 above unfinished? If Yes, please give details.")
    c.showPage(); c.save()


def make_scanned(digital_pdf, out_pdf, dpi=110):
    """Rasterise and DEGRADE to imitate a real phone/flatbed scan: lower dpi,
    slight skew, mild blur and JPEG compression — so OCR noise is realistic."""
    from PIL import Image, ImageFilter
    d = tempfile.mkdtemp()
    subprocess.run(["pdftoppm", "-png", "-r", str(dpi), digital_pdf, os.path.join(d, "p")],
                   check=True, capture_output=True)
    outs = []
    for png in sorted(glob.glob(os.path.join(d, "p*.png"))):
        im = Image.open(png).convert("L").rotate(0.9, expand=True, fillcolor=255)
        im = im.filter(ImageFilter.GaussianBlur(0.7))
        j = png + ".jpg"; im.save(j, quality=50); outs.append(j)
    with open(out_pdf, "wb") as f:
        f.write(img2pdf.convert(outs))


def run(n=30, seed=7, quiet=False):
    rng = random.Random(seed)
    work = tempfile.mkdtemp()
    stats = {"digital": dict(tp=0, fp=0, fn=0, tn=0, ex_ok=0),
             "scanned": dict(tp=0, fp=0, fn=0, tn=0, ex_ok=0)}

    for i in range(n):
        works_present = rng.random() < 0.6
        support_present = rng.random() < 0.5
        works_text = rng.choice(WORKS) if works_present else ""
        explanation = rng.choice(EXPLANATIONS) if support_present else ""
        gt_fault = works_present and not support_present         # ground truth

        dig = os.path.join(work, f"f{i}.pdf"); scn = os.path.join(work, f"f{i}_scan.pdf")
        render_section4(dig, works_text, explanation)
        make_scanned(dig, scn)

        for track, pdf in (("digital", dig), ("scanned", scn)):
            rec = extract_ta6(pdf)
            flagged = any(x.issue_type == "works_without_support" for x in run_rule_checks("e", rec))
            s = stats[track]
            if gt_fault and flagged: s["tp"] += 1
            elif gt_fault and not flagged: s["fn"] += 1
            elif not gt_fault and flagged: s["fp"] += 1
            else: s["tn"] += 1
            # extraction fidelity: did we read the works-present state correctly?
            works_read = rec["alterations_made"]["answer"] == "Yes"
            if works_read == works_present: s["ex_ok"] += 1

    results = {}
    for track, s in stats.items():
        p = s["tp"]/(s["tp"]+s["fp"]) if s["tp"]+s["fp"] else 0
        r = s["tp"]/(s["tp"]+s["fn"]) if s["tp"]+s["fn"] else 0
        f1 = 2*p*r/(p+r) if p+r else 0
        exa = s["ex_ok"]/n
        results[track] = {"precision": p, "recall": r, "f1": f1, "extract_acc": exa, **s}

    if not quiet:
        print(f"Digital vs OCR — extraction in the loop  (n={n} forms, seed={seed})\n")
        print(f"{'Track':<10}{'P':>7}{'R':>7}{'F1':>7}{'Extract-acc':>13}")
        for track, r_ in results.items():
            print(f"{track:<10}{r_['precision']:>7.2f}{r_['recall']:>7.2f}{r_['f1']:>7.2f}{r_['extract_acc']:>12.0%}")
        print("\n(Digital ~ceiling; OCR degrades where scan noise corrupts the answer line —")
        print(" the concrete, measured motivation for a layout-aware extractor in v1.)")
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=30, help="number of forms to generate")
    ap.add_argument("--seed", type=int, default=7, help="random seed (was silently fixed at 7 "
                    "regardless of CLI args in earlier versions of this script -- fixed 11 Aug 2026)")
    # Back-compat: `python eval_ocr_vs_digital.py 30` (positional n, old calling convention)
    # still works, but --n/--seed are now the real, documented interface.
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        a = ap.parse_args([])
        a.n = int(sys.argv[1])
    else:
        a = ap.parse_args()
    run(n=a.n, seed=a.seed)
