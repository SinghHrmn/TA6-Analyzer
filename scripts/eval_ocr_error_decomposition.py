"""
OCR error decomposition.
=====================================================================
eval_ocr_vs_digital.py answers "how does end-to-end detection F1 change
between digital and scanned input" -- it conflates three different possible
failure sources into one number:
  (1) Tesseract mis-reading characters (an OCR problem)
  (2) the regex parser failing to locate/anchor the answer even when OCR text
      is largely correct (a PARSING problem)
  (3) the rule engine misjudging a correctly-extracted record (a RULE-LOGIC
      problem)
This script decomposes that: for each scanned form it (a) measures character
and word error rate (CER/WER) of the OCR output against the EXACT known
ground-truth text that was rendered (not a proxy), using Levenshtein
alignment, and (b) classifies each form's outcome into one of four buckets so
low F1 can be attributed to a specific stage rather than reported as one
opaque number.

Usage:  python scripts/eval_ocr_error_decomposition.py [n]
"""
import os, sys, random, tempfile, glob, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
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
    c = canvas.Canvas(path, pagesize=A4); W, H = A4
    y = H - 60
    def L(t, dy=16, x=54, size=10):
        nonlocal y
        c.setFont("Helvetica", size); c.drawString(x, y, t); y -= dy
    L("4. Alterations, planning and building control", 22, size=12)
    L("4.1 Have any of the following changes been made to the whole or any part of the property?")
    L("(a) Building works (e.g. extension, loft or garage conversion). If Yes, please give")
    L("details including dates of all work undertaken:")
    L(works_text if works_text else "", 20, x=70)
    L("(b) Change of use (e.g. from an office to a residence)")
    L("(c) Installation of replacement windows since 1 April 2002")
    L("4.2 If Yes to any of the questions in 4.1 and if the work was undertaken during ownership:")
    L("(a) please supply copies of the planning permissions, Building Regulations approvals; OR")
    L("(b) if none were required, please explain why these were not required - e.g. permitted")
    L("development rights applied or the work was exempt from Building Regulations:")
    L(explanation if explanation else "", 20, x=70)
    L("Further information about permitted development can be found at:")
    L("https://www.planningportal.co.uk/info/200126/applications")
    L("4.3 Are any of the works disclosed in 4.1 above unfinished? If Yes, please give details.", 20)
    c.showPage(); c.save()


def make_scanned(digital_pdf, out_pdf, dpi=110):
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


def levenshtein(a, b):
    """Standard edit distance (insert/delete/substitute), O(len(a)*len(b))."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0: return lb
    if lb == 0: return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * lb
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + (ca != cb))
        prev = cur
    return prev[lb]


def best_window_cer(reference, haystack):
    """Find the substring of `haystack` most similar to `reference` (sliding by
    reference length +/- 30%, coarse but adequate for a single answer line),
    and return (CER, matched_window). Avoids penalising CER for OCR grabbing
    surrounding boilerplate rather than genuinely misreading the answer."""
    ref_len = len(reference)
    if ref_len == 0:
        return 0.0, ""
    best_cer, best_win = 1.0, ""
    step = max(1, ref_len // 6)
    for start in range(0, max(1, len(haystack) - ref_len // 2), step):
        for span in (int(ref_len * 0.7), ref_len, int(ref_len * 1.3)):
            win = haystack[start:start + span]
            if not win.strip():
                continue
            d = levenshtein(reference.lower(), win.lower())
            cer = d / ref_len
            if cer < best_cer:
                best_cer, best_win = cer, win
    return best_cer, best_win


def word_error_rate(reference, hypothesis):
    ref_w, hyp_w = reference.split(), hypothesis.split()
    if not ref_w:
        return 0.0
    return levenshtein(ref_w, hyp_w) / len(ref_w)  # levenshtein() also works on lists


def run(n=25, seed=7, quiet=False):
    rng = random.Random(seed)
    work = tempfile.mkdtemp()
    rows = []

    for i in range(n):
        works_present = rng.random() < 0.6
        support_present = rng.random() < 0.5
        works_text = rng.choice(WORKS) if works_present else ""
        explanation = rng.choice(EXPLANATIONS) if support_present else ""
        gt_fault = works_present and not support_present

        dig = os.path.join(work, f"f{i}.pdf"); scn = os.path.join(work, f"f{i}_scan.pdf")
        render_section4(dig, works_text, explanation)
        make_scanned(dig, scn)

        rec = extract_ta6(scn)
        flagged = any(x.issue_type == "works_without_support" for x in run_rule_checks("e", rec))
        works_read = rec["alterations_made"]["answer"] == "Yes"

        # OCR quality on JUST the ground-truth text that matters, not the whole page.
        ocr_text = extract_ta6.__globals__["ocr_pdf_to_text"](scn) if works_present or support_present else ""
        cer_works, win_works = (best_window_cer(works_text, ocr_text) if works_text else (0.0, ""))
        cer_expl, win_expl = (best_window_cer(explanation, ocr_text) if explanation else (0.0, ""))
        wer_works = word_error_rate(works_text, win_works) if works_text else 0.0

        extraction_correct = (works_read == works_present)
        detection_correct = (flagged == gt_fault)

        if works_present and cer_works < 0.15 and not extraction_correct:
            bucket = "PARSER_FAILURE (OCR was clean, extraction still wrong)"
        elif works_present and cer_works >= 0.15 and not extraction_correct:
            bucket = "OCR_FAILURE (OCR corrupted the text, extraction wrong)"
        elif works_present and cer_works >= 0.15 and extraction_correct:
            bucket = "OCR_NOISY_BUT_RECOVERED (OCR corrupted but parser still got it right)"
        elif not extraction_correct:
            bucket = "PARSER_FAILURE (blank/absent case misread)"
        else:
            bucket = "CLEAN"

        rows.append({"i": i, "works_present": works_present, "support_present": support_present,
                     "cer_works": round(cer_works, 3), "wer_works": round(wer_works, 3),
                     "extraction_correct": extraction_correct, "detection_correct": detection_correct,
                     "bucket": bucket})

    from collections import Counter
    buckets = Counter(r["bucket"].split(" (")[0] for r in rows)
    mean_cer = sum(r["cer_works"] for r in rows if r["works_present"]) / max(1, sum(1 for r in rows if r["works_present"]))
    mean_wer = sum(r["wer_works"] for r in rows if r["works_present"]) / max(1, sum(1 for r in rows if r["works_present"]))

    if not quiet:
        print(f"OCR error decomposition  (n={n} forms, seed={seed})\n")
        print(f"{'#':<4}{'works?':<8}{'CER':<7}{'WER':<7}{'extract_ok':<12}{'detect_ok':<11}{'bucket'}")
        for r in rows:
            print(f"{r['i']:<4}{str(r['works_present']):<8}{r['cer_works']:<7}{r['wer_works']:<7}"
                  f"{str(r['extraction_correct']):<12}{str(r['detection_correct']):<11}{r['bucket']}")
        print(f"\nMean character error rate on the answer line (works-present forms only): {mean_cer:.1%}")
        print(f"Mean word error rate on the answer line: {mean_wer:.1%}")
        print("\nFailure attribution:")
        for b, c in buckets.most_common():
            print(f"  {b:<28} {c}/{n}")

    return {"rows": rows, "n": n, "seed": seed, "mean_cer": mean_cer, "mean_wer": mean_wer,
            "buckets": dict(buckets)}


if __name__ == "__main__":
    run(n=int(sys.argv[1]) if len(sys.argv) > 1 else 25)
