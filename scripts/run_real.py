"""
Run the pipeline on the REAL AST TA6 (digital route + text route).

Usage:  python3 run_real.py "<path to TA6 amended.pdf>" ["<editable 6th-ed template.pdf>"]
"""
import sys
from ta6.pipeline import (extract_ta6, extract_acroform,
                          run_rule_checks, generate_enquiry)

ROUTE_LABEL = {"text_digital": "digital PDF (text layer)",
               "ocr_scanned": "scanned paper (OCR)",
               "acroform_digital": "digital fillable form (AcroForm)"}


def main():
    if len(sys.argv) < 2:
        print('Usage:  python scripts/run_real.py "<path to a TA6 PDF>" '
              '["<editable 6th-ed template.pdf>"]')
        sys.exit(2)
    real_ta6 = sys.argv[1]
    template = sys.argv[2] if len(sys.argv) > 2 else None

    rec = extract_ta6(real_ta6)                      # <- auto-routes: digital / scanned / fillable
    print("=" * 70)
    print(f"STAGE 1 — EXTRACTION  ·  input auto-classified as: {ROUTE_LABEL.get(rec.get('route'), rec.get('route'))}")
    print("=" * 70)
    alt = rec["alterations_made"]
    print(f"  Section 4.1  alterations answer : {alt['answer']}")
    print(f"  Section 4.1  works described    : {alt['works']}")
    print(f"  Section 4.2  supporting docs    : "
          f"{'present' if rec['building_regs_completion_certificate']['attachment_provided'] else 'NOT provided'}")

    print("\n" + "=" * 70)
    print("STAGE 2 — DETECTION (rule engine)")
    print("=" * 70)
    issues = run_rule_checks("REAL-TA6", rec)
    if not issues:
        print("  No issues flagged.")
    for iss in issues:
        print(f"  [FLAG] {iss.issue_type}  ({iss.detection_method})")
        print(f"         {iss.description}")

    print("\n" + "=" * 70)
    print("STAGE 3 — ENQUIRY GENERATION (templated baseline)")
    print("=" * 70)
    for iss in issues:
        print("  " + generate_enquiry(iss, rec).replace("\n", "\n  "))

    if template:
        print("\n" + "=" * 70)
        print("DIGITAL ROUTE CHECK  ·  editable 6th-edition template (AcroForm)")
        print("=" * 70)
        fields = extract_acroform(template)
        btn = [k for k, v in fields.items() if v["type"] == "/Btn"]
        txt = [k for k, v in fields.items() if v["type"] == "/Tx"]
        print(f"  Machine-readable form fields found: {len(fields)}  "
              f"({len(txt)} text, {len(btn)} checkboxes)")
        print("  -> a digitally-submitted TA6 can be read field-by-field with no OCR.")
        print("  Sample fields:", ", ".join(list(fields)[:6]))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError) as e:
        print(f"\nError: {e}")
        sys.exit(1)
