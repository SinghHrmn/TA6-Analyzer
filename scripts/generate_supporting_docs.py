"""
Synthetic supporting-document generator.
=====================================================================
Generates realistic-looking (clearly-marked SYNTHETIC) supporting documents
to test the pipeline against YOUR OWN test TA6 forms -- hand-filled, scanned,
or digitally completed -- not just the auto-generated dataset. This is
independent of ta6.generator (which only ever produces one matched planning
extract as part of building a whole synthetic TA6 record); this script
produces a standalone SET of documents for a property you describe, so you
can pair them with any TA6 you already have.

Produces up to four documents, each as a real PDF with realistic section
headings and wording (loosely modelled on the structure of the genuine
Reports on Title / Local Authority search supplied by the collaborating
firm, so they exercise the extractor's text parsing realistically):

  - Local Authority search extract
  - Report on Title extract
  - EPC (Energy Performance Certificate) summary
  - Planning permission / building control extract

You control, per document, whether it CONFIRMS, CONTRADICTS, or is SILENT
on a specific piece of work you describe (e.g. a loft conversion) -- this is
what lets you build test cases on demand: "seller declared a loft conversion
with no certificate, does the LA search mention it, and does that change
what the pipeline flags."

Usage:
    python scripts/generate_supporting_docs.py \\
        --address "12 Example Road, Anytown, AT1 2BC" \\
        --works "loft conversion" --works-year 2018 \\
        --scenario contradict \\
        --out my_test_docs/

    --scenario confirm     : the documents corroborate the declared works
                              (planning permission granted, referenced in the
                              LA search and report on title)
    --scenario contradict  : the documents show works with NO matching
                              consent on record (a real gap) -- or, if you
                              pass --works-not-declared, works the SELLER
                              never mentioned on the TA6 at all (the reverse
                              direction: undisclosed works)
    --scenario silent      : the documents don't mention the works either
                              way (a legitimate "not resolved by available
                              evidence" case -- the honest common case)
"""
import argparse
import os
import random
import textwrap
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas


def _write_page(path, title, body_lines, footer=None):
    c = canvas.Canvas(str(path), pagesize=A4)
    W, H = A4
    y = H - 22 * mm

    c.setFont("Helvetica-Bold", 13)
    c.drawString(20 * mm, y, title)
    y -= 6 * mm
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(20 * mm, y, "SYNTHETIC DOCUMENT — generated for pipeline testing, not a real record")
    y -= 10 * mm

    for line in body_lines:
        if line is None:
            y -= 4 * mm
            continue
        bold, text, size = False, line, 10
        if isinstance(line, tuple):
            text, bold = line[0], line[1]
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        for chunk in textwrap.wrap(text, 95) or [""]:
            if y < 25 * mm:
                c.showPage()
                y = H - 22 * mm
                c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
            c.drawString(22 * mm, y, chunk)
            y -= 5 * mm

    if footer:
        c.setFont("Helvetica-Oblique", 7)
        c.drawString(20 * mm, 15 * mm, footer)
    c.showPage()
    c.save()


def _planning_ref(rng):
    return f"{rng.randint(19, 24):02d}/{rng.randint(1000, 9999)}/{rng.choice(['FUL', 'HH', 'LDC'])}"


# --------------------------------------------------------------------------
# 12 Aug 2026 (dissertation audit Goal A4): the four documents' section
# content used to be composed AND rendered to PDF in one pass inside
# build_docs() below -- fine for one-off CLI use, but generating matched
# documents for 100+ forms at scale doesn't need (or want) 400+ PDFs
# rendered through reportlab just to hand plain text to the NLI backend.
# This function is the single source of truth for what each document SAYS,
# given (address, works, works_year, scenario, works_not_declared, seed);
# build_docs() below still renders it to PDF (byte-identical body content --
# behaviour-preserving refactor), and compose_texts() (further down) turns
# the same structure into plain text for at-scale, PDF-free NLI evaluation
# so both paths can never drift apart.
# --------------------------------------------------------------------------
def _compose_bodies(address, works, works_year, scenario, works_not_declared, seed):
    rng = random.Random(seed)
    ref = _planning_ref(rng)
    docs = {}

    mentions_works = scenario in ("confirm", "contradict") or works_not_declared
    has_consent = scenario == "confirm"

    # ---------------- Local Authority search ----------------
    body = [
        f"Property: {address}", None,
        ("1. Planning and Building Control", True),
    ]
    if mentions_works:
        if has_consent:
            body += [f"Planning permission {ref}: {works}, granted {works_year}.",
                     f"Building Regulations completion certificate on file, ref BR-{rng.randint(1000,9999)}."]
        else:
            body += [f"Site inspection notes reference a {works} at this address; no matching planning "
                     f"permission or Building Regulations completion certificate found on the register "
                     f"as at the date of this search."]
    else:
        body += ["No planning applications or Building Control entries revealed by this search."]
    body += [None, ("2. Roads and footpaths", True),
             "The roads and footpaths fronting the property are adopted highway maintainable at public expense.",
             None, ("3. Other matters", True),
             "No adverse entries revealed."]
    docs["Local Authority search"] = {
        "title": "LOCAL AUTHORITY SEARCH — Extract", "body": body,
        "footer": f"Search reference LAS-{rng.randint(10000,99999)} — synthetic test document",
        "filename": "Local_Authority_Search_SYNTHETIC.pdf"}

    # ---------------- Report on Title ----------------
    body = [
        f"Property: {address}", f"Title number: SYNTH{rng.randint(100000,999999)}", None,
        ("1. Tenure and registration", True),
        "The property is registered with title absolute.",
        None, ("2. Planning entries revealed", True),
    ]
    if mentions_works:
        if has_consent:
            body += [f"There is a planning entry relating to a {works}, permission {ref} granted "
                     f"{works_year}; we attach a copy for your records."]
        else:
            body += [f"There is a planning entry relating to a {works} at this property; we have been "
                     f"unable to identify a matching Building Regulations completion certificate and "
                     f"recommend this is raised with the seller's solicitor."]
    else:
        body += ["No adverse planning entries are revealed by the search."]
    body += [None, ("3. Charges and restrictions", True),
             "Subject to a standard mortgage restriction in favour of the registered lender; to be "
             "discharged on completion."]
    docs["Report on Title"] = {
        "title": "REPORT ON TITLE — Extract", "body": body,
        "footer": f"Matter ref RPT-{rng.randint(10000,99999)} — synthetic test document",
        "filename": "Report_on_Title_SYNTHETIC.pdf"}

    # ---------------- EPC ----------------
    band = rng.choice("CDE")
    body = [
        f"Property: {address}", f"Certificate reference: {rng.randint(1000000000,9999999999)}", None,
        (f"Energy rating: {band}", True),
        f"Current efficiency: {rng.randint(55,78)}  ·  Potential: {rng.randint(75,92)}",
        None, ("Recommendations", True),
        "Consider loft insulation top-up and draught proofing to improve the current rating.",
    ]
    if mentions_works and "loft" in works.lower():
        body += [None, ("Note", True),
                 "Roof space is used as habitable accommodation at the time of assessment; standard "
                 "loft-insulation recommendations do not apply to the converted area."]
    docs["EPC"] = {
        "title": "ENERGY PERFORMANCE CERTIFICATE — Summary", "body": body,
        "footer": "Synthetic test document — not a valid EPC", "filename": "EPC_SYNTHETIC.pdf"}

    # ---------------- Planning extract ----------------
    body = [f"Address matched: {address}", None]
    if mentions_works:
        body += [(f"Application {ref}", True),
                 f"Description: {works}",
                 f"Decision: {'Granted' if has_consent else 'No application found — see note'}  "
                 f"{'· ' + str(works_year) if has_consent else ''}"]
        if not has_consent:
            body += [None, "Note: works of this description were identified from a site inspection; "
                            "no corresponding planning application or decision is held on this register."]
    else:
        body += ["No planning applications on record for this address."]
    docs["Planning extract"] = {
        "title": "LOCAL PLANNING RECORD — Extract", "body": body,
        "footer": "Synthetic test document", "filename": "Planning_Extract_SYNTHETIC.pdf"}

    return docs


def build_docs(address, works, works_year, scenario, works_not_declared, seed, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    docs = _compose_bodies(address, works, works_year, scenario, works_not_declared, seed)
    made = []
    for d in docs.values():
        p = out_dir / d["filename"]
        _write_page(p, d["title"], d["body"], footer=d["footer"])
        made.append(str(p))
    return made


def compose_texts(address, works, works_year, scenario, works_not_declared, seed, doc_names=None):
    """PDF-free counterpart to build_docs(): same section content (via
    _compose_bodies, so the two can never disagree), flattened to plain text
    -- close to what pdftotext -layout would hand the pipeline, without the
    cost of rendering and re-reading a PDF. Returns {doc_name: text}, keyed
    by the same names build_docs() writes files for ("Local Authority
    search", "Report on Title", "EPC", "Planning extract"); pass doc_names to
    restrict to a subset (for eval_nli.py, one claim is checked against every
    key in this dict, so keep it to 1-2 documents unless you want every form
    scored against all four)."""
    docs = _compose_bodies(address, works, works_year, scenario, works_not_declared, seed)
    if doc_names:
        docs = {k: v for k, v in docs.items() if k in doc_names}
    out = {}
    for name, d in docs.items():
        lines = [d["title"], ""]
        for line in d["body"]:
            if line is None:
                lines.append("")
            elif isinstance(line, tuple):
                lines.append(line[0])
            else:
                lines.append(line)
        if d.get("footer"):
            lines += ["", d["footer"]]
        out[name] = "\n".join(lines)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--address", required=True, help='e.g. "12 Example Road, Anytown, AT1 2BC"')
    ap.add_argument("--works", default="loft conversion",
                    help="the works to reference, e.g. 'loft conversion', 'side extension'")
    ap.add_argument("--works-year", type=int, default=2019)
    ap.add_argument("--scenario", choices=["confirm", "contradict", "silent"], default="silent",
                    help="confirm: docs corroborate the works with consent on record. "
                         "contradict: docs show the works but with no matching consent (the real "
                         "gap this pipeline is designed to catch). "
                         "silent: docs don't mention the works either way (a legitimate "
                         "not-resolved case).")
    ap.add_argument("--works-not-declared", action="store_true",
                    help="use this to build the REVERSE-direction test case: works appear in the "
                         "supporting documents that the seller did NOT declare on their TA6 at all. "
                         "Pair the generated docs with a TA6 where you left the relevant question "
                         "answered 'No' or blank.")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="my_test_docs")
    a = ap.parse_args()

    made = build_docs(a.address, a.works, a.works_year, a.scenario, a.works_not_declared, a.seed, a.out)
    print(f"Generated {len(made)} synthetic supporting documents in {a.out}/:")
    for m in made:
        print(f"  - {m}")
    print(f"\nScenario: {a.scenario}"
          f"{' (+ works not declared on the TA6 — reverse-direction test)' if a.works_not_declared else ''}")
    print("Pair these with a TA6 PDF (yours, hand-filled or scanned) via the webapp's 'Supporting "
          "documents' upload, or scripts/run_real_with_evidence.py, to test cross-document checking "
          "on a scenario you control.")


if __name__ == "__main__":
    main()
