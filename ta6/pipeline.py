"""
TA6 pipeline (v0)  ·  extract -> check -> generate
==================================================
Dissertation: AI-Assisted Document Analysis and Enquiry Generation for
UK Residential Conveyancing (H. Singh, MSc Applied AI, LSBU).

This module is the runnable core of the three-stage pipeline. The parts that
run WITHOUT an LLM (digital extraction, the rule engine, structured cross-check,
templated enquiry generation) are implemented here. The LLM-dependent parts
(free-text NLI contradiction detection; higher-quality generation) are marked
with `# LLM-HOOK` and expose a clean interface to run locally with an API key.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
import re
import os
import subprocess


# ----------------------------------------------------------------------------
@dataclass
class Issue:
    record_id: str
    field: str
    issue_type: str          # missing_detail | missing_attachment | works_without_support | cross_doc_contradiction
    detection_method: str    # rule | structured_crosscheck | nli(LLM)
    description: str
    enquiry: str = ""


# ============================================================================
# STAGE 1 — EXTRACTION
# ============================================================================
def extract_acroform(pdf_path: str) -> Dict:
    """Digital route: read a fillable TA6 (e.g. the 6th-edition editable template)
    directly from its AcroForm fields. High accuracy, no OCR."""
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    fields = reader.get_fields() or {}
    out = {}
    for name, f in fields.items():
        out[name] = {"type": str(f.get("/FT")), "value": f.get("/V")}
    return out


def _pdftext(pdf_path: str) -> str:
    try:
        return subprocess.run(["pdftotext", "-layout", pdf_path, "-"],
                              capture_output=True, text=True).stdout
    except FileNotFoundError:
        raise RuntimeError("'pdftotext' not found. Install Poppler — "
                           "macOS: brew install poppler · Ubuntu: sudo apt install poppler-utils")


def _parse_section4(txt: str, source: str = "") -> Dict:
    """Parse Section 4 (Alterations, planning and building control) from the TEXT
    of a completed TA6 — whether that text came from a digital text layer OR from
    OCR of a scan. Shared by both routes, so detection is identical downstream.

    Checkbox states are only weakly recoverable from flat text (their marks lose
    column alignment) — the limitation that motivates the layout-aware model in v1."""
    # --- 4.1(a): the seller's works answer. Anchor on the prompt tail ("work
    # undertaken:"), which is stable in both digital text AND OCR, then take the
    # answer line; fall back to a content pattern. Section numbers are NOT relied
    # on, because OCR frequently mangles "4.1"/"4.3". ---
    works_text = ""
    m = re.search(r"work undertaken\s*:?\s*\n+\s*([^\n]+)", txt, re.I)
    if m:
        cand = m.group(1).strip()
        # Two different exclusion checks were being run as one, which hid a bug:
        # the leading-marker patterns ("(b)", "(c)"...) are ANCHORED (only make
        # sense at position 0), but "change of use" is boilerplate that can occur
        # anywhere in the line and was wrongly forced through re.match too — so a
        # single OCR misread of the leading "(b)" (e.g. read as "(o)") silently
        # defeated the whole exclusion and let the printed instruction line
        # ("(b) Change of use...") through as if it were the seller's own answer.
        # Confirmed via scripts/eval_ocr_error_decomposition.py: this was the sole
        # cause of every blank-works-line misread in a 25-form scanned-route test.
        leading_marker = re.match(r"^\(?[b-d]\)|^year|^\[|^yes\b|^no\b", cand, re.I)
        boilerplate_anywhere = re.search(r"change of use", cand, re.I)
        if len(cand) > 8 and not leading_marker and not boilerplate_anywhere:
            works_text = cand
    if not works_text:
        m2 = re.search(r"([A-Z][^\n]{0,90}?(?:carried out|converted|erected|installed)[^\n]*)", txt)
        if m2 and "e.g." not in m2.group(1).lower():
            works_text = m2.group(1).strip()

    # --- 4.2: did the seller provide supporting documents / an explanation?
    # Strip the STANDARD printed prompt; anything left is the seller's own answer.
    # (Principled version: diff the completed form against the blank template.) ---
    def find(*pats):
        for p in pats:
            mm = re.search(p, txt, re.I)
            if mm:
                return mm
        return None
    s = find(r"exempt from building regulations", r"4\.2\b")
    e = find(r"further information", r"unfinished", r"are any of the works", r"4\.3\b")
    sec_support = txt[s.end():e.start()] if (s and e and e.start() > s.end()) else ""
    # Strip ONLY the fixed trailing boilerplate (and any 4.3 stub that leaked in).
    # We deliberately do NOT keyword-strip "building regulations" / "permitted
    # development": sellers legitimately use those words in their own 4.2 answers,
    # and over-stripping them was a measured source of false positives (see Results).
    strip = ("further information", "http", "planning portal", "planningportal",
             "200126", "/applications", "are any of the works", "disclosed in", "unfinished")
    substantive = [l.strip() for l in sec_support.splitlines()
                   if len(l.strip()) >= 6 and not any(s in l.strip().lower() for s in strip)]
    supported = len(substantive) > 0

    return {
        "source": source,
        "alterations_made": {
            "answer": "Yes" if works_text else "Unknown",
            "works": works_text,
            "explanation": " ".join(substantive),
        },
        "building_regs_completion_certificate": {
            "answer": "Unknown", "attachment_provided": supported,
        },
    }


def classify_pdf(pdf_path: str) -> str:
    """Router — decide how a TA6 arrived, so the right extractor is used:
        'acroform' : fillable digital form  -> read fields directly (no OCR)
        'text'     : digital PDF w/ text layer -> parse text
        'scanned'  : image-only PDF -> OCR then parse."""
    try:
        from pypdf import PdfReader
        if (PdfReader(pdf_path).get_fields() or {}):
            return "acroform"
    except Exception:
        pass
    body = re.sub(r"\s", "", _pdftext(pdf_path))
    return "text" if len(body) > 200 else "scanned"


def ocr_pdf_to_text(pdf_path: str, dpi: int = 300) -> str:
    """OCR route: rasterise the scan and read it with Tesseract."""
    import tempfile, glob
    d = tempfile.mkdtemp()
    try:
        subprocess.run(["pdftoppm", "-png", "-r", str(dpi), pdf_path, os.path.join(d, "pg")],
                       check=True, capture_output=True)
    except FileNotFoundError:
        raise RuntimeError("'pdftoppm' not found. Install Poppler — "
                           "macOS: brew install poppler · Ubuntu: sudo apt install poppler-utils")
    parts = []
    for img in sorted(glob.glob(os.path.join(d, "pg*.png"))):
        try:
            parts.append(subprocess.run(["tesseract", img, "stdout", "--psm", "6"],
                                         capture_output=True, text=True).stdout)
        except FileNotFoundError:
            raise RuntimeError("'tesseract' not found. Install Tesseract — "
                               "macOS: brew install tesseract · Ubuntu: sudo apt install tesseract-ocr")
    return "\n".join(parts)


def extract_ta6(pdf_path: str) -> Dict:
    """Unified Stage-1 entry point: detect the input type and dispatch to the right
    route, returning ONE structured record so the SAME checks run on digital or scan."""
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(
            f"TA6 file not found: {pdf_path}\n"
            '  Pass a real PDF path, e.g.  python scripts/run_real.py "/full/path/to/TA6.pdf"')
    route = classify_pdf(pdf_path)
    if route == "acroform":
        rec = _parse_section4("", pdf_path)        # field->section mapping is v1
        rec["route"] = "acroform_digital"
        rec["acroform_field_count"] = len(extract_acroform(pdf_path))
        return rec
    txt = _pdftext(pdf_path) if route == "text" else ocr_pdf_to_text(pdf_path)
    rec = _parse_section4(txt, pdf_path)
    rec["route"] = "text_digital" if route == "text" else "ocr_scanned"
    return rec


def extract_real_ta6_section4(pdf_path: str) -> Dict:   # backward-compatible wrapper
    return {**_parse_section4(_pdftext(pdf_path), pdf_path), "route": "text_digital"}


# ============================================================================
# STAGE 2 — CONSISTENCY / CONTRADICTION DETECTION
# ============================================================================
YESNO_DETAIL_FIELDS = ["disputes_or_complaints", "notices_received", "flooding"]


def run_rule_checks(rid: str, ta6: Dict) -> List[Issue]:
    """Deterministic, single-field checks. No ML required — and honestly labelled
    as such, which is the point: rules handle what rules handle."""
    issues: List[Issue] = []

    # (a) "Yes" answered but the details box left blank.
    for fname in YESNO_DETAIL_FIELDS:
        f = ta6.get(fname)
        if isinstance(f, dict) and f.get("answer") == "Yes" and not str(f.get("details", "")).strip():
            issues.append(Issue(rid, fname, "missing_detail", "rule",
                                f"Section '{fname}' answered 'Yes' but no details were provided."))

    cert = ta6.get("building_regs_completion_certificate", {}) or {}
    alt = ta6.get("alterations_made", {}) or {}

    # (b) Certificate declared to exist but not attached.
    if cert.get("answer") == "Yes" and cert.get("attachment_provided") is False:
        issues.append(Issue(rid, "building_regs_completion_certificate", "missing_attachment", "rule",
                            "A building regulations completion certificate is stated to exist but was not attached."))

    # (c) Works described but no completion certificate or exemption explanation.
    works = str(alt.get("works", "")).strip()
    has_works = alt.get("answer") == "Yes" and bool(works) and works.lower() not in ("none",)
    supported = cert.get("attachment_provided") is True or bool(str(alt.get("explanation", "")).strip())
    if has_works and not supported and cert.get("answer") != "Yes":
        issues.append(Issue(rid, "alterations_made", "works_without_support", "rule",
                            f"Alterations are described ('{alt.get('works')}') but no completion certificate "
                            f"or exemption explanation is provided."))
    return issues


def run_structured_crosscheck(rid: str, ta6: Dict, planning: Optional[Dict]) -> List[Issue]:
    """Cross-document check where BOTH sides are structured (e.g. synthetic data,
    or once the supporting docs are parsed). The free-text version of this is the
    LLM-NLI job below."""
    issues: List[Issue] = []
    alt = ta6.get("alterations_made", {}) or {}
    apps = (planning or {}).get("applications", []) or []
    if alt.get("answer") == "No" and apps:
        a = apps[0]
        issues.append(Issue(rid, "alterations_made | planning_extract", "cross_doc_contradiction",
                            "structured_crosscheck",
                            f"TA6 states no alterations, but the planning record shows "
                            f"{a.get('description')} (ref {a.get('reference')}, {a.get('decision')} "
                            f"{a.get('decision_year')})."))
    return issues


def detect_free_text_contradiction(rid: str, claims: List[str],
                                   supporting: Dict[str, str], backend: str = None) -> List[Issue]:
    """Free-text, cross-document contradiction detection via document-level NLI.
    Delegates to nli.py (Ollama / Anthropic / offline stub) and wraps each
    contradiction as a pipeline Issue with a ready enquiry."""
    from ta6 import nli
    issues = []
    for cx in nli.detect_contradictions(claims, supporting, backend=backend):
        iss = Issue(rid, f"TA6 free text | {cx.document}", "cross_doc_contradiction",
                    f"nli:{cx.backend}",
                    f"The TA6 states “{cx.claim}”, but the {cx.document} indicates otherwise: "
                    f"“{cx.evidence}”.")
        iss.enquiry = (f"The seller's Property Information Form states: “{cx.claim}” However, the "
                       f"{cx.document} indicates otherwise (“{cx.evidence}”). Please confirm the "
                       f"position and, where relevant, supply the planning permission and building "
                       f"regulations completion certificate for our review.")
        issues.append(iss)
    return issues


# ============================================================================
# STAGE 3 — ENQUIRY GENERATION
# ============================================================================
_TEMPLATES = {
    "missing_detail":
        ("The Property Information Form indicates '{field}' was answered affirmatively, but no details "
         "have been provided. Please provide full details, including dates and the parties involved, and "
         "confirm whether the matter has been resolved."),
    "missing_attachment":
        ("The seller has confirmed that a building regulations completion certificate is available, but a "
         "copy has not been supplied. Please provide a copy of the certificate for our review."),
    "works_without_support":
        ("The Property Information Form discloses that works have been carried out at the property "
         "({works}), but no building regulations completion certificate or explanation of why one was not "
         "required has been provided. Please supply the relevant planning permission and building "
         "regulations completion certificate, or confirm the basis on which these were not required."),
    "cross_doc_contradiction":
        ("The Property Information Form states that no alterations have been carried out at the property. "
         "However, {detail} Please confirm whether these works were carried out and, if so, supply the "
         "planning permission and building regulations completion certificate for our review."),
}


def generate_enquiry(issue: Issue, ta6: Dict) -> str:
    """Templated, cited baseline generation (runs offline). The LLM version below
    produces the same content in a firm-specific register."""
    alt = (ta6 or {}).get("alterations_made", {}) or {}
    t = _TEMPLATES.get(issue.issue_type, "Please clarify the matter identified at '{field}'.")
    return t.format(field=issue.field.replace("_", " "),
                    works=alt.get("works", "the works described"),
                    detail=issue.description.split("However,")[-1].strip() or issue.description)


def generate_enquiry_llm(issue: Issue, ta6: Dict, context: str = ""):   # LLM-HOOK
    """Controlled, cited generation in the firm's register, constrained to the
    detected issue and its source field. Runs locally with an API key."""
    raise NotImplementedError("Wire an LLM here to run locally (see proposal: Instructor/Outlines).")


# ============================================================================
# Convenience: run all detectors on a structured record.
# ============================================================================
def analyse(rid: str, ta6: Dict, planning: Optional[Dict] = None) -> List[Issue]:
    issues = run_rule_checks(rid, ta6) + run_structured_crosscheck(rid, ta6, planning)
    for iss in issues:
        iss.enquiry = generate_enquiry(iss, ta6)
    return issues
