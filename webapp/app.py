"""
TA6 Analyser — Flask web app  (dissertation demo / evaluation harness)
======================================================================
Wraps the pipeline (ta6/pipeline.py + ta6/nli.py) behind a web UI: upload a
TA6 (+ optional supporting documents), auto-route (digital / scanned /
fillable), extract, show exactly what was extracted field-by-field, detect
issues (rules + cross-document NLI), generate enquiries, and download them.
Every run is logged to runs.jsonl.

11 Aug 2026: added the "Extracted fields" table below -- upload a form and
see what the pipeline actually read before looking at the issues it raised,
so extraction correctness can be checked visually, form by form, rather than
only trusting the downstream flags. IMPORTANT SCOPE NOTE, shown in the UI
itself: rule-based detection (run_rule_checks) only ever reads Section 4
(alterations/building control) — that is genuinely all ta6.pipeline extracts
for the "text" and "scanned" routes. For a truly fillable ("acroform") PDF
upload, the field table below instead shows a full per-question read-back via
ta6.acroform.extract_record (the same mechanism verified in
scripts/evaluate_v2.py), but that fuller read-back is NOT yet wired into the
rule engine — it is shown for visual verification only. Don't let the table
being long for a fillable form imply the rules are checking all of it; they
are not, yet.

Run:
    pip install flask
    export TA6_NLI_BACKEND=ollama      # or leave unset for the offline baseline
    python app.py                      # -> http://127.0.0.1:5000
"""
import os, sys, json, tempfile, datetime
from flask import Flask, request, render_template_string, send_file, abort

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ta6.pipeline import (extract_ta6, run_rule_checks,
                          detect_free_text_contradiction, generate_enquiry, _pdftext)
from ta6.acroform import extract_record as acroform_extract_record

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024  # 40 MB
RUN_LOG = os.path.join(os.path.dirname(__file__), "runs.jsonl")
SEVERITY = {"cross_doc_contradiction": "High", "works_without_support": "Medium",
            "missing_attachment": "Medium", "missing_detail": "Medium"}
ROUTE_LABEL = {"text_digital": "digital PDF (text layer)", "ocr_scanned": "scanned paper (OCR)",
               "acroform_digital": "digital fillable form"}


def claims_from_record(rec):
    """Turn the extracted TA6 into checkable claims for cross-document NLI."""
    claims = []
    alt = rec.get("alterations_made", {})
    if alt.get("answer") in ("No", "Unknown"):
        claims.append("No alterations, extensions or other works have been carried out at the property.")
    return claims


def build_field_table(rec, ta6_path):
    """Every field the pipeline extracted, with what it read for it -- the
    visual check requested 11 Aug 2026: upload a form, see the field-by-field
    read-back before looking at the issues raised from it.

    Returns (rows, note) where rows is a list of {field, value} dicts and
    note is an honest caption explaining what scope this table covers for
    the route that was actually used (see module docstring)."""
    route = rec.get("route")

    if route == "acroform_digital":
        # The Section-4-only parser can't read a live fillable form (it's
        # handed empty text upstream) -- fall back to the full per-question
        # AcroForm read-back instead, so a fillable-form upload doesn't just
        # show blanks. This is the same mechanism verified in evaluate_v2.py.
        try:
            full = acroform_extract_record(ta6_path)
        except Exception as e:
            return ([{"field": "(error reading AcroForm fields)", "value": str(e)}],
                    "Could not read this form's fields — see error above.")
        rows = [{"field": f"Header: {k}", "value": v} for k, v in full.get("header", {}).items()]
        for a in full.get("answers", []):
            label = a["question"] or f"Question {a['q']}"
            val = a["answer"]
            if a["details"]:
                val += f"  —  “{a['details']}”"
            rows.append({"field": f"Q{a['q']}: {label}", "value": val})
        note = (f"Fillable-form upload: showing the full per-question read-back "
                f"({len(full.get('answers', []))} answered questions found) via the AcroForm route. "
                f"Rule-based detection below still only checks the Section 4 fields, not this full "
                f"table — that integration doesn't exist yet.")
        return rows, note

    # text_digital / ocr_scanned: this really is all ta6.pipeline extracts.
    alt = rec.get("alterations_made", {}) or {}
    cert = rec.get("building_regs_completion_certificate", {}) or {}
    rows = [
        {"field": "Section 4.1: alterations made?", "value": alt.get("answer", "?")},
        {"field": "Section 4.1: works described", "value": alt.get("works", "") or "(none read)"},
        {"field": "Section 4.2: explanation / exemption text", "value": alt.get("explanation", "") or "(none read)"},
        {"field": "Building regs completion certificate: answer", "value": cert.get("answer", "?")},
        {"field": "Building regs completion certificate: attachment provided?",
         "value": "Yes" if cert.get("attachment_provided") else "No"},
    ]
    note = ("This pipeline currently extracts Section 4 (alterations, planning and building "
            "control) only — that is the true scope of ta6.pipeline for digital-text and scanned "
            "input, not a display limitation. See Chapter 7 future work for full-form extraction.")
    return rows, note


PAGE = """<!doctype html><html><head><meta charset=utf-8><title>TA6 Analyser</title>
<style>
 body{font:15px/1.5 system-ui,Segoe UI,sans-serif;margin:0;background:#f4f7fb;color:#1b2a44}
 .wrap{max-width:900px;margin:0 auto;padding:28px}
 h1{font-size:24px;margin:0 0 2px} .sub{color:#5c6b82;margin:0 0 22px}
 .card{background:#fff;border:1px solid #dfe7f2;border-radius:10px;padding:20px;margin-bottom:18px}
 label{font-weight:600;display:block;margin:10px 0 4px} input[type=file]{width:100%}
 button{background:#13233f;color:#fff;border:0;border-radius:7px;padding:11px 20px;font-size:15px;cursor:pointer;margin-top:14px}
 table{border-collapse:collapse;width:100%;font-size:14px} th,td{border:1px solid #e2e8f2;padding:8px 10px;text-align:left;vertical-align:top}
 th{background:#13233f;color:#fff} tr:nth-child(even) td{background:#f6f9fd}
 .pill{display:inline-block;padding:2px 9px;border-radius:20px;font-size:12px;font-weight:700}
 .High{background:#fce8e6;color:#9c2b2b}.Medium{background:#fdf3e0;color:#8a5a08}
 .enq{background:#f6f9fd;border-left:3px solid #2e75b6;padding:10px 12px;border-radius:5px;margin:6px 0;font-size:14px}
 .meta{color:#5c6b82;font-size:13px} .none{color:#1e6b3a;font-weight:600}
 a.btn{display:inline-block;margin-top:8px;color:#13233f;font-weight:600}
</style></head><body><div class=wrap>
<h1>TA6 Analyser</h1>
<p class=sub>AI-assisted review of seller-disclosure (TA6) forms &middot; assistive tool &mdash; a solicitor reviews every output.</p>

{% if not result %}
<form class=card method=post action="/analyse" enctype=multipart/form-data>
  <label>TA6 form (PDF &mdash; digital or scanned)</label>
  <input type=file name=ta6 accept=application/pdf required>
  <label>Supporting documents (optional &mdash; title report, EPC, search, planning)</label>
  <input type=file name=supporting multiple accept=application/pdf>
  <button type=submit>Analyse form</button>
  <p class=meta style="margin-top:14px">Detection backend: <b>{{ backend }}</b></p>
</form>
{% else %}
<div class=card>
  <b>Input:</b> {{ result.filename }} &nbsp;&middot;&nbsp; auto-classified as <b>{{ result.route }}</b><br>
  <span class=meta>Alterations declared: <b>{{ result.alterations }}</b>{% if result.works %} &mdash; &ldquo;{{ result.works }}&rdquo;{% endif %}</span>
</div>
<div class=card>
  <h3 style="margin-top:0">Extracted fields ({{ result.field_rows|length }})</h3>
  <p class=meta style="margin-top:-6px">{{ result.field_note }}</p>
  <table><tr><th style="width:45%">Field</th><th>What the pipeline read</th></tr>
  {% for r in result.field_rows %}
  <tr><td>{{ r.field }}</td><td>{{ r.value }}</td></tr>
  {% endfor %}</table>
</div>
<div class=card>
  <h3 style="margin-top:0">Issues flagged: {{ result.issues|length }}</h3>
  {% if result.issues %}
  <table><tr><th>Severity</th><th>Type</th><th>Detected by</th><th>Finding</th></tr>
  {% for i in result.issues %}
  <tr><td><span class="pill {{ i.severity }}">{{ i.severity }}</span></td>
      <td>{{ i.issue_type }}</td><td>{{ i.detection_method }}</td><td>{{ i.description }}</td></tr>
  {% endfor %}</table>
  <h3>Draft enquiries</h3>
  {% for i in result.issues %}<div class=enq>{{ i.enquiry }}</div>{% endfor %}
  <a class=btn href="/download/{{ result.token }}">&#8681; Download enquiries (.txt)</a>
  {% else %}<p class=none>No issues detected. (A solicitor should still review.)</p>{% endif %}
</div>
<a class=btn href="/">&#8592; Analyse another form</a>
{% endif %}
</div></body></html>"""

_DOWNLOADS = {}


@app.route("/")
def index():
    return render_template_string(PAGE, result=None, backend=os.getenv("TA6_NLI_BACKEND", "stub (offline)"))


@app.route("/analyse", methods=["POST"])
def analyse():
    f = request.files.get("ta6")
    if not f or not f.filename:
        abort(400, "No TA6 uploaded")
    tmp = tempfile.mkdtemp()
    ta6_path = os.path.join(tmp, f.filename)
    f.save(ta6_path)

    rec = extract_ta6(ta6_path)
    field_rows, field_note = build_field_table(rec, ta6_path)
    issues = run_rule_checks("web", rec)

    # optional supporting documents -> cross-document NLI
    supporting = {}
    for sf in request.files.getlist("supporting"):
        if sf and sf.filename:
            p = os.path.join(tmp, sf.filename); sf.save(p)
            supporting[sf.filename] = _pdftext(p)
    claims = claims_from_record(rec)
    if supporting and claims:
        issues += detect_free_text_contradiction("web", claims, supporting)

    for i in issues:
        if not i.enquiry:
            i.enquiry = generate_enquiry(i, rec)
        i.severity = SEVERITY.get(i.issue_type, "Medium")

    token = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    _DOWNLOADS[token] = "\n\n".join(f"{n}. {i.enquiry}" for n, i in enumerate(issues, 1)) or "No enquiries."
    with open(RUN_LOG, "a") as lg:
        lg.write(json.dumps({"ts": token, "file": f.filename, "route": rec.get("route"),
                             "n_supporting": len(supporting), "n_issues": len(issues)}) + "\n")

    result = {"filename": f.filename, "route": ROUTE_LABEL.get(rec.get("route"), rec.get("route")),
              "alterations": rec.get("alterations_made", {}).get("answer", "?"),
              "works": rec.get("alterations_made", {}).get("works", ""),
              "issues": issues, "token": token,
              "field_rows": field_rows, "field_note": field_note}
    return render_template_string(PAGE, result=result, backend=os.getenv("TA6_NLI_BACKEND", "stub (offline)"))


@app.route("/download/<token>")
def download(token):
    text = _DOWNLOADS.get(token)
    if text is None:
        abort(404)
    p = os.path.join(tempfile.gettempdir(), f"enquiries_{token}.txt")
    open(p, "w").write(text)
    return send_file(p, as_attachment=True, download_name="enquiries.txt")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
