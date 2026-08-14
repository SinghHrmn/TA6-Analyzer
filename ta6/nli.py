"""
Cross-document contradiction detection  (the dissertation's ML contribution)
============================================================================
Document-level NLI: given a CLAIM taken from the TA6 free text and a SUPPORTING
document (title report / EPC / search / planning), decide whether the document
ENTAILS, CONTRADICTS, or is NEUTRAL to the claim, with an evidence span.
(Formulation follows ContractNLI; Koreeda & Manning, 2021.)

Three interchangeable backends — pick with TA6_NLI_BACKEND or backend=...:
  * "anthropic" : real LLM (needs ANTHROPIC_API_KEY). Set ANTHROPIC_MODEL to a
                  model your account supports.
  * "ollama"    : local model at http://localhost:11434 (free, private, offline).
  * "stub"      : deterministic lexical fallback (negation vs. keyword). No model.
                  Lets the whole pipeline run with zero setup; clearly labelled.
"stub" is a BASELINE, not the contribution — report it as such and compare the
LLM backend against it in your results chapter.
"""
from dataclasses import dataclass, asdict
from typing import List, Dict
import os, re, json


@dataclass
class Contradiction:
    claim: str
    document: str
    label: str          # entail | contradict | neutral
    evidence: str       # span from the supporting document
    confidence: float
    backend: str


# --------------------------------------------------------------------------
# Shared low-level model call -- ANY prompt in, raw text out. Both this
# module's contradiction detection AND ta6.pipeline.generate_enquiry_llm()
# call through this one function, so there is exactly one place that knows
# how to reach Ollama/Anthropic, not two copies of the same HTTP/client code.
# 12 Aug 2026: extracted from the two backend functions below (dissertation
# audit Goal A3) -- behaviour-preserving, re-verified against
# scripts/eval_nli.py after the refactor (same F1 on the stub backend).
# --------------------------------------------------------------------------
def call_model(prompt: str, backend: str = None, max_tokens: int = 400,
               json_mode: bool = True) -> str:
    """Call the resolved backend with an arbitrary prompt; return the raw text
    response. Raises on an unreachable/misconfigured backend rather than
    silently returning an empty string -- callers decide how to handle that."""
    be = _resolve(backend)
    if be == "anthropic":
        from anthropic import Anthropic
        client = Anthropic()  # reads ANTHROPIC_API_KEY
        model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
        msg = client.messages.create(model=model, max_tokens=max_tokens,
                                     messages=[{"role": "user", "content": prompt}])
        return msg.content[0].text
    if be == "ollama":
        import urllib.request
        model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        body = json.dumps({"model": model, "format": "json" if json_mode else "",
                           "stream": False, "prompt": prompt}).encode()
        req = urllib.request.Request("http://localhost:11434/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        resp = json.loads(urllib.request.urlopen(req, timeout=120).read())
        return resp["response"]
    raise ValueError(f"call_model() has no real-model implementation for backend {be!r} "
                     f"(stub is a deterministic fallback with no generative capability -- "
                     f"there is nothing for it to call).")


def _parse_json_object(raw: str, fallback: dict) -> dict:
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        return json.loads(m.group(0)) if m else dict(fallback)
    except json.JSONDecodeError:
        return dict(fallback)


# --------------------------------------------------------------------------
# Backend 1 — Anthropic API
# --------------------------------------------------------------------------
_PROMPT = """You are assisting a conveyancing solicitor. Decide the relationship
between a CLAIM from a seller's TA6 Property Information Form and a SUPPORTING
DOCUMENT from the same transaction.

CLAIM: {claim}

SUPPORTING DOCUMENT ({doc_name}):
{doc_text}

Reply with ONLY a JSON object:
{{"label": "entail|contradict|neutral", "evidence": "<quote the smallest span from the document that justifies your decision, or empty>", "confidence": <0-1>}}
"contradict" means the document shows the claim is false or incomplete."""


def _anthropic(claim, doc_name, doc_text):
    raw = call_model(_PROMPT.format(claim=claim, doc_name=doc_name, doc_text=doc_text[:6000]),
                     backend="anthropic", max_tokens=300)
    d = _parse_json_object(raw, {"label": "neutral", "evidence": "", "confidence": 0.0})
    return d["label"], d.get("evidence", ""), float(d.get("confidence", 0.5))


# --------------------------------------------------------------------------
# Backend 2 — local Ollama
# --------------------------------------------------------------------------
def _ollama(claim, doc_name, doc_text):
    raw = call_model(_PROMPT.format(claim=claim, doc_name=doc_name, doc_text=doc_text[:6000]),
                     backend="ollama")
    d = _parse_json_object(raw, {"label": "neutral", "evidence": "", "confidence": 0.0})
    return d["label"], d.get("evidence", ""), float(d.get("confidence", 0.5))


# --------------------------------------------------------------------------
# Backend 3 — offline deterministic stub (lexical negation baseline)
# --------------------------------------------------------------------------
_NEG = re.compile(r"\b(no|not|none|never|without|are not|has not|have not|hasn't|haven't|no such)\b", re.I)
_TOPICS = {
    "alteration": ["extension", "loft conversion", "garage conversion", "conservatory",
                   "outbuilding", "alteration", "porch", "annexe"],
    "dispute":    ["dispute", "complaint", "boundary dispute", "notice of"],
    "flood":      ["flood", "flooding", "surface water"],
    "planning":   ["planning permission", "enforcement notice", "breach of condition"],
}


def _stub(claim, doc_name, doc_text):
    c, d = claim.lower(), doc_text.lower()
    if not _NEG.search(c):
        return "neutral", "", 0.3
    for _topic, kws in _TOPICS.items():
        if any(k in c for k in kws) or any(k.split()[0] in c for k in kws):
            for k in kws:
                if k in d:
                    span = re.search(r"[^.]*" + re.escape(k) + r"[^.]*\.", doc_text, re.I)
                    return "contradict", (span.group(0).strip() if span else k), 0.6
    return "neutral", "", 0.3


_BACKENDS = {"anthropic": _anthropic, "ollama": _ollama, "stub": _stub}


def _resolve(backend):
    backend = backend or os.getenv("TA6_NLI_BACKEND", "auto")
    if backend != "auto":
        return backend
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "stub"


def detect_contradictions(claims: List[str], supporting: Dict[str, str],
                          backend: str = None) -> List[Contradiction]:
    """Check each claim against each supporting document; return contradictions."""
    be = _resolve(backend)
    fn = _BACKENDS[be]
    out = []
    for claim in claims:
        for doc_name, doc_text in supporting.items():
            try:
                label, evidence, conf = fn(claim, doc_name, doc_text)
            except Exception as e:
                label, evidence, conf = "error", str(e)[:80], 0.0
            if label == "contradict":
                out.append(Contradiction(claim, doc_name, label, evidence, conf, be))
    return out


if __name__ == "__main__":
    # Demo: a TA6 that declares no alterations, checked against real-style docs.
    claims = ["No alterations, extensions or other works have been carried out at the property."]
    supporting = {
        "Local planning record": "Application 19/0421/HH — erection of a single-storey rear extension "
                                  "— Decision: Granted, 2019. No enforcement notices recorded.",
        "EPC": "Walls: solid brick, as built. Roof: pitched, insulated. A rear extension is present.",
        "Local Authority search": "No adverse entries. Building control completion certificate on file.",
    }
    be = _resolve(None)
    print(f"backend = {be}\n")
    for cx in detect_contradictions(claims, supporting):
        print(f"[CONTRADICTION vs {cx.document}]  conf={cx.confidence}")
        print(f"   claim   : {cx.claim}")
        print(f"   evidence: {cx.evidence}\n")
