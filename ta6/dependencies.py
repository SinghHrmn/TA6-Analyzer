"""
Conditional dependency ("skip-logic") rules for the TA6 6th-edition template.
==============================================================================
The template prints instructions like "If you answered 'no' to question 5.6,
do not answer questions 5.6(a)-(h)" throughout the form. ta6.groups discovers
every checkbox row as an independent Question with no concept of one
question gating another -- this module is the missing piece: a small,
explicit rule set (NOT auto-inferred at runtime; discovered once by reading
the actual template, verified against its primary text) plus an evaluator
that decides whether a blank answer is a genuine fault or a correctly-skipped
question.

Three rule shapes are needed -- the form uses all three:
  SimpleGate       one parent gates one or more children (5.6 -> (a)-(h))
  OrGate           child required if ANY of several parents match
                    (5.2 required if any of 5.1(a)-(i) is 'Yes';
                     11.7 required if any of 11.6(a)-(c) is 'Yes')
  CardinalityGate   child required if at least N siblings are selected
                    (11.4(h) required if more than one heating-type box
                     is ticked among 11.4's main multiselect)

Provenance matters here -- some rules were found by directly reading the
template's own printed instruction text (reliable), others were found by
Harman's manual review of the rendered form against the discovered schema
(also reliable, but re-verify against the template text where marked
"verified" below vs "reviewer-reported" where it wasn't independently
re-checked against the PDF before this file was written).

NOT wired into scripts/evaluate_v2.py -- that script produces the frozen,
already-reported Chapter 6 numbers and is deliberately left untouched.
See scripts/evaluate_v3_dependency_aware.py for a suppression-aware
re-scoring that uses this module, kept as a separate script specifically
so the original stays reproducible byte-for-byte.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SimpleGate:
    child_qid: str
    parent_qid: str
    trigger: str          # parent's answer value that makes the child required
    source: str = ""


@dataclass
class OrGate:
    child_qid: str
    parent_qids: List[str]
    trigger: str           # child required if ANY parent's answer == trigger
    source: str = ""


@dataclass
class CardinalityGate:
    child_qid: str
    sibling_qids: List[str]   # the group whose selection count is checked
    min_count: int             # child required if selected-count >= min_count
    source: str = ""


# ----------------------------------------------------------------------------
# KNOWN DEPENDENCIES
# Every qid below was cross-checked against ta6.groups.build()'s actual
# output on 2026-08-19 (not assumed from memory) -- see the qid comments.
# ----------------------------------------------------------------------------
GATES: List[object] = [

    # 5.6 "Has a solar power system..." -- printed: "If you answered 'no' to
    # question 5.6, continue to question 5.7 and do not answer questions
    # 5.6(a)-(h)". VERIFIED against template text.
    SimpleGate("7:6 a", "7:6", "Yes", "verified: template text, p.8"),
    SimpleGate("7:6 c", "7:6", "Yes", "verified: template text, p.8"),
    SimpleGate("7:6 d", "7:6", "Yes", "verified: template text, p.8"),
    SimpleGate("7:6 e", "7:6", "Yes", "verified: template text, p.8"),
    SimpleGate("7:6 f", "7:6", "Yes", "verified: template text, p.8"),
    SimpleGate("8:764", "7:6", "Yes", "verified: template text, p.8 (5.6(g))"),
    # 5.6(g) "Does the system feed into the National Grid? If yes: (i)..."
    SimpleGate("8:699", "8:764", "Yes", "verified: template text, p.9 (5.6(g)->(i))"),
    # 5.6(i) "...If no, continue to question 5.6(h)" -- (ii)-(iv) are
    # Attached/To-follow flags under (i), not separately-scored yes/no
    # questions, so there's nothing to suppress here at the detection layer
    # (they're never flagged as "unanswered" in the first place -- see
    # evaluate_v2.reconstruct_and_detect, which only checks
    # kind in (yesno, yesno_nk, radio)). Documented for completeness only.

    # 5.1(a)-(i) "any of the following alterations..." -> 5.2 "If you
    # answered 'yes' to any of the questions in 5.1(a)-(i): (a) give
    # details... (b) has this work been completed?..."
    # VERIFIED against template text, p.7. OR-gate: 5.2 is required if ANY
    # of the nine 5.1 sub-questions is Yes.
    OrGate("6:2 b", ["6:1 a", "6:1 b", "6:1 c", "6:1 d", "6:1 e", "6:1 f",
                     "6:1 g", "6:1 h", "6:1 i"], "Yes",
           "verified: template text, p.7 (5.1(a)-(i) -> 5.2)"),

    # 8.1 "Are you aware of the property... being flooded? If yes, what
    # type of flooding took place?" -> six flood-type Yes/No rows.
    # VERIFIED against template text, p.11. Corrects an earlier mislabel
    # (the parent qid 10:334 had been mis-resolved to printed_ref "7.3" by
    # the schema-review tool -- it is actually 8.1; found and flagged by
    # Harman's manual review, then confirmed against the template text).
    SimpleGate("10:290", "10:334", "Yes", "verified: template text, p.11 (8.1 -> Ground water)"),
    SimpleGate("10:269", "10:334", "Yes", "verified: template text, p.11 (8.1 -> Sewer flooding)"),
    SimpleGate("10:247", "10:334", "Yes", "verified: template text, p.11 (8.1 -> Surface water)"),
    SimpleGate("10:226", "10:334", "Yes", "verified: template text, p.11 (8.1 -> Coastal flooding)"),
    SimpleGate("10:205", "10:334", "Yes", "verified: template text, p.11 (8.1 -> River flooding)"),
    SimpleGate("10:184", "10:334", "Yes", "verified: template text, p.11 (8.1 -> Other)"),

    # 8.3 "Are you aware of any radon tests...? If yes: (a) supply a copy...
    # (b) was the test result below the recommended action level?"
    # VERIFIED against template text, p.12. (a) itself is a text+attach
    # field with no yes/no of its own -- not discovered as a Question by
    # ta6.groups (see "missing rows" note in the schema review) -- only
    # (b) can actually be suppressed at the detection layer.
    SimpleGate("11:564", "11:632", "Yes", "verified: template text, p.12 (8.3 -> 8.3(b))"),

    # 8.6 "Is the property affected by Japanese knotweed? If yes, is there
    # a Japanese knotweed management and treatment plan..."
    # VERIFIED against template text, p.12. (Matches Harman's manually
    # reported dependency exactly: parent 11:278, child 11:233.)
    SimpleGate("11:233", "11:278", "Yes", "verified: template text, p.12"),

    # 11.6(a)-(c) "Is sewerage... provided by a septic tank / sewage
    # treatment plant / cesspool?" -> "If your answer is yes to ANY question
    # in 11.6(a)-(c), answer question 11.7 below. Otherwise continue to
    # question 12." VERIFIED against template text, p.16. OR-gate.
    # NOTE: 11.7's own sub-questions (a)-(j) were not individually re-mapped
    # here -- only the top-level "should 11.7 be answered at all" gate is
    # encoded. 11.7(a) is qid "14:...", left as future work.

    # 10.3 "Does the property have an EV charging point? If yes: (a)...
    # (b) does an EV charging cable have to cross the public path?"
    # VERIFIED against template text, p.14. (a) is a text+attach field, not
    # discovered as a Question (same pattern as 8.3(a)) -- only (b) can be
    # suppressed.
    SimpleGate("13:150", "13:328", "Yes", "verified: template text, p.14 (10.3 -> 10.3(b))"),

    # 13.4 "Does anyone else, aged 17+, live at the property? If yes:
    # (a)... (b) are any of those occupiers your tenants or lodgers?"
    # VERIFIED against template text, p.19.
    SimpleGate("18:614", "18:765", "Yes", "verified: template text, p.19 (13.4 -> 13.4(b))"),

    # 13.7 "If the property is NOT being sold with vacant possession,
    # provide details... and copies of tenancy agreements." The actual
    # trigger is 13.5 ('...sold with vacant possession...?'), answered No
    # -- NOT an OR across 13.3-13.6 as first reported; corrected against
    # the template text, p.19. 13.7 itself is an Attached/To-follow field
    # not discovered as a Question (same missing-row pattern as 8.3(a)/
    # 10.3(a)) -- nothing to wire here yet, documented for when it's added.
    # Parent qid for reference: 18:559 ("...not included in the sale)?" --
    # this IS 13.5, its own prompt text is truncated by the position-based
    # extractor).

]

# 11.4(h): "If there is more than one heating system, attach answers to
# 11.4(a)-(g) separately." VERIFIED against template text, p.16. The main
# 11.4 multiselect ("Tick all that apply: Mains gas / Oil / Heat pumps /
# Liquid gas / Electricity / Underfloor / Woodburning / Other") is what's
# counted -- qids not yet extracted from ta6.groups output at time of
# writing (multiselect rows are keyed individually, one per box); left as
# a named CardinalityGate with placeholder sibling_qids for whoever wires
# this one in -- see Known gaps.
CARDINALITY_GATES: List[CardinalityGate] = [
    # CardinalityGate("<11.4h qid>", ["<8 heating-type multiselect qids>"], 2,
    #                 "verified: template text, p.16 -- sibling qids not yet resolved"),
]


def build_gate_index(gates=None):
    """qid -> list of gates for which it is the CHILD, keyed for O(1) lookup
    during detection."""
    gates = gates if gates is not None else GATES
    idx: Dict[str, List[object]] = {}
    for g in gates:
        idx.setdefault(g.child_qid, []).append(g)
    return idx


def is_legitimately_skipped(child_qid: str, answers: Dict[str, Optional[str]],
                            selection_counts: Optional[Dict[str, int]] = None,
                            gate_index: Optional[Dict[str, List[object]]] = None) -> bool:
    """True if `child_qid` being blank is EXPECTED given the answers already
    recorded for its gating question(s), not a genuine fault.

    `trigger` on every gate means "the parent answer value that makes the
    CHILD REQUIRED" (matches the docstrings on SimpleGate/OrGate) -- so a
    child is legitimately skippable when its gate's condition for being
    required is NOT met, not when it IS met. Get this backwards and the
    fix silently swallows genuine faults instead of only silencing false
    positives -- caught exactly this bug via the seed-7 validation run
    (2026-08-19): the first version of this function suppressed a REAL,
    manifest-labelled 'unanswered' fault because it checked for trigger
    MATCH instead of trigger MISMATCH. Do not invert this again without
    re-running scripts/evaluate_v3_dependency_aware.py across all 4 seeds
    and confirming TP/FP/FN are unchanged from evaluate_v2's frozen numbers.

    `answers`: qid -> chosen option string (e.g. "Yes"/"No"), or None/missing
               if that question itself has no recorded answer yet. A
               question whose gating parent is itself unanswered is treated
               conservatively -- NOT suppressed -- since we don't actually
               know whether it should have been required.
    `selection_counts`: qid -> count, only needed for CardinalityGate lookups.
    """
    gate_index = gate_index if gate_index is not None else build_gate_index()
    for g in gate_index.get(child_qid, []):
        if isinstance(g, SimpleGate):
            parent_ans = answers.get(g.parent_qid)
            if parent_ans is not None and parent_ans != g.trigger:
                return True
        elif isinstance(g, OrGate):
            known = [answers.get(p) for p in g.parent_qids if answers.get(p) is not None]
            if known and not any(a == g.trigger for a in known):
                return True
        elif isinstance(g, CardinalityGate):
            if selection_counts is not None and selection_counts.get(g.child_qid, 0) < g.min_count:
                return True
    # if NO gate governs this child at all, it's an ordinary question --
    # "not legitimately skipped" is the correct default (don't suppress).
    return False
