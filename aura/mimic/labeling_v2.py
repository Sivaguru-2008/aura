"""Improved rule-based CXR report labeler (v2).

Fixes the failure modes measured in the retrain_v2 error analysis (vision_audit Phase 5):
  1. POST-modifier / temporal negation: "pneumothorax ... resolved", "no longer seen",
     "interval resolution of X" — v1 only looked BEFORE the concept, so trailing
     negations were missed (caused false-positive labels on resolved findings).
  2. Cardiomegaly synonyms: "heart size is prominent / mildly enlarged", "cardiac
     silhouette is enlarged/prominent" — v1 required the literal word "cardiomegaly".
  3. Opacity hierarchy: consolidation / pneumonia / infiltrate ARE opacities, so they
     set opacity=1 (v1 labelled them opacity-negative -> false-positive model errors).
  4. Tighter uncertainty scope: a definite finding ("mild cardiomegaly") is not made
     uncertain by an earlier "possibly" attached to a DIFFERENT finding.

Returns {finding_value: 1/0} using the training convention (present=1, uncertain/absent=0).
NOT the official Stanford CheXpert labeler (Python2/credentialed) — an improved,
transparent rule labeler validated against report reading.
"""
import re
from schemas.clinical import Finding

CONCEPTS = {
    "effusion":      r"effusion|pleural fluid",
    "consolidation": r"consolidat",
    "pneumonia":     r"pneumonia|pneumonic",
    "opacity":       r"opacit|opacification|infiltrat|airspace (?:disease|opacit)",
    "cardiomegaly":  (r"cardiomegaly"
                      r"|(?:cardiac silhouette|cardiomediastinal silhouette|heart|cardiac)"
                      r"(?:\s+\w+){0,3}\s+(?:enlarge\w*|prominent|prominence)"
                      r"|enlarge\w*\s+(?:of\s+)?(?:the\s+)?(?:cardiac|heart|cardiomediastin)"
                      r"|(?:mild|moderate|severe|marked)\w*\s+cardiomegaly"),
    "edema":         r"edema|oedema|vascular congestion|pulmonary congestion|fluid overload",
    "pneumothorax":  r"pneumothora",
    "nodule":        r"nodul|\bmass(?:es)?\b|neoplasm|malignan|metasta",
    "hyperinflation":r"hyperinflat|hyperexpan|emphysema|hyperlucen|flattened diaphragm|\bblebs?\b",
}
CONCEPT_TO_FINDING = {
    "opacity": Finding.OPACITY, "consolidation": Finding.CONSOLIDATION,
    "effusion": Finding.EFFUSION, "cardiomegaly": Finding.CARDIOMEGALY,
    "nodule": Finding.NODULE, "pneumothorax": Finding.PNEUMOTHORAX,
    "hyperinflation": Finding.HYPERINFLATION,
}

_NEG = re.compile(r"\b(no|not|without|never|free of|negative for|clear of|absence of|absent|"
                  r"ruled out|rule out|resolved|resolution of|no evidence|no sign|no definite|"
                  r"no new|unremarkable|removal of|removed|clear(?:ed)?)\b")
# negation that appears AFTER the concept ("pneumothorax has resolved")
_POST_NEG = re.compile(r"\b(resolved|resolution|cleared|no longer|has (?:been )?removed|"
                       r"not (?:seen|identified|present|visualized)|now absent|"
                       r"interval (?:resolution|improvement|clearance))\b")
_UNC = re.compile(r"\b(may|maybe|possible|possibly|could|cannot exclude|cannot be excluded|"
                  r"questionable|question of|suspicious|concerning for|likely|probable|"
                  r"probably|worrisome|differential|versus|\bvs\b|borderline|equivocal)\b")
# NOTE: comma is deliberately NOT a breaker — "no A, B, or C" must carry negation
# across the list to C (dropping this was a measured v2 regression).
_BREAK = re.compile(r"\b(but|however|except|although|though|otherwise|aside from|other than)\b|;")
_SENT = re.compile(r"[.!?\n]+")


def _label_in_sentence(sent: str, m: re.Match) -> int:
    start, end = m.start(), m.end()
    # Negation can fall INSIDE a multi-word match ("heart is not enlarged",
    # "cardiac silhouette is not enlarged") — the pre/post windows miss it.
    if _NEG.search(m.group(0)):
        return 0
    breaks_before = [b.end() for b in _BREAK.finditer(sent) if b.end() <= start]
    pre = sent[(max(breaks_before) if breaks_before else 0):start]
    breaks_after = [b.start() for b in _BREAK.finditer(sent) if b.start() >= end]
    post_full = sent[end:(min(breaks_after) if breaks_after else len(sent))]
    # POST-negation must refer to THIS concept: stop at the next "and"/"with"
    # clause (else "consolidation ... and edema cleared" wrongly negates consolidation)
    # and cap distance so a far-away resolution word doesn't reach back.
    post = re.split(r"\b(?:and|with)\b", post_full)[0][:45]
    if _NEG.search(pre):        return 0
    if _POST_NEG.search(post):  return 0
    if _UNC.search(pre):        return -1
    return 1


def label_v2(text: str) -> dict:
    if not text or not text.strip():
        return {f.value: 0.0 for f in Finding}
    clean = re.sub(r"_{2,}", " ", text.lower())
    clean = clean.replace("findings:", " . ").replace("impression:", " . ")
    concepts = {}
    for sent in (s.strip() for s in _SENT.split(clean) if s.strip()):
        for name, pat in CONCEPTS.items():
            best = concepts.get(name)
            for m in re.finditer(pat, sent):
                lbl = _label_in_sentence(sent, m)
                order = {1: 3, -1: 2, 0: 1}
                if best is None or order[lbl] > order.get(best, 0):
                    best = lbl
            if best is not None:
                concepts[name] = best
    # Opacity hierarchy: consolidation / pneumonia are opacities.
    if concepts.get("consolidation") == 1 or concepts.get("pneumonia") == 1:
        concepts["opacity"] = 1
    out = {f.value: 0.0 for f in Finding}
    for c, v in concepts.items():
        f = CONCEPT_TO_FINDING.get(c)
        if f is not None:
            out[f.value] = 1.0 if v == 1 else 0.0
    return out
