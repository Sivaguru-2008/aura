"""Stage S1 of the field audit: medical-term match and not a review.

Reproduces ``stage1_medical.json`` from ``pubmed_records.json`` (PubMed metadata for
the 231 PMIDs in ``all_pmids.txt``). The regex is deliberately broad: S1 only removes
records that are plainly not about human health; the judgement calls happen at S2.
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).parent
MEDICAL = re.compile(
    r"medic|clinic|patient|disease|cancer|tumou?r|diagnos|health|cardi|heart|diabet|"
    r"breast|lung|covid|pneumon|alzheim|parkinson|brain|mri|x-ray|ct scan|ecg|eeg|retin|"
    r"skin|lesion|malignan|mammogra|hepat|liver|kidney|renal|stroke|sepsis|icu|ehr|"
    r"electronic health|genom|protein|drug|molecul|biomed|patholog|histopath|dementia|"
    r"autism|epilep|schizo|glioma|leukemi|thyroid|blood", re.I)
REVIEW = re.compile(r"review|survey|perspective", re.I)


def main() -> None:
    records = json.loads((HERE / "pubmed_records.json").read_text(encoding="utf-8"))
    order = (HERE / "all_pmids.txt").read_text().split()
    keep = []
    for pmid in order:
        a = records[pmid]
        text = a.get("title", "") + " " + a.get("abstract", "")
        types = " ".join(a.get("article_types", []))
        is_review = bool(REVIEW.search(types)) or bool(
            re.match(r"(?i).*\b(review|survey)\b", a.get("title", "")))
        if MEDICAL.search(text) and not is_review:
            keep.append(pmid)
    (HERE / "stage1_medical.json").write_text(json.dumps(keep, indent=1))
    print(f"{len(order)} records -> {len(keep)} pass S1")


if __name__ == "__main__":
    main()
