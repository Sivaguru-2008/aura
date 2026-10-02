"""Loaders for the public datasets of the included studies. Raw files in ``data/``.

Each loader returns ``(X, y, feature_names, meta)`` with raw (unscaled, possibly
missing) features. Scaling, imputation and selection happen inside the CV folds, in the
study's pipeline — never here — so nothing a loader does can leak across a split.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "data"


# --------------------------------------------------------------------------- #
def ckd():
    """UCI Chronic Kidney Disease (Rubini 2015): 400 patients, 24 features.

    The distributed ARFF has known formatting faults (tabs, doubled commas, stray
    spaces inside nominal values). They are repaired by stripping whitespace and
    dropping empty fields, which recovers exactly 25 fields on all 400 rows.
    """
    path = DATA / "ckd" / "Chronic_Kidney_Disease" / "chronic_kidney_disease_full.arff"
    lines = path.read_text(encoding="latin-1").splitlines()
    names, nominal = [], {}
    rows = []
    in_data = False
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("%"):
            continue
        if s.lower().startswith("@attribute"):
            m = re.match(r"@attribute\s+'?([^'\s]+)'?\s+(.*)", s, re.I)
            nm, typ = m.group(1), m.group(2).strip()
            names.append(nm)
            if typ.startswith("{"):
                nominal[nm] = [v.strip() for v in typ.strip("{}").split(",")]
            continue
        if s.lower().startswith("@data"):
            in_data = True
            continue
        if in_data:
            fields = [f.strip() for f in s.replace("\t", "").split(",")]
            fields = [f for f in fields if f != ""]
            rows.append(fields)
    rows = [r for r in rows if len(r) == len(names)]
    target = names[-1]
    feats = names[:-1]
    X = np.full((len(rows), len(feats)), np.nan)
    for i, r in enumerate(rows):
        for j, nm in enumerate(feats):
            v = r[j]
            if v == "?":
                continue
            if nm in nominal:
                # binary nominal: first listed level -> 1 (normal/present/yes/good ... ),
                # coded consistently per attribute; only the ordering matters for trees,
                # and linear models are indifferent to a sign flip.
                X[i, j] = float(nominal[nm].index(v)) if v in nominal[nm] else np.nan
            else:
                X[i, j] = float(v)
    y = np.array([1 if r[-1] == "ckd" else 0 for r in rows])
    return X, y, feats, {"n": len(rows), "positives": int(y.sum()),
                         "source": "UCI id 336"}


def glioma():
    """UCI Glioma Grading (TCGA LGG/GBM): 839 patients, 23 features, Grade 1 = GBM."""
    path = DATA / "glioma" / "TCGA_InfoWithGrade.csv"
    with path.open() as f:
        r = list(csv.reader(f))
    head, body = r[0], r[1:]
    arr = np.array(body, dtype=float)
    y = arr[:, head.index("Grade")].astype(int)
    feats = [h for h in head if h != "Grade"]
    X = arr[:, [head.index(h) for h in feats]]
    return X, y, feats, {"n": len(y), "positives": int(y.sum()),
                         "positive_class": "GBM (HGG)", "source": "UCI id 759"}


def parkinsons():
    """UCI Parkinson's voice (Little 2008): 195 recordings of 31 people, status 1 = PD.

    Recordings are grouped by subject (``phon_R01_S01_*``); the subject id is returned
    in ``meta['groups']`` because a recording-level split puts the same voice on both
    sides. The study under audit split by recording; both are reported.
    """
    path = DATA / "parkinsons" / "parkinsons.data"
    with path.open() as f:
        r = list(csv.reader(f))
    head, body = r[0], r[1:]
    y = np.array([int(row[head.index("status")]) for row in body])
    feats = [h for h in head if h not in ("name", "status")]
    X = np.array([[float(row[head.index(h)]) for h in feats] for row in body])
    groups = np.array([re.match(r"phon_R01_(S\d+)_", row[0]).group(1) for row in body])
    return X, y, feats, {"n": len(y), "positives": int(y.sum()),
                         "subjects": int(len(set(groups))), "groups": groups,
                         "source": "UCI id 174"}


def pima():
    """Pima Indians Diabetes (NIDDK), OpenML 37: 768 women, 8 features."""
    lines = (DATA / "pima.arff").read_text().splitlines()
    i = next(k for k, l in enumerate(lines) if l.strip().lower() == "@data")
    rows = [l.split(",") for l in lines[i + 1:] if l.strip() and not l.startswith("%")]
    X = np.array([[float(v) for v in r[:8]] for r in rows])
    y = np.array([1 if r[8].strip() == "tested_positive" else 0 for r in rows])
    feats = ["preg", "plas", "pres", "skin", "insu", "mass", "pedi", "age"]
    return X, y, feats, {"n": len(y), "positives": int(y.sum()), "source": "OpenML 37"}


def heart918():
    """UCI Heart Disease, the four ``processed.*`` cohorts, exact duplicates removed.

    This is the corpus behind the widely used 918-row 'Heart Failure Prediction' file
    (Cleveland 303, Hungary 294, Switzerland 123, VA Long Beach 200, less duplicates;
    Statlog's 270 rows are a re-coded subset of Cleveland). That file is Kaggle-only and
    its imputation is undocumented, so it is rebuilt here from UCI: the same 11
    features (ca and thal dropped), target = diagnosis > 0, '?' and the Swiss
    cholesterol zeros left missing for in-fold imputation.
    """
    cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
            "exang", "oldpeak", "slope", "ca", "thal", "num"]
    keep = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
            "exang", "oldpeak", "slope"]
    rows = []
    for name in ("cleveland", "hungarian", "switzerland", "va"):
        for ln in (DATA / "heart" / f"processed.{name}.data").read_text().splitlines():
            if not ln.strip():
                continue
            vals = [np.nan if v.strip() == "?" else float(v) for v in ln.split(",")]
            rows.append(tuple(vals[cols.index(c)] for c in keep) + (vals[-1],))
    # exact duplicates (NaN-aware)
    seen, uniq = set(), []
    for r in rows:
        key = tuple("nan" if (isinstance(v, float) and np.isnan(v)) else v for v in r)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    arr = np.array(uniq, dtype=float)
    X = arr[:, :-1]
    X[:, keep.index("chol")] = np.where(X[:, keep.index("chol")] == 0, np.nan,
                                        X[:, keep.index("chol")])
    y = (arr[:, -1] > 0).astype(int)
    return X, y, keep, {"n": len(y), "positives": int(y.sum()),
                        "rows_before_dedup": len(rows), "source": "UCI id 45"}


# --------------------------------------------------------------------------- #
# EEG (Borisov et al., MSU): 39 healthy, 45 schizophrenia; 16 ch x 7680 @128 Hz
# --------------------------------------------------------------------------- #
EEG_CHANNELS = ["F7", "F3", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3",
                "Pz", "P4", "T6", "O1", "O2"]
_DB2_LO = np.array([-0.12940952255092145, 0.22414386804185735,
                    0.836516303737469, 0.48296291314469025])
_DB2_HI = np.array([-0.48296291314469025, 0.836516303737469,
                    -0.22414386804185735, -0.12940952255092145])


def _dwt_db2(x: np.ndarray):
    """One level of the db2 DWT, symmetric extension (PyWavelets' default mode)."""
    F = len(_DB2_LO)
    ext = np.pad(x, F - 1, mode="symmetric")
    lo = np.convolve(ext, _DB2_LO, mode="valid")[1::2]
    hi = np.convolve(ext, _DB2_HI, mode="valid")[1::2]
    return lo, hi


def wavedec_db2(x: np.ndarray, level: int = 4):
    """[cA_level, cD_level, ..., cD1], like ``pywt.wavedec(x, 'db2', level=level)``."""
    out = []
    a = np.asarray(x, float)
    for _ in range(level):
        a, d = _dwt_db2(a)
        out.append(d)
    return [a] + out[::-1]


def _band_stats(c: np.ndarray) -> list[float]:
    e = c ** 2
    e_safe = np.where(e > 0, e, 1e-300)
    return [float(c.mean()), float(c.var()), float(c.std()),
            float(-(e * np.log(e_safe)).sum()),      # Shannon entropy (wentropy)
            float(np.log(e_safe).sum())]              # log-energy entropy


def eeg_schizophrenia(channel: str):
    """25 DWT features (5 sub-bands x mean, var, std, Shannon, log-energy) for one channel."""
    ch = EEG_CHANNELS.index(channel)
    X, y, ids = [], [], []
    for label, folder in ((0, "norm"), (1, "sch")):
        for f in sorted((DATA / "eeg" / folder).glob("*.eea")):
            sig = np.loadtxt(f).reshape(16, 7680)[ch]
            feats = []
            for band in wavedec_db2(sig, 4):
                feats += _band_stats(band)
            X.append(feats)
            y.append(label)
            ids.append(f.stem)
    names = [f"{b}_{s}" for b in ("A4", "D4", "D3", "D2", "D1")
             for s in ("mean", "var", "std", "shannon", "logenergy")]
    return (np.array(X), np.array(y), names,
            {"n": len(y), "positives": int(sum(y)), "channel": channel, "ids": ids})
