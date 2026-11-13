import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import concurrent.futures
import json
import time
from pathlib import Path
import numpy as np
import scipy.stats as stats
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from aura.common.config import ARTIFACTS
from aura.common.mathx import softmax
from aura.ml.evaluation import v6_confirmatory as v6
from aura.ml.evaluation import matched_optimiser as mo
from aura.ml.evaluation.matched_optimiser import _full_width_logits
from aura.ml.evaluation.v6_mechanism import _features
from aura.services.safety.calibration import fit_temperature, expected_calibration_error

OUT_DIR = Path(__file__).resolve().parent

SEEDS = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41]

def load_data():
    qse = np.load(ARTIFACTS / "quantum_study_evidence.npz")
    Xtr_m, ytr_m = qse["Xtr"], qse["ytr"]
    Xcal_m, ycal_m = qse["Xcal"], qse["ycal"]
    Xte_m, yte_m = qse["Xte"], qse["yte"]

    chex = np.load(ARTIFACTS / "chexpert_evidence.npz")
    perm = np.random.default_rng(7).permutation(len(chex["X"]))
    n_cal = int(0.40 * len(chex["X"]))
    cal_idx, te_idx = perm[:n_cal], perm[n_cal:]

    Xcal_cx = chex["X"][cal_idx]
    ycal_cx = chex["y_u0"][cal_idx]
    Xte_cx = chex["X"][te_idx]
    yte_cx = chex["y_u0"][te_idx]

    return {
        "mimic": (Xtr_m, ytr_m, Xcal_m, ycal_m, Xte_m, yte_m),
        "chexpert": (Xcal_cx, ycal_cx, Xte_cx, yte_cx),
    }

def analyze_input_shift(Xtr_m, Xte_m, Xte_cx):
    informative = 7
    shift_report = {}
    for i in range(informative):
        m_tr = Xtr_m[:, i]
        m_te = Xte_m[:, i]
        cx = Xte_cx[:, i]
        
        min_tr, max_tr = float(m_tr.min()), float(m_tr.max())
        outside_frac = float((cx < min_tr).sum() + (cx > max_tr).sum()) / len(cx)
        
        def quantiles(a):
            return {
                "mean": round(float(a.mean()), 4),
                "sd": round(float(a.std(ddof=1)), 4),
                "p05": round(float(np.percentile(a, 5)), 4),
                "p25": round(float(np.percentile(a, 25)), 4),
                "p50": round(float(np.percentile(a, 50)), 4),
                "p75": round(float(np.percentile(a, 75)), 4),
                "p95": round(float(np.percentile(a, 95)), 4),
                "min": round(float(a.min()), 4),
                "max": round(float(a.max()), 4),
            }
        
        shift_report[f"channel_{i+1}"] = {
            "mimic_train": quantiles(m_tr),
            "mimic_test": quantiles(m_te),
            "chexpert_transfer": quantiles(cx),
            "fraction_chexpert_outside_mimic_train_range": round(outside_frac, 5),
        }
    return shift_report

def train_eval_vqc_worker(task_info):
    arm_type, s = task_info
    torch.set_num_threads(2)
    data = load_data()
    Xtr_m, ytr_m, Xcal_m, ycal_m, Xte_m, yte_m = data["mimic"]
    Xcal_cx, ycal_cx, Xte_cx, yte_cx = data["chexpert"]
    d = Xtr_m.shape[1]

    if arm_type == "vqc_default":
        p_path = OUT_DIR / f"temp_vqcd_s{s}.npz"
        build = v6._build_quantum_fast(8, 3, "ring", 1.0, 0.1)
        lr = 0.05
    else:
        p_path = OUT_DIR / f"temp_vqcc2_s{s}.npz"
        build = v6._build_quantum_fast(8, 4, "ring", 0.5, 0.01)
        lr = 0.01

    if not p_path.exists():
        t0 = time.time()
        mo.SEED = s
        mo.WEIGHT_DECAY = 0.0
        mo.MAX_EPOCHS = 150
        mo.PATIENCE = 20
        rec, params, _ = mo._train_under_protocol(build, Xtr_m, ytr_m, Xcal_m, ycal_m, lr)
        v6._save_params(p_path, params)
        print(f"[{arm_type}:seed{s}] Trained in {time.time()-t0:.1f}s, cal_nll={rec['best_calibration_nll']:.4f}", flush=True)

    fwd = v6._load_forward(build, p_path)
    Lcal_m = v6._logits(fwd, Xcal_m)
    Lte_m = v6._logits(fwd, Xte_m)
    Lcal_cx = v6._logits(fwd, Xcal_cx)
    Lte_cx = v6._logits(fwd, Xte_cx)

    T_m = fit_temperature(Lcal_m, ycal_m)
    T_cx = fit_temperature(Lcal_cx, ycal_cx)

    Pm = np.array([softmax(r / T_m) for r in Lte_m])
    P_froz = np.array([softmax(r / T_m) for r in Lte_cx])
    P_adap = np.array([softmax(r / T_cx) for r in Lte_cx])

    nll_m = float(-np.log(np.clip(Pm[np.arange(len(yte_m)), yte_m], 1e-12, 1)).mean())
    nll_froz = float(-np.log(np.clip(P_froz[np.arange(len(yte_cx)), yte_cx], 1e-12, 1)).mean())
    nll_adap = float(-np.log(np.clip(P_adap[np.arange(len(yte_cx)), yte_cx], 1e-12, 1)).mean())

    return arm_type, {
        "seed": s, "mimic_nll": round(nll_m, 5),
        "frozen_chexpert_nll": round(nll_froz, 5), "adapted_chexpert_nll": round(nll_adap, 5),
        "frozen_transfer_delta": round(nll_froz - nll_m, 5),
    }

def main():
    data = load_data()
    Xtr_m, ytr_m, Xcal_m, ycal_m, Xte_m, yte_m = data["mimic"]
    Xcal_cx, ycal_cx, Xte_cx, yte_cx = data["chexpert"]
    d = Xtr_m.shape[1]

    print("Pre-building trig features...")
    Ftr_t1, _ = _features(Xtr_m, 8, 1)
    keep_t1 = Ftr_t1.std(0) > 1e-12
    keep_t1[0] = False
    Ftr_t1 = Ftr_t1[:, keep_t1]
    Fcal_m_t1 = _features(Xcal_m, 8, 1)[0][:, keep_t1]
    Fte_m_t1 = _features(Xte_m, 8, 1)[0][:, keep_t1]
    Fcal_cx_t1 = _features(Xcal_cx, 8, 1)[0][:, keep_t1]
    Fte_cx_t1 = _features(Xte_cx, 8, 1)[0][:, keep_t1]

    Ftr_t2, _ = _features(Xtr_m, 8, 2)
    keep_t2 = Ftr_t2.std(0) > 1e-12
    keep_t2[0] = False
    Ftr_t2 = Ftr_t2[:, keep_t2]
    Fcal_m_t2 = _features(Xcal_m, 8, 2)[0][:, keep_t2]
    Fte_m_t2 = _features(Xte_m, 8, 2)[0][:, keep_t2]
    Fcal_cx_t2 = _features(Xcal_cx, 8, 2)[0][:, keep_t2]
    Fte_cx_t2 = _features(Xte_cx, 8, 2)[0][:, keep_t2]

    results = {
        "vqc_default": [],
        "vqc_c2": [],
        "poe": [],
        "gated_head": [],
        "logistic_regression": [],
        "trig_lr_order1": [],
        "trig_lr_order2": [],
    }

    print("Fitting classical linear & trig LR...")
    for s in SEEDS:
        m_lr = LogisticRegression(C=1.0, max_iter=2000, random_state=s).fit(Xtr_m[:, :7], ytr_m)
        Lcal_m = _full_width_logits(m_lr, Xcal_m[:, :7])
        Lte_m = _full_width_logits(m_lr, Xte_m[:, :7])
        Lcal_cx = _full_width_logits(m_lr, Xcal_cx[:, :7])
        Lte_cx = _full_width_logits(m_lr, Xte_cx[:, :7])
        T_m = fit_temperature(Lcal_m, ycal_m)
        T_cx = fit_temperature(Lcal_cx, ycal_cx)
        Pm = np.array([softmax(r / T_m) for r in Lte_m])
        P_froz = np.array([softmax(r / T_m) for r in Lte_cx])
        P_adap = np.array([softmax(r / T_cx) for r in Lte_cx])
        nll_m = float(-np.log(np.clip(Pm[np.arange(len(yte_m)), yte_m], 1e-12, 1)).mean())
        nll_froz = float(-np.log(np.clip(P_froz[np.arange(len(yte_cx)), yte_cx], 1e-12, 1)).mean())
        nll_adap = float(-np.log(np.clip(P_adap[np.arange(len(yte_cx)), yte_cx], 1e-12, 1)).mean())
        results["logistic_regression"].append({
            "seed": s, "mimic_nll": round(nll_m, 5),
            "frozen_chexpert_nll": round(nll_froz, 5), "adapted_chexpert_nll": round(nll_adap, 5),
            "frozen_transfer_delta": round(nll_froz - nll_m, 5),
        })

        m_t1 = LogisticRegression(C=1.0, max_iter=2000, random_state=s).fit(Ftr_t1, ytr_m)
        Lcal_m1 = _full_width_logits(m_t1, Fcal_m_t1)
        Lte_m1 = _full_width_logits(m_t1, Fte_m_t1)
        Lcal_cx1 = _full_width_logits(m_t1, Fcal_cx_t1)
        Lte_cx1 = _full_width_logits(m_t1, Fte_cx_t1)
        T_m1 = fit_temperature(Lcal_m1, ycal_m)
        T_cx1 = fit_temperature(Lcal_cx1, ycal_cx)
        Pm1 = np.array([softmax(r / T_m1) for r in Lte_m1])
        nll_m1 = float(-np.log(np.clip(Pm1[np.arange(len(yte_m)), yte_m], 1e-12, 1)).mean())
        P_froz1 = np.array([softmax(r / T_m1) for r in Lte_cx1])
        nll_froz1 = float(-np.log(np.clip(P_froz1[np.arange(len(yte_cx)), yte_cx], 1e-12, 1)).mean())
        P_adap1 = np.array([softmax(r / T_cx1) for r in Lte_cx1])
        nll_adap1 = float(-np.log(np.clip(P_adap1[np.arange(len(yte_cx)), yte_cx], 1e-12, 1)).mean())
        results["trig_lr_order1"].append({
            "seed": s, "mimic_nll": round(nll_m1, 5),
            "frozen_chexpert_nll": round(nll_froz1, 5), "adapted_chexpert_nll": round(nll_adap1, 5),
            "frozen_transfer_delta": round(nll_froz1 - nll_m1, 5),
        })

        m_t2 = LogisticRegression(C=0.1, max_iter=2000, random_state=s).fit(Ftr_t2, ytr_m)
        Lcal_m2 = _full_width_logits(m_t2, Fcal_m_t2)
        Lte_m2 = _full_width_logits(m_t2, Fte_m_t2)
        Lcal_cx2 = _full_width_logits(m_t2, Fcal_cx_t2)
        Lte_cx2 = _full_width_logits(m_t2, Fte_cx_t2)
        T_m2 = fit_temperature(Lcal_m2, ycal_m)
        T_cx2 = fit_temperature(Lcal_cx2, ycal_cx)
        Pm2 = np.array([softmax(r / T_m2) for r in Lte_m2])
        nll_m2 = float(-np.log(np.clip(Pm2[np.arange(len(yte_m)), yte_m], 1e-12, 1)).mean())
        P_froz2 = np.array([softmax(r / T_m2) for r in Lte_cx2])
        nll_froz2 = float(-np.log(np.clip(P_froz2[np.arange(len(yte_cx)), yte_cx], 1e-12, 1)).mean())
        P_adap2 = np.array([softmax(r / T_cx2) for r in Lte_cx2])
        nll_adap2 = float(-np.log(np.clip(P_adap2[np.arange(len(yte_cx)), yte_cx], 1e-12, 1)).mean())
        results["trig_lr_order2"].append({
            "seed": s, "mimic_nll": round(nll_m2, 5),
            "frozen_chexpert_nll": round(nll_froz2, 5), "adapted_chexpert_nll": round(nll_adap2, 5),
            "frozen_transfer_delta": round(nll_froz2 - nll_m2, 5),
        })

    print("Loading/Training PoE and Gated Head...")
    for s in SEEDS:
        p_path_poe = OUT_DIR / f"temp_poe_s{s}.npz"
        b_poe = v6._build_linear_c(d, 0.01)
        if not p_path_poe.exists():
            mo.SEED = s
            mo.WEIGHT_DECAY = 1e-4
            rec_poe, params_poe, _ = mo._train_under_protocol(b_poe, Xtr_m, ytr_m, Xcal_m, ycal_m, 0.05)
            v6._save_params(p_path_poe, params_poe)
        fwd_poe = v6._load_forward(b_poe, p_path_poe)
        Lcal_m = v6._logits(fwd_poe, Xcal_m)
        Lte_m = v6._logits(fwd_poe, Xte_m)
        Lcal_cx = v6._logits(fwd_poe, Xcal_cx)
        Lte_cx = v6._logits(fwd_poe, Xte_cx)
        T_m = fit_temperature(Lcal_m, ycal_m)
        T_cx = fit_temperature(Lcal_cx, ycal_cx)
        Pm = np.array([softmax(r / T_m) for r in Lte_m])
        P_froz = np.array([softmax(r / T_m) for r in Lte_cx])
        P_adap = np.array([softmax(r / T_cx) for r in Lte_cx])
        nll_m = float(-np.log(np.clip(Pm[np.arange(len(yte_m)), yte_m], 1e-12, 1)).mean())
        nll_froz = float(-np.log(np.clip(P_froz[np.arange(len(yte_cx)), yte_cx], 1e-12, 1)).mean())
        nll_adap = float(-np.log(np.clip(P_adap[np.arange(len(yte_cx)), yte_cx], 1e-12, 1)).mean())
        results["poe"].append({
            "seed": s, "mimic_nll": round(nll_m, 5),
            "frozen_chexpert_nll": round(nll_froz, 5), "adapted_chexpert_nll": round(nll_adap, 5),
            "frozen_transfer_delta": round(nll_froz - nll_m, 5),
        })

        p_path_g = OUT_DIR / f"temp_gated_s{s}.npz"
        b_gated = v6._build_gated_c(d, 0.01)
        if not p_path_g.exists():
            mo.SEED = s
            mo.WEIGHT_DECAY = 1e-4
            rec_gated, params_gated, _ = mo._train_under_protocol(b_gated, Xtr_m, ytr_m, Xcal_m, ycal_m, 0.1)
            v6._save_params(p_path_g, params_gated)
        fwd_gated = v6._load_forward(b_gated, p_path_g)
        Lcal_m = v6._logits(fwd_gated, Xcal_m)
        Lte_m = v6._logits(fwd_gated, Xte_m)
        Lcal_cx = v6._logits(fwd_gated, Xcal_cx)
        Lte_cx = v6._logits(fwd_gated, Xte_cx)
        T_m = fit_temperature(Lcal_m, ycal_m)
        T_cx = fit_temperature(Lcal_cx, ycal_cx)
        Pm = np.array([softmax(r / T_m) for r in Lte_m])
        P_froz = np.array([softmax(r / T_m) for r in Lte_cx])
        P_adap = np.array([softmax(r / T_cx) for r in Lte_cx])
        nll_m = float(-np.log(np.clip(Pm[np.arange(len(yte_m)), yte_m], 1e-12, 1)).mean())
        nll_froz = float(-np.log(np.clip(P_froz[np.arange(len(yte_cx)), yte_cx], 1e-12, 1)).mean())
        nll_adap = float(-np.log(np.clip(P_adap[np.arange(len(yte_cx)), yte_cx], 1e-12, 1)).mean())
        results["gated_head"].append({
            "seed": s, "mimic_nll": round(nll_m, 5),
            "frozen_chexpert_nll": round(nll_froz, 5), "adapted_chexpert_nll": round(nll_adap, 5),
            "frozen_transfer_delta": round(nll_froz - nll_m, 5),
        })

    print("Launching VQC runs across 5 parallel workers...")
    vqc_tasks = [(arm, s) for arm in ["vqc_default", "vqc_c2"] for s in SEEDS]
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        for arm, res in executor.map(train_eval_vqc_worker, vqc_tasks):
            results[arm].append(res)

    for k in results:
        results[k].sort(key=lambda r: r["seed"])

    print("Running VQC transfer ablations...")
    ablations = {}
    ablations["scale_sweep"] = []
    for sc in [0.5, 1.0, 2.0]:
        mo.SEED = 7
        b_sc = v6._build_quantum_fast(8, 3, "ring", sc, 0.1)
        p_sc_path = OUT_DIR / f"temp_sc_{sc}.npz"
        if not p_sc_path.exists():
            _, p_sc, _ = mo._train_under_protocol(b_sc, Xtr_m, ytr_m, Xcal_m, ycal_m, 0.05)
            v6._save_params(p_sc_path, p_sc)
        f_sc = v6._load_forward(b_sc, p_sc_path)
        
        max_angle = float(np.pi * sc * Xte_cx[:, :7].max())
        exceeds_2pi = bool(max_angle > 2 * np.pi)
        
        T_m = fit_temperature(v6._logits(f_sc, Xcal_m), ycal_m)
        T_cx = fit_temperature(v6._logits(f_sc, Xcal_cx), ycal_cx)
        
        Pm = np.array([softmax(r / T_m) for r in v6._logits(f_sc, Xte_m)])
        P_froz = np.array([softmax(r / T_m) for r in v6._logits(f_sc, Xte_cx)])
        P_adap = np.array([softmax(r / T_cx) for r in v6._logits(f_sc, Xte_cx)])
        
        nll_m = float(-np.log(np.clip(Pm[np.arange(len(yte_m)), yte_m], 1e-12, 1)).mean())
        nll_froz = float(-np.log(np.clip(P_froz[np.arange(len(yte_cx)), yte_cx], 1e-12, 1)).mean())
        nll_adap = float(-np.log(np.clip(P_adap[np.arange(len(yte_cx)), yte_cx], 1e-12, 1)).mean())
        
        ablations["scale_sweep"].append({
            "scale": sc,
            "max_angle_radians": round(max_angle, 4),
            "max_angle_over_pi": round(max_angle / np.pi, 4),
            "exceeds_2pi_wrapping": exceeds_2pi,
            "mimic_nll": round(nll_m, 5),
            "frozen_chexpert_nll": round(nll_froz, 5),
            "adapted_chexpert_nll": round(nll_adap, 5),
            "transfer_delta": round(nll_froz - nll_m, 5),
        })

    mins = Xtr_m.min(axis=0)
    maxs = Xtr_m.max(axis=0)
    Xte_cx_clipped = np.clip(Xte_cx, mins, maxs)

    fwd_vd7 = v6._load_forward(v6._build_quantum_fast(8, 3, "ring", 1.0, 0.1), OUT_DIR / "temp_vqcd_s7.npz")
    T_m7 = fit_temperature(v6._logits(fwd_vd7, Xcal_m), ycal_m)
    P_clip_froz = np.array([softmax(r / T_m7) for r in v6._logits(fwd_vd7, Xte_cx_clipped)])
    nll_clip_froz = float(-np.log(np.clip(P_clip_froz[np.arange(len(yte_cx)), yte_cx], 1e-12, 1)).mean())
    
    unclipped_nll = results["vqc_default"][0]["frozen_chexpert_nll"]
    ablations["input_clipping"] = {
        "unclipped_frozen_nll": unclipped_nll,
        "clipped_frozen_nll": round(nll_clip_froz, 5),
        "clipping_benefit": round(unclipped_nll - nll_clip_froz, 5),
    }

    sc_m = StandardScaler().fit(Xtr_m[:, :7])
    sc_cx = StandardScaler().fit(Xcal_cx[:, :7])
    def to_01(arr):
        return 1.0 / (1.0 + np.exp(-arr))
    Xtr_m_std = np.hstack([to_01(sc_m.transform(Xtr_m[:, :7])), np.zeros((len(Xtr_m), 1))])
    Xcal_m_std = np.hstack([to_01(sc_m.transform(Xcal_m[:, :7])), np.zeros((len(Xcal_m), 1))])
    Xte_m_std = np.hstack([to_01(sc_m.transform(Xte_m[:, :7])), np.zeros((len(Xte_m), 1))])
    Xcal_cx_std = np.hstack([to_01(sc_cx.transform(Xcal_cx[:, :7])), np.zeros((len(Xcal_cx), 1))])
    Xte_cx_std = np.hstack([to_01(sc_cx.transform(Xte_cx[:, :7])), np.zeros((len(Xte_cx), 1))])

    p_std_path = OUT_DIR / "temp_std.npz"
    b_std = v6._build_quantum_fast(8, 3, "ring", 1.0, 0.1)
    if not p_std_path.exists():
        mo.SEED = 7
        _, p_std, _ = mo._train_under_protocol(b_std, Xtr_m_std, ytr_m, Xcal_m_std, ycal_m, 0.05)
        v6._save_params(p_std_path, p_std)
    fwd_std = v6._load_forward(b_std, p_std_path)
    T_m_std = fit_temperature(v6._logits(fwd_std, Xcal_m_std), ycal_m)
    P_std_m = np.array([softmax(r / T_m_std) for r in v6._logits(fwd_std, Xte_m_std)])
    P_std_cx = np.array([softmax(r / T_m_std) for r in v6._logits(fwd_std, Xte_cx_std)])
    nll_std_m = float(-np.log(np.clip(P_std_m[np.arange(len(yte_m)), yte_m], 1e-12, 1)).mean())
    nll_std_cx = float(-np.log(np.clip(P_std_cx[np.arange(len(yte_cx)), yte_cx], 1e-12, 1)).mean())

    ablations["cohort_standardization"] = {
        "standardized_mimic_nll": round(nll_std_m, 5),
        "standardized_chexpert_frozen_nll": round(nll_std_cx, 5),
        "standardized_transfer_delta": round(nll_std_cx - nll_std_m, 5),
    }

    summary = {}
    for k, v in results.items():
        deltas = [r["frozen_transfer_delta"] for r in v]
        froz_nlls = [r["frozen_chexpert_nll"] for r in v]
        m_nlls = [r["mimic_nll"] for r in v]
        adap_nlls = [r["adapted_chexpert_nll"] for r in v]
        summary[k] = {
            "mimic_nll_mean": round(float(np.mean(m_nlls)), 5),
            "mimic_nll_sd": round(float(np.std(m_nlls, ddof=1)), 5),
            "frozen_chexpert_nll_mean": round(float(np.mean(froz_nlls)), 5),
            "frozen_chexpert_nll_sd": round(float(np.std(froz_nlls, ddof=1)), 5),
            "frozen_transfer_delta_mean": round(float(np.mean(deltas)), 5),
            "frozen_transfer_delta_sd": round(float(np.std(deltas, ddof=1)), 5),
            "adapted_chexpert_nll_mean": round(float(np.mean(adap_nlls)), 5),
            "seeds": v,
        }

    vqc_delta = summary["vqc_default"]["frozen_transfer_delta_mean"]
    t1_delta = summary["trig_lr_order1"]["frozen_transfer_delta_mean"]
    poe_delta = summary["poe"]["frozen_transfer_delta_mean"]
    clip_benefit = ablations["input_clipping"]["clipping_benefit"]

    if abs(t1_delta - vqc_delta) < 0.03:
        mechanism_verdict = "trigonometric_basis"
    elif clip_benefit > 0.5 * (vqc_delta - poe_delta):
        mechanism_verdict = "scale_shift"
    elif vqc_delta > t1_delta + 0.05:
        mechanism_verdict = "circuit_structure_or_entanglement"
    else:
        mechanism_verdict = "unexplained"

    output = {
        "cohorts": {
            "mimic_train_n": len(ytr_m),
            "mimic_cal_n": len(ycal_m),
            "mimic_test_n": len(yte_m),
            "chexpert_cal_n": len(ycal_cx),
            "chexpert_test_n": len(yte_cx),
        },
        "input_feature_shift": analyze_input_shift(Xtr_m, Xte_m, Xte_cx),
        "models_10seeds": summary,
        "ablations": ablations,
        "decision_rule": {
            "vqc_default_mean_delta": vqc_delta,
            "vqc_c2_mean_delta": summary["vqc_c2"]["frozen_transfer_delta_mean"],
            "trig_lr_order1_mean_delta": t1_delta,
            "poe_mean_delta": poe_delta,
            "clipping_benefit": clip_benefit,
            "verdict": mechanism_verdict,
        }
    }

    out_file = OUT_DIR / "domain_shift_analysis.json"
    out_file.write_text(json.dumps(output, indent=2))
    print(f"\nWrote {out_file}", flush=True)
    print(f"Decision rule verdict: {mechanism_verdict}", flush=True)

if __name__ == "__main__":
    main()
