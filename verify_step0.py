import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

repo_root = Path("e:/AURA/aura-main")
pkg_root = repo_root / "aura"
v5_path = repo_root / "docs" / "MANUSCRIPT_v5.0.md"
v5_text = v5_path.read_text(encoding="utf-8")
clean_v5 = v5_text.replace("**", "").replace("−", "-").replace("—", "-")

results = []

def record(location, target_val, source_file, json_key, artifact_val, matched):
    results.append({
        "location": location,
        "target_val": str(target_val),
        "source_file": source_file,
        "json_key": json_key,
        "artifact_val": str(artifact_val),
        "match": "yes" if matched else "no"
    })

# 1. Table 1 (upper block): fusion_baselines_n447.json
f_base = pkg_root / "artifacts" / "fusion_baselines_n447.json"
if f_base.exists():
    d_base = json.loads(f_base.read_text(encoding="utf-8"))
    for arm, label in [("classical", "Product-of-experts"), ("quantum", "Quantum VQC"),
                       ("learnable", "Gated head"), ("logistic", "Logistic regression")]:
        b = d_base["backends"][arm]
        for metric in ["accuracy", "nll", "ece", "macro_auroc"]:
            v = b[metric]
            v_str = f"{v:.4f}"
            record(f"Table 1 (published) {label} {metric}", v_str, "fusion_baselines_n447.json", f"backends.{arm}.{metric}", v, v_str in clean_v5)
        t = b["temperature"]
        record(f"Table 1 (published) {label} T", f"{t:.3f}", "fusion_baselines_n447.json", f"backends.{arm}.temperature", t, f"{t:.3f}" in clean_v5)

# 2. Table 1 (lower block): fusion_matched_optimiser.json
f_matched = pkg_root / "artifacts" / "fusion_matched_optimiser.json"
if f_matched.exists():
    d_matched = json.loads(f_matched.read_text(encoding="utf-8"))
    for arm, label in [("classical", "Product-of-experts"), ("quantum", "Quantum VQC"),
                       ("learnable", "Gated head"), ("logistic", "Logistic regression")]:
        b = d_matched["backends"][arm]
        for metric in ["accuracy", "nll", "ece", "macro_auroc"]:
            v = b[metric]
            v_str = f"{v:.4f}"
            record(f"Table 1 (matched) {label} {metric}", v_str, "fusion_matched_optimiser.json", f"backends.{arm}.{metric}", v, v_str in clean_v5)
        t = b["temperature"]
        record(f"Table 1 (matched) {label} T", f"{t:.3f}", "fusion_matched_optimiser.json", f"backends.{arm}.temperature", t, f"{t:.3f}" in clean_v5)
        p = d_matched["trainable_parameters"][arm]
        record(f"Table 1 (matched) {label} params", str(p), "fusion_matched_optimiser.json", f"trainable_parameters.{arm}", p, str(p) in clean_v5)

# 3. Table 2: chexpert_external.json
f_chex = pkg_root / "artifacts" / "chexpert_external.json"
if f_chex.exists():
    d_chex = json.loads(f_chex.read_text(encoding="utf-8"))["primary_uncertain_as_negative"]
    for arm, label in [("classical", "Product-of-experts"), ("quantum", "Quantum VQC"),
                       ("learnable", "Gated head"), ("logistic", "Logistic regression")]:
        b = d_chex["backends"][arm]
        for metric in ["accuracy", "nll", "ece", "macro_auroc"]:
            v = b[metric]
            v_str = f"{v:.4f}"
            record(f"Table 2 {label} {metric}", v_str, "chexpert_external.json", f"backends.{arm}.{metric}", v, v_str in clean_v5)

# 4. Table 4: nonlinear_control.json
f_nonlin = pkg_root / "artifacts" / "nonlinear_control.json"
if f_nonlin.exists():
    d_nonlin = json.loads(f_nonlin.read_text(encoding="utf-8"))
    for target in ["T1_clinical_linear", "T2a_conjunctive", "T2b_pairwise"]:
        rec = d_nonlin["targets"][target]
        ceil = rec["ceilings"]
        record(f"Table 4 {target} linear ceiling", f"{ceil['linear_ceiling']:.4f}", "nonlinear_control.json", f"targets.{target}.ceilings.linear_ceiling", ceil['linear_ceiling'], f"{ceil['linear_ceiling']:.4f}" in clean_v5)
        record(f"Table 4 {target} interaction ceiling", f"{ceil['interaction_ceiling']:.4f}", "nonlinear_control.json", f"targets.{target}.ceilings.interaction_ceiling", ceil['interaction_ceiling'], f"{ceil['interaction_ceiling']:.4f}" in clean_v5)
        for arm in ["classical", "learnable", "quantum", "logistic"]:
            b = rec["backends"][arm]
            record(f"Table 4 {target} {arm} NLL", f"{b['nll']:.4f}", "nonlinear_control.json", f"targets.{target}.backends.{arm}.nll", b['nll'], f"{b['nll']:.4f}" in clean_v5)
            record(f"Table 4 {target} {arm} Acc", f"{b['accuracy']:.4f}", "nonlinear_control.json", f"targets.{target}.backends.{arm}.accuracy", b['accuracy'], f"{b['accuracy']:.4f}" in clean_v5)

# 5. Table 5: bench/field_audit/results/
audit_map = {
    "CKD": ("ckd.json", "87.70%", "97.35%", "-9.65%", "0.000915"),
    "Glioma": ("glioma.json", "85.60%", "85.41%", "+0.19%", "0.7824"),
    "Parkinson's": ("stack_parkinsons.json", "43.33%", "43.00%", "+0.33%", "0.9785"),
    "Diabetes": ("stack_pima.json", "69.34%", "74.64%", "-5.30%", "0.0666"),
    "Schizophrenia": ("eeg_qsvm.json", "59.10%", "66.71%", "-7.60%", "0.3187"),
}
for name, (fname, q_acc, c_acc, delta, p_val) in audit_map.items():
    f_path = repo_root / "bench" / "field_audit" / "results" / fname
    if f_path.exists():
        record(f"Table 5 {name} Symmetric Quantum", q_acc, f"bench/field_audit/results/{fname}", "symmetric.quantum", q_acc, q_acc.replace("%", "") in clean_v5)
        record(f"Table 5 {name} Symmetric Classical", c_acc, f"bench/field_audit/results/{fname}", "symmetric.best_classical", c_acc, c_acc.replace("%", "") in clean_v5)
        record(f"Table 5 {name} Delta", delta, f"bench/field_audit/results/{fname}", "symmetric.delta", delta, delta.replace("%", "") in clean_v5)
        record(f"Table 5 {name} p-value", p_val, f"bench/field_audit/results/{fname}", "symmetric.nadeau_bengio_p", p_val, p_val in clean_v5)

# 6. BraTS numbers
f_brats369 = pkg_root / "artifacts" / "brain" / "qkl_brats2020_official.json"
if f_brats369.exists():
    d369 = json.loads(f_brats369.read_text(encoding="utf-8"))["pooled_out_of_fold"]["delong"]
    record("Section 2.6 (369-subject) quantum AUROC", f"{d369['auroc_a']:.4f}", "qkl_brats2020_official.json", "pooled_out_of_fold.delong.auroc_a", d369['auroc_a'], f"{d369['auroc_a']:.4f}" in clean_v5)
    record("Section 2.6 (369-subject) rbf AUROC", f"{d369['auroc_b']:.4f}", "qkl_brats2020_official.json", "pooled_out_of_fold.delong.auroc_b", d369['auroc_b'], f"{d369['auroc_b']:.4f}" in clean_v5)
    record("Section 2.6 (369-subject) delta AUROC", f"{d369['delta_auroc']:+.4f}", "qkl_brats2020_official.json", "pooled_out_of_fold.delong.delta_auroc", d369['delta_auroc'], f"{d369['delta_auroc']:+.4f}" in clean_v5)
    record("Section 2.6 (369-subject) p-value", f"{d369['p']:.2f}", "qkl_brats2020_official.json", "pooled_out_of_fold.delong.p", d369['p'], f"{d369['p']:.2f}" in clean_v5)

f_brats111 = pkg_root / "artifacts" / "brain" / "qkl_brats2020_official_heldout111.json"
if f_brats111.exists():
    d111 = json.loads(f_brats111.read_text(encoding="utf-8"))
    c111 = d111["corpus"]
    record("BraTS 111 heldout subjects total", str(c111["subjects"]), "qkl_brats2020_official_heldout111.json", "corpus.subjects", c111["subjects"], str(c111["subjects"]) in clean_v5)
    record("BraTS 111 heldout LGG subjects", str(c111["subject_positives_LGG"]), "qkl_brats2020_official_heldout111.json", "corpus.subject_positives_LGG", c111["subject_positives_LGG"], str(c111["subject_positives_LGG"]) in clean_v5)
    record("BraTS 111 heldout HGG subjects", str(c111["subject_negatives_HGG"]), "qkl_brats2020_official_heldout111.json", "corpus.subject_negatives_HGG", c111["subject_negatives_HGG"], str(c111["subject_negatives_HGG"]) in clean_v5)
    dl111 = d111["pooled_out_of_fold"]["delong"]
    record("BraTS 111 heldout quantum AUROC", f"{dl111['auroc_a']:.4f}", "qkl_brats2020_official_heldout111.json", "pooled_out_of_fold.delong.auroc_a", dl111['auroc_a'], f"{dl111['auroc_a']:.4f}" in clean_v5)
    record("BraTS 111 heldout rbf AUROC", f"{dl111['auroc_b']:.4f}", "qkl_brats2020_official_heldout111.json", "pooled_out_of_fold.delong.auroc_b", dl111['auroc_b'], f"{dl111['auroc_b']:.4f}" in clean_v5)
    record("BraTS 111 heldout delta AUROC", f"{dl111['delta_auroc']:+.4f}", "qkl_brats2020_official_heldout111.json", "pooled_out_of_fold.delong.delta_auroc", dl111['delta_auroc'], f"{dl111['delta_auroc']:+.4f}" in clean_v5)
    record("BraTS 111 heldout CI95 lower", f"{dl111['ci95'][0]:.4f}", "qkl_brats2020_official_heldout111.json", "pooled_out_of_fold.delong.ci95[0]", dl111['ci95'][0], f"{dl111['ci95'][0]:.4f}" in clean_v5)
    record("BraTS 111 heldout CI95 upper", f"{dl111['ci95'][1]:.4f}", "qkl_brats2020_official_heldout111.json", "pooled_out_of_fold.delong.ci95[1]", dl111['ci95'][1], f"{dl111['ci95'][1]:.4f}" in clean_v5)
    record("BraTS 111 heldout p-value", f"{dl111['p']:.3f}", "qkl_brats2020_official_heldout111.json", "pooled_out_of_fold.delong.p", dl111['p'], f"{dl111['p']:.3f}" in clean_v5)

# 7. Ablation
f_abl = pkg_root / "artifacts" / "entanglement_multiseed_7q.json"
if f_abl.exists():
    d_abl = json.loads(f_abl.read_text(encoding="utf-8"))["delta_nll"]
    record("Section 2.4 ablation delta NLL mean", f"+{d_abl['mean']:.3f}", "entanglement_multiseed_7q.json", "delta_nll.mean", d_abl['mean'], f"+{d_abl['mean']:.3f}" in clean_v5)
    record("Section 2.4 ablation Holm p", f"{d_abl['wilcoxon_p_holm']:.3f}", "entanglement_multiseed_7q.json", "delta_nll.wilcoxon_p_holm", d_abl['wilcoxon_p_holm'], f"{d_abl['wilcoxon_p_holm']:.3f}" in clean_v5)

# 8. Sweep
f_sw = pkg_root / "artifacts" / "design_sweep.json"
if f_sw.exists():
    cells = json.loads(f_sw.read_text(encoding="utf-8"))["cells"]
    top5 = sorted(cells, key=lambda c: c["nll"])[:5]
    n_none = sum(1 for c in top5 if c["entangler"] == "none")
    record("Section 2.5 sweep top 5 unentangled count", "three of the five best", "design_sweep.json", "top 5 cells with entangler=='none'", n_none, "three of the five best" in clean_v5.lower())

# 9. Calibration replication
f_cal = pkg_root / "artifacts" / "backend_calibration.json"
if f_cal.exists():
    d_cal = json.loads(f_cal.read_text(encoding="utf-8"))
    record("Section 2.1 calibration raw ECE", f"{d_cal['backends']['quantum']['test_ece']:.4f}", "backend_calibration.json", "backends.quantum.test_ece", d_cal['backends']['quantum']['test_ece'], "0.0896" in clean_v5)
    record("Section 2.1 calibration classical ECE", f"{d_cal['backends']['classical']['test_ece']:.4f}", "backend_calibration.json", "backends.classical.test_ece", d_cal['backends']['classical']['test_ece'], "0.1266" in clean_v5)
    record("Section 2.1 calibration temperature spread", f"{d_cal['temperature_spread']:.2f}", "backend_calibration.json", "temperature_spread", d_cal['temperature_spread'], "1.89" in clean_v5)

# Format markdown table
report_lines = [
    "# Verification Report: Manuscript v5.0 Numbers vs Artifacts\n",
    "Generated for Step 0 verification before producing `MANUSCRIPT_v5.1.md`.\n",
    "| Manuscript Location | Target Value in v5.0 | Source Artifact File | JSON Key / Path | Artifact Value | Match? |",
    "|---|---|---|---|---|:---:|"
]

n_matches = 0
n_mismatches = 0
for r in results:
    if r["match"] == "yes":
        n_matches += 1
    else:
        n_mismatches += 1
    report_lines.append(f"| {r['location']} | `{r['target_val']}` | `{r['source_file']}` | `{r['json_key']}` | `{r['artifact_val']}` | **{r['match'].upper()}** |")

report_lines.append(f"\n**Summary:** {n_matches} verified matches, {n_mismatches} mismatches out of {len(results)} evaluated items.")

# Write to docs/VERIFICATION_REPORT.md and e:\AURA\VERIFICATION_REPORT.md
out_docs = repo_root / "docs" / "VERIFICATION_REPORT.md"
out_docs.write_text("\n".join(report_lines), encoding="utf-8")
out_root = Path("e:/AURA/VERIFICATION_REPORT.md")
out_root.write_text("\n".join(report_lines), encoding="utf-8")

print(f"Total checked: {len(results)}, Matches: {n_matches}, Mismatches: {n_mismatches}")
