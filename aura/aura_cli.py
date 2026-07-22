"""AURA command-line entry point.

Model / training / serving (unchanged):
    py -m aura_cli train [n]        fit vision detectors + fusion models + calibration
    py -m aura_cli train-cnn [arch] fine-tune a CNN vision backbone on GPU (timm)
    py -m aura_cli bench [n]        quantum-vs-classical fusion benchmark + full metrics
    py -m aura_cli serve [port]     start gateway + dashboard
    py -m aura_cli demo             train (if needed) then serve

Production evaluation (new):
    py -m aura_cli predict --image sample.jpg [--out DIR] [--no-explain]
    py -m aura_cli evaluate [--limit N] [--bootstrap N] [--no-plots]
    py -m aura_cli explain --image sample.jpg [--out DIR] [--no-scorecam]
    py -m aura_cli benchmark [--iters N]

Run from the aura/ directory.  train-cnn arch in {densenet121, efficientnetv2,
convnext, swin}; pass a manifest CSV via AURA_CNN_MANIFEST to train on real CXRs.
The predict/evaluate/explain/benchmark commands use the trained DenseNet-121 in
artifacts/best_model.pt, loaded automatically by the vision engine.
"""
from __future__ import annotations

import argparse
import os
import sys

from common.config import ARTIFACTS


def _utf8_stdout() -> None:
    """Make stdout/stderr UTF-8 so reports with non-ASCII glyphs never crash on Windows."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _need_training() -> bool:
    return not (ARTIFACTS / "fusion_quantum.npz").exists()


# --------------------------------------------------------------------------- #
# Existing commands (behavior preserved exactly)
# --------------------------------------------------------------------------- #
def cmd_train(argv: list[str]) -> None:
    n = int(argv[0]) if argv else 700
    from ml.training import train_vision, train_fusion
    train_vision.run(n)
    train_fusion.run(n)


def cmd_train_cnn(argv: list[str]) -> None:
    from ml.training.train_cnn import run, TrainConfig
    arch = argv[0] if argv else "densenet121"
    manifest = os.environ.get("AURA_CNN_MANIFEST")
    run(manifest=manifest, synthetic=manifest is None, cfg=TrainConfig(arch=arch))


def cmd_bench(argv: list[str]) -> None:
    n = int(argv[0]) if argv else 500
    from ml.evaluation import benchmark
    benchmark.run(n)


def cmd_serve(argv: list[str]) -> None:
    port = int(argv[0]) if argv else 8000
    import uvicorn
    if _need_training():
        print("[aura] no trained models found - training first ...")
        cmd_train([])
    uvicorn.run("gateway.app:app", host="127.0.0.1", port=port, log_level="info")


def cmd_demo(argv: list[str]) -> None:
    if _need_training():
        cmd_train([])
        cmd_bench([])
    cmd_serve(argv)


# --------------------------------------------------------------------------- #
# New production-evaluation commands
# --------------------------------------------------------------------------- #
def cmd_predict(args: argparse.Namespace) -> None:
    from services.inference.predict import predict_image

    if not args.image:
        print("error: predict requires --image PATH")
        sys.exit(2)
    res = predict_image(
        args.image, out_dir=args.out,
        save_explain=not args.no_explain, save_reports=not args.no_report,
        include_scorecam=not args.no_scorecam,
    )
    _print_prediction(res, show_report=not args.brief)


def cmd_evaluate(args: argparse.Namespace) -> None:
    from ml.evaluation import clinical_eval

    clinical_eval.evaluate_validation(
        limit=args.limit, n_bootstrap=args.bootstrap, make_plots=not args.no_plots,
    )
    rep_path = clinical_eval.EVAL_DIR / "EVALUATION_SUMMARY.md"
    print(rep_path.read_text(encoding="utf-8"))

    if args.calibrate:
        from ml.evaluation import vision_calibration
        vision_calibration.run_calibration(limit=args.limit, make_plots=not args.no_plots)
        print((vision_calibration.CAL_DIR / "CALIBRATION_SUMMARY.md").read_text(encoding="utf-8"))


def cmd_explain(args: argparse.Namespace) -> None:
    from services.inference.predict import predict_image

    if not args.image:
        print("error: explain requires --image PATH")
        sys.exit(2)
    res = predict_image(
        args.image, out_dir=args.out, save_explain=True, save_reports=False,
        include_scorecam=not args.no_scorecam,
    )
    print(f"\nExplainability for study {res['study_id']}")
    print(f"  target finding : {res['clinical_report']['vision_findings'][0]['finding'] if res['clinical_report']['vision_findings'] else '-'}")
    methods = [k.replace("method_", "") for k in res["artifacts"] if k.startswith("method_")]
    print(f"  saliency methods: {', '.join(methods)}")
    print("  artifacts:")
    for k, v in res["artifacts"].items():
        if k.startswith("method_") or k.endswith("_png") or k == "explanation_html":
            print(f"    {k:22} {v}")


def cmd_benchmark(args: argparse.Namespace) -> None:
    from ml.evaluation import perf_benchmark

    perf_benchmark.run(iters=args.iters)


def cmd_calibrate(args: argparse.Namespace) -> None:
    from ml.evaluation import vision_calibration

    vision_calibration.run_calibration(limit=args.limit, make_plots=not args.no_plots)
    print((vision_calibration.CAL_DIR / "CALIBRATION_SUMMARY.md").read_text(encoding="utf-8"))


def cmd_agent(args: argparse.Namespace) -> None:
    import cv2
    import numpy as np
    from pathlib import Path
    from services.vision.engine import VisionEngine
    from services.fusion.engine import FusionEngine
    from services.fusion.evidence import encode
    from services.agent.active_diagnosis import ActiveDiagnosisAgent
    from schemas.contracts import StructuredPriors

    if not args.image:
        print("error: agent requires --image PATH")
        sys.exit(2)
        
    image_path = Path(args.image)
    if not image_path.is_file():
        print(f"error: image file not found: {image_path}")
        sys.exit(1)
        
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"error: failed to load/decode image: {image_path}")
        sys.exit(1)
        
    vision_engine = VisionEngine.load()
    fusion_engine = FusionEngine()
    
    # 1) Run Vision Model to obtain raw probabilities
    vision_result = vision_engine.analyze("STU-CLI-AGENT", img)
    
    # 2) Encode evidence
    priors = StructuredPriors()
    x = encode(vision_result, priors)
    
    # 3) Run agent
    agent = ActiveDiagnosisAgent(
        fusion_model=fusion_engine,
        entropy_target_bits=args.entropy_target,
        confidence=args.confidence,
        max_tests=args.max_tests
    )
    trajectory = agent.diagnose(x)
    _print_agent_trajectory(trajectory)


def _print_agent_trajectory(trajectory) -> None:
    print(f"\n=== Active Diagnosis Trajectory ({trajectory.backend.upper()}) ===")
    print(f"Initial Uncertainty: {trajectory.initial_entropy:.3f} bits")
    print(f"Final Uncertainty  : {trajectory.final_entropy:.3f} bits (reduction: {trajectory.bits_resolved:+.3f} bits)")
    print(f"Acquired Tests     : {trajectory.n_tests}")
    print(f"Status             : {trajectory.status.upper()}")
    print(f"Final Diagnosis    : {trajectory.final_diagnosis} (p={trajectory.final_probability:.1%})")
    print("-" * 68)
    for step in trajectory.steps:
        print(f"\nStep {step.step}:")
        print(f"  calibrated differential:")
        for dx, prob in list(step.posterior.items())[:3]:
            print(f"    {dx:<25}: {prob:.1%}")
        if len(step.posterior) > 3:
            print(f"    ...")
        print(f"  predictive entropy      : {step.entropy_bits:.3f} bits")
        if step.action_display:
            print(f"  recommended next step   : {step.action_display} (EIG={step.action_eig_bits} bits)")
            print(f"  acquired findings       : {step.resolved}")
        if step.decision:
            print(f"  decision                : {step.decision}")


# --------------------------------------------------------------------------- #
# Pretty printing
# --------------------------------------------------------------------------- #
def _print_prediction(res: dict, show_report: bool = True) -> None:
    print(f"\n=== AURA prediction: {res['study_id']} ===\n")
    print("Vision findings (DenseNet-121):")
    for f in res["findings"]:
        mark = "POSITIVE" if f["present"] else "        "
        bar = "#" * int(round(f["probability"] * 20))
        print(f"  {f['finding']:<20} {f['probability']:.3f}  {mark}  {bar}")
    c = res["clinical_report"]["confidence"]
    print(f"\nDifferential (calibrated):")
    for d in res["clinical_report"]["differential_diagnosis"]:
        print(f"  {d['diagnosis']:<40} {d['probability']:.1%}")
    print(f"\nTop diagnosis   : {res['top_diagnosis']} ({res['top_probability']:.1%})")
    cal = res["clinical_report"]["calibration"]
    print(f"Confidence set  : {', '.join(cal['conformal_set'])} "
          f"({int((cal['conformal_coverage'] or 0)*100)}% {cal['conformal_method']})")
    print(f"Risk level      : {res['risk_level']['level']} - {res['risk_level']['rationale']}")
    rt = res["clinical_report"]["recommended_tests"]
    if rt:
        print(f"Recommended     : {rt[0]['test']} ({rt[0]['cost']} cost)")
    print(f"Inference time  : {res['inference_time_s']*1000:.0f} ms")
    print("\nArtifacts:")
    for k in ("report_markdown", "overlay_png", "overlay_hires_png", "explanation_html"):
        if k in res["artifacts"]:
            print(f"  {k:22} {res['artifacts'][k]}")
    if show_report:
        from services.report.clinical_report import render_text
        print("\n" + "-" * 68)
        print(render_text(res["clinical_report"]))


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
_LEGACY = {"train": cmd_train, "train-cnn": cmd_train_cnn, "bench": cmd_bench,
           "serve": cmd_serve, "demo": cmd_demo}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aura_cli", add_help=True,
                                description="AURA clinical AI command-line interface.")
    sub = p.add_subparsers(dest="command")

    pr = sub.add_parser("predict", help="run full inference on one radiograph")
    pr.add_argument("--image", required=False, help="path to a CXR (jpg/png/dicom)")
    pr.add_argument("--out", default=None, help="output directory for artifacts")
    pr.add_argument("--no-explain", action="store_true", help="skip saliency artifacts")
    pr.add_argument("--no-report", action="store_true", help="skip report files")
    pr.add_argument("--no-scorecam", action="store_true", help="skip Score-CAM (faster)")
    pr.add_argument("--brief", action="store_true", help="summary only (no full report text)")
    pr.set_defaults(func=cmd_predict)

    ev = sub.add_parser("evaluate", help="evaluate on the MIMIC-CXR validation set")
    ev.add_argument("--limit", type=int, default=None, help="cap #patients (debug)")
    ev.add_argument("--bootstrap", type=int, default=1000, help="bootstrap iterations")
    ev.add_argument("--no-plots", action="store_true")
    ev.add_argument("--calibrate", action="store_true", help="also run calibration suite")
    ev.set_defaults(func=cmd_evaluate)

    ex = sub.add_parser("explain", help="write saliency overlays for one radiograph")
    ex.add_argument("--image", required=False)
    ex.add_argument("--out", default=None)
    ex.add_argument("--no-scorecam", action="store_true")
    ex.set_defaults(func=cmd_explain)

    bm = sub.add_parser("benchmark", help="latency / throughput / memory benchmark")
    bm.add_argument("--iters", type=int, default=50)
    bm.set_defaults(func=cmd_benchmark)

    cal = sub.add_parser("calibrate", help="temperature scaling / MC-dropout / conformal")
    cal.add_argument("--limit", type=int, default=None)
    cal.add_argument("--no-plots", action="store_true")
    cal.set_defaults(func=cmd_calibrate)

    ag = sub.add_parser("agent", help="run sequential diagnostic agent on one radiograph")
    ag.add_argument("--image", required=True, help="path to a CXR (jpg/png/dicom)")
    ag.add_argument("--entropy-target", type=float, default=0.6, help="entropy threshold for commit")
    ag.add_argument("--confidence", type=float, default=0.85, help="confidence threshold for commit")
    ag.add_argument("--max-tests", type=int, default=3, help="max tests to acquire")
    ag.set_defaults(func=cmd_agent)
    return p


def main() -> None:
    _utf8_stdout()
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd = sys.argv[1]

    # Legacy positional commands keep their exact original interface.
    if cmd in _LEGACY:
        _LEGACY[cmd](sys.argv[2:])
        return

    parser = _build_parser()
    if cmd in {"predict", "evaluate", "explain", "benchmark", "calibrate", "agent", "-h", "--help"}:
        args = parser.parse_args(sys.argv[1:])
        if getattr(args, "func", None) is None:
            print(__doc__)
            sys.exit(1)
        args.func(args)
        return

    print(__doc__)
    sys.exit(1)


if __name__ == "__main__":
    main()
