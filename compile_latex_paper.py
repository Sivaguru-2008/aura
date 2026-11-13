# -*- coding: utf-8 -*-
"""Compile Nature / npj Digital Medicine manuscript to PDF using pdflatex and verify zero broken references."""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("e:/AURA")

def compile_doc(tex_name, job_name):
    tex_path = ROOT / tex_name
    print(f"\n--- Compiling {tex_path.name} (jobname: {job_name}) ---")
    
    # Run 2 passes to resolve all citations (\cite), labels (\ref), and references
    for pass_num in (1, 2):
        cmd = [
            "pdflatex",
            "-interaction=nonstopmode",
            f"-jobname={job_name}",
            tex_path.name
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
        if res.returncode != 0:
            print(f"Pass {pass_num} error output:")
            print(res.stdout[-1500:])
            print(res.stderr[-1000:])
            raise RuntimeError(f"pdflatex failed on pass {pass_num} for {tex_name}")
        print(f"Pass {pass_num} succeeded.")

    pdf_path = ROOT / f"{job_name}.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"Expected {pdf_path} was not generated.")
    print(f"Successfully generated {pdf_path.name} ({pdf_path.stat().st_size:,} bytes)")
    return pdf_path

def verify_and_distribute():
    # 1. Compile manuscript
    main_pdf = compile_doc("manuscript.tex", "main")
    
    # 2. Compile supplementary
    supp_pdf = compile_doc("supplementary_information.tex", "supplementary_information")
    
    # 3. Verify zero [?] and zero ?? using pypdf
    import pypdf
    for p in (main_pdf, supp_pdf):
        reader = pypdf.PdfReader(str(p))
        num_pages = len(reader.pages)
        full_text = ""
        q_count = 0
        broken_ref_count = 0
        for idx, page in enumerate(reader.pages):
            txt = page.extract_text() or ""
            full_text += txt
            q_page = txt.count("[?]")
            b_page = txt.count("??")
            if q_page > 0:
                print(f"ERROR: {p.name} page {idx+1} contains {q_page} occurrences of [?]")
                q_count += q_page
            if b_page > 0:
                print(f"ERROR: {p.name} page {idx+1} contains {b_page} occurrences of ??")
                broken_ref_count += b_page
        
        print(f"VERIFICATION {p.name}: {num_pages} pages, [?]: {q_count}, ??: {broken_ref_count}")
        if q_count > 0 or broken_ref_count > 0:
            raise ValueError(f"Verification failed for {p.name}: broken references detected.")

    # 4. Mirror copies everywhere
    ms_dir = ROOT / "manuscript"
    ms_dir.mkdir(exist_ok=True)
    
    # Main mirrors
    targets_main = [
        ROOT / "manuscript.pdf",
        ROOT / "MANUSCRIPT_FINAL_npjDM.pdf",
        ROOT / "manuscript_latex.pdf",
        ms_dir / "main.pdf",
        ms_dir / "manuscript.pdf",
        ms_dir / "MANUSCRIPT_FINAL_npjDM.pdf",
        ms_dir / "manuscript_latex.pdf"
    ]
    for t in targets_main:
        shutil.copy2(main_pdf, t)
        print(f"Mirrored main -> {t}")

    # Supplementary mirrors
    targets_supp = [
        ROOT / "supplementary_latex.pdf",
        ms_dir / "supplementary_information.pdf",
        ms_dir / "supplementary_latex.pdf"
    ]
    for t in targets_supp:
        shutil.copy2(supp_pdf, t)
        print(f"Mirrored supp -> {t}")

    print("\nALL PDF BUILDS AND VERIFICATIONS COMPLETED SUCCESSFULLY WITH ZERO UNRESOLVED REFERENCES!")

if __name__ == "__main__":
    verify_and_distribute()
