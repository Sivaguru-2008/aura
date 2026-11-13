"""
Edit line 260 in manuscript.tex and corresponding section in MANUSCRIPT_FINAL_npjDM.md
to tone down causal language regarding trigonometric representations and transfer sensitivity.
"""
from pathlib import Path

tex_path = Path("E:/AURA/aura-main/manuscript.tex")
md_path = Path("E:/AURA/aura-main/MANUSCRIPT_FINAL_npjDM.md")

tex_content = tex_path.read_text(encoding="utf-8")
md_content = md_path.read_text(encoding="utf-8")

old_tex_sentence = "Crucially, a classical order-1 trigonometric Fourier surrogate experienced comparable domain shift degradation ($\\Delta\\text{NLL}_{\\text{shift}} = +0.1434$), confirming that transfer degradation reflects the geometry of periodic trigonometric basis functions under distribution shift. Under our pre-specified decision rule, this confirms that transfer sensitivity is an inherent property of trigonometric feature spaces under distribution shift, rather than quantum state geometry or angle wrapping per se."

new_tex_sentence = "Crucially, a classical order-1 trigonometric Fourier surrogate experienced comparable domain shift degradation ($\\Delta\\text{NLL}_{\\text{shift}} = +0.1434$ nats), which is consistent with sensitivity to trigonometric basis geometries under distribution shift, rather than an effect unique to quantum state spaces or input-scale angle wrapping. Under our pre-specified decision rule, this observation is consistent with trigonometric feature sensitivity in this benchmark rather than proof of a universal property across all trigonometric representations."

assert old_tex_sentence in tex_content, "old_tex_sentence not found in manuscript.tex"
tex_content = tex_content.replace(old_tex_sentence, new_tex_sentence)
tex_path.write_text(tex_content, encoding="utf-8")
print("Updated manuscript.tex successfully!")

# Now check Markdown
old_md_sentence = "Crucially, a classical order-1 trigonometric Fourier surrogate experienced comparable domain shift degradation ($\\Delta\\text{NLL}_{\\text{shift}} = +0.1434$), confirming that transfer degradation reflects the geometry of periodic trigonometric basis functions under distribution shift. Under our pre-specified decision rule, this confirms that transfer sensitivity is an inherent property of trigonometric feature spaces under distribution shift, rather than quantum state geometry or angle wrapping per se."

if old_md_sentence in md_content:
    md_content = md_content.replace(old_md_sentence, new_tex_sentence)
    md_path.write_text(md_content, encoding="utf-8")
    print("Updated MANUSCRIPT_FINAL_npjDM.md successfully!")
else:
    print("Note: old_md_sentence not found verbatim in markdown; searching substring...")
    for line in md_content.splitlines():
        if "Fourier surrogate experienced comparable domain shift" in line:
            print("Found line in md:", line[:100])
            md_content = md_content.replace(line, line.replace(
                "confirming that transfer degradation reflects the geometry of periodic trigonometric basis functions under distribution shift. Under our pre-specified decision rule, this confirms that transfer sensitivity is an inherent property of trigonometric feature spaces under distribution shift, rather than quantum state geometry or angle wrapping per se.",
                "which is consistent with sensitivity to trigonometric basis geometries under distribution shift, rather than an effect unique to quantum state spaces or input-scale angle wrapping. Under our pre-specified decision rule, this observation is consistent with trigonometric feature sensitivity in this benchmark rather than proof of a universal property across all trigonometric representations."
            ))
            md_path.write_text(md_content, encoding="utf-8")
            print("Updated MANUSCRIPT_FINAL_npjDM.md with line replace!")
            break
