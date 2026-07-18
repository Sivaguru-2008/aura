# -*- coding: utf-8 -*-
"""Generate the hand-drawn / engineering-notebook architecture slide as a
self-contained HTML file (Google Fonts is the only external dep).

Emits two sizes from one template:
  slide_architecture.html        1920x1080 (16:9) — the standalone deliverable
  slide_architecture_4x3.html    1440x1080 (4:3)  — matches the 10x7.5in deck slide

QPU/arrow/note auto-position over the 'fusion' box via JS, so it stays correct at
any width. Logos are pulled from the deck's media, grayscaled, base64-embedded.
"""
import base64, io
from PIL import Image

MEDIA = "_media_extract/ppt/media"
LOGOS = ["image4.png", "image10.png", "image7.png", "image8.jpeg", "image11.png"]

def logo_datauri(fn, h=120):
    im = Image.open(f"{MEDIA}/{fn}").convert("RGBA")
    w = int(im.width * h / im.height)
    im = im.resize((w, h), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

LOGO_TAGS = "\n        ".join(
    f'<img class="logo" src="{logo_datauri(f)}" alt="">' for f in LOGOS)

def gen(W, H, out):
    svc_font = 20 if W >= 1700 else 16      # narrower canvas -> smaller service labels
    title_font = 76 if W >= 1700 else 60
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AURA — Architecture</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Architects+Daughter&family=Caveat:wght@500;700&family=Nunito:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
  :root{{ --paper:#faf7f0; --ink:#2b2b2b; --ink2:#524d45; --pencil:#8a8378; --hl:#ffd23f; }}
  *{{box-sizing:border-box; margin:0; padding:0;}}
  html,body{{width:{W}px; height:{H}px;}}
  body{{
    background-color:var(--paper);
    background-image:radial-gradient(rgba(120,110,90,.16) 1.2px, transparent 1.3px);
    background-size:34px 34px; background-position:12px 12px;
    font-family:'Nunito',sans-serif; color:var(--ink); position:relative; overflow:hidden;
  }}
  .defs{{position:absolute; width:0; height:0;}}
  .slide{{position:absolute; inset:0; padding:52px 84px 36px; display:flex; flex-direction:column;}}

  .stamp{{
    align-self:flex-start; font-family:'Architects Daughter',sans-serif;
    font-size:19px; letter-spacing:3px; color:var(--ink2);
    border:2px solid var(--ink2); border-radius:20px 6px 18px 6px/6px 18px 6px 20px;
    padding:5px 14px; transform:rotate(-1.4deg); opacity:.85;
  }}
  .title{{
    font-family:'Caveat',cursive; font-weight:700; color:var(--ink);
    font-size:{title_font}px; line-height:1.02; margin:12px 0 4px; letter-spacing:.5px;
  }}
  .hl{{position:relative; white-space:nowrap; padding:0 4px;}}
  .hl::before{{
    content:""; position:absolute; left:-2px; right:-2px; top:16%; bottom:10%;
    background:var(--hl); opacity:.55; z-index:-1; transform:rotate(-1.3deg) skewX(-7deg);
    border-radius:14px 8px 12px 10px/10px 12px 8px 14px;
  }}

  .board{{flex:1; display:flex; flex-direction:column; justify-content:space-between; padding:8px 0 2px;}}
  .tier{{position:relative; padding-left:148px;}}
  .tierlabel{{
    position:absolute; left:0; top:6px; width:130px;
    font-family:'Architects Daughter',sans-serif; font-size:22px; color:var(--ink2);
    letter-spacing:1px; text-align:right; padding-right:14px;
    border-bottom:2px solid var(--pencil); padding-bottom:2px;
  }}
  .tierlabel small{{display:block; font-size:13px; color:var(--pencil); letter-spacing:.5px; border:0;}}
  .rule{{height:14px; margin:5px 0;}}
  .rule svg{{width:100%; height:100%; display:block;}}

  .row{{display:flex; gap:24px; align-items:stretch;}}
  .box{{
    background:#fff; border:2px solid var(--ink);
    border-radius:255px 15px 225px 15px/15px 225px 15px 255px;
    font-family:'Architects Daughter',sans-serif; font-size:21px; color:var(--ink);
    padding:12px 16px; min-height:56px; display:flex; align-items:center; justify-content:center;
    text-align:center; flex:1; line-height:1.05;
  }}
  .box.alt{{border-radius:15px 225px 15px 255px/225px 15px 255px 15px; transform:rotate(.4deg);}}
  .box.alt2{{border-radius:225px 15px 255px 15px/15px 255px 15px 225px; transform:rotate(-.5deg);}}

  .app{{padding-bottom:4px;}}
  .apphead{{
    font-family:'Nunito',sans-serif; font-weight:600; font-size:16px; color:var(--ink2);
    text-align:right; margin-bottom:66px;
  }}
  .bay{{position:relative;}}
  .services{{gap:18px;}}
  .services .box{{font-size:{svc_font}px; min-height:60px; padding:10px 10px;}}
  .box.fusion{{border-width:3.5px; font-weight:700;}}
  .box.openbay{{
    border:2px dashed var(--pencil); color:var(--pencil); flex:.8;
    flex-direction:column; gap:0; background:transparent;
  }}
  .box.openbay .plus{{font-size:24px; line-height:1; font-weight:700;}}
  .box.openbay small{{font-size:12px;}}

  .qpu{{
    position:absolute; left:172px; top:-112px; width:250px; height:94px;
    background:rgba(255,210,63,.34); border:2.5px solid var(--ink);
    border-radius:225px 18px 235px 18px/18px 235px 18px 225px;
    display:flex; flex-direction:column; align-items:center; justify-content:center; gap:4px;
    transform:rotate(-.7deg);
  }}
  .qpu::after{{content:""; position:absolute; inset:-7px; border:2px solid var(--ink);
    border-radius:inherit; opacity:.8;}}
  .qpu .t{{font-family:'Architects Daughter',sans-serif; font-size:23px; font-weight:700; letter-spacing:.5px;}}
  .qpu .eq{{font-family:'Architects Daughter',sans-serif; font-size:17px; color:var(--ink2);}}
  .dockarrow{{position:absolute; left:264px; top:-20px; width:64px; height:100px; z-index:3;}}
  .docknote{{
    position:absolute; left:466px; top:-98px; width:200px;
    font-family:'Caveat',cursive; font-weight:700; font-size:28px; color:var(--ink);
    line-height:1.0; transform:rotate(-3deg);
  }}

  .foot{{display:flex; flex-direction:column; gap:4px; margin-top:6px;}}
  .offline{{font-family:'Nunito',sans-serif; font-weight:700; font-size:16px; color:var(--ink);
    border-bottom:2px solid var(--pencil); align-self:flex-start; padding-bottom:2px;}}
  .stack{{font-family:'Nunito',sans-serif; font-size:15px; color:var(--ink2);}}
  .stack b{{font-weight:800; color:var(--ink);}}
  .baseline{{display:flex; align-items:flex-end; justify-content:space-between; margin-top:10px;}}
  .logostrip{{display:flex; align-items:center; gap:24px; height:38px;}}
  .logostrip .logo{{height:40px; width:auto; filter:grayscale(1) contrast(.95) opacity(.82); mix-blend-mode:multiply;}}
  .pagefoot{{font-family:'Nunito',sans-serif; font-size:15px; color:var(--pencil); letter-spacing:.5px;}}
</style>
</head>
<body>
  <svg class="defs">
    <filter id="rough"><feTurbulence type="fractalNoise" baseFrequency="0.014" numOctaves="2" seed="7" result="n"/>
      <feDisplacementMap in="SourceGraphic" in2="n" scale="3.5"/></filter>
  </svg>

  <div class="slide">
    <div class="stamp">04 · ARCHITECTURE &amp; STACK</div>
    <h1 class="title">Modular 3-tier — quantum as a <span class="hl">swappable</span> service</h1>

    <div class="board">
      <div class="tier">
        <div class="tierlabel">PRESENTATION</div>
        <div class="row">
          <div class="box">doctor dashboard</div>
          <div class="box alt">worklist</div>
          <div class="box alt2">case console</div>
          <div class="box">report signing</div>
        </div>
      </div>

      <div class="rule"><svg viewBox="0 0 1600 14" preserveAspectRatio="none">
        <path d="M2,8 C220,3 380,12 600,7 S1010,3 1220,9 S1500,5 1598,8" fill="none"
              stroke="#2b2b2b" stroke-width="2" stroke-linecap="round" filter="url(#rough)"/></svg></div>

      <div class="tier app">
        <div class="tierlabel">APPLICATION<small>services</small></div>
        <div class="apphead">FastAPI gateway · async event bus · typed Pydantic contracts</div>
        <div class="bay">
          <div class="qpu">
            <div class="t">QPU · 8-qubit VQC</div>
            <div class="eq">k(x,x′) = |⟨φ(x)|φ(x′)⟩|²</div>
          </div>
          <svg class="dockarrow" viewBox="0 0 64 100">
            <path d="M32,6 C27,34 37,58 32,82" fill="none" stroke="#2b2b2b" stroke-width="3"
                  stroke-linecap="round" filter="url(#rough)"/>
            <path d="M19,70 L32,88 L45,70" fill="none" stroke="#2b2b2b" stroke-width="3"
                  stroke-linecap="round" stroke-linejoin="round" filter="url(#rough)"/></svg>
          <div class="docknote">docks in — swap anytime</div>
          <div class="row services">
            <div class="box">vision</div>
            <div class="box fusion">fusion</div>
            <div class="box alt">safety</div>
            <div class="box">explain</div>
            <div class="box alt2">recommend</div>
            <div class="box">report</div>
            <div class="box alt">memory</div>
            <div class="box openbay"><span class="plus">+</span><span>OPEN BAY</span><small>add svc</small></div>
          </div>
        </div>
      </div>

      <div class="rule"><svg viewBox="0 0 1600 14" preserveAspectRatio="none">
        <path d="M2,7 C240,11 420,3 640,8 S1040,12 1240,6 S1520,10 1598,7" fill="none"
              stroke="#2b2b2b" stroke-width="2" stroke-linecap="round" filter="url(#rough)"/></svg></div>

      <div class="tier">
        <div class="tierlabel">DATA</div>
        <div class="row">
          <div class="box">audit ledger</div>
          <div class="box alt">case memory</div>
          <div class="box alt2">model registry</div>
          <div class="box">provenance {{seed · circuit · shots}}</div>
        </div>
      </div>
    </div>

    <div class="foot">
      <div class="offline">Offline-capable · no PHI leaves the site</div>
      <div class="stack"><b>Stack:</b> Python 3 · FastAPI · PennyLane · NumPy · scikit-learn · SciPy · SQLite · Pydantic</div>
    </div>

    <div class="baseline">
      <div class="logostrip">
        {LOGO_TAGS}
      </div>
      <div class="pagefoot">RIT – QUANT-A-THAN 2026</div>
    </div>
  </div>

  <script>
    function place(){{
      const bay=document.querySelector('.bay'); if(!bay) return;
      const f=document.querySelector('.box.fusion');
      const qpu=document.querySelector('.qpu'), arrow=document.querySelector('.dockarrow'), note=document.querySelector('.docknote');
      const br=bay.getBoundingClientRect(), fr=f.getBoundingClientRect();
      const fcx=(fr.left+fr.width/2)-br.left;
      const aw=arrow.getBoundingClientRect().width;  // SVG has no offsetWidth
      qpu.style.left=Math.round(fcx-qpu.offsetWidth/2)+'px';
      arrow.style.left=Math.round(fcx-aw/2)+'px';
      let nl=fcx+qpu.offsetWidth/2+18; const maxl=br.width-note.offsetWidth-4;
      if(nl>maxl) nl=maxl; note.style.left=Math.round(nl)+'px';
    }}
    if(document.fonts&&document.fonts.ready) document.fonts.ready.then(place);
    window.addEventListener('load', place); setTimeout(place,250); setTimeout(place,700);
  </script>
</body>
</html>
"""
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print("wrote %s  (%dx%d, %d KB)" % (out, W, H, len(html) // 1024))

gen(1920, 1080, "slide_architecture.html")
gen(1440, 1080, "slide_architecture_4x3.html")
