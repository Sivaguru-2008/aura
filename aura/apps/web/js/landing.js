/* ============================================================
   LANDING — five scenes, each with its own physics.
   hero: intelligence forming · pipeline: scroll-built system ·
   benchmark: instrumentation · safety: tension · portal: exit.
   ============================================================ */
window.LANDING = (() => {
  "use strict";
  const { Field, scene, countTo, api, clamp, lerp, REDUCED } = FX;

  /* ================= HERO — a living constellation ================= */
  function hero() {
    const cv = document.getElementById("hero-field");
    if (!cv) return;
    const f = new Field(cv, { count: 150, hue: 168, links: true, linkDist: 120, mode: "orbit", ringR: 0.36, size: 1.5 });
    f.start();
    window._heroField = f;
  }

  /* ================= PIPELINE — the system assembles on scroll ================= */
  const STAGES = [
    { key: "VISION",    title: "Vision",            desc: "Reads the study and reports observations — opacity, effusion, nodule — never conclusions. Observations are evidence, not verdicts." },
    { key: "EVIDENCE",  title: "Evidence graph",    desc: "Findings and structured priors become eight calibrated evidence channels. What is absent is recorded as loudly as what is present." },
    { key: "FUSION",    title: "Quantum fusion",    desc: "An 8-qubit variational circuit angle-encodes the evidence; entangling layers capture interactions a product of experts cannot. A classical Bayesian twin runs beside it." },
    { key: "SAFETY",    title: "Safety",            desc: "Deep-ensemble epistemic variance, temperature scaling, conformal sets with a 90% coverage guarantee, OOD energy — and the license to say “I don't know.”" },
    { key: "EXPLAIN",   title: "Explainability",    desc: "Occlusion saliency over the image; Shapley attribution and counterfactuals over every evidence node. Nothing the system believes is opaque." },
    { key: "RECOMMEND", title: "Missing evidence",  desc: "Ranks the next test by expected information gain per unit cost and risk — the question that shrinks the differential fastest." },
    { key: "REPORT",    title: "Grounded report",   desc: "Every sentence traces to the evidence nodes that produced it. The clinician signs; the system learns from the verdict." },
  ];

  // glyphs = 16-point radial profiles → true SVG path morphing
  const GLYPHS = {
    eye:      [.9,.55,.4,.35,.32,.35,.4,.55,.9,.55,.4,.35,.32,.35,.4,.55],
    graph:    [.95,.3,.7,.3,.95,.3,.7,.3,.95,.3,.7,.3,.95,.3,.7,.3],
    quantum:  [.9,.45,.9,.45,.9,.45,.9,.45,.9,.45,.9,.45,.9,.45,.9,.45],
    shield:   [.55,.62,.72,.85,.95,.85,.72,.62,.55,.52,.5,.5,.5,.5,.5,.52],
    lens:     [.85,.8,.85,.8,.85,.8,.85,.8,.5,.45,.5,.45,.5,.45,.5,.45],
    compass:  [1,.35,.55,.35,1,.35,.55,.35,1,.35,.55,.35,1,.35,.55,.35],
    doc:      [.8,.72,.66,.72,.8,.72,.66,.72,.8,.72,.66,.72,.8,.72,.66,.72],
  };
  const STAGE_GLYPH = ["eye", "graph", "quantum", "shield", "lens", "compass", "doc"];

  function glyphPath(profile, spin = 0) {
    const n = profile.length, pts = [];
    for (let i = 0; i < n; i++) {
      const a = (i / n) * Math.PI * 2 - Math.PI / 2 + spin;
      const r = profile[i] * 46;
      pts.push([Math.cos(a) * r, Math.sin(a) * r]);
    }
    // smooth closed catmull-rom → bezier
    let d = `M ${pts[0][0].toFixed(1)} ${pts[0][1].toFixed(1)} `;
    for (let i = 0; i < n; i++) {
      const p0 = pts[(i - 1 + n) % n], p1 = pts[i], p2 = pts[(i + 1) % n], p3 = pts[(i + 2) % n];
      const c1 = [p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6];
      const c2 = [p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6];
      d += `C ${c1[0].toFixed(1)} ${c1[1].toFixed(1)} ${c2[0].toFixed(1)} ${c2[1].toFixed(1)} ${p2[0].toFixed(1)} ${p2[1].toFixed(1)} `;
    }
    return d + "Z";
  }

  function pipeline() {
    const sec = document.getElementById("pipeline");
    const svg = document.getElementById("pipe-svg");
    if (!sec || !svg) return;
    const NS = "http://www.w3.org/2000/svg";
    const N = STAGES.length;

    // node positions on a gentle wave
    const pos = STAGES.map((_, i) => {
      const x = 60 + (880 * i) / (N - 1);
      const y = 150 + Math.sin((i / (N - 1)) * Math.PI * 1.7) * 58;
      return [x, y];
    });
    let railD = `M ${pos[0][0]} ${pos[0][1]}`;
    for (let i = 1; i < N; i++) {
      const [px, py] = pos[i - 1], [x, y] = pos[i];
      railD += ` C ${px + 70} ${py}, ${x - 70} ${y}, ${x} ${y}`;
    }
    const mk = (t, attrs) => { const e = document.createElementNS(NS, t); for (const k in attrs) e.setAttribute(k, attrs[k]); svg.appendChild(e); return e; };
    mk("path", { d: railD, class: "rail" });
    const lit = mk("path", { d: railD, class: "rail-lit" });
    const railLen = lit.getTotalLength();
    lit.style.strokeDasharray = railLen;
    lit.style.strokeDashoffset = railLen;

    const nodes = pos.map(([x, y], i) => {
      const c = mk("circle", { cx: x, cy: y, r: 11, class: "node" });
      mk("text", { x, y: y + (i % 2 ? 34 : -26), "text-anchor": "middle", class: "node-lbl" }).textContent = STAGES[i].key;
      return c;
    });
    const lbls = [...svg.querySelectorAll(".node-lbl")];
    // motes flowing along the rail
    const motes = Array.from({ length: 9 }, (_, i) => mk("circle", { r: 2.2, class: "mote", opacity: 0 }));

    const glyphEl = document.getElementById("pipe-glyph-path");
    const titleEl = document.getElementById("pipe-title");
    const descEl = document.getElementById("pipe-desc");
    const idxEl = document.getElementById("pipe-index");
    let cur = -1, morphFrom = GLYPHS.eye, morphTo = GLYPHS.eye, morphP = 1, spin = 0;

    // continuous glyph morph/rotation
    (function glyphTick() {
      spin += 0.0035;
      morphP = Math.min(1, morphP + 0.045);
      const e = 1 - Math.pow(1 - morphP, 3);
      const prof = morphFrom.map((v, i) => lerp(v, morphTo[i], e));
      glyphEl.setAttribute("d", glyphPath(prof, spin));
      requestAnimationFrame(glyphTick);
    })();

    scene(sec.querySelector(".pin-space"), (p) => {
      const draw = clamp(p * 1.12, 0, 1);
      lit.style.strokeDashoffset = railLen * (1 - draw);
      const idx = clamp(Math.floor(draw * N), 0, N - 1);
      nodes.forEach((n, i) => { n.classList.toggle("lit", i <= draw * N - 0.5); lbls[i].classList.toggle("lit", i <= draw * N - 0.5); });
      motes.forEach((m, i) => {
        const t = (draw - i * 0.035);
        if (t <= 0 || draw >= 1) { m.setAttribute("opacity", 0); return; }
        const pt = lit.getPointAtLength(railLen * clamp(t, 0, 1));
        m.setAttribute("cx", pt.x); m.setAttribute("cy", pt.y);
        m.setAttribute("opacity", 0.9);
      });
      if (idx !== cur) {
        cur = idx;
        const s = STAGES[idx];
        idxEl.textContent = String(idx + 1).padStart(2, "0") + " / 07";
        titleEl.textContent = s.title;
        descEl.textContent = s.desc;
        [titleEl, descEl].forEach((el) => { el.classList.remove("sw"); void el.offsetWidth; el.classList.add("sw"); });
        morphFrom = morphFrom.map((v, i) => lerp(v, morphTo[i], 1 - Math.pow(1 - morphP, 3)));
        morphTo = GLYPHS[STAGE_GLYPH[idx]];
        morphP = 0;
      }
    });
  }

  /* ================= BENCHMARK — scientific instrumentation ================= */
  const FALLBACK_BENCH = {
    quantum:   { accuracy: 0.96, nll: 0.0925, ece: 0.02,   brier: 0.0597 },
    classical: { accuracy: 0.93, nll: 0.4877, ece: 0.2755, brier: 0.2035 },
  };

  function gaugeSVG(el, color) {
    el.innerHTML = `<svg viewBox="0 0 190 120">
      <g class="ticks"></g>
      <path class="arc-bg" d="M 20 100 A 75 75 0 0 1 170 100"/>
      <path class="arc-val" d="M 20 100 A 75 75 0 0 1 170 100" stroke="${color}"/>
      <line class="needle" x1="95" y1="100" x2="95" y2="34" transform="rotate(-90 95 100)"/>
      <circle cx="95" cy="100" r="4" fill="${color}"/>
      <text class="g-num" x="95" y="88">0%</text>
      <text class="g-lbl" x="95" y="116">ACCURACY</text>
    </svg>`;
    const ticks = el.querySelector(".ticks");
    for (let i = 0; i <= 10; i++) {
      const a = -Math.PI + (i / 10) * Math.PI;
      const x1 = 95 + Math.cos(a) * 80, y1 = 100 + Math.sin(a) * 80;
      const x2 = 95 + Math.cos(a) * (i % 5 ? 74 : 70), y2 = 100 + Math.sin(a) * (i % 5 ? 74 : 70);
      const t = document.createElementNS("http://www.w3.org/2000/svg", "line");
      t.setAttribute("x1", x1); t.setAttribute("y1", y1); t.setAttribute("x2", x2); t.setAttribute("y2", y2);
      t.setAttribute("class", "tick");
      ticks.appendChild(t);
    }
    const arc = el.querySelector(".arc-val");
    const L = arc.getTotalLength();
    arc.style.strokeDasharray = L; arc.style.strokeDashoffset = L;
    return {
      set(v) {
        arc.style.strokeDashoffset = L * (1 - v);
        el.querySelector(".needle").setAttribute("transform", `rotate(${-90 + v * 180} 95 100)`);
        countTo(el.querySelector(".g-num"), v * 100, { dur: 1600, fmt: (x) => x.toFixed(1) + "%" });
      },
    };
  }

  function meterRow(name, val, max, good) {
    return `<div class="meter ${good ? "good" : "bad"}" data-v="${val / max}">
      <div class="m-row"><span>${name}</span><span class="m-val">${val.toFixed(4)}</span></div>
      <div class="m-track"><div class="m-fill"></div></div></div>`;
  }

  function scope(bench) {
    const cv = document.getElementById("scope");
    if (!cv) return;
    const g = cv.getContext("2d");
    let w, h;
    const fit = () => {
      const dpr = Math.min(devicePixelRatio || 1, 2);
      const r = cv.getBoundingClientRect();
      w = r.width; h = r.height;
      cv.width = w * dpr; cv.height = h * dpr;
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    fit(); addEventListener("resize", fit);
    // reliability curves synthesized from ECE (visual model: overconfidence bends the trace below the diagonal)
    const curve = (ece, x) => Math.pow(clamp(x, 0, 1), 1 + ece * 5.5);
    let t = 0, mx = -1, my = -1;
    cv.addEventListener("pointermove", (e) => { const r = cv.getBoundingClientRect(); mx = e.clientX - r.left; my = e.clientY - r.top; });
    cv.addEventListener("pointerleave", () => { mx = my = -1; });
    const P = 34; // padding
    (function draw(now) {
      t += 0.016;
      g.clearRect(0, 0, w, h);
      // grid
      g.strokeStyle = "rgba(255,255,255,0.05)"; g.lineWidth = 1;
      for (let i = 0; i <= 10; i++) {
        const x = P + ((w - 2 * P) * i) / 10, y = P + ((h - 2 * P) * i) / 10;
        g.beginPath(); g.moveTo(x, P); g.lineTo(x, h - P); g.stroke();
        g.beginPath(); g.moveTo(P, y); g.lineTo(w - P, y); g.stroke();
      }
      // diagonal (perfect calibration)
      g.strokeStyle = "rgba(255,255,255,0.22)"; g.setLineDash([4, 5]);
      g.beginPath(); g.moveTo(P, h - P); g.lineTo(w - P, P); g.stroke();
      g.setLineDash([]);
      // sweep progress (draws forever, like a scope trace)
      const sweep = REDUCED ? 1 : (Math.sin(t * 0.5) * 0.5 + 0.5) * 0.25 + 0.75;
      const traces = [
        { ece: bench.classical.ece, col: "rgba(255,93,93,0.9)", glow: "rgba(255,93,93,0.4)" },
        { ece: bench.quantum.ece, col: "rgba(139,124,247,1)", glow: "rgba(139,124,247,0.6)" },
      ];
      for (const tr of traces) {
        g.beginPath();
        for (let i = 0; i <= 100 * sweep; i++) {
          const x = i / 100;
          const px = P + (w - 2 * P) * x;
          const py = h - P - (h - 2 * P) * curve(tr.ece, x);
          i === 0 ? g.moveTo(px, py) : g.lineTo(px, py);
        }
        g.shadowColor = tr.glow; g.shadowBlur = 8;
        g.strokeStyle = tr.col; g.lineWidth = 1.8; g.stroke();
        g.shadowBlur = 0;
      }
      // crosshair
      if (mx >= 0) {
        g.strokeStyle = "rgba(255,255,255,0.25)";
        g.setLineDash([3, 4]);
        g.beginPath(); g.moveTo(mx, P); g.lineTo(mx, h - P); g.stroke();
        g.beginPath(); g.moveTo(P, my); g.lineTo(w - P, my); g.stroke();
        g.setLineDash([]);
        const conf = clamp((mx - P) / (w - 2 * P), 0, 1);
        g.fillStyle = "rgba(233,237,245,0.85)";
        g.font = "10px ui-monospace, monospace";
        g.fillText(`conf ${conf.toFixed(2)} → q:${curve(bench.quantum.ece, conf).toFixed(2)} c:${curve(bench.classical.ece, conf).toFixed(2)}`, Math.min(mx + 8, w - 170), Math.max(my - 8, 14));
      }
      // axes labels
      g.fillStyle = "rgba(86,95,116,0.9)"; g.font = "9px ui-monospace, monospace";
      g.fillText("stated confidence →", P, h - 10);
      g.save(); g.translate(12, h - P); g.rotate(-Math.PI / 2); g.fillText("observed accuracy →", 0, 0); g.restore();
      requestAnimationFrame(draw);
    })();
  }

  async function benchmark() {
    const rig = document.getElementById("bench-rig");
    if (!rig) return;
    let bench = FALLBACK_BENCH;
    try {
      const d = await api("/v1/admin/safety");
      if (d.benchmark && d.benchmark.quantum) bench = d.benchmark;
    } catch { /* offline — artifact values */ }

    const gq = gaugeSVG(document.getElementById("gauge-q"), "#8b7cf7");
    const gc = gaugeSVG(document.getElementById("gauge-c"), "#ff5d5d");
    const q = bench.quantum, c = bench.classical;
    const maxNll = Math.max(q.nll, c.nll) * 1.15, maxEce = Math.max(q.ece, c.ece) * 1.15, maxBr = Math.max(q.brier, c.brier) * 1.15;
    document.getElementById("meters-q").innerHTML =
      meterRow("ECE · calibration error", q.ece, maxEce, true) +
      meterRow("NLL · surprise", q.nll, maxNll, true) +
      meterRow("BRIER", q.brier, maxBr, true);
    document.getElementById("meters-c").innerHTML =
      meterRow("ECE · calibration error", c.ece, maxEce, false) +
      meterRow("NLL · surprise", c.nll, maxNll, false) +
      meterRow("BRIER", c.brier, maxBr, false);
    const ratio = c.ece / Math.max(q.ece, 1e-6);
    document.getElementById("bench-verdict").innerHTML =
      `Same evidence. Same conformal guarantee. The quantum posterior is <b>${ratio.toFixed(1)}× better calibrated</b> — when AURA says ${Math.round(q.accuracy * 100)}%, it means ${Math.round(q.accuracy * 100)}%.`;

    // fire when instrumentation scrolls into view
    const io = new IntersectionObserver((es) => {
      es.forEach((e) => {
        if (!e.isIntersecting) return;
        io.disconnect();
        gq.set(q.accuracy); gc.set(c.accuracy);
        rig.querySelectorAll(".meter").forEach((m, i) => {
          setTimeout(() => { m.querySelector(".m-fill").style.width = clamp(+m.dataset.v, 0.02, 1) * 100 + "%"; }, 200 + i * 120);
        });
      });
    }, { threshold: 0.35 });
    io.observe(rig);
    scope(bench);
  }

  /* ================= SAFETY — tension, then the refusal ================= */
  function safety() {
    const sec = document.getElementById("safety");
    if (!sec) return;
    const ecg = document.getElementById("ecg");
    const g = ecg.getContext("2d");
    let w, h;
    const fit = () => {
      const dpr = Math.min(devicePixelRatio || 1, 2);
      const r = ecg.getBoundingClientRect();
      w = r.width; h = r.height;
      ecg.width = w * dpr; ecg.height = h * dpr;
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    fit(); addEventListener("resize", fit);

    const numEl = document.getElementById("conf-num");
    const chipsEl = document.getElementById("safety-chips");
    const slab = document.getElementById("abstain-slab");
    const h2 = document.getElementById("safety-h2");
    const pinEl = sec.querySelector(".pin");
    const SET = ["Pneumonia", "Heart failure", "Malignancy", "COPD"];
    let tension = 0, abst = false, phase = 0, chipCount = 0;

    scene(sec.querySelector(".pin-space"), (p) => {
      const drop = clamp((p - 0.08) / 0.62, 0, 1);
      const conf = lerp(0.93, 0.41, 1 - Math.pow(1 - drop, 2));
      tension = drop;
      numEl.textContent = conf.toFixed(2);
      numEl.classList.toggle("warn", conf < 0.75 && conf >= 0.55);
      numEl.classList.toggle("crit", conf < 0.55);
      pinEl.style.setProperty("--tension", (drop * 0.9).toFixed(3));
      const want = 1 + Math.floor(drop * 3.6);
      if (want !== chipCount) {
        chipCount = want;
        chipsEl.innerHTML = SET.slice(0, want)
          .map((s, i) => `<span class="conf-chip ${i > 0 ? "hot" : ""}">${s}</span>`).join("");
      }
      const nowAbst = p > 0.78;
      if (nowAbst !== abst) {
        abst = nowAbst;
        slab.classList.toggle("on", abst);
        h2.textContent = abst ? "So it doesn't." : "It refuses to guess.";
      }
    });

    // ECG — speeds and destabilizes with tension; goes calm+flat after abstention
    (function beat(now) {
      phase += 0.016 * (1 + tension * 2.4);
      g.clearRect(0, 0, w, h);
      const mid = h * 0.55;
      g.beginPath();
      const abstained = abst;
      for (let x = 0; x < w; x += 2) {
        const tt = phase + x * 0.02;
        let y;
        if (abstained) {
          y = mid + Math.sin(tt * 0.8) * 2.5; // sedated — the system took over
        } else {
          const cyc = tt % 3;
          const spike = cyc < 0.25 ? Math.sin((cyc / 0.25) * Math.PI) * (26 + tension * 34) : 0;
          const jitter = tension * 7 * Math.sin(tt * 13.7) * Math.sin(tt * 5.3);
          y = mid - spike + Math.sin(tt * 1.4) * 4 + jitter;
        }
        x === 0 ? g.moveTo(x, y) : g.lineTo(x, y);
      }
      const col = abstained ? "rgba(75,225,195,0.9)" : `rgba(${Math.round(lerp(75, 255, tension))}, ${Math.round(lerp(225, 93, tension))}, ${Math.round(lerp(195, 93, tension))}, 0.9)`;
      g.shadowColor = col; g.shadowBlur = 10;
      g.strokeStyle = col; g.lineWidth = 1.6; g.stroke();
      g.shadowBlur = 0;
      requestAnimationFrame(beat);
    })();
  }

  /* ================= PORTAL ================= */
  function portal() {
    const cv = document.getElementById("portal-canvas");
    if (!cv) return;
    const f = new Field(cv, { count: 170, hue: 172, links: true, linkDist: 100, mode: "orbit", ringR: 0.4, size: 1.4, speed: 1.2 });
    f.start();
    // live footer stats
    api("/v1/health").then((hh) => {
      document.getElementById("portal-sub").textContent =
        `gateway online · fusion: ${hh.backend} · ${hh.cases} live cases · conformal coverage 90%`;
    }).catch(() => {});
  }

  function init() { hero(); pipeline(); benchmark(); safety(); portal(); }
  return { init };
})();
