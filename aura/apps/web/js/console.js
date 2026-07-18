/* ============================================================
   CONSOLE — the living clinical system behind /app.
   Every panel is wired to the real gateway. Nothing is static:
   cases converge into reasoning, charts spring, reports type
   themselves, and the system reacts to the clinician's verdict.
   ============================================================ */
window.CONSOLE = (() => {
  "use strict";
  const { Field, api, toast, typeInto, clamp, REDUCED } = FX;

  const DX_LABEL = {
    normal: "No acute abnormality", pneumonia: "Pneumonia",
    heart_failure: "Heart failure", copd: "COPD",
    malignancy: "Malignancy", pneumothorax_dx: "Pneumothorax",
  };
  const EV_LABEL = {
    opacity: "opacity", consolidation: "consolidation", effusion: "effusion",
    cardiomegaly: "cardiomegaly", nodule: "nodule", hyperinflation: "hyperinflation",
    pneumothorax: "pneumothorax", prior_risk: "prior risk",
  };
  // report grounding uses Finding enum values; evidence nodes use channel names
  const FINDING_TO_CHANNEL = {
    opacity: "opacity", consolidation: "consolidation", pleural_effusion: "effusion",
    cardiomegaly: "cardiomegaly", nodule: "nodule", pneumothorax: "pneumothorax",
    hyperinflation: "hyperinflation",
  };
  const ABSTAIN_TEXT = {
    low_confidence: "No diagnosis cleared the confidence threshold.",
    large_conformal_set: "Too many diagnoses remain statistically plausible.",
    out_of_distribution: "This study sits outside the model's validated distribution.",
    high_epistemic_uncertainty: "The model's own uncertainty about itself is too high.",
  };

  const S = { cases: [], current: null, bundles: new Map(), booted: false, offline: false };
  const $ = (id) => document.getElementById(id);

  /* ================= boot & assembly ================= */
  async function boot() {
    if (S.booted) { return; }
    S.booted = true;
    // ambient particles behind everything
    const amb = new Field($("c-ambient"), { count: 60, hue: 190, mode: "drift", size: 1.1, mouse: false, speed: 0.5 });
    amb.start();
    clock();
    bindChrome();
    const grid = $("c-grid");
    grid.classList.add("assembling");
    try {
      const [health, cases] = await Promise.all([api("/v1/health"), api("/v1/cases")]);
      S.cases = cases.cases || [];
      renderChips(health);
      renderWorklist();
      // panels assemble themselves — staggered spring-in
      requestAnimationFrame(() => {
        grid.classList.remove("assembling");
        [...grid.children].forEach((p, i) => { p.style.transitionDelay = i * 90 + "ms"; setTimeout(() => (p.style.transitionDelay = ""), 1200); });
      });
      if (S.cases.length) selectCase(S.cases[0].case_id, { first: true });
    } catch (err) {
      S.offline = true;
      grid.classList.remove("assembling");
      $("c-chips").innerHTML = `<span class="c-chip" style="color:var(--red)">GATEWAY OFFLINE — run \`py -m aura_cli serve\`</span>`;
    }
  }

  function renderChips(h) {
    $("c-chips").innerHTML = `
      <span class="c-chip ${h.backend === "quantum" ? "q" : ""}">fusion <b>${h.backend}</b></span>
      <span class="c-chip">coverage <b>90%</b></span>
      <span class="c-chip">worklist <b>${h.cases}</b></span>
      <span class="c-chip">status <b>${h.trained ? "trained" : "untrained"}</b></span>`;
  }

  function clock() {
    const el = $("c-clock");
    setInterval(() => {
      el.textContent = new Date().toISOString().slice(11, 19) + " UTC";
    }, 1000);
  }

  function bindChrome() {
    $("btn-exit").addEventListener("click", () => window.ROUTER.surface());
    $("tg-sal").addEventListener("click", (e) => {
      e.target.classList.toggle("on");
      $("xray-sal").classList.toggle("on", e.target.classList.contains("on"));
    });
    $("tg-reg").addEventListener("click", (e) => {
      e.target.classList.toggle("on");
      $("xray-regions").classList.toggle("off", !e.target.classList.contains("on"));
    });
    // feedback verdicts
    document.querySelectorAll("#fb-tools .tg[data-verdict]").forEach((b) => {
      b.addEventListener("click", () => feedback(b.dataset.verdict, b));
    });
    $("btn-sign").addEventListener("click", sign);
    // simulate
    const picker = $("sim-picker");
    picker.innerHTML = ["random", ...Object.keys(DX_LABEL)]
      .map((d) => `<button class="sim-chip" data-dx="${d}">${d === "random" ? "⚄ random" : DX_LABEL[d]}</button>`).join("");
    $("btn-sim").addEventListener("click", () => { picker.hidden = !picker.hidden; });
    picker.addEventListener("click", (e) => {
      const b = e.target.closest("[data-dx]");
      if (b) { picker.hidden = true; simulate(b.dataset.dx); }
    });
    // upload
    $("btn-upload").addEventListener("click", () => { $("input-file").click(); });
    $("input-file").addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        uploadImage(e.target.files[0]);
      }
    });
  }

  /* ================= worklist ================= */
  function renderWorklist() {
    const ol = $("worklist");
    $("rail-count").textContent = S.cases.length + " cases";
    ol.innerHTML = "";
    S.cases.forEach((c, i) => {
      const li = document.createElement("li");
      li.className = "wl-item" + (c.abstained ? " abst" : "") + (c.case_id === S.current ? " sel" : "");
      li.style.setProperty("--pri", clamp(c.priority_score, 0.12, 1).toFixed(2));
      li.style.animation = `capIn .5s var(--ease) ${i * 45}ms backwards`;
      li.innerHTML = `
        <div class="wl-id"><span>${c.case_id}</span><span class="wl-state ${c.state}">${c.state}</span></div>
        <div class="wl-dx">${DX_LABEL[c.top_diagnosis] || c.top_diagnosis || "—"}</div>
        <div class="wl-sub">p ${(c.top_probability || 0).toFixed(2)} · pri ${(c.priority_score || 0).toFixed(2)} · ${c.backend || ""}</div>`;
      li.addEventListener("click", () => selectCase(c.case_id));
      ol.appendChild(li);
    });
  }

  /* ================= case selection — evidence converges ================= */
  async function selectCase(id, { first = false } = {}) {
    if (S.current === id && !first) return;
    S.current = id;
    document.querySelectorAll(".wl-item").forEach((el) => {
      el.classList.toggle("sel", el.querySelector(".wl-id span").textContent === id);
    });
    const grid = $("c-grid");
    if (!first) {
      grid.classList.add("switching");
      await wait(REDUCED ? 0 : 300);
    }
    let b = S.bundles.get(id);
    if (!b) {
      try { b = await api(`/v1/cases/${id}`); S.bundles.set(id, b); }
      catch { toast("failed to load case"); grid.classList.remove("switching"); return; }
    }
    populate(b);
    grid.classList.remove("switching");
    loadSimilar(id);
  }

  const wait = (ms) => new Promise((r) => setTimeout(r, ms));

  function populate(b) {
    drawXray(b);
    drawEvidence(b);
    drawPosterior(b);
    drawSafety(b);
    drawRecs(b);
    drawReport(b);
  }

  /* ================= x-ray + saliency + findings ================= */
  function paintGrid(canvas, flat, shape, colormap) {
    const [rows, cols] = shape;
    const off = document.createElement("canvas");
    off.width = cols; off.height = rows;
    const og = off.getContext("2d");
    const im = og.createImageData(cols, rows);
    let mn = Infinity, mx = -Infinity;
    for (const v of flat) { if (v < mn) mn = v; if (v > mx) mx = v; }
    const rng = mx - mn || 1;
    for (let i = 0; i < flat.length; i++) {
      const t = (flat[i] - mn) / rng;
      const [r, g2, bb, a] = colormap(t);
      im.data[i * 4] = r; im.data[i * 4 + 1] = g2; im.data[i * 4 + 2] = bb; im.data[i * 4 + 3] = a;
    }
    og.putImageData(im, 0, 0);
    const g = canvas.getContext("2d");
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const r = canvas.getBoundingClientRect();
    canvas.width = r.width * dpr; canvas.height = r.height * dpr;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.imageSmoothingEnabled = true; g.imageSmoothingQuality = "high";
    g.drawImage(off, 0, 0, r.width, r.height);
  }

  function drawXray(b) {
    if (b.image && b.image.length) {
      paintGrid($("xray"), b.image, b.image_shape, (t) => {
        const v = Math.round(Math.pow(t, 0.9) * 255);
        return [v, v, Math.min(255, v + 6), 255];
      });
    }
    const sal = (b.explanation && b.explanation.saliency) || [];
    if (sal.length) {
      paintGrid($("xray-sal"), sal, b.explanation.saliency_shape || b.image_shape, (t) => {
        // heat: transparent → cyan → amber
        const a = Math.round(Math.pow(t, 1.4) * 235);
        return t < 0.55 ? [40, 210, 190, a * 0.7] : [245, 182, 78, a];
      });
    }
    // finding regions materialize
    const wrap = $("xray-regions");
    wrap.innerHTML = "";
    const found = ((b.vision && b.vision.findings) || []).filter((f) => f.probability >= 0.5);
    found.forEach((f, i) => {
      const [r0, c0, r1, c1] = f.region;
      const d = document.createElement("div");
      d.className = "region";
      d.style.cssText = `top:${r0 * 100}%;left:${c0 * 100}%;height:${(r1 - r0) * 100}%;width:${(c1 - c0) * 100}%;animation-delay:${0.25 + i * 0.18}s`;
      d.innerHTML = `<span class="r-lbl">${(EV_LABEL[FINDING_TO_CHANNEL[f.finding]] || f.finding)} · ${f.probability.toFixed(2)}</span>`;
      wrap.appendChild(d);
    });
    const p = b.priors || {};
    $("xray-meta").innerHTML = `
      <span>${b.study_id} · CXR 64×64</span>
      <span>${p.age_band || "?"} · ${p.sex || "?"}${p.smoker ? " · smoker" : ""}${p.fever ? " · fever" : ""}${p.prior_cancer ? " · prior ca" : ""}</span>`;
  }

  async function loadSimilar(id) {
    const row = $("similar-row");
    row.innerHTML = "";
    try {
      const d = await api(`/v1/cases/${id}/similar`);
      if (!d.similar || !d.similar.length) return;
      row.innerHTML = `<span>memory recalls:</span>` + d.similar
        .map((s) => `<button class="sim-link" data-id="${s.case_id}">${s.case_id} · ${DX_LABEL[s.label] || s.label || ""} · ${(s.similarity ?? s.score ?? 0).toFixed ? (s.similarity ?? s.score ?? 0).toFixed(2) : ""}</button>`).join("");
      row.querySelectorAll(".sim-link").forEach((btn) =>
        btn.addEventListener("click", () => selectCase(btn.dataset.id)));
    } catch { /* memory quiet */ }
  }

  /* ================= evidence graph — evidence converging into reasoning ================= */
  function drawEvidence(b) {
    const svg = $("ev-svg");
    const NS = "http://www.w3.org/2000/svg";
    svg.innerHTML = "";
    const tip = $("ev-tip");
    tip.hidden = true;
    if (!b.evidence || !b.safety) return;
    const attr = (b.explanation && b.explanation.evidence_attribution) || {};
    const cfs = (b.explanation && b.explanation.counterfactuals) || {};
    const CX = 280, CY = 205, RX = 205, RY = 150;
    const items = b.evidence;
    const maxAttr = Math.max(1e-6, ...Object.values(attr).map((v) => Math.abs(v)));

    const mk = (t, attrs, parent = svg) => {
      const e = document.createElementNS(NS, t);
      for (const k in attrs) e.setAttribute(k, attrs[k]);
      parent.appendChild(e); return e;
    };

    // edges first (under nodes)
    const edgeEls = [];
    const nodePos = items.map((it, i) => {
      const a = (i / items.length) * Math.PI * 2 - Math.PI / 2;
      return [CX + Math.cos(a) * RX, CY + Math.sin(a) * RY];
    });
    items.forEach((it, i) => {
      const w = Math.abs(attr[it.name] || 0) / maxAttr;
      const pos = attr[it.name] >= 0;
      const [x, y] = nodePos[i];
      const e = mk("path", {
        d: `M ${x} ${y} Q ${(x + CX) / 2 + (y - CY) * 0.12} ${(y + CY) / 2 - (x - CX) * 0.12} ${CX} ${CY}`,
        class: "ev-edge",
        stroke: pos ? "rgba(75,225,195,0.75)" : "rgba(255,93,93,0.65)",
        "stroke-width": (0.7 + w * 4.2).toFixed(2),
        opacity: 0.25 + w * 0.75,
      });
      const L = e.getTotalLength();
      e.style.strokeDasharray = L;
      e.style.strokeDashoffset = L;
      e.style.transition = `stroke-dashoffset .9s cubic-bezier(.16,1,.3,1) ${0.15 + i * 0.07}s`;
      edgeEls.push(e);
    });

    // diagnosis core
    const core = mk("g", { class: "dx-core" });
    mk("circle", { cx: CX, cy: CY, r: 44, fill: "rgba(75,225,195,0.09)", stroke: "rgba(75,225,195,0.85)", "stroke-width": 1.6 }, core);
    const ring = mk("circle", { cx: CX, cy: CY, r: 52, fill: "none", stroke: "rgba(75,225,195,0.3)", "stroke-width": 1, "stroke-dasharray": "3 6" }, core);
    (function spinRing() { // the reasoning core is alive
      if (!ring.isConnected) return;
      const t = performance.now() / 1000;
      ring.setAttribute("transform", `rotate(${(t * 14) % 360} ${CX} ${CY})`);
      requestAnimationFrame(spinRing);
    })();
    const dxl = mk("text", { x: CX, y: CY - 1, class: "dx-lbl" }, core);
    dxl.textContent = DX_LABEL[b.safety.top] || b.safety.top;
    const dxs = mk("text", { x: CX, y: CY + 17, class: "dx-sub" }, core);
    dxs.textContent = (b.safety.top_probability * 100).toFixed(0) + "% CALIBRATED";
    core.style.opacity = 0; core.style.transition = "opacity .8s .5s";

    // evidence nodes converge from the core outward
    items.forEach((it, i) => {
      const [x, y] = nodePos[i];
      const gEl = mk("g", { class: "ev-node", "data-name": it.name });
      const absent = it.kind === "absent_evidence";
      const col = it.name === "prior_risk" ? "#8b7cf7" : absent ? "#565f74" : "#4be1c3";
      const r = 5 + it.value * 11;
      const c = mk("circle", {
        cx: 0, cy: 0, r,
        fill: absent ? "transparent" : col,
        "fill-opacity": absent ? 0 : 0.22,
        stroke: col, "stroke-width": absent ? 1 : 1.6,
        "stroke-dasharray": absent ? "3 4" : "none",
        color: col,
      }, gEl);
      // uncertainty halo pulses on uncertain nodes
      if (it.uncertainty > 0.3) {
        const halo = mk("circle", { cx: 0, cy: 0, r: r + 4, fill: "none", stroke: "rgba(244,182,78,0.5)", "stroke-width": 1 }, gEl);
        halo.innerHTML = `<animate attributeName="r" values="${r + 3};${r + 9};${r + 3}" dur="2.2s" repeatCount="indefinite"/><animate attributeName="stroke-opacity" values=".5;.05;.5" dur="2.2s" repeatCount="indefinite"/>`;
      }
      const lbl = mk("text", { x: 0, y: -r - 8, "text-anchor": "middle", class: "ev-lbl" }, gEl);
      lbl.textContent = (EV_LABEL[it.name] || it.name) + " " + it.value.toFixed(2);
      // converge animation: node flies from core to its slot
      gEl.style.transform = `translate(${CX}px, ${CY}px) scale(.2)`;
      gEl.style.opacity = 0;
      gEl.style.transition = `transform .9s cubic-bezier(.34,1.56,.64,1) ${0.1 + i * 0.07}s, opacity .5s ${0.1 + i * 0.07}s`;
      requestAnimationFrame(() => requestAnimationFrame(() => {
        gEl.style.transform = `translate(${x}px, ${y}px) scale(1)`;
        gEl.style.opacity = 1;
        edgeEls[i].style.strokeDashoffset = 0;
        core.style.opacity = 1;
      }));
      // tooltip with counterfactual reasoning
      gEl.addEventListener("pointerenter", () => {
        const cf = cfs[it.name];
        const stage = svg.closest(".ev-stage").getBoundingClientRect();
        const pt = svg.getBoundingClientRect();
        const sx = pt.left + (x / 560) * pt.width - stage.left;
        const sy = pt.top + (y / 420) * pt.height - stage.top;
        tip.style.left = sx + "px"; tip.style.top = sy + "px";
        tip.innerHTML = `<b>${EV_LABEL[it.name] || it.name}</b><br>
          strength ${it.value.toFixed(2)} · unc ${it.uncertainty.toFixed(2)}<br>
          ${cf !== undefined ? `remove it → top belief <span class="${cf <= 0 ? "cf-neg" : "cf-pos"}">${cf > 0 ? "+" : ""}${(cf * 100).toFixed(1)}%</span>` : (absent ? "absent — see next best evidence" : "")}`;
        tip.hidden = false;
      });
      gEl.addEventListener("pointerleave", () => { tip.hidden = true; });
    });
  }

  function lightNodes(names, on) {
    document.querySelectorAll("#ev-svg .ev-node").forEach((n) => {
      n.classList.toggle("lit", on && names.includes(n.dataset.name));
    });
  }

  /* ================= posterior ================= */
  function drawPosterior(b) {
    const wrap = $("post-bars");
    wrap.innerHTML = "";
    if (!b.safety) return;
    $("post-backend").textContent = b.fusion ? `${b.fusion.backend} fusion · ${b.fusion.n_shots || 0} shots` : "";
    const conf = new Set((b.safety.conformal_set || []).map(String));
    const stds = (b.fusion && b.fusion.posterior_std) || {};
    const preds = [...b.safety.predictions].sort((a, c) => c.probability - a.probability);
    preds.forEach((p, i) => {
      const row = document.createElement("div");
      row.className = "post-row" + (i === 0 ? " top" : "") + (conf.has(p.diagnosis) ? " inset" : "");
      const std = stds[p.diagnosis] || 0;
      row.innerHTML = `
        <span class="post-lbl">${DX_LABEL[p.diagnosis] || p.diagnosis}</span>
        <div class="post-track">
          <div class="post-fill"></div>
          <div class="post-ci"></div>
        </div>
        <span class="post-num">0%</span>`;
      wrap.appendChild(row);
      const fill = row.querySelector(".post-fill");
      const ci = row.querySelector(".post-ci");
      setTimeout(() => {
        fill.style.width = p.probability * 100 + "%";
        const lo = clamp(p.ci_low - std, 0, 1), hi = clamp(p.ci_high + std, 0, 1);
        ci.style.left = lo * 100 + "%"; ci.style.width = Math.max(0.5, (hi - lo) * 100) + "%";
        FX.countTo(row.querySelector(".post-num"), p.probability * 100, { dur: 1100, fmt: (v) => v.toFixed(1) + "%" });
      }, 80 + i * 90);
    });
  }

  /* ================= safety panel ================= */
  function dial(el, label, val, norm, color) {
    const v = clamp(norm, 0, 1);
    el.innerHTML = `<svg viewBox="0 0 86 60">
      <path class="d-arc-bg" d="M 10 52 A 34 34 0 0 1 76 52"/>
      <path class="d-arc" d="M 10 52 A 34 34 0 0 1 76 52" stroke="${color}"/>
      <text class="d-num" x="43" y="50">${val}</text>
    </svg><span class="d-lbl">${label}</span>`;
    const arc = el.querySelector(".d-arc");
    const L = arc.getTotalLength();
    arc.style.strokeDasharray = L; arc.style.strokeDashoffset = L;
    setTimeout(() => { arc.style.strokeDashoffset = L * (1 - v); }, 150);
  }

  function drawSafety(b) {
    const s = b.safety;
    if (!s) return;
    const flag = $("safety-flag");
    const banner = $("abstain-banner");
    if (s.abstained) {
      flag.textContent = "ABSTAINED"; flag.className = "flag abst";
      banner.hidden = false;
      banner.innerHTML = `<b>AURA declined to commit.</b> ${ABSTAIN_TEXT[s.abstention_reason] || s.abstention_reason}
        Escalated with its full uncertainty state — no silent failure.`;
    } else {
      flag.textContent = "WITHIN ENVELOPE"; flag.className = "flag ok";
      banner.hidden = true;
    }
    const dials = $("dials");
    dials.innerHTML = `<div class="dial" id="d-epi"></div><div class="dial" id="d-ale"></div><div class="dial" id="d-ood"></div>`;
    dial($("d-epi"), "EPISTEMIC", s.epistemic_uncertainty.toFixed(3), s.epistemic_uncertainty / 0.25, s.epistemic_uncertainty > 0.12 ? "#f4b64e" : "#4be1c3");
    dial($("d-ale"), "ALEATORIC", s.aleatoric_uncertainty.toFixed(3), s.aleatoric_uncertainty / 1.8, "#8b7cf7");
    dial($("d-ood"), "OOD ENERGY", s.ood_energy.toFixed(2), (s.ood_energy + 6) / 12, s.is_ood ? "#ff5d5d" : "#4be1c3");
    $("conf-lbl").textContent = `${Math.round((s.conformal_coverage || 0.9) * 100)}% CONFORMAL SET — truth in this set ${Math.round((s.conformal_coverage || 0.9) * 100)}/100 times`;
    $("conf-chips").innerHTML = (s.conformal_set || [])
      .map((d, i) => `<span class="conf-chip ${s.conformal_set.length > 2 && i > 0 ? "hot" : ""}" style="animation-delay:${i * 0.1}s">${DX_LABEL[d] || d}</span>`).join("");
  }

  /* ================= recommendations ================= */
  function drawRecs(b) {
    const ol = $("recs");
    ol.innerHTML = "";
    const recs = b.recommendations || [];
    if (!recs.length) { ol.innerHTML = `<li class="p-hint mono">information-gain analysis: nothing further indicated</li>`; return; }
    const maxE = Math.max(...recs.map((r) => r.expected_info_gain), 1e-6);
    recs.forEach((r, i) => {
      const li = document.createElement("li");
      li.className = "rec";
      li.style.animation = `capIn .5s var(--ease) ${i * 90}ms backwards`;
      li.innerHTML = `
        <div class="rec-top"><span class="rec-name">${r.display}</span>
          <span class="rec-eig">+${r.expected_info_gain.toFixed(3)} bits</span></div>
        <div class="rec-track"><div class="rec-fill"></div></div>
        <div class="rec-meta"><span>cost ${r.cost_tier}</span><span>risk ${r.risk_tier}</span><span>utility ${r.utility.toFixed(2)}</span></div>
        <p class="rec-rationale">${r.rationale}</p>`;
      li.addEventListener("click", () => li.classList.toggle("open"));
      ol.appendChild(li);
      setTimeout(() => { li.querySelector(".rec-fill").style.width = (r.expected_info_gain / maxE) * 100 + "%"; }, 250 + i * 120);
    });
  }

  /* ================= grounded report — reasoning writes itself ================= */
  async function drawReport(b) {
    const body = $("report-body");
    body.innerHTML = "";
    if (!b.report) return;
    const ground = b.report.grounding || {};
    const groundNodes = (key) => (ground[key] || []).map((g) => FINDING_TO_CHANNEL[g] || g);
    const blocks = [
      ["FINDINGS", b.report.findings_text, groundNodes("findings")],
      ["IMPRESSION", b.report.impression_text, groundNodes("impression")],
      ["RECOMMENDATION", b.report.recommendation_text, groundNodes("recommendation")],
    ];
    for (const [lbl, text, nodes] of blocks) {
      const div = document.createElement("div");
      div.className = "rep-block";
      div.innerHTML = `<span class="rb-lbl">${lbl}</span><span class="rb-text"></span>
        <span class="rep-grounding">grounded in: <i>${nodes.map((n) => EV_LABEL[n] || n).join(" · ") || "—"}</i></span>`;
      body.appendChild(div);
      div.addEventListener("pointerenter", () => lightNodes(nodes, true));
      div.addEventListener("pointerleave", () => lightNodes(nodes, false));
      await typeInto(div.querySelector(".rb-text"), text, { cps: 220 });
    }
  }

  /* ================= actions ================= */
  async function feedback(verdict, btn) {
    if (!S.current) return;
    btn.classList.remove("pulse"); void btn.offsetWidth; btn.classList.add("pulse");
    try {
      const d = await api(`/v1/cases/${S.current}/feedback`, {
        method: "POST", headers: { "Content-Type": "application/json", "x-aura-user": "clinician" },
        body: JSON.stringify({ verdict }),
      });
      toast(`verdict "${verdict}" recorded — ${d.stats.total} feedback events in the learning loop`);
    } catch { toast("feedback failed — gateway offline?"); }
  }

  async function sign() {
    if (!S.current) return;
    try {
      await api(`/v1/cases/${S.current}/report/sign`, {
        method: "POST", headers: { "Content-Type": "application/json", "x-aura-user": "clinician" },
        body: JSON.stringify({ signed_by: "clinician" }),
      });
      const panel = $("panel-report");
      panel.classList.remove("signed-sweep"); void panel.offsetWidth; panel.classList.add("signed-sweep");
      toast(`${S.current} signed — audit trail updated`);
      const c = S.cases.find((x) => x.case_id === S.current);
      if (c) { c.state = "signed"; renderWorklist(); }
    } catch { toast("sign failed — gateway offline?"); }
  }

  /* ================= simulate — intelligence forming, live ================= */
  const FORM_STAGES = [
    "synthesizing study …",
    "vision engine reading film",
    "encoding 8 evidence channels",
    "entangling qubits — fusion posterior",
    "conformal calibration · OOD sweep",
    "counterfactual attribution",
    "ranking next-best evidence",
    "grounding report",
  ];

  async function simulate(dx) {
    const overlay = $("case-forming");
    const txt = $("forming-text");
    overlay.hidden = false;
    txt.innerHTML = "";
    const f = new Field($("forming-canvas"), { count: 160, hue: 172, mode: "collapse", size: 1.6, speed: 1.1 });
    f.start();
    // staged boot text while the real pipeline runs
    let alive = true;
    (async () => {
      for (const line of FORM_STAGES) {
        if (!alive) return;
        txt.innerHTML += `<span class="ok">▸</span> ${line}\n`;
        await wait(REDUCED ? 30 : 340);
      }
    })();
    try {
      const d = await api("/v1/studies/simulate", {
        method: "POST", headers: { "Content-Type": "application/json", "x-aura-user": "clinician" },
        body: JSON.stringify({ diagnosis: dx }),
      });
      const cases = await api("/v1/cases");
      S.cases = cases.cases || [];
      const h = await api("/v1/health").catch(() => null);
      if (h) renderChips(h);
      await wait(REDUCED ? 0 : 900); // let the convergence land
      alive = false;
      overlay.hidden = true; f.destroy();
      renderWorklist();
      S.current = null;
      selectCase(d.case_id, { first: true });
      toast(`${d.case_id} analyzed live by the full pipeline`);
    } catch {
      alive = false; overlay.hidden = true; f.destroy();
      toast("simulation failed — gateway offline?");
    }
  }

  async function uploadImage(file) {
    const overlay = $("case-forming");
    const txt = $("forming-text");
    overlay.hidden = false;
    txt.innerHTML = "";
    const f = new Field($("forming-canvas"), { count: 160, hue: 172, mode: "collapse", size: 1.6, speed: 1.1 });
    f.start();
    // staged boot text while the real pipeline runs
    let alive = true;
    (async () => {
      for (const line of FORM_STAGES) {
        if (!alive) return;
        txt.innerHTML += `<span class="ok">▸</span> ${line}\n`;
        await wait(REDUCED ? 30 : 340);
      }
    })();
    try {
      const fd = new FormData();
      fd.append("file", file);
      const d = await api("/v1/studies/upload", {
        method: "POST",
        headers: { "x-aura-user": "clinician" },
        body: fd,
      });
      const cases = await api("/v1/cases");
      S.cases = cases.cases || [];
      const h = await api("/v1/health").catch(() => null);
      if (h) renderChips(h);
      await wait(REDUCED ? 0 : 900); // let the convergence land
      alive = false;
      overlay.hidden = true; f.destroy();
      renderWorklist();
      S.current = null;
      selectCase(d.case_id, { first: true });
      toast(`${d.case_id} uploaded and analyzed live`);
    } catch (err) {
      alive = false; overlay.hidden = true; f.destroy();
      toast("upload failed — gateway offline or bad file?");
    } finally {
      $("input-file").value = "";
    }
  }


  return { boot };
})();
