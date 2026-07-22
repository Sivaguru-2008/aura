/* ============================================================
   AURA — History, Image, and Clinical Report Portal Script
   ============================================================ */

window.HISTORY_PORTAL = (() => {
  "use strict";

  const { Field, api, toast, typeInto, clamp, countTo } = FX;

  let DX_LABEL = {};
  let EV_LABEL = {};

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

  const S = {
    cases: [],
    bundles: new Map(),
    booted: false,
    activeCaseId: null,
    viewMode: "history-list", // "history-list", "image", "report"
  };

  const $ = (id) => document.getElementById(id);

  /* ================= boot & routing ================= */
  async function boot() {
    if (S.booted) return;
    S.booted = true;

    // ambient particles behind everything
    const amb = new Field($("c-ambient"), { count: 60, hue: 190, mode: "drift", size: 1.1, mouse: false, speed: 0.5 });
    amb.start();
    clock();
    bindEvents();
    FX.init();

    await fetchCases();
    route();
  }

  function clock() {
    const el = $("h-clock");
    setInterval(() => {
      el.textContent = new Date().toISOString().slice(11, 19) + " UTC";
    }, 1000);
  }

  async function fetchCases() {
    try {
      const data = await api("/v1/cases");
      S.cases = data.cases || [];
      if (data.dx_labels) DX_LABEL = data.dx_labels;
      if (data.ev_labels) EV_LABEL = data.ev_labels;
    } catch (err) {
      toast("Failed to load history archive");
      $("history-grid").innerHTML = `<div class="loading-state mono" style="color:var(--red)">GATEWAY OFFLINE — Run uvicorn server</div>`;
    }
  }

  function bindEvents() {
    window.addEventListener("hashchange", route);
    
    // Search / Filter inputs
    $("history-search").addEventListener("input", renderHistoryList);
    $("history-filter-state").addEventListener("change", renderHistoryList);

    // Image toggles
    $("img-tg-sal").addEventListener("click", (e) => {
      e.target.classList.toggle("on");
      $("img-xray-sal").classList.toggle("on", e.target.classList.contains("on"));
    });
    
    $("img-tg-reg").addEventListener("click", (e) => {
      e.target.classList.toggle("on");
      $("img-xray-regions").classList.toggle("off", !e.target.classList.contains("on"));
    });
  }

  function route() {
    const hash = window.location.hash || "#/";
    const parts = hash.split("/");
    
    // Hide all view containers first
    document.querySelectorAll(".h-view").forEach(el => el.hidden = true);
    
    if (hash === "#/" || hash === "") {
      S.viewMode = "history-list";
      document.body.dataset.view = "history-list";
      $("view-history-list").hidden = false;
      $("nav-history").classList.add("active");
      renderHistoryList();
    } else if (parts[1] === "case" && parts[2] && parts[3] === "image") {
      S.viewMode = "image";
      document.body.dataset.view = "image";
      $("view-image").hidden = false;
      $("nav-history").classList.remove("active");
      
      const caseId = parts[2];
      S.activeCaseId = caseId;
      loadAndShowCase(caseId, "image");
    } else if (parts[1] === "case" && parts[2] && parts[3] === "report") {
      S.viewMode = "report";
      document.body.dataset.view = "report";
      $("view-report").hidden = false;
      $("nav-history").classList.remove("active");
      
      const caseId = parts[2];
      S.activeCaseId = caseId;
      loadAndShowCase(caseId, "report");
    } else {
      // Fallback
      window.location.hash = "#/";
    }
  }

  /* ================= view 1: history list ================= */
  function renderHistoryList() {
    const grid = $("history-grid");
    const emptyState = $("empty-state");
    grid.innerHTML = "";

    const query = $("history-search").value.toLowerCase().trim();
    const filterState = $("history-filter-state").value;

    // Filter to uploaded cases (CASE-UPLOAD-X) by default, or show all if needed.
    // Let's filter to CASE-UPLOAD- so it's specifically the "history for every image uploaded".
    // We can also allow viewing simulated live studies if query matches them.
    const filtered = S.cases.filter(c => {
      // Must match search query on ID or Diagnosis
      const matchesQuery = c.case_id.toLowerCase().includes(query) || 
                           (DX_LABEL[c.top_diagnosis] || c.top_diagnosis || "").toLowerCase().includes(query);
      
      // Filter by state
      let matchesState = true;
      if (filterState === "signed") matchesState = c.state === "signed";
      else if (filterState === "abstained") matchesState = c.abstained === true;
      else if (filterState === "within-envelope") matchesState = c.abstained === false;
      
      // Filter to only uploaded films (CASE-UPLOAD-*)
      const isUpload = c.case_id.startsWith("CASE-UPLOAD-");

      return matchesQuery && matchesState && isUpload;
    });

    if (filtered.length === 0) {
      grid.hidden = true;
      emptyState.hidden = false;
      return;
    }

    grid.hidden = false;
    emptyState.hidden = true;

    filtered.forEach((c, idx) => {
      const card = document.createElement("div");
      card.className = "history-card";
      card.style.animation = `capIn .4s var(--ease) ${idx * 45}ms backwards`;
      
      const dateStr = c.created_at ? new Date(c.created_at).toLocaleString() : "Date Unknown";
      
      card.innerHTML = `
        <div class="card-top">
          <span class="card-id">${c.case_id}</span>
          <span class="card-status ${c.abstained ? 'abstained' : c.state}">${c.abstained ? 'abstained' : c.state}</span>
        </div>
        <div class="card-dx">${c.abstained ? 'Abstained (Uncertain)' : (c.top_diagnosis_label || c.top_diagnosis || '')}</div>
        <div class="card-meta">
          <span>p ${(c.top_probability || 0).toFixed(2)}</span>
          <span>pri ${(c.priority_score || 0).toFixed(2)}</span>
          <span>${c.backend}</span>
          <span>${dateStr}</span>
        </div>
        <div class="card-actions">
          <a href="#/case/${c.case_id}/image" class="card-btn primary">View Film</a>
          <a href="#/case/${c.case_id}/report" class="card-btn">View Report</a>
        </div>
      `;
      
      // Allow clicking card (except buttons) to navigate to report
      card.addEventListener("click", (e) => {
        if (e.target.closest(".card-btn")) return;
        window.location.hash = `#/case/${c.case_id}/image`;
      });

      grid.appendChild(card);
    });
  }

  /* ================= case loading ================= */
  async function loadAndShowCase(caseId, view) {
    let b = S.bundles.get(caseId);
    if (!b) {
      try {
        b = await api(`/v1/cases/${caseId}`);
        S.bundles.set(caseId, b);
      } catch (err) {
        toast("Failed to load case data");
        window.location.hash = "#/";
        return;
      }
    }
    if (b.dx_labels) DX_LABEL = b.dx_labels;
    if (b.ev_labels) EV_LABEL = b.ev_labels;

    if (view === "image") {
      populateImagePage(b);
    } else {
      populateReportPage(b);
    }
  }

  /* ================= view 2: image viewer page ================= */
  function populateImagePage(b) {
    $("img-case-title").textContent = `${b.case_id} Image Viewer`;
    $("img-meta-id").textContent = `${b.study_id} · CXR ${b.image_shape ? b.image_shape.join('x') : '64x64'}`;
    
    // Bind navigation switchers
    $("btn-goto-report").onclick = () => window.location.hash = `#/case/${b.case_id}/report`;

    const p = b.priors || {};
    $("img-priors-bar").textContent = `Age: ${p.age_band || "?"} · Sex: ${p.sex || "?"}${p.smoker ? " · Smoker" : ""}${p.fever ? " · Fever" : ""}${p.prior_cancer ? " · Prior Ca" : ""}`;

    // Draw X-ray canvas
    if (b.image && b.image.length) {
      paintGrid($("img-xray"), b.image, b.image_shape, (t) => {
        const v = Math.round(Math.pow(t, 0.9) * 255);
        return [v, v, Math.min(255, v + 6), 255];
      });
    }

    // Draw Saliency overlay
    const sal = (b.explanation && b.explanation.saliency) || [];
    if (sal.length) {
      paintGrid($("img-xray-sal"), sal, b.explanation.saliency_shape || b.image_shape, (t) => {
        const a = Math.round(Math.pow(t, 1.4) * 235);
        return t < 0.55 ? [40, 210, 190, a * 0.7] : [245, 182, 78, a];
      });
    }

    // Paint detected finding bounding boxes
    const wrap = $("img-xray-regions");
    wrap.innerHTML = "";
    const found = ((b.vision && b.vision.findings) || []).filter((f) => f.probability >= 0.5);
    found.forEach((f, i) => {
      const [r0, c0, r1, c1] = f.region;
      const d = document.createElement("div");
      d.className = "region";
      d.style.cssText = `top:${r0 * 100}%;left:${c0 * 100}%;height:${(r1 - r0) * 100}%;width:${(c1 - c0) * 100}%;animation-delay:${0.2 + i * 0.1}s`;
      d.innerHTML = `<span class="r-lbl">${(EV_LABEL[FINDING_TO_CHANNEL[f.finding]] || f.finding)} · ${f.probability.toFixed(2)}</span>`;
      wrap.appendChild(d);
    });

    // Populate priors grid
    const prGrid = $("img-priors-grid");
    prGrid.innerHTML = "";
    Object.entries(p).forEach(([k, val]) => {
      const chip = document.createElement("div");
      chip.className = "prior-chip" + (val ? " active" : "");
      chip.innerHTML = `
        <span class="prior-lbl">${k.replace('_', ' ')}</span>
        <span class="prior-val">${val === true ? "YES" : val === false ? "NO" : val}</span>
      `;
      prGrid.appendChild(chip);
    });

    // Populate vision findings checklist
    const list = $("img-findings-list");
    list.innerHTML = "";
    const allFindings = (b.vision && b.vision.findings) || [];
    allFindings.forEach(f => {
      const li = document.createElement("li");
      const isPresent = f.probability >= 0.5;
      li.className = isPresent ? "present" : "absent";
      li.innerHTML = `
        <span class="finding-name">${EV_LABEL[FINDING_TO_CHANNEL[f.finding]] || f.finding}</span>
        <span class="finding-prob ${f.probability >= 0.75 ? 'high' : ''}">${(f.probability * 100).toFixed(0)}%</span>
      `;
      list.appendChild(li);
    });

    // Populate safety indicators
    const safeCard = $("img-safety-card");
    const s = b.safety || {};
    if (s.abstained) {
      safeCard.className = "safety-indicator-card abst";
      safeCard.innerHTML = `
        <strong>AURA Abstained</strong>
        <span class="safety-reason">${ABSTAIN_TEXT[s.abstention_reason] || s.abstention_reason}</span>
      `;
    } else {
      safeCard.className = "safety-indicator-card ok";
      safeCard.innerHTML = `
        <strong>Safety Envelope Secure</strong>
        <span class="safety-reason">Diagnosis is within calibrated model parameters.</span>
      `;
    }
  }

  /* ================= view 3: clinical report page ================= */
  async function populateReportPage(b) {
    $("rep-case-title").textContent = `${b.case_id} Clinical Report`;
    
    // Navigation switchers
    $("btn-goto-image").onclick = () => window.location.hash = `#/case/${b.case_id}/image`;
    
    // Status Badge
    const stateBadge = $("rep-doc-state");
    stateBadge.textContent = b.state.toUpperCase();
    stateBadge.className = "rep-doc-status-badge " + b.state.toLowerCase();

    // Fill metadata fields
    $("rep-val-case").textContent = b.case_id;
    $("rep-val-study").textContent = b.study_id;
    $("rep-val-date").textContent = b.created_at ? new Date(b.created_at).toLocaleString() : "2026-07-19 10:00 UTC";
    
    const p = b.priors || {};
    $("rep-val-profile").textContent = `Age ${p.age_band || "?"}, ${p.sex || "?"}${p.smoker ? " · Smoker" : ""}`;
    $("rep-val-engine").textContent = b.fusion ? `${b.fusion.backend} engine · ${b.fusion.n_shots || 0} shots` : "PennyLane Q1 (8 qubits)";

    // Render grounded texts (Findings, Impression, Recommendations)
    if (b.report) {
      const ground = b.report.grounding || {};
      const groundNodes = (key) => (ground[key] || []).map((g) => FINDING_TO_CHANNEL[g] || g);
      
      const findingsNodes = groundNodes("findings");
      $("rep-body-findings").textContent = b.report.findings_text;
      $("rep-ground-findings").textContent = "grounded in: " + (findingsNodes.map((n) => EV_LABEL[n] || n).join(" · ") || "—");
      
      const impressionNodes = groundNodes("impression");
      $("rep-body-impression").textContent = b.report.impression_text;
      $("rep-ground-impression").textContent = "grounded in: " + (impressionNodes.map((n) => EV_LABEL[n] || n).join(" · ") || "—");

      const recsNodes = groundNodes("recommendation");
      $("rep-body-recommendation").textContent = b.report.recommendation_text;
      $("rep-ground-recommendation").textContent = "grounded in: " + (recsNodes.map((n) => EV_LABEL[n] || n).join(" · ") || "—");
    }

    // Render calibrated differential bars
    const barsContainer = $("rep-diff-bars");
    barsContainer.innerHTML = "";
    if (b.safety && b.safety.predictions) {
      const stds = (b.fusion && b.fusion.posterior_std) || {};
      const preds = [...b.safety.predictions].sort((a, c) => c.probability - a.probability);
      
      preds.forEach((pred, i) => {
        const row = document.createElement("div");
        row.className = "rep-bar-row";
        row.innerHTML = `
          <div class="rep-bar-lbl-row">
            <span class="rep-bar-lbl">${DX_LABEL[pred.diagnosis] || pred.diagnosis}</span>
            <span class="rep-bar-val">0%</span>
          </div>
          <div class="rep-bar-track">
            <div class="rep-bar-fill"></div>
          </div>
        `;
        barsContainer.appendChild(row);

        const fill = row.querySelector(".rep-bar-fill");
        const valEl = row.querySelector(".rep-bar-val");
        setTimeout(() => {
          fill.style.width = pred.probability * 100 + "%";
          countTo(valEl, pred.probability * 100, { dur: 1000, fmt: (v) => v.toFixed(1) + "%" });
        }, 100 + i * 80);
      });
    }

    // Render Conformal Set
    const confTitle = $("rep-conformal-title");
    const confChips = $("rep-conformal-chips");
    const s = b.safety || {};
    confChips.innerHTML = "";
    
    if (s.conformal_set && s.conformal_set.length) {
      confTitle.textContent = `${Math.round((s.conformal_coverage || 0.9) * 100)}% Conformal Set`;
      s.conformal_set.forEach(d => {
        const chip = document.createElement("span");
        chip.className = "rep-conf-chip";
        chip.textContent = DX_LABEL[d] || d;
        confChips.appendChild(chip);
      });
    } else {
      confChips.innerHTML = `<span class="text-muted mono font-small">No diagnoses cleared the conformal set threshold.</span>`;
    }

    // Render dials
    const epiDial = $("rep-dial-epi");
    const aleDial = $("rep-dial-ale");
    const oodDial = $("rep-dial-ood");
    
    if (s.epistemic_uncertainty !== undefined) {
      drawDial(epiDial, s.epistemic_uncertainty.toFixed(3), s.epistemic_uncertainty / 0.25, s.epistemic_uncertainty > 0.12 ? "#f4b64e" : "#4be1c3");
      drawDial(aleDial, s.aleatoric_uncertainty.toFixed(3), s.aleatoric_uncertainty / 1.8, "#8b7cf7");
      drawDial(oodDial, s.ood_energy.toFixed(2), (s.ood_energy + 6) / 12, s.is_ood ? "#ff5d5d" : "#4be1c3");
      
      const statusBox = $("rep-safety-status-box");
      if (s.abstained) {
        statusBox.textContent = `ABSTAINED: ${s.abstention_reason.toUpperCase().replace(/_/g, ' ')}`;
        statusBox.className = "safety-desc-box abst";
      } else {
        statusBox.textContent = "WITHIN ENVELOPE";
        statusBox.className = "safety-desc-box ok";
      }
    }

    // Signature Block
    const sigLine = $("rep-sig-line");
    if (b.state === "signed") {
      sigLine.innerHTML = `<span class="sig-name">Electronically signed by Clinician</span>`;
    } else {
      sigLine.innerHTML = `<button class="btn-core" id="rep-btn-sign"><span>Sign Diagnostic Report</span></button>`;
      $("rep-btn-sign").addEventListener("click", () => signReport(b.case_id));
    }

    // Export & Print actions
    $("rep-btn-export").onclick = () => exportReportText(b);
    $("rep-btn-print").onclick = () => window.print();
  }

  function drawDial(el, val, norm, color) {
    const v = clamp(norm, 0, 1);
    el.innerHTML = `<svg viewBox="0 0 86 60">
      <path class="d-arc-bg" d="M 10 52 A 34 34 0 0 1 76 52"/>
      <path class="d-arc" d="M 10 52 A 34 34 0 0 1 76 52" stroke="${color}"/>
      <text class="d-num" x="43" y="50">${val}</text>
    </svg>`;
    const arc = el.querySelector(".d-arc");
    const L = arc.getTotalLength();
    arc.style.strokeDasharray = L; arc.style.strokeDashoffset = L;
    setTimeout(() => { arc.style.strokeDashoffset = L * (1 - v); }, 150);
  }

  async function signReport(caseId) {
    try {
      await api(`/v1/cases/${caseId}/report/sign`, {
        method: "POST", 
        headers: { "Content-Type": "application/json", "x-aura-user": "clinician" },
        body: JSON.stringify({ signed_by: "clinician" }),
      });
      
      toast(`${caseId} Report signed — audit trail updated`);
      
      // Update local state caches
      const b = S.bundles.get(caseId);
      if (b) b.state = "signed";
      
      const c = S.cases.find(x => x.case_id === caseId);
      if (c) c.state = "signed";

      route(); // reload view state
    } catch {
      toast("Report signing failed — gateway offline?");
    }
  }

  function exportReportText(b) {
    if (!b || !b.report) { toast("No report loaded to export"); return; }
    const s = b.safety || {};
    const dxLabel = (d) => DX_LABEL[d] || d || "—";
    const lines = [
      "AURA CLINICAL INTEGRITY SYSTEM REPORT",
      "=".repeat(56),
      `case         ${b.case_id}`,
      `study        ${b.study_id}`,
      `state        ${b.state.toUpperCase()}`,
      `exported     ${new Date().toISOString()}`,
      `fusion       ${(b.fusion && b.fusion.backend) || "—"} · conformal coverage ${Math.round((s.conformal_coverage || 0.9) * 100)}%`,
      "",
      "1. CLINICAL FINDINGS",
      b.report.findings_text || "—",
      "",
      "2. DIAGNOSTIC IMPRESSION",
      b.report.impression_text || "—",
      "",
      "3. CLINICAL RECOMMENDATIONS",
      b.report.recommendation_text || "—",
      "",
      "SAFETY ASSESSMENT",
      `top diagnosis        ${dxLabel(s.top)} (p=${s.top_probability ?? "—"})`,
      `conformal set        [${(s.conformal_set || []).map(dxLabel).join(", ")}] · ${s.conformal_method || "—"}`,
      `epistemic / aleatoric ${s.epistemic_uncertainty ?? "—"} / ${s.aleatoric_uncertainty ?? "—"}`,
      `ood energy (z)       ${s.ood_energy ?? "—"}`,
      `abstained            ${s.abstained ? `YES — ${s.abstention_reason}` : "no"}`,
      "",
      "Generated by AURA. Decision support only — not a medical diagnosis.",
    ];
    const blob = new Blob([lines.join("\n")], { type: "text/plain" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${b.case_id}_clinical_report.txt`;
    a.click();
    URL.revokeObjectURL(a.href);
    toast(`${b.case_id} report downloaded`);
  }

  /* ================= image drawing helper ================= */
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

  // Auto boot when DOM ready
  document.readyState === "loading"
    ? document.addEventListener("DOMContentLoaded", boot)
    : boot();

  return { boot };
})();
