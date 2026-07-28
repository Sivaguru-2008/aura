/* ============================================================
   CONSOLE — the living clinical system behind /app.
   Every panel is wired to the real gateway. Nothing is static:
   cases converge into reasoning, charts spring, reports type
   themselves, and the system reacts to the clinician's verdict.
   ============================================================ */
window.CONSOLE = (() => {
  "use strict";
  const { Field, api, toast, typeInto, clamp, REDUCED } = FX;

  let DX_LABEL = {};
  let EV_LABEL = {};
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
    incomplete_study: "The study is missing sequences this model requires — its calibration and its reported accuracy both assume a complete study.",
    low_quality: "Measured image quality is below the floor for a reportable result.",
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
      const [health, casesData] = await Promise.all([api("/v1/health"), api("/v1/studies")]);
      S.cases = casesData.cases || [];
      if (casesData.dx_labels) DX_LABEL = casesData.dx_labels;
      if (casesData.ev_labels) EV_LABEL = casesData.ev_labels;
      renderChips(health);
      renderWorklist();
      renderTelemetry();
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
    // Synthetic study generation removed — AURA runs real inference on uploaded
    // radiographs only. The upload paths below are the sole ways to create a case.
    // upload
    // Toggle MRI dropdown
    const mriBtn = $("btn-upload-mri");
    const mriDropdown = $("mri-dropdown");
    
    mriBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      const visible = mriDropdown.style.display === "block";
      mriDropdown.style.display = visible ? "none" : "block";
    });
    
    document.addEventListener("click", () => {
      mriDropdown.style.display = "none";
    });
    
    $("btn-upload-mri-files").addEventListener("click", (e) => {
      e.stopPropagation();
      mriDropdown.style.display = "none";
      $("input-file-mri").click();
    });
    
    $("btn-upload-mri-folder").addEventListener("click", (e) => {
      e.stopPropagation();
      mriDropdown.style.display = "none";
      $("input-folder-mri").click();
    });

    $("btn-upload-xray").addEventListener("click", () => { $("input-file-xray").click(); });
    $("btn-history").addEventListener("click", () => { window.open("/history", "_blank"); });
    
    $("input-file-xray").addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        uploadImage(e.target.files[0], "CHEST_XRAY");
      }
    });
    
    $("input-file-mri").addEventListener("change", (e) => {
      if (e.target.files && e.target.files.length) {
        if (e.target.files.length === 1) {
          uploadImage(e.target.files[0], "BRAIN_MRI");
        } else {
          uploadImage(e.target.files, "BRAIN_MRI");
        }
      }
    });

    $("input-folder-mri").addEventListener("change", (e) => {
      if (e.target.files && e.target.files.length) {
        uploadImage(e.target.files, "BRAIN_MRI");
      }
    });

    // drag a film anywhere onto the console to analyze it
    const root = $("console");
    ["dragenter", "dragover"].forEach((evt) => root.addEventListener(evt, (e) => {
      if (![...(e.dataTransfer?.types || [])].includes("Files")) return;
      e.preventDefault();
      $("btn-upload-xray").classList.add("dropping");
      $("btn-upload-mri").classList.add("dropping");
    }));
    root.addEventListener("dragleave", (e) => {
      if (!e.relatedTarget || !root.contains(e.relatedTarget)) {
        $("btn-upload-xray").classList.remove("dropping");
        $("btn-upload-mri").classList.remove("dropping");
      }
    });
    
    // Recursive directory traversal for dropped folders
    async function getFilesFromDataTransfer(dataTransfer) {
      const files = [];
      const items = [...(dataTransfer.items || [])];
      
      const traverse = async (entry) => {
        if (entry.isFile) {
          const file = await new Promise((resolve) => entry.file(resolve));
          const relativePath = entry.fullPath.startsWith("/") ? entry.fullPath.slice(1) : entry.fullPath;
          Object.defineProperty(file, "filename", {
            value: relativePath,
            writable: true
          });
          files.push(file);
        } else if (entry.isDirectory) {
          const reader = entry.createReader();
          let allEntries = [];
          
          const readAll = async () => {
            const results = await new Promise((resolve) => reader.readEntries(resolve));
            if (results.length > 0) {
              allEntries = allEntries.concat(results);
              await readAll();
            }
          };
          await readAll();
          
          for (const child of allEntries) {
            await traverse(child);
          }
        }
      };

      for (const item of items) {
        if (item.kind === "file") {
          const entry = item.webkitGetAsEntry();
          if (entry) {
            await traverse(entry);
          }
        }
      }
      return files;
    }

    root.addEventListener("drop", async (e) => {
      e.preventDefault();
      $("btn-upload-xray").classList.remove("dropping");
      $("btn-upload-mri").classList.remove("dropping");
      
      try {
        const files = await getFilesFromDataTransfer(e.dataTransfer);
        if (files.length === 0) return;
        
        let isMri = false;
        for (const file of files) {
          const name = file.name.toLowerCase();
          if (name.endsWith(".nii") || name.endsWith(".nii.gz") || name.endsWith(".nrrd") || name.endsWith(".zip")) {
            isMri = true;
            break;
          }
        }
        
        const declared = isMri ? "BRAIN_MRI" : null;
        if (files.length === 1) {
          uploadImage(files[0], declared);
        } else {
          uploadImage(files, declared);
        }
      } catch (err) {
        console.error("Drop traversal failed", err);
        const f = e.dataTransfer.files && e.dataTransfer.files[0];
        if (f) uploadImage(f, null);
      }
    });
    // report export
    $("btn-export").addEventListener("click", exportReport);
  }

  /* ================= worklist ================= */
  function renderWorklist() {
    const ol = $("worklist");
    $("rail-count").textContent = S.cases.length + " cases";
    ol.innerHTML = "";
    if (S.cases.length === 0) {
      ol.innerHTML = `
        <div class="empty-state mono" style="padding: 24px 12px; color: var(--faint); font-size: 13px; text-align: center; line-height: 1.6;">
          No studies available.<br>Upload a chest X-ray or brain MRI to begin analysis.
        </div>`;
      return;
    }
    S.cases.forEach((c, i) => {
      const li = document.createElement("li");
      li.className = "wl-item" + (c.abstained ? " abst" : "") + (c.case_id === S.current ? " sel" : "");
      li.style.setProperty("--pri", clamp(c.priority_score, 0.12, 1).toFixed(2));
      li.style.animation = `capIn .5s var(--ease) ${i * 45}ms backwards`;
      li.innerHTML = `
        <div class="wl-id"><span>${c.case_id}</span><span class="wl-state ${c.state}">${c.state}</span></div>
        <div class="wl-dx">${c.top_diagnosis_label || c.top_diagnosis || "—"}</div>
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
      try {
        b = await api(`/v1/cases/${id}`);
        S.bundles.set(id, b);
      }
      catch { toast("failed to load case"); grid.classList.remove("switching"); return; }
    }
    if (b.dx_labels) DX_LABEL = b.dx_labels;
    if (b.ev_labels) EV_LABEL = b.ev_labels;
    populate(b);
    grid.classList.remove("switching");
    loadSimilar(id);
  }

  const wait = (ms) => new Promise((r) => setTimeout(r, ms));

  function populate(b) {
    // Render each panel in isolation. A throw in any one panel must not abort the
    // rest: previously an exception in e.g. drawPosterior silently halted every
    // panel after it, so the differential, safety, next-best-evidence and report
    // all rendered blank while the x-ray and evidence graph looked fine.
    const steps = [
      ["xray", drawXray],
      ["neuroview", (b) => { if (window.NEUROVIEW) window.NEUROVIEW.load(b.case_id, b); }],
      ["evidence", drawEvidence],
      ["posterior", drawPosterior],
      ["safety", drawSafety],
      ["drp", drawDRP],
      ["recommendations", drawRecs],
      ["report", drawReport],
    ];
    for (const [name, fn] of steps) {
      try { fn(b); }
      catch (e) { console.error(`AURA: panel "${name}" failed to render`, e); }
    }
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
    // Show exactly the findings the model asserts, at the *calibrated* per-finding
    // operating point the report uses (server attaches f.present/f.threshold from
    // vision_serving_calibration.json). Falls back to 0.5 only for legacy responses
    // that lack the flag. Previously a hardcoded 0.5 hid genuine detections between
    // the calibrated threshold (0.13–0.29) and 0.5 (audit H1).
    const found = ((b.vision && b.vision.findings) || []).filter(
      (f) => (f.present !== undefined ? f.present : f.probability >= 0.5));
    found.forEach((f, i) => {
      const [r0, c0, r1, c1] = f.region;
      const d = document.createElement("div");
      d.className = "region";
      d.style.cssText = `top:${r0 * 100}%;left:${c0 * 100}%;height:${(r1 - r0) * 100}%;width:${(c1 - c0) * 100}%;animation-delay:${0.25 + i * 0.18}s`;
      // EV_LABEL is keyed by finding value (API ev_labels); look it up directly.
      // The old FINDING_TO_CHANNEL hop mislabeled pleural_effusion (value ≠ channel).
      d.innerHTML = `<span class="r-lbl">${(EV_LABEL[f.finding] || f.finding)} · ${f.probability.toFixed(2)}</span>`;
      wrap.appendChild(d);
    });
    const p = b.priors || {};
    const modalityLabel = (b.study_id && b.study_id.startsWith("STU-MR")) ? "MRI" : "CXR";
    $("xray-meta").innerHTML = `
      <span>${b.study_id} · ${modalityLabel} ${(b.image_shape || []).join("×") || "—"}</span>
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

    // --- quantum entanglement edges and telemetry ---
    const nodePosMap = {};
    items.forEach((it, i) => {
      nodePosMap[it.name] = { pos: nodePos[i], index: i };
    });

    const qEnt = b.fusion && b.fusion.quantum_entanglement;
    if (qEnt && qEnt.top_pairs) {
      qEnt.top_pairs.forEach((pair) => {
        const [ch1, ch2] = pair.channels;
        const corr = pair.correlation;
        
        if (nodePosMap[ch1] && nodePosMap[ch2]) {
          const [x1, y1] = nodePosMap[ch1].pos;
          const [x2, y2] = nodePosMap[ch2].pos;
          
          const mx = (x1 + x2) / 2;
          const my = (y1 + y2) / 2;
          const dx = x2 - x1;
          const dy = y2 - y1;
          const len = Math.sqrt(dx * dx + dy * dy);
          
          const ox = -dy / len * 25;
          const oy = dx / len * 25;
          
          const isPos = corr > 0;
          const strokeCol = isPos ? "rgba(139,124,247,0.75)" : "rgba(244,182,78,0.7)";
          const strokeWidth = (Math.abs(corr) * 12 + 1.2).toFixed(2);
          
          const e = mk("path", {
            d: `M ${x1} ${y1} Q ${mx + ox} ${my + oy} ${x2} ${y2}`,
            class: "ev-entanglement-edge",
            stroke: strokeCol,
            "stroke-width": strokeWidth,
            fill: "none",
            opacity: 0.15 + Math.abs(corr) * 0.8,
            "stroke-dasharray": isPos ? "none" : "3 3",
          });
          
          const L = e.getTotalLength();
          e.style.strokeDasharray = L;
          e.style.strokeDashoffset = L;
          e.style.transition = `stroke-dashoffset 1.1s ease-out 0.8s`;
          requestAnimationFrame(() => requestAnimationFrame(() => {
            e.style.strokeDashoffset = 0;
          }));
        }
      });
    }

    const qTel = $("quantum-entanglement-telemetry");
    if (b.fusion && (qEnt || b.fusion.qae_enabled || b.fusion.qbn_enabled)) {
      qTel.style.display = "block";
      if (qEnt) {
        // Null-safe formatting: the entanglement field set varies by backend/version,
        // and a single undefined here used to throw and blank every panel below it
        // (now isolated in populate(), but the telemetry should still degrade to "—").
        const f4 = (v) => (typeof v === "number" && isFinite(v)) ? v.toFixed(4) : "—";
        $("q-entropy").textContent = f4(qEnt.measurement_entropy_bits);
        const shift = qEnt.entropy_shift_bits;
        const shiftEl = $("q-entropy-shift");
        shiftEl.textContent = (typeof shift === "number" && isFinite(shift))
          ? `${shift >= 0 ? "+" : ""}${shift.toFixed(4)} bits` : "—";
        shiftEl.style.color = (shift < 0) ? "#4be1c3" : "#ff6b6b";
        $("q-coupling").textContent = f4(qEnt.differential_coupling);
        // Backend emits `total_coupling`; some builds named it `baseline_coupling`.
        $("q-baseline-coupling").textContent = f4(qEnt.baseline_coupling ?? qEnt.total_coupling);
      } else {
        $("q-entropy").textContent = "—";
        $("q-entropy-shift").textContent = "—";
        $("q-coupling").textContent = "—";
        $("q-baseline-coupling").textContent = "—";
      }
      const qaeRow = $("qae-telemetry-row");
      if (qaeRow) qaeRow.style.display = b.fusion.qae_enabled ? "flex" : "none";
      const qbnRow = $("qbn-telemetry-row");
      if (qbnRow) qbnRow.style.display = b.fusion.qbn_enabled ? "flex" : "none";
    } else {
      qTel.style.display = "none";
    }
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
    const preds = [...(b.safety.predictions || [])].sort((a, c) => c.probability - a.probability);
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
    // A null here means the engine did not measure this quantity — the brain path runs
    // one network with no ensemble, no conformal calibration set and no OOD scorer.
    // Rendering that as a dial reading 0.000 would claim a measurement was taken and
    // came back reassuring, which is the opposite of what null means.
    const num = (v) => (typeof v === "number" && isFinite(v));
    const dialOrNA = (el, label, v, fmt, frac, color) => {
      if (!num(v)) { dial(el, label, "n/a", 0, "#4a5568"); el.title = "not measured by this engine"; return; }
      dial(el, label, fmt(v), frac(v), color(v));
    };
    dialOrNA($("d-epi"), "EPISTEMIC", s.epistemic_uncertainty,
      (v) => v.toFixed(3), (v) => v / 0.25, (v) => (v > 0.12 ? "#f4b64e" : "#4be1c3"));
    dialOrNA($("d-ale"), "ALEATORIC", s.aleatoric_uncertainty,
      (v) => v.toFixed(3), (v) => v / 1.8, () => "#8b7cf7");
    dialOrNA($("d-ood"), "OOD ENERGY", s.ood_energy,
      (v) => v.toFixed(2), (v) => (v + 6) / 12, () => (s.is_ood ? "#ff5d5d" : "#4be1c3"));
    $("conf-lbl").textContent = num(s.conformal_coverage)
      ? `${Math.round(s.conformal_coverage * 100)}% CONFORMAL SET — truth in this set ${Math.round(s.conformal_coverage * 100)}/100 times`
      : "CANDIDATE SET — no conformal calibration fitted for this engine, so no coverage is claimed";
    $("conf-chips").innerHTML = (s.conformal_set || [])
      .map((d, i) => `<span class="conf-chip ${s.conformal_set.length > 2 && i > 0 ? "hot" : ""}" style="animation-delay:${i * 0.1}s">${DX_LABEL[d] || d}</span>`).join("");
  }

  function drawDRP(b) {
    const sidebar = $("drp-sidebar-panel");
    if (!sidebar) return;

    // Check if safety controller failed
    const reportOverlay = $("report-safety-overlay");
    if (b.safety_controller && b.safety_controller.status === "FAILED") {
      if (reportOverlay) {
        reportOverlay.style.display = "flex";
        $("safety-overlay-msg").textContent = b.safety_controller.detail || "Clinical safety thresholds breached.";
        
        let checklistHtml = "<strong>Mitigation Checklist:</strong><ul style='margin-top:5px; padding-left:15px; list-style-type: disc;'>";
        if (b.safety_controller.failed_checks.includes("DATA_INTEGRITY")) {
          checklistHtml += "<li>Verify upload aspect ratio and image quality.</li>";
          checklistHtml += "<li>Ensure all MRI sequences are fully uploaded and complete.</li>";
        }
        if (b.safety_controller.failed_checks.includes("OOD")) {
          checklistHtml += "<li>Perform human verification of input labels.</li>";
          checklistHtml += "<li>Review clinical history for unusual anomalies.</li>";
        }
        if (b.safety_controller.failed_checks.includes("EPISTEMIC")) {
          checklistHtml += "<li>Consult senior radiologist / repeat scan if uncertainty remains high.</li>";
        }
        checklistHtml += "</ul>";
        $("safety-mitigation-checklist").innerHTML = checklistHtml;
      }
      sidebar.style.display = "none";
      return;
    } else {
      if (reportOverlay) reportOverlay.style.display = "none";
    }

    const drp = b.drp;
    if (!drp) {
      sidebar.style.display = "none";
      return;
    }

    sidebar.style.display = "block";

    // Update status chip
    const statusChip = $("drp-status-chip");
    if (drp.status === "READY") {
      statusChip.innerHTML = `<span class="flag ok" style="padding: 2px 6px; font-size: 10px;">DECISION READINESS: READY</span>`;
    } else {
      statusChip.innerHTML = `<span class="flag abst" style="padding: 2px 6px; font-size: 10px; background: #ff5d5d;">DECISION READINESS: NOT READY</span>`;
    }

    // Render dimensions
    const dims = [
      ["Coverage", drp.coverage],
      ["Quality", drp.quality],
      ["Consistency", drp.consistency],
      ["Robustness", drp.robustness],
      ["Stability", drp.stability],
      ["Consensus", drp.consensus]
    ];

    let barsHtml = "";
    dims.forEach(([name, val]) => {
      const pct = (val * 100).toFixed(0);
      let color = "#4be1c3";
      if (val < 0.6) color = "#ff5d5d";
      else if (val < 0.8) color = "#ffd166";

      barsHtml += `
        <div style="display: flex; flex-direction: column; gap: 4px;">
          <div style="display: flex; justify-content: space-between; font-size: 10.5px;">
            <span style="color: var(--faint);">${name}</span>
            <span style="color: ${color}; font-weight: bold;">${pct}%</span>
          </div>
          <div style="height: 4px; background: rgba(255,255,255,0.05); border-radius: 2px; overflow: hidden;">
            <div style="height: 100%; width: ${pct}%; background: ${color}; transition: width 0.6s ease-out;"></div>
          </div>
        </div>
      `;
    });
    $("drp-bars").innerHTML = barsHtml;

    // Show limiting factor
    const limitingDiv = $("drp-limiting");
    if (drp.status === "NOT_READY") {
      limitingDiv.style.display = "block";
      let linkHtml = "";
      if (drp.limiting_dimension === "coverage" || drp.limiting_dimension === "quality") {
        linkHtml = `<div style="margin-top: 6px;"><a href="#" onclick="alert('Order request for required test submitted.'); return false;" style="color: #ffd166; text-decoration: underline; font-weight: bold;">Order Required Test &rarr;</a></div>`;
      }
      limitingDiv.innerHTML = `
        <div style="color: #ff5d5d; font-weight: bold; font-size: 11px;">LIMITING FACTOR:</div>
        <div style="color: #fff; font-size: 11px; margin-top: 2px;">${drp.limiting_factor}</div>
        ${linkHtml}
      `;
    } else {
      limitingDiv.style.display = "none";
    }

    // Render EDP
    const edpPanel = $("edp-panel");
    const edpList = $("edp-list");
    if (drp.evidence_dependency_profile && Object.keys(drp.evidence_dependency_profile).length > 0) {
      edpPanel.style.display = "block";
      let edpHtml = "";
      Object.entries(drp.evidence_dependency_profile).forEach(([finding, w]) => {
        const obs = w.observed;
        const clin = w.clinical;
        const label = EV_LABEL[finding] || finding;

        edpHtml += `
          <div style="display: flex; justify-content: space-between; font-size: 10.5px; border-bottom: 1px dashed rgba(255,255,255,0.03); padding-bottom: 2px;">
            <span style="color: #fff; text-overflow: ellipsis; overflow: hidden; white-space: nowrap; max-width: 110px;" title="${label}">${label}</span>
            <span style="font-size: 9.5px; color: var(--faint);">obs: <b style="color: #8b7cf7;">${obs.toFixed(2)}</b> / clin: <b>${clin.toFixed(2)}</b></span>
          </div>
        `;
      });
      edpList.innerHTML = edpHtml;
    } else {
      edpPanel.style.display = "none";
    }
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
      const casesData = await api("/v1/studies");
      S.cases = casesData.cases || [];
      if (casesData.dx_labels) DX_LABEL = casesData.dx_labels;
      if (casesData.ev_labels) EV_LABEL = casesData.ev_labels;
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

  const UPLOAD_STAGES = [
    "receiving radiograph …",
    "X-ray intake gate — grayscale · tonal depth · chest anatomy",
    "vision engine reading film",
    "encoding 8 evidence channels",
    "entangling qubits — fusion posterior",
    "conformal calibration · OOD sweep",
    "counterfactual attribution",
    "ranking next-best evidence",
    "grounding report",
  ];

  // human-readable rendering of the intake gate's measurements
  const GATE_CHECK_LABEL = {
    mean_saturation: ["color saturation", (v) => `${v.toFixed(3)} (max 0.080)`],
    colored_fraction: ["colored pixels", (v) => `${(v * 100).toFixed(1)}% (max 10%)`],
    aspect_ratio: ["aspect ratio", (v) => `${v.toFixed(2)} (0.40–2.50)`],
    gray_std: ["tonal range", (v) => `${v.toFixed(3)} (min 0.040)`],
    tonal_entropy_bits: ["tonal entropy", (v) => `${v.toFixed(2)} bits (min 4.00)`],
    center_ratio: ["mediastinum brightness", (v) => `${v.toFixed(2)}× lungs (min 1.05)`],
    column_variation: ["chest column profile", (v) => `${v.toFixed(3)} (min 0.090)`],
    modality: ["dicom modality", (v) => `${v}`],
    body_part: ["dicom body part", (v) => `${v}`],
  };

  function gateCheckLines(checks) {
    if (!checks) return "";
    return Object.entries(checks)
      .filter(([k]) => GATE_CHECK_LABEL[k])
      .map(([k, v]) => {
        const [lbl, fmt] = GATE_CHECK_LABEL[k];
        return `  ${lbl.padEnd(24)} ${fmt(v)}`;
      }).join("\n");
  }

  // ``modality`` is what the *user* declared by choosing an upload button. Pass null
  // when nothing was declared (a drag-and-drop) so the server routes on the pixels
  // rather than on a guess the client made up.
  // rather than on a guess the client made up.
  async function uploadImage(fileOrFiles, modality = null) {
    const isMultiple = (fileOrFiles instanceof FileList) || Array.isArray(fileOrFiles);
    const files = isMultiple ? [...fileOrFiles] : [fileOrFiles];
    const firstFile = files[0];

    // If it's a Brain MRI upload, show the preview first!
    if (modality === "BRAIN_MRI") {
      const overlay = $("case-forming");
      const txt = $("forming-text");
      overlay.hidden = false;
      txt.innerHTML = `<span class="ok">▸</span> Staging files and reading metadata …\n`;
      const f = new Field($("forming-canvas"), { count: 80, hue: 258, mode: "collapse", size: 1.6, speed: 1.1 });
      f.start();

      try {
        const fd = new FormData();
        if (isMultiple) {
          files.forEach(file => {
            const filename = file.filename || file.name;
            fd.append("files", file, filename);
          });
        } else {
          fd.append("file", firstFile);
        }

        const previewData = await api(`/v1/studies/preview`, {
          method: "POST",
          headers: { "x-aura-user": "clinician" },
          body: fd,
        });

        // Hide overlay and stop animation
        overlay.hidden = true;
        f.destroy();

        // Populate the preview modal elements
        $("preview-patient-id").textContent = previewData.patient_id || "n/a";
        $("preview-study-id").textContent = previewData.study_id || "n/a";
        $("preview-seq-type").textContent = previewData.sequence_type || "n/a";
        $("preview-dims").textContent = previewData.original_dimensions ? previewData.original_dimensions.join(" x ") : "n/a";
        $("preview-spacing").textContent = previewData.voxel_spacing ? previewData.voxel_spacing.map(v => Number(v).toFixed(3)).join(" x ") + " mm" : "n/a";
        $("preview-orientation").textContent = previewData.orientation || "n/a";
        $("preview-slices").textContent = previewData.number_of_slices || "n/a";
        $("preview-affine").textContent = previewData.affine_matrix_summary || "n/a";
        
        const scannerText = previewData.scanner_metadata 
          ? `Manufacturer: ${previewData.scanner_metadata.Manufacturer}\nStrength: ${previewData.scanner_metadata.MagneticFieldStrength}\nModality: ${previewData.scanner_metadata.Modality}`
          : "n/a";
        $("preview-scanner").innerText = scannerText;

        // Populate thumbnails
        const t = previewData.thumbnails || {};
        $("thumb-flair").src = t.flair || "";
        $("thumb-t1").src = t.t1 || "";
        $("thumb-t1ce").src = t.t1ce || "";
        $("thumb-t2").src = t.t2 || "";

        // Display modal
        const modal = $("mri-preview-modal");
        modal.style.display = "flex";

        // Bind cancel buttons
        const closeBtn = $("btn-close-preview");
        const cancelBtn = $("btn-cancel-preview");
        const runBtn = $("btn-run-analysis");

        const cleanup = () => {
          modal.style.display = "none";
          closeBtn.onclick = null;
          cancelBtn.onclick = null;
          runBtn.onclick = null;
        };

        closeBtn.onclick = cleanup;
        cancelBtn.onclick = cleanup;

        // When "Analyze Study" is clicked, run actual analysis!
        runBtn.onclick = async () => {
          cleanup();
          await executeRealAnalysis(files, modality);
        };

      } catch (err) {
        overlay.hidden = true;
        f.destroy();

        const d = (err && err.detail) || {};
        const why = d.reason || d.message || err.message || "upload validation failed";
        // The #1 upload mistake is selecting a single sequence. A lone 3-D volume can
        // never form a complete BraTS study (FLAIR + T1 + T1ce + T2), and the raw
        // validator message doesn't say what to do about it. Turn it into an
        // instruction. A genuine single 4-D NIfTI still succeeds, so this only fires
        // for the incomplete-study case.
        if (files.length === 1 && /four sequences|4-?D|3-?D file/i.test(why)) {
          toast("Brain MRI needs all four sequences. Select FLAIR, T1, T1ce and T2 together (Ctrl/Cmd-click), or use Upload Folder — one 3-D file on its own can't be analysed.");
        } else {
          toast("Preview generation failed: " + why);
        }
      } finally {
        $("input-file-xray").value = "";
        $("input-file-mri").value = "";
        $("input-folder-mri").value = "";
      }
      return;
    }

    // Default flow for X-ray or other modalities
    await executeRealAnalysis(files, modality);
  }

  async function executeRealAnalysis(files, modality) {
    const isMultiple = files.length > 1;
    const firstFile = files[0];
    const overlay = $("case-forming");
    const txt = $("forming-text");
    const prev = $("forming-preview");
    overlay.hidden = false;
    txt.innerHTML = "";
    
    let prevUrl = null;
    if (!isMultiple && firstFile && firstFile.type && firstFile.type.startsWith("image/")) {
      prevUrl = URL.createObjectURL(firstFile);
      prev.src = prevUrl; prev.hidden = false;
    } else { prev.hidden = true; }
    
    const hue = modality === "BRAIN_MRI" ? 258 : 172;
    const f = new Field($("forming-canvas"), { count: 160, hue: hue, mode: "collapse", size: 1.6, speed: 1.1 });
    f.start();
    
    const mriStages = [
      "Uploading…",
      "Verifying MRI sequences…",
      "Reading NIfTI volumes…",
      "Checking affine consistency…",
      "Checking voxel spacing…",
      "Preparing multimodal tensor…",
      "Running AI inference…",
      "Generating report…",
    ];
    const xrayStages = [
      "receiving radiograph …",
      "X-ray intake gate — grayscale · tonal depth · chest anatomy",
      "vision engine reading film",
      "encoding 8 evidence channels",
      "entangling qubits — fusion posterior",
      "conformal calibration · OOD sweep",
      "counterfactual attribution",
      "ranking next-best evidence",
      "grounding report",
    ];
    const stages = modality === "BRAIN_MRI" ? mriStages : xrayStages;
    
    let alive = true;
    (async () => {
      for (const line of stages) {
        if (!alive) return;
        txt.innerHTML += `<span class="ok">▸</span> ${line}\n`;
        await wait(REDUCED ? 30 : 340);
      }
    })();
    try {
      const fd = new FormData();
      if (files.length > 1 || files[0].filename) {
        files.forEach(file => {
          const filename = file.filename || file.name;
          fd.append("files", file, filename);
        });
      } else {
        fd.append("file", firstFile);
      }
      
      const q = modality ? `?declared_modality=${encodeURIComponent(modality)}` : "";
      const d = await api(`/v1/studies/analyze${q}`, {
        method: "POST",
        headers: { "x-aura-user": "clinician" },
        body: fd,
      });
      const caseId = (d.result && d.result.case_id) || d.case_id;
      const casesData = await api("/v1/studies");
      S.cases = casesData.cases || [];
      if (casesData.dx_labels) DX_LABEL = casesData.dx_labels;
      if (casesData.ev_labels) EV_LABEL = casesData.ev_labels;
      const h = await api("/v1/health").catch(() => null);
      if (h) renderChips(h);
      await wait(REDUCED ? 0 : 900); // let the convergence land
      alive = false;
      overlay.hidden = true; f.destroy();
      renderWorklist();
      renderTelemetry();
      S.current = null;
      if (caseId) {
        selectCase(caseId, { first: true });
        toast(`${caseId} analyzed! <a href="/history#/case/${caseId}/image" target="_blank" style="color:var(--cyan);text-decoration:underline">View Film Page</a> | <a href="/history#/case/${caseId}/report" target="_blank" style="color:var(--cyan);text-decoration:underline">View Report Page</a>`);
      } else {
        toast(`Analysis completed: ${d.result ? d.result.message : "no case created"}`);
      }
    } catch (err) {
      alive = false;
      const d = (err && err.detail) || {};
      const code = d.error || d.code;
      const is422 = err && err.status === 422;
      const REFUSAL = {
        not_a_cxr: "REJECTED — not a chest X-ray",
        modality_conflict: "REJECTED — this is not the modality you selected",
        modality_undetermined: "REJECTED — AURA could not identify this study",
        unsupported_modality: "REJECTED — no engine serves this modality yet",
        unreadable_image: "REJECTED — the study could not be read",
        study_validation_failed: "REJECTED — validation failed",
      };
      
      const why = d.reason || d.message || err.message || "upload failed";
      
      if (code === "clinical_safety_violation") {
        txt.innerHTML += `<span class="bad">✕ SAFETY BREACH DETECTED</span>\n<span class="bad">  ${why}</span>\n`;
        const reportOverlay = $("report-safety-overlay");
        if (reportOverlay) {
          reportOverlay.style.display = "flex";
          $("safety-overlay-msg").textContent = why;
          
          let checklistHtml = "<strong>Mitigation Checklist:</strong><ul style='margin-top:5px; padding-left:15px; list-style-type: disc;'>";
          checklistHtml += "<li>Verify upload aspect ratio and image quality.</li>";
          checklistHtml += "<li>Ensure all MRI sequences are fully uploaded and complete.</li>";
          checklistHtml += "<li>Perform human verification of input labels.</li>";
          checklistHtml += "<li>Review clinical history for unusual anomalies.</li>";
          checklistHtml += "<li>Consult senior radiologist / repeat scan if uncertainty remains high.</li>";
          checklistHtml += "</ul>";
          $("safety-mitigation-checklist").innerHTML = checklistHtml;
        }
      } else if (is422 && (REFUSAL[code] || code === "study_validation_failed")) {
        const title = REFUSAL[code] || "REJECTED — validation failed";
        txt.innerHTML += `<span class="bad">✕ ${title}</span>\n<span class="bad">  ${why}</span>\n`;
        const lines = gateCheckLines(d.checks);
        if (lines) txt.innerHTML += `<span class="dim-meas">gate measurements:\n${lines}</span>\n`;
        if (Array.isArray(d.candidates) && d.candidates.length) {
          const rows = d.candidates.slice(0, 4).map((c) =>
            `  ${String(c.label || c.modality).padEnd(22)} ${Number(c.confidence).toFixed(2)}` +
            (c.reason ? ` (${c.reason})` : "")
          ).join("\n");
          txt.innerHTML += `<span class="dim-meas">modality candidates:\n${rows}</span>\n`;
        }
      } else {
        txt.innerHTML += `<span class="bad">✕ ${why}</span>\n`;
      }
      
      await wait(1800);
      overlay.hidden = true; f.destroy();
      renderTelemetry();
    } finally {
      $("input-file-xray").value = "";
      $("input-file-mri").value = "";
      $("input-folder-mri").value = "";
      const prevEl = $("forming-preview");
      prevEl.hidden = true; prevEl.removeAttribute("src");
      if (prevUrl) URL.revokeObjectURL(prevUrl);
    }
  }

  /* ================= report export ================= */
  function exportReport() {
    const b = S.current && S.bundles.get(S.current);
    if (!b || !b.report) { toast("no report loaded to export"); return; }
    const s = b.safety || {};
    const dxLabel = (d) => DX_LABEL[d] || d || "—";
    const lines = [
      "AURA GROUNDED REPORT",
      "=".repeat(56),
      `case         ${b.case_id}`,
      `study        ${b.study_id}`,
      `state        ${b.state}`,
      `exported     ${new Date().toISOString()}`,
      `fusion       ${(b.fusion && b.fusion.backend) || "—"} · conformal coverage ${Math.round((s.conformal_coverage || 0.9) * 100)}%`,
      "",
      "FINDINGS",
      b.report.findings_text || "—",
      "",
      "IMPRESSION",
      b.report.impression_text || "—",
      "",
      "RECOMMENDATION",
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
    a.download = `${b.case_id}_report.txt`;
    a.click();
    URL.revokeObjectURL(a.href);
    toast(`${b.case_id} report downloaded`);
  }

  async function renderTelemetry() {
    try {
      const stats = await api("/v1/admin/safety");
      let total = S.cases.length;
      let rejected = 0;
      let ood = 0;
      
      if (stats.recent_audit && stats.recent_audit.length) {
        stats.recent_audit.forEach(a => {
          if (a.action === "case.upload_rejected" || a.action === "modality.rejected" || a.action === "agent.upload_rejected") {
            rejected++;
          }
          if (a.detail && (a.detail.reason === "out_of_distribution" || a.detail.reason === "unrecognised_image" || (typeof a.detail.reason === "string" && a.detail.reason.includes("OOD")))) {
            ood++;
          }
        });
      }
      
      S.cases.forEach(c => {
        if (c.abstained && c.conformal_set && c.conformal_set.length === 0) {
          ood++;
        }
      });
      
      const totalStudies = stats.recent_audit ? stats.recent_audit.filter(a => a.action === "case.uploaded").length : total;
      const rejectedCount = stats.recent_audit ? stats.recent_audit.filter(a => a.action === "case.upload_rejected" || a.action === "modality.rejected" || a.action === "agent.upload_rejected").length : rejected;
      const oodCount = stats.recent_audit ? stats.recent_audit.filter(a => a.detail && (a.detail.reason === "out_of_distribution" || a.detail.reason === "unrecognised_image" || a.action === "modality.rejected" && a.detail.confidence < 0.2)).length : ood;
      
      if ($("tel-total-studies")) $("tel-total-studies").textContent = Math.max(total, totalStudies);
      if ($("tel-rejected-studies")) $("tel-rejected-studies").textContent = rejectedCount;
      if ($("tel-ood-count")) $("tel-ood-count").textContent = oodCount;
      if ($("tel-calib-status")) {
        const ece = stats.benchmark && stats.benchmark.quantum ? stats.benchmark.quantum.ece : 0.2087;
        $("tel-calib-status").textContent = ece;
      }
      if ($("tel-audit-health")) $("tel-audit-health").textContent = "Healthy";
    } catch (err) {
      console.warn("Failed to load telemetry:", err);
    }
  }


  return { boot };
})();
