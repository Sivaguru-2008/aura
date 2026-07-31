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
  let currentSaliencyTab = "heatmap";
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

  const S = { cases: [], current: null, bundles: new Map(), booted: false, offline: false, health: null };
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
      $("c-chips").innerHTML = `<span class="c-chip" style="color:var(--red)">GATEWAY OFFLINE — run \`py -m aura.aura_cli serve\`</span>`;
    }
  }

  function renderChips(h) {
    S.health = h;   // every health payload flows through here; telemetry reads the
                    // serving backend from it to pick the right calibration column.
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

  /* ================= workflow pager =================
     Three views over ONE case. Deliberately a visibility switch and nothing more:
     populate() keeps rendering every panel for every case, so a page change cannot
     drop panel state, cannot re-fire a fetch, and cannot introduce an ordering bug.
     Everything below only sets attributes and classes.                            */
  const PAGES = ["read", "assess", "report"];
  const PAGE_KEY = "aura.console.page";
  let PAGE = "read";

  function setPage(name, { remember = true } = {}) {
    if (!PAGES.includes(name)) return;
    PAGE = name;
    const grid = $("c-grid");
    if (grid) grid.dataset.page = name;
    document.querySelectorAll("#c-pager .pg-tab").forEach((t) => {
      const on = t.dataset.goto === name;
      t.classList.toggle("on", on);
      t.setAttribute("aria-selected", on ? "true" : "false");
    });
    // A page change is a viewport change: start the new view at the top rather
    // than at whatever scroll offset the previous (taller) page happened to have.
    const main = $("c-main");
    if (main) main.scrollTo({ top: 0, behavior: REDUCED ? "auto" : "smooth" });
    if (remember) { try { localStorage.setItem(PAGE_KEY, name); } catch (e) {} }
  }

  function bindPager() {
    document.querySelectorAll("#c-pager .pg-tab").forEach((t) => {
      t.addEventListener("click", () => setPage(t.dataset.goto));
    });
    // 1/2/3 jump between views; ignored while typing so the report editor is safe.
    document.addEventListener("keydown", (e) => {
      if (document.body.dataset.view !== "console") return;
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const el = document.activeElement;
      if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.isContentEditable)) return;
      const i = ["1", "2", "3"].indexOf(e.key);
      if (i >= 0) { e.preventDefault(); setPage(PAGES[i]); }
    });
    let start = "read";
    try { start = localStorage.getItem(PAGE_KEY) || "read"; } catch (e) {}
    setPage(start, { remember: false });
  }

  /** Flag on the Assess tab when the answer for this case is "AURA abstained". */
  function updatePagerState(b) {
    const dot = $("pg-dot-assess");
    if (dot) dot.hidden = !(b.safety && b.safety.abstained);
  }

  function bindChrome() {
    bindPager();
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
    
    // FHIR & HL7 exports (Task 6)
    $("btn-export-fhir").addEventListener("click", exportFHIR);
    $("btn-export-hl7").addEventListener("click", exportHL7);
    
    // Admin Audit modal (Task 6)
    const auditTrigger = $("tel-audit-health");
    if (auditTrigger) {
      auditTrigger.style.cursor = "pointer";
      auditTrigger.style.textDecoration = "underline";
      auditTrigger.addEventListener("click", showAuditLog);
    }
    $("btn-close-audit").addEventListener("click", () => {
      $("audit-modal").style.display = "none";
    });
    
    // Explainability Dashboard Tabs (Task 5)
    setupExplainTabs();

    // Clinical Priors sidebar checkboxes (Task 2)
    const priorsList = ["fever", "cough", "bnp", "cardiopathy", "diabetes", "hypertension", "smoking", "crp"];
    priorsList.forEach((c) => {
      const checkbox = $(`prior-${c}`);
      if (checkbox) {
        checkbox.addEventListener("change", async () => {
          const id = S.current;
          if (!id) return;
          const grid = $("c-grid");
          grid.classList.add("switching");
          try {
            const payload = {
              priors: {
                fever: $("prior-fever").checked,
                cough: $("prior-cough").checked,
                high_bnp: $("prior-bnp").checked,
                prior_cardiopathy: $("prior-cardiopathy").checked,
                diabetes: $("prior-diabetes").checked,
                hypertension: $("prior-hypertension").checked,
                smoker: $("prior-smoking").checked,
                elevated_crp: $("prior-crp").checked
              }
            };
            const updated = await api(`/v1/cases/${id}/recompute`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(payload)
            });
            S.bundles.set(id, updated);
            populate(updated);
            toast("Clinical priors updated & reasoning recomputed");
          } catch (err) {
            console.error("Recomputation failed:", err);
            toast("Recomputation failed — check input");
          } finally {
            grid.classList.remove("switching");
          }
        });
      }
    });

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
    $("btn-agent").addEventListener("click", () => { $("input-file-agent").click(); });

    $("input-file-xray").addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        uploadImage(e.target.files[0], "CHEST_XRAY");
      }
    });

    $("input-file-agent").addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        runAgent(e.target.files[0]);
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
      ["measurement", drawMeasurement],
      ["safety", drawSafety],
      ["drp", drawDRP],
      ["recommendations", drawRecs],
      ["report", drawReport],
      ["evolution", drawEvolution],
      ["longitudinal", drawLongitudinal],
      ["discussion", drawDiscussion],
      ["priors_sidebar", updatePriorsSidebar],
      ["pager", updatePagerState],
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

  /* Draw the measured Grad-CAM++ region outlines.
     The polygons come from services/explain/geometry.py: simplified contours of
     the actual activation, normalized to [0,1] in (x, y) order. Returns false
     when the case carries no geometry, so the caller can fall back to the
     static anatomical boxes. */
  function drawSaliencyGeometry(b, wrap) {
    const geo = b.explanation && b.explanation.geometry;
    const regions = geo && geo.regions;
    if (!regions || !regions.length) return false;

    const NS = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("class", "sal-geo");
    svg.setAttribute("viewBox", "0 0 1 1");
    svg.setAttribute("preserveAspectRatio", "none");

    // Iso-level contours first, so region outlines sit on top of them.
    (geo.contours || []).forEach((c) => {
      if (!c.points || c.points.length < 3) return;
      const path = document.createElementNS(NS, "polygon");
      path.setAttribute("points", c.points.map((p) => `${p[0]},${p[1]}`).join(" "));
      path.setAttribute("class", "sal-contour");
      path.setAttribute("stroke-width", String(0.0016 + 0.0022 * c.level));
      path.setAttribute("opacity", String(0.18 + 0.42 * c.level));
      svg.appendChild(path);
    });

    regions.forEach((r, i) => {
      if (r.polygon && r.polygon.length >= 3) {
        const poly = document.createElementNS(NS, "polygon");
        poly.setAttribute("points", r.polygon.map((p) => `${p[0]},${p[1]}`).join(" "));
        poly.setAttribute("class", "sal-region");
        poly.style.animationDelay = `${0.25 + i * 0.18}s`;
        svg.appendChild(poly);
      }
      if (r.peak_point) {
        const dot = document.createElementNS(NS, "circle");
        dot.setAttribute("cx", r.peak_point[0]);
        dot.setAttribute("cy", r.peak_point[1]);
        dot.setAttribute("r", "0.008");
        dot.setAttribute("class", "sal-peak");
        svg.appendChild(dot);
      }
    });
    wrap.appendChild(svg);

    // Labels are HTML, not SVG text: the viewBox is unit-square and
    // non-uniformly scaled, which would distort any glyphs drawn inside it.
    const label = geo.finding ? (EV_LABEL[geo.finding] || geo.finding) : "saliency";
    regions.forEach((r, i) => {
      const [cx, cy] = r.centroid || r.peak_point || [0.5, 0.5];
      const tag = document.createElement("div");
      tag.className = "sal-tag";
      tag.style.cssText = `left:${cx * 100}%;top:${cy * 100}%;animation-delay:${0.35 + i * 0.18}s`;
      const pct = geo.probability != null ? ` · ${(geo.probability * 100).toFixed(0)}%` : "";
      tag.textContent = `${label}${pct} · peak ${r.peak.toFixed(2)}`;
      wrap.appendChild(tag);
    });
    return true;
  }

  function drawXray(b) {
    if (b.image && b.image.length) {
      paintGrid($("xray"), b.image, b.image_shape, (t) => {
        const v = Math.round(Math.pow(t, 0.9) * 255);
        return [v, v, Math.min(255, v + 6), 255];
      });
    }
    
    let sal = [];
    if (currentSaliencyTab === "heatmap") {
      sal = (b.explanation && b.explanation.saliency) || [];
    } else if (currentSaliencyTab === "occlusion") {
      sal = (b.explanation && b.explanation.saliency_methods && b.explanation.saliency_methods.occlusion) || [];
    } else if (currentSaliencyTab === "integrated") {
      sal = (b.explanation && b.explanation.saliency_methods && b.explanation.saliency_methods.integrated_gradients) || [];
    } else if (currentSaliencyTab === "importance" || currentSaliencyTab === "counterfactual") {
      sal = (b.explanation && b.explanation.saliency) || [];
    }
    
    if (sal.length) {
      paintGrid($("xray-sal"), sal, (b.explanation && b.explanation.saliency_shape) || b.image_shape, (t) => {
        // heat: transparent → cyan → amber
        const a = Math.round(Math.pow(t, 1.4) * 235);
        return t < 0.55 ? [40, 210, 190, a * 0.7] : [245, 182, 78, a];
      });
    } else {
      const cv = $("xray-sal");
      if (cv) {
        const ctx = cv.getContext("2d");
        ctx.clearRect(0, 0, cv.width, cv.height);
      }
    }
    
    const wrap = $("xray-regions");
    wrap.innerHTML = "";
    
    if (currentSaliencyTab === "importance") {
      showFeatureImportanceOverlay(b);
    } else if (currentSaliencyTab === "counterfactual") {
      showCounterfactualOverlay(b);
    } else if (drawSaliencyGeometry(b, wrap)) {
      // Real Grad-CAM++ region outlines were drawn — nothing further to add.
    } else {
      // Fallback: static per-finding anatomical boxes. Used when the case
      // carries no heatmap geometry (older cases, or geometry extraction
      // skipped). These are anatomy priors, not measured localisation.
      const found = ((b.vision && b.vision.findings) || []).filter(
        (f) => (f.present !== undefined ? f.present : f.probability >= 0.5));
      found.forEach((f, i) => {
        const [r0, c0, r1, c1] = f.region;
        const d = document.createElement("div");
        d.className = "region region-static";
        d.style.cssText = `top:${r0 * 100}%;left:${c0 * 100}%;height:${(r1 - r0) * 100}%;width:${(c1 - c0) * 100}%;animation-delay:${0.25 + i * 0.18}s`;
        d.innerHTML = `<span class="r-lbl">${(EV_LABEL[f.finding] || f.finding)} · ${f.probability.toFixed(2)}</span>`;
        wrap.appendChild(d);
      });
    }
    
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
    if (b.fusion && (qEnt || b.fusion.qae_applied)) {
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
      // qae_applied, not the qae_enabled setting: the row must mean "a quantum
      // autoencoder produced this evidence vector", which additionally requires a
      // vision embedding and loaded QAE weights. The QBN row was removed with the
      // flag behind it — QuantumBayesianNetwork is not on the serving path, so the
      // row could only ever have announced a component that did not run.
      const qaeRow = $("qae-telemetry-row");
      if (qaeRow) qaeRow.style.display = b.fusion.qae_applied ? "flex" : "none";
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

  /* ================= measurement budget (quantum only) =================
     The one panel in this console showing something a classical model cannot
     produce. Quantum precision is bought with shots — Var[<Z>] = (1-<Z>²)/n — so
     an unresolved case splits into two states with opposite instructions:
       measurement-limited : the answer exists, buy more shots (we say how many)
       model-limited       : tied at infinite precision, escalate to a human
     A classical softmax of 0.55 means "unsure" and cannot tell you which.        */
  function drawMeasurement(b) {
    const panel = $("panel-measure");
    if (!panel) return;
    const m = b.measurement;
    // Classical backend (or a skipped budget) -> hide the panel entirely rather
    // than render zeros that look like a measured result.
    if (!m) { panel.hidden = true; return; }
    panel.hidden = false;

    const FIXED = 512;                       // the non-adaptive budget it replaces
    const spent = m.shots_spent || 0;
    const saving = spent > 0 ? FIXED / spent : 0;
    const committed = !!m.committed;
    const limited = m.limiting_factor || null;

    const verdict = committed
      ? { cls: "commit", tag: "committed",
          note: `resolved at ${spent.toLocaleString()} shots` }
      : limited === "measurement"
        ? { cls: "measure", tag: "measurement-limited",
            note: m.predicted_shots
              ? `~${Number(m.predicted_shots).toLocaleString()} shots would resolve this`
              : "more measurement would resolve this" }
        : { cls: "model", tag: "model-limited",
            note: m.floor_limited
              ? "lead is below the clinical-significance floor — escalate"
              : "tied at infinite precision — escalate" };

    $("mb-hint").textContent = committed
      ? `${saving >= 1 ? saving.toFixed(1) + "× under" : (1 / saving).toFixed(1) + "× over"} a fixed ${FIXED}-shot budget`
      : "no achievable budget settles this case";

    const top = DX_LABEL[m.top] || m.top;
    const run = DX_LABEL[m.runner_up] || m.runner_up;

    // Margin bar: the observed ±1σ shot-noise band against the analytic margin the
    // serving path uses. Seeing the band straddle zero is the whole point.
    const scale = Math.max(0.35, Math.abs(m.analytic_margin) * 1.6, Math.abs(m.margin) * 1.6);
    const pct = (v) => clamp((v / scale) * 100, 0, 100);

    $("mb-body").innerHTML = `
      <div class="mb-verdict ${verdict.cls}">
        <span class="mb-tag">${verdict.tag}</span>
        <span class="mb-note">${verdict.note}</span>
      </div>
      <div class="mb-pair mono">
        <span class="mb-top">${top}</span>
        <span class="mb-vs">vs</span>
        <span class="mb-run">${run}</span>
      </div>
      <div class="mb-track" title="decision margin ±1σ shot noise">
        <div class="mb-band"></div>
        <div class="mb-analytic" title="margin at infinite shots"></div>
      </div>
      <dl class="mb-stats mono">
        <div><dt>shots spent</dt><dd>${spent.toLocaleString()}</dd></div>
        <div><dt>margin</dt><dd>${m.margin.toFixed(3)} ± ${m.margin_std.toFixed(3)}</dd></div>
        <div><dt>separation</dt><dd>${m.separation_z.toFixed(2)} σ</dd></div>
        <div><dt>at ∞ shots</dt><dd>${m.analytic_margin.toFixed(3)}</dd></div>
      </dl>
      <p class="mb-reason">${m.reason || ""}</p>`;

    const band = $("mb-body").querySelector(".mb-band");
    const mark = $("mb-body").querySelector(".mb-analytic");
    const lo = pct(Math.max(0, m.margin - m.margin_std));
    const hi = pct(m.margin + m.margin_std);
    mark.style.left = pct(Math.abs(m.analytic_margin)) + "%";
    setTimeout(() => { band.style.left = lo + "%"; band.style.width = Math.max(1, hi - lo) + "%"; }, 120);
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

    // The bundle carries the profile under both names; prefer the canonical one.
    const drp = b.decision_readiness || b.drp;
    if (!drp) {
      sidebar.style.display = "none";
      return;
    }

    sidebar.style.display = "block";

    // Update status chip
    // DecisionReadinessProfile.state is a lowercase enum ("ready" / "not_ready" /
    // "conditional"), not a `status` field — reading drp.status made this chip
    // report NOT READY for every case, including ready ones.
    const statusChip = $("drp-status-chip");
    const ready = String(drp.state || "").toLowerCase() === "ready";
    statusChip.innerHTML = ready
      ? `<span class="flag ok" style="padding: 2px 6px; font-size: 10px;">DECISION READINESS: READY</span>`
      : `<span class="flag abst" style="padding: 2px 6px; font-size: 10px; background: #ff5d5d;">DECISION READINESS: NOT READY</span>`;

    // Render dimensions. The contract names these s_* (see
    // schemas/contracts.py::DecisionReadinessProfile); reading drp.coverage etc.
    // yielded undefined, so every bar rendered "NaN%" with width:NaN% — and since
    // NaN < 0.6 is false, all six kept the healthy cyan colour while showing NaN.
    // There is no `stability` dimension in the profile, so it is not listed: an
    // invented row would render 0% and read as a failing score that does not exist.
    const dims = [
      ["Coverage", drp.s_coverage],
      ["Quality", drp.s_quality],
      ["Consistency", drp.s_consistency],
      ["Robustness", drp.s_robustness],
      ["Consensus", drp.s_consensus]
    ];

    let barsHtml = "";
    dims.forEach(([name, raw]) => {
      // A dimension the backend did not send is shown as unavailable, never as 0%.
      if (raw === undefined || raw === null || !isFinite(raw)) {
        barsHtml += `
        <div style="display: flex; justify-content: space-between; font-size: 10.5px;">
          <span style="color: var(--faint);">${name}</span>
          <span style="color: var(--faint);">n/a</span>
        </div>`;
        return;
      }
      const val = clamp(Number(raw), 0, 1);
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

        // Populate the preview modal elements. A field the server could not read
        // off the file comes back null and must render as an explicit "not in file"
        // — never as a plausible-looking value.
        const NOT_IN_FILE = "— not present in study";
        $("preview-patient-id").textContent = previewData.patient_id || NOT_IN_FILE;
        $("preview-study-id").textContent = previewData.study_id || "n/a";
        $("preview-seq-type").textContent = previewData.sequence_type || "n/a";
        $("preview-dims").textContent = previewData.original_dimensions ? previewData.original_dimensions.join(" x ") : "n/a";
        $("preview-spacing").textContent = previewData.voxel_spacing ? previewData.voxel_spacing.map(v => Number(v).toFixed(3)).join(" x ") + " mm" : "n/a";
        $("preview-orientation").textContent = previewData.orientation || NOT_IN_FILE;
        $("preview-slices").textContent = previewData.number_of_slices || "n/a";
        $("preview-affine").textContent = previewData.affine_matrix_summary || "n/a";

        const sm = previewData.scanner_metadata;
        $("preview-scanner").innerText = sm
          ? Object.entries(sm).map(([k, v]) => `${k}: ${v}`).join("\n")
          : NOT_IN_FILE;

        // Real, measured: which sequences the intake layer actually identified.
        const mods = previewData.detected_modalities || [];
        $("preview-modalities").textContent = mods.length ? mods.join(" · ") : NOT_IN_FILE;
        $("preview-order-source").textContent = previewData.channel_order_source || NOT_IN_FILE;

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
    
    drawInferenceTimeline(0);
    let alive = true;
    (async () => {
      let idx = 0;
      for (const line of stages) {
        if (!alive) return;
        txt.innerHTML += `<span class="ok">▸</span> ${line}\n`;
        drawInferenceTimeline(Math.min(8, Math.round(idx * (8 / (stages.length - 1)))));
        idx++;
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
      // Calibration ECE is read from the benchmark artifact for the backend that is
      // actually serving (health.backend), not fixed to the quantum column — the two
      // are calibrated separately and differ. No artifact -> "—", never a stand-in
      // number: an invented ECE on a calibration readout is the one value in this
      // console that must never be guessed.
      if ($("tel-calib-status")) {
        const el = $("tel-calib-status");
        const bench = stats.benchmark || {};
        const served = (S.health && S.health.backend) || null;
        const row = (served && bench[served]) || null;
        if (row && typeof row.ece === "number") {
          el.textContent = row.ece.toFixed(4);
          el.title = `ECE for the serving ${served} backend, from artifacts/benchmark.json (n=${bench.n_eval ?? "?"})`;
          el.style.color = "#4be1c3";
        } else {
          el.textContent = "—";
          el.title = "no benchmark artifact on this deployment — run `aura_cli evaluate`";
          el.style.color = "var(--faint)";
        }
      }
      // Audit health is a real check: the store must be returning audit rows.
      if ($("tel-audit-health")) {
        const el = $("tel-audit-health");
        const rows = Array.isArray(stats.recent_audit) ? stats.recent_audit.length : 0;
        el.textContent = rows > 0 ? "Healthy" : "No audit rows";
        el.style.color = rows > 0 ? "#4be1c3" : "#ffd166";
      }
    } catch (err) {
      console.warn("Failed to load telemetry:", err);
    }
  }

  /* ================= explainability tabs (Task 5) ================= */
  function setupExplainTabs() {
    const tabs = ["heatmap", "importance", "counterfactual", "occlusion", "integrated"];
    tabs.forEach((t) => {
      const el = $(`tab-${t}`);
      if (el) {
        // Remove old listener
        const newEl = el.cloneNode(true);
        el.parentNode.replaceChild(newEl, el);
        
        newEl.addEventListener("click", () => {
          tabs.forEach((x) => $(`tab-${x}`).classList.remove("active-tab"));
          newEl.classList.add("active-tab");
          currentSaliencyTab = t;
          const case_id = S.current;
          if (case_id) {
            const b = S.bundles.get(case_id);
            if (b) drawXray(b);
          }
        });
      }
    });
  }

  function showFeatureImportanceOverlay(b) {
    const wrap = $("xray-regions");
    wrap.innerHTML = "";
    const overlay = document.createElement("div");
    overlay.className = "explain-hud mono";
    overlay.style.cssText = "position:absolute; inset: 12px; background:rgba(10,10,15,0.85); border: 1px solid rgba(75, 225, 195, 0.2); border-radius: 8px; padding: 12px; font-size:11px; overflow-y:auto; color:#fff; z-index:5;";
    
    let html = `<h5 style="color:#4be1c3; margin:0 0 10px 0; text-transform:uppercase;">Shapley Feature Importance</h5>`;
    const attr = (b.explanation && b.explanation.evidence_attribution) || {};
    const sorted = Object.entries(attr).sort((a,b) => Math.abs(b[1]) - Math.abs(a[1]));
    
    if (sorted.length === 0) {
      html += `<div style="color:var(--dim)">No attributions available.</div>`;
    } else {
      sorted.forEach(([k, v]) => {
        const pct = Math.min(100, Math.round(Math.abs(v) * 100));
        const color = v >= 0 ? "#4be1c3" : "#ff5d5d";
        const sign = v >= 0 ? "+" : "";
        html += `
          <div style="margin-bottom:8px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:2px;">
              <span>${EV_LABEL[k] || k}</span>
              <span style="color:${color}; font-weight:bold;">${sign}${v.toFixed(4)}</span>
            </div>
            <div style="height:4px; background:rgba(255,255,255,0.05); border-radius:2px;">
              <div style="width:${pct}%; height:100%; background:${color}; border-radius:2px;"></div>
            </div>
          </div>
        `;
      });
    }
    overlay.innerHTML = html;
    wrap.appendChild(overlay);
  }

  function showCounterfactualOverlay(b) {
    const wrap = $("xray-regions");
    wrap.innerHTML = "";
    const overlay = document.createElement("div");
    overlay.className = "explain-hud mono";
    overlay.style.cssText = "position:absolute; inset: 12px; background:rgba(10,10,15,0.85); border: 1px solid rgba(139, 124, 247, 0.2); border-radius: 8px; padding: 12px; font-size:11px; overflow-y:auto; color:#fff; z-index:5;";
    
    let html = `<h5 style="color:#8b7cf7; margin:0 0 10px 0; text-transform:uppercase;">Counterfactual Scenarios</h5>`;
    html += `<p style="color:var(--dim); margin-bottom:12px; font-size:10px;">Change in top diagnosis probability if evidence is removed:</p>`;
    const cf = (b.explanation && b.explanation.counterfactuals) || {};
    const sorted = Object.entries(cf).sort((a,b) => Math.abs(b[1]) - Math.abs(a[1]));
    
    if (sorted.length === 0) {
      html += `<div style="color:var(--dim)">No counterfactuals available.</div>`;
    } else {
      sorted.forEach(([k, v]) => {
        const pct = Math.min(100, Math.round(Math.abs(v) * 100));
        const color = v < 0 ? "#ff5d5d" : "#4be1c3";
        const sign = v >= 0 ? "+" : "";
        html += `
          <div style="margin-bottom:8px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:2px;">
              <span>Remove "${EV_LABEL[k] || k}"</span>
              <span style="color:${color}; font-weight:bold;">${sign}${v.toFixed(4)}</span>
            </div>
            <div style="height:4px; background:rgba(255,255,255,0.05); border-radius:2px;">
              <div style="width:${pct}%; height:100%; background:${color}; border-radius:2px;"></div>
            </div>
          </div>
        `;
      });
    }
    overlay.innerHTML = html;
    wrap.appendChild(overlay);
  }

  /* ================= priors sidebar (Task 2) ================= */
  function updatePriorsSidebar(b) {
    const sidebar = $("priors-sidebar-panel");
    if (!b) {
      sidebar.style.display = "none";
      return;
    }
    
    // Only show priors for X-rays (since MRI has no classical reasoning backend rules)
    const is_brain = b.study_id.startsWith("STU-MR") || (b.fusion && b.fusion.backend === "brain-vision-presence-head");
    if (is_brain) {
      // For brain cases, let's keep the panel visible but only enable Age band or relevant fields
      sidebar.style.display = "block";
      // disable most checkboxes except fever/smoking/prior-cancer which act as clinical contexts
      document.querySelectorAll(".priors-checkboxes input").forEach(cb => cb.disabled = false);
    } else {
      sidebar.style.display = "block";
      document.querySelectorAll(".priors-checkboxes input").forEach(cb => cb.disabled = false);
    }
    
    const p = b.priors || {};
    const m = b.multimodal || { symptoms: {}, history: {}, labs: {} };
    
    $("prior-fever").checked = !!p.fever;
    $("prior-cough").checked = !!(m.symptoms && m.symptoms.productive_cough);
    $("prior-bnp").checked = !!(m.labs && m.labs.bnp && m.labs.bnp >= 400);
    $("prior-cardiopathy").checked = !!(m.history && m.history.heart_failure);
    $("prior-diabetes").checked = !!(m.history && m.history.diabetes);
    $("prior-hypertension").checked = !!(m.history && m.history.hypertension);
    $("prior-smoking").checked = !!p.smoker;
    $("prior-crp").checked = !!(m.labs && m.labs.crp && m.labs.crp > 50);
  }

  /* ================= confidence evolution flow (Task 3) ================= */
  function drawEvolution(b) {
    const top_dx = b.safety && b.safety.top;
    
    const vis_max = b.vision && b.vision.findings && b.vision.findings.length ? 
      Math.max(...b.vision.findings.map(f => f.probability)) : 0.0;
      
    const fus_val = b.fusion && b.fusion.posterior && top_dx && b.fusion.posterior[top_dx] !== undefined ? 
      b.fusion.posterior[top_dx] : 0.0;
      
    const rea_val = b.reasoning && b.reasoning.adjusted_posterior && top_dx && b.reasoning.adjusted_posterior[top_dx] !== undefined ? 
      b.reasoning.adjusted_posterior[top_dx] : fus_val;
      
    const cal_pred = b.safety && b.safety.predictions ? 
      b.safety.predictions.find(p => p.diagnosis === top_dx) : null;
    const cal_val = cal_pred ? cal_pred.probability : rea_val;
    
    const final_val = b.safety ? b.safety.top_probability : cal_val;
    
    $("val-vision").textContent = vis_max ? `${Math.round(vis_max * 100)}%` : "—";
    $("val-fusion").textContent = fus_val ? `${Math.round(fus_val * 100)}%` : "—";
    $("val-reasoning").textContent = rea_val ? `${Math.round(rea_val * 100)}%` : "—";
    $("val-calibration").textContent = cal_val ? `${Math.round(cal_val * 100)}%` : "—";
    $("val-final").textContent = final_val ? `${Math.round(final_val * 100)}%` : "—";
    
    const setupNodeHover = (nodeId, title, content) => {
      const el = $(nodeId);
      if (!el) return;
      
      const newEl = el.cloneNode(true);
      el.parentNode.replaceChild(newEl, el);
      
      newEl.addEventListener("mouseenter", () => {
        const tip = $("evo-tooltip-area");
        tip.innerHTML = `<strong style="color:#8b7cf7">${title}</strong><br><div style="margin-top:6px; font-family:var(--sans);">${content}</div>`;
        tip.style.display = "block";
      });
      newEl.addEventListener("mouseleave", () => {
        $("evo-tooltip-area").style.display = "none";
      });
    };
    
    const vis_content = b.vision && b.vision.findings && b.vision.findings.length ? 
      b.vision.findings.map(f => `• ${(EV_LABEL[f.finding] || f.finding)}: <b>${Math.round(f.probability*100)}%</b>`).join("<br>") : 
      "No vision findings detected.";
      
    const fus_content = b.fusion && b.fusion.posterior ? 
      Object.entries(b.fusion.posterior).map(([k, v]) => `• ${(DX_LABEL[k] || k)}: <b>${Math.round(v*100)}%</b>`).join("<br>") : 
      "No multimodal fusion output.";
      
    const rea_content = b.reasoning && b.reasoning.steps && b.reasoning.steps.length ? 
      b.reasoning.steps.map(s => `• Step ${s.step}: ${s.rationale}`).join("<br>") : 
      "No clinical reasoning adjustments applied (behaved as prior).";
      
    const cal_content = b.safety ? 
      `Conformal Set: [<b>${b.safety.conformal_set.map(d => DX_LABEL[d] || d).join(", ")}</b>]<br>` +
      `Calibration Method: <b>${b.safety.conformal_method}</b><br>` +
      `Aleatoric Uncertainty: <b>${b.safety.aleatoric_uncertainty ? b.safety.aleatoric_uncertainty.toFixed(4) : "N/A"}</b>` : 
      "No safety calibration assessments.";
      
    const final_content = b.safety ? 
      `Top Diagnosis: <b>${DX_LABEL[b.safety.top] || b.safety.top}</b><br>` +
      `Final Calibrated Probability: <b>${Math.round(b.safety.top_probability*100)}%</b><br>` +
      `Abstained: <b>${b.safety.abstained ? "YES (" + b.safety.abstention_reason + ")" : "NO"}</b>` : 
      "No final diagnosis signed.";
      
    setupNodeHover("node-vision", "Vision Stage Findings", vis_content);
    setupNodeHover("node-fusion", "Multimodal Fusion Posterior", fus_content);
    setupNodeHover("node-reasoning", "Clinical Reasoning Adjustments", rea_content);
    setupNodeHover("node-calibration", "Safety Conformal Calibration", cal_content);
    setupNodeHover("node-final", "Final Signed Diagnosis", final_content);
  }

  /* ================= longitudinal tracking (Task 1) ================= */
  async function drawLongitudinal(b) {
    const is_brain = b.study_id.startsWith("STU-MR") || (b.fusion && b.fusion.backend === "brain-vision-presence-head");
    const panel = $("panel-longitudinal");
    
    if (!is_brain) {
      panel.style.display = "none";
      return;
    }
    
    panel.style.display = "block";
    
    // Clear previous details
    $("long-studies-list").innerHTML = `<span style="color:var(--faint); font-size:11px;">Loading patient scans...</span>`;
    $("long-timeline-flow").innerHTML = "";
    $("long-comparison-metrics").innerHTML = "";
    $("btn-compare-studies").style.display = "none";
    $("long-changed-regions").innerHTML = "No comparison run yet. Select a prior scan and click \"Compare Study\".";
    $("long-comparison-report").style.display = "none";
    
    try {
      const data = await api(`/v1/cases/${b.case_id}/tracking`);
      const timeline = data.timeline || [];
      
      if (timeline.length <= 1) {
        $("long-studies-list").innerHTML = `<span style="color:var(--faint); font-size:11px;">No prior scans found.</span>`;
        return;
      }
      
      // Populate previous scans sidebar
      let listHtml = "";
      timeline.forEach((scan) => {
        if (scan.case_id === b.case_id) return;
        const volText = scan.wt_volume_mm3 ? `${(scan.wt_volume_mm3 / 1000).toFixed(1)} cc` : "N/A";
        listHtml += `
          <button class="tg slim long-scan-btn" data-prev-id="${scan.case_id}" style="width:100%; text-align:left; justify-content:flex-start; margin-bottom:4px; font-size:11px;">
            📅 ${scan.date}<br>
            <span style="color:var(--faint); font-size:10px;">${scan.case_id} · WT: ${volText}</span>
          </button>
        `;
      });
      $("long-studies-list").innerHTML = listHtml;
      
      let selectedPrevId = null;
      document.querySelectorAll(".long-scan-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
          document.querySelectorAll(".long-scan-btn").forEach(x => x.classList.remove("on"));
          btn.classList.add("on");
          selectedPrevId = btn.dataset.prevId;
          $("btn-compare-studies").style.display = "block";
        });
      });
      
      // Populate timeline flow
      let timelineHtml = "";
      timeline.forEach((scan) => {
        const isCurrent = scan.case_id === b.case_id;
        const color = isCurrent ? "#8b7cf7" : "#4be1c3";
        timelineHtml += `
          <div style="display:flex; flex-direction:column; gap:4px; min-width:80px; align-items:center; opacity:${isCurrent ? 1.0 : 0.7}">
            <div style="font-weight:bold; color:${color};">${scan.date.slice(5)}</div>
            <div style="width:10px; height:10px; border-radius:50%; background:${color}; box-shadow:0 0 6px ${color};"></div>
            <div style="font-size:10px; color:var(--faint);">${scan.case_id.slice(-6)}</div>
          </div>
        `;
      });
      $("long-timeline-flow").innerHTML = timelineHtml;
      
      plotVolumeTrend(timeline, b.case_id);
      
      const compareBtn = $("btn-compare-studies");
      compareBtn.onclick = null;
      compareBtn.onclick = async () => {
        if (!selectedPrevId) return;
        
        compareBtn.disabled = true;
        compareBtn.querySelector("span").textContent = "Comparing...";
        
        try {
          const comp = await api(`/v1/cases/progression?previous_case_id=${selectedPrevId}&current_case_id=${b.case_id}`, {
            method: "POST"
          });
          
          $("long-comparison-report").innerHTML = parseMarkdown(comp.report_md || "");
          $("long-comparison-report").style.display = "block";
          
          let metricsHtml = "";
          let changedHtml = `<h6 style="color:#ffd166; margin-bottom:6px; font-size:10.5px;">CHANGED REGIONS:</h6>`;
          let anyChange = false;
          
          if (comp.volume_changes) {
            Object.entries(comp.volume_changes).forEach(([region, c]) => {
              const diffPct = c.percent_change;
              const sign = diffPct >= 0 ? "+" : "";
              const color = diffPct > 10 ? "#ff5d5d" : (diffPct < -10 ? "#4be1c3" : "#ffd166");
              const label = region.replace("_", " ").toUpperCase();
              
              metricsHtml += `
                <div style="font-size:11px; margin-bottom:8px;">
                  <div style="display:flex; justify-content:space-between; margin-bottom:2px;">
                    <span>${label}</span>
                    <span style="color:${color}; font-weight:bold;">${sign}${diffPct.toFixed(1)}%</span>
                  </div>
                  <div style="height:3px; background:rgba(255,255,255,0.05); border-radius:1.5px;">
                    <div style="width:${Math.min(100, Math.abs(diffPct))}%; height:100%; background:${color}; border-radius:1.5px;"></div>
                  </div>
                </div>
              `;
              
              if (Math.abs(diffPct) >= 5.0) {
                anyChange = true;
                const changeType = diffPct > 0 ? "Growth" : "Regression";
                changedHtml += `
                  <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                    <span style="color:#fff;">${label}</span>
                    <span style="color:${color}">${changeType} (${sign}${diffPct.toFixed(1)}%)</span>
                  </div>
                `;
              }
            });
          }
          
          if (!anyChange) {
            changedHtml += `<div style="color:var(--dim)">No significant changes detected (stable &lt; 5%).</div>`;
          }
          
          $("long-comparison-metrics").innerHTML = metricsHtml;
          $("long-changed-regions").innerHTML = changedHtml;
          
          toast("Volumetric progression compared");
        } catch (e) {
          console.error("Progression comparison failed", e);
          toast("Failed to run study comparison");
        } finally {
          compareBtn.disabled = false;
          compareBtn.querySelector("span").textContent = "Compare Study";
        }
      };
      
    } catch (e) {
      console.error("Longitudinal tracking failed", e);
      $("long-studies-list").innerHTML = `<span style="color:var(--red); font-size:11px;">Failed to load history.</span>`;
    }
  }

  function plotVolumeTrend(timeline, currentCaseId) {
    const canvas = $("long-trend-canvas");
    if (!canvas) return;
    
    const ctx = canvas.getContext("2d");
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const rect = canvas.getBoundingClientRect();
    
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    
    ctx.fillStyle = "rgba(10, 10, 15, 0.3)";
    ctx.fillRect(0, 0, rect.width, rect.height);
    
    ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
    ctx.lineWidth = 1;
    for (let y = 20; y < rect.height; y += 40) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(rect.width, y);
      ctx.stroke();
    }
    
    const volumes = timeline.map(s => s.wt_volume_mm3 || 0);
    const maxVol = Math.max(...volumes, 1000);
    
    const n = timeline.length;
    const paddingX = 40;
    const paddingY = 20;
    const graphWidth = rect.width - paddingX * 2;
    const graphHeight = rect.height - paddingY * 2;
    
    const points = timeline.map((scan, idx) => {
      const x = paddingX + (idx / Math.max(1, n - 1)) * graphWidth;
      const y = rect.height - paddingY - ((scan.wt_volume_mm3 || 0) / maxVol) * graphHeight;
      return { x, y, case_id: scan.case_id, wt: scan.wt_volume_mm3 };
    });
    
    ctx.strokeStyle = "#8b7cf7";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    points.forEach((p, idx) => {
      if (idx === 0) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    });
    ctx.stroke();
    
    points.forEach((p) => {
      const isCurrent = p.case_id === currentCaseId;
      ctx.fillStyle = isCurrent ? "#8b7cf7" : "#4be1c3";
      ctx.beginPath();
      ctx.arc(p.x, p.y, isCurrent ? 6 : 4, 0, Math.PI * 2);
      ctx.fill();
      
      if (isCurrent) {
        ctx.strokeStyle = "rgba(139, 124, 247, 0.4)";
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 9, 0, Math.PI * 2);
        ctx.stroke();
      }
      
      ctx.fillStyle = "rgba(255, 255, 255, 0.7)";
      ctx.font = "9px monospace";
      ctx.fillText(`${(p.wt / 1000).toFixed(1)} cc`, p.x - 15, p.y - 10);
    });
  }

  /* ================= multi-agent specialist discussion (Task 7) ================= */
  async function drawDiscussion(b) {
    const chat = $("discussion-chat");
    const cons = $("discussion-consensus");
    
    chat.innerHTML = `<span style="color:var(--faint); font-size:11px;">Generating clinical specialist debate...</span>`;
    cons.innerHTML = "";
    
    try {
      const data = await api(`/v1/cases/${b.case_id}/discussion`);
      
      let chatHtml = "";
      data.opinions.forEach((o) => {
        const colors = {
          "Radiologist": { bg: "rgba(75, 225, 195, 0.08)", border: "rgba(75, 225, 195, 0.25)", color: "#4be1c3" },
          "Pulmonologist": { bg: "rgba(247, 124, 163, 0.08)", border: "rgba(247, 124, 163, 0.25)", color: "#f77ca3" },
          "Cardiologist": { bg: "rgba(253, 209, 102, 0.08)", border: "rgba(253, 209, 102, 0.25)", color: "#ffd166" },
          "Neurologist": { bg: "rgba(93, 193, 247, 0.08)", border: "rgba(93, 193, 247, 0.25)", color: "#5dc1f7" },
          "Oncologist": { bg: "rgba(139, 124, 247, 0.08)", border: "rgba(139, 124, 247, 0.25)", color: "#8b7cf7" }
        }[o.specialist] || { bg: "rgba(255,255,255,0.03)", border: "rgba(255,255,255,0.1)", color: "#fff" };
        
        chatHtml += `
          <div style="background:${colors.bg}; border:1px solid ${colors.border}; padding:10px 14px; border-radius:10px; font-family:var(--sans); font-size:12px; line-height:1.45;">
            <div style="display:flex; justify-content:space-between; margin-bottom:4px; font-family:var(--mono); font-size:11px; font-weight:bold; color:${colors.color};">
              <span>👨‍⚕️ ${o.specialist.toUpperCase()}</span>
              <span title="${o.confidence_basis === "rule_weight"
                ? "Strength of the rule that fired — not a calibrated model probability."
                : "Read from the model's own output for this case."}">${
                o.confidence_basis === "rule_weight" ? "Rule strength" : "Model confidence"
              }: ${Math.round(o.confidence * 100)}%</span>
            </div>
            <div style="color:#e2e8f0;">${o.opinion}</div>
            ${o.supporting_evidence.length ? `
              <div style="margin-top:6px; font-family:var(--mono); font-size:10px; color:var(--faint);">
                Supporting Evidence: ${o.supporting_evidence.join(", ")}
              </div>
            ` : ""}
          </div>
        `;
      });
      chat.innerHTML = chatHtml;
      
      cons.innerHTML = `
        <div style="font-weight:bold; color:#8b7cf7; margin-bottom:4px; text-transform:uppercase; letter-spacing:0.05em;">
          🤝 Consensus Recommendation
        </div>
        <div style="color:#e2e8f0; font-family:var(--sans); font-size:12px; line-height:1.4;">
          ${data.consensus.opinion}
        </div>
        <div style="margin-top:6px; font-size:10.5px; color:var(--faint);">
          ${data.consensus.confidence_basis === "model" ? "Consensus Confidence" : "Consensus Rule Strength"}:
          <span style="color:#4be1c3; font-weight:bold;">${Math.round(data.consensus.confidence * 100)}%</span>
          <span style="opacity:0.7;">· mean of contributing specialists</span>
        </div>
      `;
      
    } catch (e) {
      console.error("Discussion generation failed", e);
      chat.innerHTML = `<span style="color:var(--red); font-size:11px;">Failed to generate discussion.</span>`;
    }
  }

  /* ================= Active Diagnosis Agent (sequential EIG testing) ================= */
  const dxLabel = (v) => DX_LABEL[v] || String(v || "").replace(/_/g, " ");

  async function runAgent(file) {
    const panel = $("panel-agent");
    const summary = $("agent-summary");
    const stepsWrap = $("agent-steps");

    // reveal the panel and scroll it into view
    panel.style.display = "";
    stepsWrap.innerHTML = "";
    summary.innerHTML = `<span style="color:var(--faint); font-size:12px;">
      <i class="dot" style="display:inline-block; width:6px; height:6px; background:#ffd166; border-radius:50%; margin-right:6px;"></i>
      running sequential agent on <b style="color:#ffd166;">${file.name}</b> …</span>`;
    panel.scrollIntoView({ behavior: REDUCED ? "auto" : "smooth", block: "center" });

    try {
      const fd = new FormData();
      fd.append("file", file);
      const traj = await api(`/v1/studies/agent`, {
        method: "POST",
        headers: { "x-aura-user": "clinician" },
        body: fd,
      });
      renderAgent(traj, file.name);
    } catch (err) {
      const d = (err && err.detail) || {};
      const why = d.reason || d.message || err.message || "agent run failed";
      const rejected = err && err.status === 422;
      summary.innerHTML = `<div style="background:var(--red-dim); border:1px solid rgba(255,93,93,0.35); border-radius:8px; padding:12px; color:#ff8a8a; font-size:12px;">
        <b>${rejected ? "✕ REJECTED — not a chest X-ray" : "✕ agent run failed"}</b><br>
        <span style="color:var(--dim);">${why}</span></div>`;
      toast(rejected ? "agent: not a chest X-ray" : "agent run failed");
    } finally {
      $("input-file-agent").value = "";
    }
  }

  function renderAgent(t, fileName) {
    const summary = $("agent-summary");
    const stepsWrap = $("agent-steps");

    const committed = !!t.committed;
    const badge = committed
      ? `<span style="background:var(--cyan-dim); border:1px solid rgba(75,225,195,0.4); color:#4be1c3; padding:3px 10px; border-radius:999px; font-weight:bold;">COMMITTED</span>`
      : `<span style="background:rgba(255,209,102,0.12); border:1px solid rgba(255,209,102,0.4); color:#ffd166; padding:3px 10px; border-radius:999px; font-weight:bold;">ABSTAINED</span>`;

    const e0 = Number(t.initial_entropy) || 0;
    const eN = Number(t.final_entropy) || 0;
    const remaining = e0 > 1e-9 ? Math.max(0, Math.min(1, eN / e0)) : 0;

    const cell = (label, val, color) => `
      <div style="flex:1 1 120px; min-width:120px; background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06); border-radius:8px; padding:10px 12px;">
        <div style="color:var(--faint); font-size:10px; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:4px;">${label}</div>
        <div style="color:${color || "#fff"}; font-weight:bold; font-size:15px;">${val}</div>
      </div>`;

    summary.innerHTML = `
      <div style="display:flex; align-items:center; gap:12px; flex-wrap:wrap; margin-bottom:12px; font-size:12px;">
        ${badge}
        <span style="color:var(--dim);">on <b style="color:#fff;">${fileName || "study"}</b></span>
        <span style="color:var(--faint);">status: <b style="color:var(--dim);">${t.status || "—"}</b></span>
        <span style="color:var(--faint); margin-left:auto;">fusion backend: <b style="color:#8b7cf7;">${t.backend || "—"}</b></span>
      </div>
      <div style="display:flex; gap:10px; flex-wrap:wrap; margin-bottom:12px;">
        ${cell("Final diagnosis", committed ? dxLabel(t.final_diagnosis) : "— (escalated)", committed ? "#4be1c3" : "#ffd166")}
        ${cell("Final confidence", (Number(t.final_probability) * 100).toFixed(1) + "%", "#fff")}
        ${cell("Tests ordered", t.n_tests, "#8b7cf7")}
        ${cell("Bits resolved", Number(t.bits_resolved).toFixed(2) + " bits", "#4be1c3")}
      </div>
      <div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06); border-radius:8px; padding:10px 12px;">
        <div style="display:flex; justify-content:space-between; color:var(--faint); font-size:10.5px; margin-bottom:6px;">
          <span>ENTROPY  ${e0.toFixed(2)} bits <span style="color:#4be1c3;">&rarr;</span> ${eN.toFixed(2)} bits</span>
          <span>uncertainty resolved: <b style="color:#4be1c3;">${(100 * (1 - remaining)).toFixed(0)}%</b></span>
        </div>
        <div style="height:8px; border-radius:999px; background:rgba(255,255,255,0.06); overflow:hidden;">
          <div style="height:100%; width:${(100 * (1 - remaining)).toFixed(1)}%; background:linear-gradient(90deg,#8b7cf7,#4be1c3); border-radius:999px;"></div>
        </div>
      </div>`;

    const steps = Array.isArray(t.steps) ? t.steps : [];
    stepsWrap.innerHTML = steps.map((s) => {
      // Intermediate steps carry no `decision` — the agent is still gathering
      // evidence, so the action it chose *is* the decision worth showing.
      const dec = String(s.decision || "").toLowerCase();
      const label = s.decision || (s.action_display ? "order test" : "step");
      const accent = /commit/.test(dec) ? "#4be1c3"
        : /abstain|stop/.test(dec) ? "#ff8a8a"
        : "#8b7cf7"; // order test / continue
      const top = (Array.isArray(s.top) ? s.top : []).slice(0, 3);
      const bars = top.map(([dx, p]) => {
        const pct = (Number(p) * 100);
        return `
          <div style="display:flex; align-items:center; gap:8px; margin-bottom:3px;">
            <span style="width:130px; color:var(--dim); font-size:11px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">${dxLabel(dx)}</span>
            <div style="flex:1; height:6px; background:rgba(255,255,255,0.06); border-radius:999px; overflow:hidden;">
              <div style="height:100%; width:${pct.toFixed(1)}%; background:${accent}; opacity:0.85;"></div>
            </div>
            <span style="width:44px; text-align:right; color:#fff; font-size:11px;">${pct.toFixed(1)}%</span>
          </div>`;
      }).join("");

      const eig = (s.action_eig_bits !== null && s.action_eig_bits !== undefined)
        ? ` · EIG <b style="color:#4be1c3;">${Number(s.action_eig_bits).toFixed(3)}</b> bits` : "";
      const action = s.action_display
        ? `<div style="color:var(--dim); font-size:11.5px; margin-top:6px;">action: <b style="color:${accent};">${s.action_display}</b>${eig}</div>` : "";
      const resolved = (Array.isArray(s.resolved) && s.resolved.length)
        ? `<div style="color:var(--faint); font-size:10.5px; margin-top:4px;">resolved: ${s.resolved.map(([c, v]) => `${c}=${Number(v).toFixed(2)}`).join(" · ")}</div>` : "";
      const rationale = s.rationale
        ? `<div style="color:var(--dim); font-size:11.5px; margin-top:8px; font-family:var(--sans); line-height:1.4;">${s.rationale}</div>` : "";

      return `
        <div style="border:1px solid rgba(255,255,255,0.06); border-left:3px solid ${accent}; border-radius:8px; padding:12px 14px; background:rgba(255,255,255,0.015);">
          <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px; font-family:var(--mono); font-size:11.5px;">
            <span style="width:22px; height:22px; border-radius:50%; border:2px solid ${accent}; color:${accent}; display:flex; align-items:center; justify-content:center; font-weight:bold; font-size:11px;">${s.step}</span>
            <span style="color:${accent}; font-weight:bold; text-transform:uppercase; letter-spacing:0.04em;">${label}</span>
            ${s.confident ? `<span style="color:#4be1c3; font-size:10px;">✓ confident</span>` : ""}
            <span style="margin-left:auto; color:var(--faint);">entropy <b style="color:#fff;">${Number(s.entropy_bits).toFixed(2)}</b> bits</span>
          </div>
          ${bars}
          ${action}
          ${resolved}
          ${rationale}
        </div>`;
    }).join("");
  }

  /* ================= live evidence timeline (Task 4) ================= */
  function drawInferenceTimeline(activeIndex) {
    const timeline = $("inference-timeline");
    if (!timeline) return;
    
    const stages = [
      { name: "Upload", val: 0 },
      { name: "Vision", val: 1 },
      { name: "Evidence", val: 2 },
      { name: "Fusion", val: 3 },
      { name: "Reasoning", val: 4 },
      { name: "Safety", val: 5 },
      { name: "Explanation", val: 6 },
      { name: "Recommendation", val: 7 },
      { name: "Report", val: 8 }
    ];
    
    let html = "";
    stages.forEach((s, idx) => {
      const isActive = idx === activeIndex;
      const isCompleted = idx < activeIndex;
      const color = isActive ? "#8b7cf7" : (isCompleted ? "#4be1c3" : "var(--faint)");
      const glow = isActive ? "box-shadow: 0 0 10px rgba(139, 124, 247, 0.4);" : "";
      
      html += `
        <div style="display:flex; flex-direction:column; align-items:center; gap:4px; opacity:${isActive ? 1.0 : (isCompleted ? 0.8 : 0.4)}">
          <div style="width:24px; height:24px; border-radius:50%; background:rgba(0,0,0,0.3); border:2px solid ${color}; display:flex; align-items:center; justify-content:center; color:${color}; font-weight:bold; font-size:10px; ${glow}">
            ${idx + 1}
          </div>
          <span style="color:${color}; font-size:9.5px; font-weight:${isActive ? 'bold' : 'normal'}">${s.name}</span>
        </div>
      `;
      if (idx < stages.length - 1) {
        const nextColor = (idx < activeIndex) ? "#4be1c3" : "rgba(255,255,255,0.05)";
        html += `<div style="width:20px; height:1px; background:${nextColor}; margin-bottom:12px; align-self:center;"></div>`;
      }
    });
    timeline.innerHTML = html;
  }

  /* ================= FHIR & HL7 exports (Task 6) ================= */
  async function exportFHIR() {
    const id = S.current;
    if (!id) { toast("no case loaded to export"); return; }
    
    try {
      const data = await api(`/v1/cases/${id}/export/fhir`);
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `FHIR_DiagnosticReport_${id}.json`;
      a.click();
      URL.revokeObjectURL(url);
      toast("FHIR DiagnosticReport exported");
    } catch (e) {
      console.error(e);
      toast("FHIR export failed");
    }
  }
  
  async function exportHL7() {
    const id = S.current;
    if (!id) { toast("no case loaded to export"); return; }
    
    try {
      const resp = await fetch(`/v1/cases/${id}/export/hl7`);
      const hl7Text = await resp.text();
      const blob = new Blob([hl7Text], { type: "text/plain" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `HL7_ORU_R01_${id}.txt`;
      a.click();
      URL.revokeObjectURL(url);
      toast("HL7 ORU^R01 exported");
    } catch (e) {
      console.error(e);
      toast("HL7 export failed");
    }
  }

  /* ================= admin audit viewer (Task 6) ================= */
  async function showAuditLog() {
    const modal = $("audit-modal");
    const rowsWrap = $("audit-log-rows");
    rowsWrap.innerHTML = `<tr><td colspan="5" style="color:var(--faint); text-align:center; padding:20px;">Fetching system audit trail...</td></tr>`;
    modal.style.display = "flex";
    
    try {
      const data = await api("/v1/admin/safety");
      const logs = data.recent_audit || [];
      
      if (!logs.length) {
        rowsWrap.innerHTML = `<tr><td colspan="5" style="color:var(--faint); text-align:center; padding:20px;">No audit records found.</td></tr>`;
        return;
      }
      
      let rowsHtml = "";
      logs.forEach((log) => {
        const date = new Date(log.created_at).toISOString().replace("T", " ").slice(0, 19);
        rowsHtml += `
          <tr style="border-bottom:1px solid rgba(255,255,255,0.03);">
            <td style="padding:6px 4px; color:var(--dim); white-space:nowrap;">${date}</td>
            <td style="padding:6px 4px; color:#4be1c3;">${log.actor}</td>
            <td style="padding:6px 4px; color:#fff; font-weight:bold;">${log.action}</td>
            <td style="padding:6px 4px; color:var(--dim);">${log.entity_type} (${log.entity_id || "—"})</td>
            <td style="padding:6px 4px; color:var(--faint); font-size:10.5px; word-break:break-all;">${JSON.stringify(log.detail || {})}</td>
          </tr>
        `;
      });
      rowsWrap.innerHTML = rowsHtml;
      
    } catch (e) {
      console.error("Audit log fetch failed", e);
      rowsWrap.innerHTML = `<tr><td colspan="5" style="color:var(--red); text-align:center; padding:20px;">Failed to load audit logs.</td></tr>`;
    }
  }

  function parseMarkdown(md) {
    return md
      .replace(/\n\n/g, "<br><br>")
      .replace(/\n/g, "<br>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/\*([^*]+)\*/g, "<em>$1</em>")
      .replace(/- `\[ \]` (.*)/g, '<div class="todo"><input type="checkbox" disabled> $1</div>')
      .replace(/- `\[x\]` (.*)/g, '<div class="todo"><input type="checkbox" checked disabled> $1</div>')
      .replace(/### (.*)/g, "<h3 style='color:#8b7cf7; margin-top:10px; margin-bottom:6px;'>$1</h3>")
      .replace(/## (.*)/g, "<h2 style='color:#4be1c3; margin-top:14px; margin-bottom:8px;'>$1</h2>")
      .replace(/# (.*)/g, "<h1 style='color:#fff; margin-top:18px; margin-bottom:10px;'>$1</h1>");
  }

  return { boot };

})();
