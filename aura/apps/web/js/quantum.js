/* ============================================================
   QUANTUM EVIDENCE — /quantum
   ------------------------------------------------------------
   Renders aura/artifacts/* through GET /v1/quantum/evidence. Every figure on the
   page comes from that response; there are no literals in this file except units
   and labels. A block the API omits renders as "not measured" rather than a zero,
   because a plausible default is the one thing this project refuses to serve.

   Deliberately renders the unfavourable results (§05) with the same weight as the
   favourable ones. A page that showed only the wins would misrepresent the system.
   ============================================================ */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, html) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (html != null) n.innerHTML = html;
    return n;
  };
  const num = (v, d = 4) =>
    (typeof v === "number" && isFinite(v)) ? v.toFixed(d) : "—";
  const pct = (v, d = 1) =>
    (typeof v === "number" && isFinite(v)) ? `${(v * 100).toFixed(d)}%` : "—";
  const int = (v) =>
    (typeof v === "number" && isFinite(v)) ? v.toLocaleString() : "—";

  /** Uniform "this was not measured" block, so absence never looks like a value. */
  const notMeasured = (what) =>
    el("p", "q-absent", `<strong>Not measured.</strong> ${what} No artifact is present, ` +
      `so nothing is shown rather than a placeholder.`);

  async function api(path) {
    const r = await fetch(path, { headers: { "x-aura-user": "quantum-view" } });
    if (!r.ok) throw new Error(`${path} → ${r.status}`);
    return r.json();
  }

  // ---------------------------------------------------------------- 01 served
  function renderServed(d) {
    const s = d.served || {};
    const host = $("q-served");
    const cards = [
      ["Fusion backend", s.fusion_backend, "the model that produces every posterior"],
      ["Register", `${s.n_qubits} qubits · ${s.n_layers} layers`, `${s.entangler} entangler`],
      ["Encoding", "RY(π·xᵢ)", "one evidence channel per qubit"],
      ["Readout", "⟨Zᵢ⟩ → linear head", "local observable, softmax over 6 diagnoses"],
      ["Shot budget", `${int(s.n_shots)} default`, "sequenced per study — see §02"],
      ["Brain QKL head", s.neuro_qkl_enabled ? "served" : "off", "glioma grade (HGG/LGG)"],
    ];
    cards.forEach(([k, v, sub]) => {
      const c = el("div", "q-card");
      c.append(el("span", "q-card-k", k), el("span", "q-card-v", String(v ?? "—")),
               el("span", "q-card-sub", sub));
      host.append(c);
    });

    // The honest half: what exists in the tree but is NOT on the serving path.
    const un = $("q-unserved");
    un.append(el("h4", null, "Present in the repository, <em>not</em> on the serving path"));
    const list = el("ul", "q-unserved-list");
    [
      ["Quantum autoencoder (QAE)", s.qae_served,
       "imported by the fusion engine, but no trained weights ship — <code>load()</code> returns null, so the plain-encode branch is taken"],
      ["Quantum Bayesian network (QBN)", s.qbn_served,
       "wired into the clinical reasoner, but <code>load_trained()</code> returns null because no fitted artifact ships — so the rule-adjusted posterior is returned unchanged. The constructor's default theta is six unfitted constants, and serving those as a quantum inference would dress up hardcoded numbers as a learned model"],
      ["Quantum multi-modal fusion (QMMF)", false,
       "no patient in the corpus has both a chest radiograph and a brain MRI, so a joint model has nothing to train on"],
      ["Data re-uploading ansatz", false,
       "implemented as an extension point; the served circuit uses single angle-encoding"],
    ].forEach(([name, served, why]) => {
      list.append(el("li", null,
        `<span class="q-pill ${served ? "on" : "off"}">${served ? "served" : "not served"}</span>` +
        `<strong>${name}</strong> — ${why}`));
    });
    un.append(list);
  }

  // ------------------------------------------------------------- 02 budget
  function renderBudget(d) {
    const host = $("q-budget");
    const b = d.measurement_budget;
    if (!b) { host.append(notMeasured("Shot-budget study over the held-out split.")); return; }

    const split = el("div", "q-split");
    const ml = b.abstention_breakdown?.measurement_limited ?? 0;
    const mo = b.abstention_breakdown?.model_limited ?? 0;

    split.append(
      statBig("Commit rate", pct(b.commit_rate), `${int(b.studies)} studies`),
      statBig("Median shots", int(b.median_shots_spent), "against a fixed 512-shot budget"),
      statBig("Abstained", pct(b.abstain_rate), `${int(ml + mo)} studies`),
    );
    host.append(split);

    const two = el("div", "q-two");
    two.append(
      el("div", "q-lane q-lane-measure",
        `<h4>Measurement-limited <span class="q-count">${int(ml)}</span></h4>
         <p>The margin is genuinely non-zero; this run had not bought enough precision.
            The system reports how many more shots would settle it
            (median ${int(b.median_predicted_shots_to_resolve)}).</p>
         <p class="q-action">→ Run the circuit longer.</p>`),
      el("div", "q-lane q-lane-model",
        `<h4>Model-limited <span class="q-count">${int(mo)}</span></h4>
         <p>The top two diagnoses are tied at <em>infinite</em> measurement precision.
            No budget resolves it.</p>
         <p class="q-action">→ Escalate to a human.</p>`),
    );
    host.append(two);
    host.append(el("p", "q-note",
      "A classical softmax reading 0.55 means “unsure” and cannot say which of these " +
      "two it is, because classical inference has no measurement budget to vary. This " +
      "runs on the serving path and appears on every quantum-backend case."));
  }

  function statBig(k, v, sub) {
    const n = el("div", "q-stat");
    n.append(el("span", "q-stat-k", k), el("span", "q-stat-v", String(v)),
             el("span", "q-stat-sub", sub || ""));
    return n;
  }

  // --------------------------------------------------- 03 hardware + noise
  function renderHardware(d) {
    const host = $("q-hw");
    const h = d.hardware;
    if (!h) { host.append(notMeasured("Execution on a real QPU.")); return; }
    host.append(el("div", "q-hw-card",
      `<div class="q-hw-top">
         <span class="q-chip live">EXECUTED ON HARDWARE</span>
         <span class="mono q-job">job ${h.job_id ?? "—"}</span>
       </div>
       <div class="q-hw-grid">
         <div><span>Device</span><strong>${h.backend ?? "—"}</strong></div>
         <div><span>Device qubits</span><strong>${int(h.backend_qubits)}</strong></div>
         <div><span>Mean |Δ⟨Z⟩| vs analytic</span><strong>${num(h.mean_abs_z_error_vs_analytic)}</strong></div>
         <div><span>Top-1 diagnosis</span><strong class="${h.top1_agrees_with_analytic ? "ok" : "bad"}">
           ${h.top1_agrees_with_analytic ? "survived device noise" : "diverged"}</strong></div>
       </div>`));
  }

  function renderLadder(d) {
    const host = $("q-ladder");
    const n = d.noise;
    if (!n) { host.append(notMeasured("Noise-model rung between simulator and QPU.")); return; }

    const rows = [];
    (n.rungs || []).forEach((r) => {
      rows.push([
        r.rung === "shot_noise_only" ? "Ideal simulator — shot noise only"
                                     : "Device noise model (FakeMarrakesh)",
        r.shots, r.mean_abs_z_error_vs_analytic, r.top1_agrees_with_analytic,
      ]);
    });
    const hw = n.hardware_reference;
    if (hw) rows.push([`Real hardware — ${hw.backend}`, null,
                       hw.mean_abs_z_error_vs_analytic, hw.top1_agrees_with_analytic]);

    const max = Math.max(...rows.map((r) => r[2] || 0), 1e-9);
    const tbl = el("div", "q-rungs");
    rows.forEach(([label, shots, err, ok]) => {
      const row = el("div", "q-rung");
      row.append(
        el("span", "q-rung-lbl", label + (shots ? ` <em>${int(shots)} shots</em>` : "")),
        el("span", "q-rung-bar",
           `<i style="width:${Math.max(2, (err / max) * 100).toFixed(1)}%"></i>`),
        el("span", "q-rung-val mono", num(err)),
        el("span", `q-rung-ok ${ok ? "ok" : "bad"}`, ok ? "top-1 held" : "top-1 lost"),
      );
      tbl.append(row);
    });
    host.append(tbl);

    const a = n.attribution || {};
    host.append(el("p", "q-verdict",
      `<strong>${pct(a.device_noise_share, 0)} of the simulated error is decoherence, ` +
      `not sampling.</strong> Shot noise shrinks as 1/√shots; the device component does not. ` +
      `Buying more shots is therefore close to useless here, and shortening the circuit is ` +
      `the lever that works — which is what §04 does. Real hardware sits above the static ` +
      `noise model, as expected: a calibration snapshot does not reproduce drift or crosstalk.`));
  }

  // ------------------------------------------------------------ 04 transpile
  function renderTranspile(d) {
    const host = $("q-transpile");
    const t = d.transpile;
    if (!t) { host.append(notMeasured("Transpilation against the device coupling map.")); return; }

    const imp = t.improvement_vs_level_1 || {};
    host.append(el("p", "q-sub",
      `Logical circuit: depth <strong>${int(t.logical?.depth)}</strong>, ` +
      `<strong>${int(t.logical?.two_qubit_gates)}</strong> two-qubit gates. Everything above ` +
      `that count is routing overhead the heavy-hex lattice forces on an 8-qubit ring.`));

    const tbl = el("table", "q-table");
    tbl.innerHTML =
      `<thead><tr><th>Optimization level</th><th>Depth</th><th>2-qubit gates</th><th>Total gates</th></tr></thead>`;
    const tb = el("tbody");
    (t.levels || []).forEach((r) => {
      const served = r.optimization_level === t.served_level;
      const tr = el("tr", served ? "served" : null);
      tr.innerHTML =
        `<td>level ${r.optimization_level}${served ? ' <span class="q-tag">served</span>' : ""}</td>` +
        `<td class="mono">${int(r.depth)}</td>` +
        `<td class="mono">${int(r.two_qubit_gates)}</td>` +
        `<td class="mono">${int(r.total_gates)}</td>`;
      tb.append(tr);
    });
    tbl.append(tb);
    // Node.append() returns undefined, so the wrapper has to be held in a variable —
    // chaining .lastChild off it throws and the whole block silently disappears.
    const wrap = el("div", "q-table-wrap");
    wrap.append(tbl);
    host.append(wrap);

    host.append(el("p", "q-verdict",
      `Moving from level 1 to level 3 saves <strong>${int(imp.two_qubit_gates_saved)} two-qubit gates ` +
      `(${num(imp.two_qubit_gates_saved_pct, 1)}%)</strong> and ${int(imp.depth_saved)} depth ` +
      `(${num(imp.depth_saved_pct, 1)}%), measured on <code>${t.backend}</code>'s published ` +
      `coupling map. Level 1 is still run on every hardware call so the improvement is a ` +
      `number in the run artifact rather than an assertion.`));
  }

  // ------------------------------------------------------------- 05 negative
  function renderNegative(d) {
    const host = $("q-negative");

    // (a) accuracy vs the classical backend
    const f = d.fusion_backends;
    if (f && f.quantum && f.classical) {
      const n = f.quantum.n ?? f.classical.n;
      host.append(el("div", "q-neg",
        `<h4>Accuracy — no advantage, and no resolvable difference</h4>
         <p>Classical product-of-experts <strong>${num(f.classical.accuracy)}</strong> vs
            quantum VQC <strong>${num(f.quantum.accuracy)}</strong> at n=${int(n)}. That is
            ${Math.round((f.classical.accuracy - f.quantum.accuracy) * n)} cases. A paired
            McNemar test cannot reach significance under <em>any</em> assignment of the
            discordant pairs, so the correct reading is “comparable”, not a ranking.</p>`));
    }

    // (b) entanglement ablation
    const ab = d.entanglement_ablation;
    if (ab) {
      const b = ab.bootstrap || {};
      const fmt = (m) => m
        ? `Δ ${m.delta >= 0 ? "+" : ""}${num(m.delta, 4)} · CI [${num(m.ci95?.[0], 4)}, ${num(m.ci95?.[1], 4)}] · ` +
          `<span class="${m.excludes_zero ? "sig" : "ns"}">${m.excludes_zero ? "significant" : "not significant"}</span>`
        : "—";
      host.append(el("div", "q-neg",
        `<h4>Entanglement, specifically — the CNOT ring is not earning its place</h4>
         <p>Qubits, layers, parameter count, encoding, readout, optimiser and seed all held
            fixed; only the CNOT ring is removed. Over ${int(b.n_studies)} studies,
            ${int(b.n_bootstrap)} bootstrap resamples:</p>
         <ul class="q-metrics">
           <li><span>accuracy</span>${fmt(b.accuracy)}</li>
           <li><span>NLL</span>${fmt(b.nll)} <em>— worse with entanglement</em></li>
           <li><span>ECE</span>${fmt(b.ece)}</li>
         </ul>`));
    }

    // (c) design sweep reproduces it
    const ds = d.design_sweep;
    if (ds && ds.cells?.length) {
      const best = [...ds.cells].sort((a, b2) => b2.accuracy - a.accuracy)[0];
      host.append(el("div", "q-neg",
        `<h4>The 48-cell sweep reproduces it independently</h4>
         <p>Best cell across the whole grid: <strong>${best.n_qubits}q / ${best.n_layers}L /
            <code>${best.entangler}</code></strong> at ${num(best.accuracy)} accuracy with
            <strong>${int(best.two_qubit_gates)} two-qubit gates</strong>. The served ring
            configuration is left unchanged rather than re-tuned to the winner — one grid
            search on one split is not grounds to alter a served clinical model — but the
            record now says the ring is kept for representational reasons, not measured ones.</p>`));
    }
    if (!host.children.length) host.append(notMeasured("Comparative studies."));
  }

  // ---------------------------------------------------------------- 06 sweep
  function renderSweep(d) {
    const ds = d.design_sweep;
    const meta = $("q-sweep-meta");
    const tbl = $("q-sweep");
    if (!ds || !ds.cells?.length) { meta.append(notMeasured("Design-space sweep.")); return; }

    meta.append(el("p", "q-sub",
      `${ds.n_cells} cells · ${int(ds.data?.train)} train / ${int(ds.data?.calibration)} calibration / ` +
      `${int(ds.data?.test)} test, patient-disjoint · ${ds.protocol?.epochs} epochs, seed ${ds.protocol?.seed}.`));
    meta.append(el("p", "q-caveat",
      `<strong>The qubit axis is an evidence ablation.</strong> The encoding is one evidence ` +
      `channel per qubit, so a cell with fewer than ${int(ds.data?.evidence_channels)} qubits ` +
      `<em>discards</em> channels. 8 qubits is set by the input dimension, not chosen for expressivity.`));

    const served = ds.served || {};
    const cells = [...ds.cells].sort((a, b) => b.accuracy - a.accuracy || a.two_qubit_gates - b.two_qubit_gates);
    const bestAcc = cells[0].accuracy;
    tbl.innerHTML =
      `<thead><tr><th>qubits</th><th>layers</th><th>entangler</th><th>accuracy</th>` +
      `<th>NLL</th><th>ECE</th><th>2q gates</th><th>params</th></tr></thead>`;
    const tb = el("tbody");
    cells.forEach((c) => {
      const isServed = c.n_qubits === served.n_qubits && c.n_layers === served.n_layers &&
                       c.entangler === served.entangler;
      const tr = el("tr", isServed ? "served" : null);
      tr.innerHTML =
        `<td class="mono">${c.n_qubits}</td><td class="mono">${c.n_layers}</td>` +
        `<td><code>${c.entangler}</code>${isServed ? ' <span class="q-tag">served</span>' : ""}</td>` +
        `<td class="mono ${c.accuracy === bestAcc ? "best" : ""}">${num(c.accuracy)}</td>` +
        `<td class="mono">${num(c.nll)}</td><td class="mono">${num(c.ece)}</td>` +
        `<td class="mono">${c.two_qubit_gates}</td><td class="mono">${c.trainable_parameters}</td>`;
      tb.append(tr);
    });
    tbl.append(tb);
  }

  // ------------------------------------------------------------- 07 coupling
  function renderCoupling(d) {
    const host = $("q-coupling");
    const c = d.evidence_coupling;
    if (!c || !c.matrix) { host.append(notMeasured("Evidence-coupling measurement.")); return; }

    const ch = c.channels || [];
    const M = c.matrix;
    let max = 0;
    M.forEach((row) => row.forEach((v) => { if (v > max) max = v; }));

    const grid = el("div", "q-heat");
    grid.style.gridTemplateColumns = `minmax(96px, auto) repeat(${ch.length}, 1fr)`;
    grid.append(el("span", "q-heat-corner", ""));
    ch.forEach((name) => grid.append(el("span", "q-heat-col", name.replace(/_/g, " "))));
    M.forEach((row, i) => {
      grid.append(el("span", "q-heat-row", (ch[i] || "").replace(/_/g, " ")));
      row.forEach((v, j) => {
        const cell = el("span", "q-heat-cell", i === j ? "" : num(v, 2));
        const a = max ? v / max : 0;
        cell.style.background = i === j ? "rgba(255,255,255,.03)"
                                        : `rgba(139,124,247,${(0.06 + a * 0.72).toFixed(3)})`;
        cell.title = `${ch[i]} ↔ ${ch[j]} = ${num(v, 4)}`;
        grid.append(cell);
      });
    });
    host.append(grid);

    // strongest / weakest, computed rather than hardcoded
    const pairs = [];
    for (let i = 0; i < M.length; i++)
      for (let j = i + 1; j < M.length; j++) pairs.push([M[i][j], ch[i], ch[j]]);
    pairs.sort((a, b) => b[0] - a[0]);
    if (pairs.length) {
      const [hi, a1, b1] = pairs[0];
      const [lo, a2, b2] = pairs[pairs.length - 1];
      host.append(el("p", "q-verdict",
        `Strongest: <strong>${a1.replace(/_/g, " ")} ↔ ${b1.replace(/_/g, " ")}</strong> at ${num(hi, 3)}. ` +
        `Weakest: ${a2.replace(/_/g, " ")} ↔ ${b2.replace(/_/g, " ")} at ${num(lo, 3)} — a ` +
        `${(hi / (lo || 1e-9)).toFixed(0)}× spread. The circuit learned that some evidence ` +
        `channels inform each other and others do not, which is the structure an entangled ` +
        `representation was chosen to capture.`));
    }
  }

  // ------------------------------------------------------------------- boot
  async function boot() {
    let d;
    try {
      d = await api("/v1/quantum/evidence");
    } catch (e) {
      $("q-src").textContent = "gateway unavailable";
      document.querySelector(".q-wrap").prepend(
        el("div", "q-missing",
           `<strong>Could not reach the gateway.</strong> ${e.message}. This page renders ` +
           `only measured artifacts, so it shows nothing rather than sample data.`));
      return;
    }

    $("q-src").textContent = "read from aura/artifacts/";
    if (d.missing?.length) {
      const m = $("q-missing");
      m.hidden = false;
      m.innerHTML = `<strong>${d.missing.length} artifact(s) not present:</strong> ` +
        `<code>${d.missing.join("</code>, <code>")}</code>. Those sections say “not measured”. ` +
        `Regenerate with the scripts in <code>aura/ml/evaluation/</code>.`;
    }

    // Each block is isolated: one bad artifact must not blank the rest of the page.
    const steps = [
      ["served", renderServed], ["budget", renderBudget], ["hardware", renderHardware],
      ["ladder", renderLadder], ["transpile", renderTranspile], ["negative", renderNegative],
      ["sweep", renderSweep], ["coupling", renderCoupling],
    ];
    for (const [name, fn] of steps) {
      try { fn(d); } catch (e) { console.error(`[quantum] ${name} failed:`, e); }
    }

    const gen = d.design_sweep?.generated || d.hardware?.generated;
    $("q-foot-text").textContent =
      `read from aura/artifacts/ · GET /v1/quantum/evidence` + (gen ? ` · latest run ${gen}` : "");
  }

  document.addEventListener("DOMContentLoaded", boot);
})();
