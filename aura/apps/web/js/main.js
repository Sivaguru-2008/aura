/* ============================================================
   MAIN — router + the warp.
   Entering /app is not navigation. The landing collapses,
   particles converge into a core, real engines report in,
   the core detonates, and the console assembles from the flash.
   ============================================================ */
window.ROUTER = (() => {
  "use strict";
  const { Field, api, REDUCED } = FX;
  const $ = (id) => document.getElementById(id);
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  let warping = false;

  async function bootLines(pre) {
    // real telemetry, streamed like a launch checklist
    let h = null;
    try { h = await api("/v1/health"); } catch { /* offline */ }
    const lines = h ? [
      ["GATEWAY", "reached"],
      ["VISION ENGINE", "online"],
      ["FUSION", `${h.backend} · 8 qubits ${h.backend === "quantum" ? "entangled" : "bypassed"}`],
      ["SAFETY", h.trained ? "calibrated · conformal 90%" : "UNTRAINED — run aura_cli train"],
      ["MEMORY", `${h.cases} cases indexed`],
      ["CONSOLE", "materializing"],
    ] : [
      ["GATEWAY", "unreachable — demo shell only"],
      ["CONSOLE", "materializing"],
    ];
    for (const [k, v] of lines) {
      pre.innerHTML += `<span class="ok">▸ ${k.padEnd(14, " ")}</span>${v}\n`;
      await wait(REDUCED ? 20 : 230);
    }
  }

  async function launch({ instant = false } = {}) {
    if (warping) return;
    warping = true;
    const warp = $("warp");
    const landing = $("landing");
    const consoleEl = $("console");
    const pre = $("boot-lines");
    pre.textContent = "";

    if (instant || REDUCED) {
      landing.style.display = "none";
      document.body.dataset.view = "console";
      consoleEl.hidden = false;
      CONSOLE.boot();
      warping = false;
      return;
    }

    // 1) the surface collapses
    warp.hidden = false;
    landing.classList.add("collapsing");
    if (window._heroField) window._heroField.setMode("collapse");

    // 2) particles converge — a core forms
    const f = new Field($("warp-canvas"), { count: 220, hue: 172, mode: "drift", size: 1.7, speed: 1.3, mouse: false });
    f.start();
    await wait(350);
    f.setMode("collapse");

    // 3) engines report in (real /v1/health during the sequence)
    await bootLines(pre);
    await wait(250);

    // 4) detonation → the console assembles from the light
    $("warp-flash").classList.add("go");
    await wait(240);
    landing.style.display = "none";
    document.body.dataset.view = "console";
    consoleEl.hidden = false;
    CONSOLE.boot();
    await wait(600);
    warp.hidden = true;
    $("warp-flash").classList.remove("go");
    f.destroy();
    pre.textContent = "";
    if (location.pathname !== "/app") history.pushState({ v: "app" }, "", "/app");
    warping = false;
  }

  function surface() {
    // leaving the deep system — reverse, gentler
    const landing = $("landing");
    document.body.dataset.view = "landing";
    $("console").hidden = true;
    landing.style.display = "";
    requestAnimationFrame(() => landing.classList.remove("collapsing"));
    if (window._heroField) window._heroField.setMode("orbit");
    if (location.pathname !== "/") history.pushState({ v: "land" }, "", "/");
  }

  function init() {
    FX.init();
    LANDING.init();
    // live gateway telemetry in the top bar
    api("/v1/health").then((h) => {
      $("sys-chip").classList.add("ok");
      $("sys-chip-text").textContent = `gateway live · ${h.backend} fusion · ${h.cases} cases`;
    }).catch(() => { $("sys-chip-text").textContent = "gateway offline — static preview"; });
    document.querySelectorAll("[data-launch]").forEach((b) =>
      b.addEventListener("click", () => launch()));
    addEventListener("popstate", () => {
      location.pathname === "/app" ? launch({ instant: true }) : surface();
    });
    // deep-linked straight into the system? still earn the entrance.
    if (location.pathname === "/app" || location.hash === "#/app") launch();
  }

  document.readyState === "loading"
    ? addEventListener("DOMContentLoaded", init)
    : init();

  return { launch, surface };
})();
