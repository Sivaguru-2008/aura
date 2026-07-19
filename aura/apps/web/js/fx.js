/* ============================================================
   FX — AURA's motion engine.
   Springs, particle fields, magnetic surfaces, depth tilt,
   scroll choreography, contextual lighting. Zero dependencies.
   Everything runs on transform/opacity/canvas → GPU-composited.
   ============================================================ */
window.FX = (() => {
  "use strict";

  const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const lerp = (a, b, t) => a + (b - a) * t;

  /* ---------------- pointer state ---------------- */
  const pointer = { x: innerWidth / 2, y: innerHeight / 2, sx: innerWidth / 2, sy: innerHeight / 2 };
  addEventListener("pointermove", (e) => { pointer.x = e.clientX; pointer.y = e.clientY; }, { passive: true });

  /* ---------------- contextual light (halo) ---------------- */
  function initHalo() {
    const halo = document.getElementById("halo");
    if (!halo) return;
    let hue = 172, hueT = 172;
    // watch which [data-hue] scene owns the viewport centre
    const scenes = [...document.querySelectorAll("[data-hue]")];
    const io = new IntersectionObserver((es) => {
      es.forEach((e) => { if (e.isIntersecting) hueT = +e.target.dataset.hue || 172; });
    }, { rootMargin: "-40% 0px -40% 0px" });
    scenes.forEach((s) => io.observe(s));
    (function tick() {
      pointer.sx = lerp(pointer.sx, pointer.x, 0.08);
      pointer.sy = lerp(pointer.sy, pointer.y, 0.08);
      hue = lerp(hue, hueT, 0.04);
      halo.style.setProperty("--hx", pointer.sx + "px");
      halo.style.setProperty("--hy", pointer.sy + "px");
      halo.style.setProperty("--halo-h", hue.toFixed(1));
      requestAnimationFrame(tick);
    })();
  }

  /* ---------------- magnetic surfaces ---------------- */
  const magnets = new Set();
  function magnetize(root = document) {
    root.querySelectorAll("[data-magnetic]").forEach((el) => magnets.add(el));
  }
  if (!REDUCED) {
    (function magTick() {
      magnets.forEach((el) => {
        if (!el.isConnected) { magnets.delete(el); el = null; return; }
        const r = el.getBoundingClientRect();
        if (!r.width) return;
        const cx = r.left + r.width / 2, cy = r.top + r.height / 2;
        const dx = pointer.x - cx, dy = pointer.y - cy;
        const d = Math.hypot(dx, dy);
        const reach = Math.max(r.width, 110);
        let tx = 0, ty = 0;
        if (d < reach) { const pull = (1 - d / reach) * 0.32; tx = dx * pull; ty = dy * pull; }
        const cur = el._mag || { x: 0, y: 0 };
        cur.x = lerp(cur.x, tx, 0.18); cur.y = lerp(cur.y, ty, 0.18);
        el._mag = cur;
        if (Math.abs(cur.x) > 0.05 || Math.abs(cur.y) > 0.05)
          el.style.transform = `translate(${cur.x.toFixed(2)}px, ${cur.y.toFixed(2)}px)`;
        else el.style.transform = "";
      });
      requestAnimationFrame(magTick);
    })();
  }

  /* ---------------- depth tilt (panels lean toward cursor) ---------------- */
  function tiltify(root = document) {
    if (REDUCED) return;
    root.querySelectorAll("[data-tilt]").forEach((el) => {
      el.addEventListener("pointermove", (e) => {
        const r = el.getBoundingClientRect();
        const px = (e.clientX - r.left) / r.width - 0.5;
        const py = (e.clientY - r.top) / r.height - 0.5;
        el.style.transform = `rotateX(${(-py * 1.6).toFixed(2)}deg) rotateY(${(px * 1.6).toFixed(2)}deg) translateZ(0)`;
      });
      el.addEventListener("pointerleave", () => { el.style.transform = ""; });
    });
  }

  /* ---------------- reveal on scroll ---------------- */
  function reveals() {
    const io = new IntersectionObserver((es) => {
      es.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } });
    }, { threshold: 0.18 });
    document.querySelectorAll(".reveal").forEach((el) => io.observe(el));
  }

  /* ---------------- pinned-scene progress (scroll choreography) ---------------- */
  const sceneCbs = [];
  function scene(el, cb) { sceneCbs.push([el, cb]); }
  function sceneTick() {
    const vh = innerHeight;
    sceneCbs.forEach(([el, cb]) => {
      const r = el.getBoundingClientRect();
      const total = r.height - vh;
      if (total <= 0) return;
      const p = clamp(-r.top / total, 0, 1);
      if (el._p !== p) { el._p = p; cb(p); }
    });
    requestAnimationFrame(sceneTick);
  }

  /* ---------------- number animation ---------------- */
  function countTo(el, target, { dur = 1400, fmt = (v) => v.toFixed(2) } = {}) {
    if (REDUCED) { el.textContent = fmt(target); return; }
    const t0 = performance.now();
    (function step(t) {
      const p = clamp((t - t0) / dur, 0, 1);
      const e = 1 - Math.pow(1 - p, 4); // easeOutQuart
      el.textContent = fmt(target * e);
      if (p < 1) requestAnimationFrame(step);
    })(t0);
  }

  /* ---------------- typewriter ---------------- */
  function typeInto(el, text, { cps = 90 } = {}) {
    if (REDUCED) { el.textContent = text; return Promise.resolve(); }
    el.textContent = "";
    const tn = document.createTextNode("");
    const caret = document.createElement("span");
    caret.className = "caret";
    el.append(tn, caret);
    return new Promise((res) => {
      let i = 0;
      const iv = setInterval(() => {
        i = Math.min(text.length, i + Math.max(1, Math.round(cps / 30)));
        tn.nodeValue = text.slice(0, i);
        if (i >= text.length) { clearInterval(iv); caret.remove(); res(); }
      }, 33);
    });
  }

  /* ---------------- toast ---------------- */
  let toastT = null;
  function toast(msg) {
    const t = document.getElementById("toast");
    if (!t) return;
    t.hidden = false; t.textContent = msg;
    requestAnimationFrame(() => t.classList.add("show"));
    clearTimeout(toastT);
    toastT = setTimeout(() => t.classList.remove("show"), 2600);
  }

  /* ============================================================
     Field — the particle system.
     Modes: "drift" ambient · "orbit" forms a living ring ·
     "collapse" everything converges (the warp) · "links" constellation.
     ============================================================ */
  class Field {
    constructor(canvas, opts = {}) {
      this.cv = canvas;
      this.cx = canvas.getContext("2d");
      this.o = Object.assign({
        count: 140, hue: 172, links: false, linkDist: 110,
        mouse: true, mode: "drift", size: 1.6, speed: 1, ringR: 0.32,
      }, opts);
      this.parts = [];
      this.running = false;
      this.t = 0;
      this.resize();
      this._onResize = () => this.resize();
      addEventListener("resize", this._onResize);
      this.seed();
      // pause when offscreen
      this._io = new IntersectionObserver((es) => {
        es.forEach((e) => (e.isIntersecting ? this.start() : this.stop()));
      });
      this._io.observe(canvas);
    }
    resize() {
      const dpr = Math.min(devicePixelRatio || 1, 1.75);
      const r = this.cv.getBoundingClientRect();
      this.w = Math.max(1, r.width); this.h = Math.max(1, r.height);
      this.cv.width = this.w * dpr; this.cv.height = this.h * dpr;
      this.cx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    seed() {
      const n = REDUCED ? Math.min(40, this.o.count) : this.o.count;
      this.parts = Array.from({ length: n }, (_, i) => ({
        x: Math.random() * this.w, y: Math.random() * this.h,
        vx: 0, vy: 0,
        th: (i / n) * Math.PI * 2,
        sp: 0.15 + Math.random() * 0.5,
        r: this.o.size * (0.5 + Math.random()),
        tw: Math.random() * Math.PI * 2,
      }));
    }
    setMode(m) { this.o.mode = m; }
    start() { if (!this.running) { this.running = true; this._raf = requestAnimationFrame(this.tick.bind(this)); } }
    stop() { this.running = false; cancelAnimationFrame(this._raf); }
    destroy() { this.stop(); this._io.disconnect(); removeEventListener("resize", this._onResize); }

    tick(now) {
      if (!this.running) return;
      const dt = Math.min(0.05, (now - (this._last || now)) / 1000) * this.o.speed;
      this._last = now;
      this.t += dt;
      const { cx: g, w, h, o, t } = this;
      g.clearRect(0, 0, w, h);
      const rect = this.cv.getBoundingClientRect();
      const mx = pointer.x - rect.left, my = pointer.y - rect.top;
      const CX = w / 2, CY = h / 2;
      const R = Math.min(w, h) * o.ringR;

      for (const p of this.parts) {
        if (o.mode === "orbit") {
          p.th += dt * p.sp * 0.5;
          const wob = Math.sin(t * 0.9 + p.th * 3) * R * 0.13;
          const tx = CX + Math.cos(p.th) * (R + wob);
          const ty = CY + Math.sin(p.th) * (R * 0.62 + wob) ;
          p.vx += (tx - p.x) * 2.6 * dt; p.vy += (ty - p.y) * 2.6 * dt;
        } else if (o.mode === "collapse") {
          p.vx += (CX - p.x) * 14 * dt; p.vy += (CY - p.y) * 14 * dt;
        } else { // drift — pseudo-curl field
          p.vx += Math.sin(p.y * 0.013 + t * 0.7) * 9 * dt;
          p.vy += Math.cos(p.x * 0.011 - t * 0.6) * 9 * dt;
        }
        if (o.mouse && !REDUCED) {
          const dx = p.x - mx, dy = p.y - my, d2 = dx * dx + dy * dy;
          if (d2 < 150 * 150 && d2 > 1) {
            const f = (o.mode === "collapse" ? 0 : 1300) / d2;
            p.vx += dx * f * dt; p.vy += dy * f * dt;
          }
        }
        const damp = o.mode === "collapse" ? 0.965 : 0.94;
        p.vx *= damp; p.vy *= damp;
        p.x += p.vx * dt * 60 * 0.16; p.y += p.vy * dt * 60 * 0.16;
        if (o.mode === "drift") { // wrap
          if (p.x < -20) p.x = w + 20; if (p.x > w + 20) p.x = -20;
          if (p.y < -20) p.y = h + 20; if (p.y > h + 20) p.y = -20;
        }
      }

      // links (constellation)
      if (o.links) {
        g.lineWidth = 0.6;
        for (let i = 0; i < this.parts.length; i++) {
          const a = this.parts[i];
          for (let j = i + 1; j < this.parts.length; j++) {
            const b = this.parts[j];
            const dx = a.x - b.x, dy = a.y - b.y;
            const d2 = dx * dx + dy * dy, max = o.linkDist * o.linkDist;
            if (d2 < max) {
              g.strokeStyle = `hsla(${o.hue}, 75%, 68%, ${(1 - d2 / max) * 0.16})`;
              g.beginPath(); g.moveTo(a.x, a.y); g.lineTo(b.x, b.y); g.stroke();
            }
          }
        }
      }
      // points
      for (const p of this.parts) {
        const twk = 0.55 + 0.45 * Math.sin(t * 2 + p.tw);
        g.fillStyle = `hsla(${o.hue}, 80%, 70%, ${0.55 * twk})`;
        g.beginPath(); g.arc(p.x, p.y, p.r * twk, 0, 7); g.fill();
      }
      // collapse core glow
      if (o.mode === "collapse") {
        const gl = g.createRadialGradient(CX, CY, 0, CX, CY, R * 1.4);
        gl.addColorStop(0, `hsla(${o.hue}, 90%, 75%, 0.5)`);
        gl.addColorStop(1, "transparent");
        g.fillStyle = gl;
        g.fillRect(0, 0, w, h);
      }
      this._raf = requestAnimationFrame(this.tick.bind(this));
    }
  }

  /* ---------------- api helpers ---------------- */
  async function api(path, opts) {
    const r = await fetch(path, opts);
    if (!r.ok) {
      const err = new Error(`${r.status} ${path}`);
      err.status = r.status;
      try { err.detail = (await r.json()).detail; } catch { err.detail = null; }
      throw err;
    }
    return r.json();
  }

  function init() {
    initHalo(); magnetize(); tiltify(); reveals();
    requestAnimationFrame(sceneTick);
    const tb = document.getElementById("topbar");
    if (tb) addEventListener("scroll", () => tb.classList.toggle("scrolled", scrollY > 30), { passive: true });
  }

  return { REDUCED, clamp, lerp, pointer, init, magnetize, tiltify, scene, countTo, typeInto, toast, Field, api };
})();
