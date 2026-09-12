/* AURA sign-in. Loaded first on every page.
 *
 * The gateway accepts two credentials: the API token in a header (scripts, PACS
 * bridges, deploy/smoke_test.py) or a signed, HttpOnly session cookie (browsers).
 * This file is the browser half. It asks /v1/auth/status whether this viewer has to
 * sign in, shows a sign-in card when they do, and shows it again whenever an API call
 * comes back 401. The cookie rides on every same-origin request by itself, so no other
 * script on the page needs to know that authentication exists.
 */
(() => {
  "use strict";

  const nativeFetch = window.fetch.bind(window);
  let scrim = null;
  let scrimKind = "";

  const STYLE = `
    .aura-auth-scrim { position: fixed; inset: 0; z-index: 10000; display: grid; place-items: center;
      padding: 16px; background: rgba(6, 10, 18, 0.72); backdrop-filter: blur(6px); }
    .aura-auth-card { width: min(380px, 100%); box-sizing: border-box; padding: 24px;
      background: #0f1623; color: #e6ebf2; border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 12px; box-shadow: 0 20px 60px rgba(0, 0, 0, 0.45);
      font: 14px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
    .aura-auth-card h2 { margin: 0 0 4px; font-size: 18px; font-weight: 600; }
    .aura-auth-card p { margin: 0 0 12px; color: #9aa7b8; }
    .aura-auth-card label { display: block; margin: 12px 0 4px; font-size: 12px; color: #9aa7b8; }
    .aura-auth-card input { width: 100%; box-sizing: border-box; padding: 9px 10px; border-radius: 8px;
      border: 1px solid rgba(255, 255, 255, 0.18); background: #0a0f18; color: inherit; font: inherit; }
    .aura-auth-card input:focus-visible { outline: 2px solid #5b9dff; outline-offset: 1px; }
    .aura-auth-card button { width: 100%; margin-top: 18px; padding: 10px; border: 0; border-radius: 8px;
      background: #3b82f6; color: #fff; font: inherit; font-weight: 600; cursor: pointer; }
    .aura-auth-card button:focus-visible { outline: 2px solid #9cc2ff; outline-offset: 2px; }
    .aura-auth-card button[disabled] { opacity: 0.6; cursor: default; }
    .aura-auth-msg { min-height: 20px; margin-top: 10px; color: #f7a4a4; font-size: 13px; }
    .aura-auth-chip { position: fixed; right: 12px; bottom: 12px; z-index: 9999; display: flex;
      gap: 10px; align-items: center; padding: 6px 12px; border-radius: 999px;
      background: rgba(15, 22, 35, 0.92); color: #c9d3e0; border: 1px solid rgba(255, 255, 255, 0.12);
      font: 12px/1.2 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
    .aura-auth-chip button { padding: 0; border: 0; background: none; color: #7fb0ff;
      font: inherit; cursor: pointer; }
  `;

  function whenBodyReady(fn) {
    if (document.body) fn();
    else document.addEventListener("DOMContentLoaded", fn, { once: true });
  }

  function injectStyle() {
    if (document.getElementById("aura-auth-style")) return;
    const s = document.createElement("style");
    s.id = "aura-auth-style";
    s.textContent = STYLE;
    document.head.appendChild(s);
  }

  // Errors reach the browser in two shapes: the gateway middleware returns the body
  // bare ({error, reason}); an HTTPException raised in a route wraps it in {detail}.
  const codeOf = (b) => (b && (b.error || (b.detail && b.detail.error))) || "";
  const reasonOf = (b) => (b && (b.reason || (b.detail && b.detail.reason))) || "";

  function isApiCall(input) {
    const raw = typeof input === "string" ? input : (input && input.url) || "";
    try {
      const u = new URL(raw, location.href);
      return u.origin === location.origin && u.pathname.startsWith("/v1/")
        && !u.pathname.startsWith("/v1/auth/");
    } catch {
      return false;
    }
  }

  function openScrim(kind, build) {
    whenBodyReady(() => {
      injectStyle();
      if (scrim) scrim.remove();
      scrim = document.createElement("div");
      scrimKind = kind;
      scrim.className = "aura-auth-scrim";
      scrim.setAttribute("role", "dialog");
      scrim.setAttribute("aria-modal", "true");
      scrim.setAttribute("aria-labelledby", "aura-auth-title");
      const card = document.createElement("div");
      card.className = "aura-auth-card";
      build(card);
      scrim.appendChild(card);
      document.body.appendChild(scrim);
      const first = card.querySelector("input");
      if (first) first.focus();
    });
  }

  function showSignIn(message) {
    // Several fetches can 401 at once; rebuilding the card would wipe what the
    // clinician is typing, so an open sign-in card only gets its message updated.
    if (scrim && scrimKind === "signin") {
      const msg = scrim.querySelector(".aura-auth-msg");
      if (msg && message) msg.textContent = message;
      return;
    }
    openScrim("signin", (card) => {
      card.innerHTML = `
        <h2 id="aura-auth-title">Sign in to AURA</h2>
        <p>This server requires an access token. Your name is recorded against every
           action you take, in the audit log.</p>
        <form novalidate>
          <label for="aura-auth-user">Your name</label>
          <input id="aura-auth-user" name="username" autocomplete="username" maxlength="64" required>
          <label for="aura-auth-token">Access token</label>
          <input id="aura-auth-token" name="password" type="password"
                 autocomplete="current-password" required>
          <button type="submit">Sign in</button>
          <div class="aura-auth-msg" role="alert"></div>
        </form>`;
      const form = card.querySelector("form");
      const msg = card.querySelector(".aura-auth-msg");
      const btn = card.querySelector("button");
      if (message) msg.textContent = message;
      form.addEventListener("submit", async (ev) => {
        ev.preventDefault();
        const user = card.querySelector("#aura-auth-user").value.trim();
        const token = card.querySelector("#aura-auth-token").value;
        if (!user || !token) {
          msg.textContent = "Enter your name and the access token.";
          return;
        }
        btn.disabled = true;
        msg.textContent = "";
        try {
          const r = await nativeFetch("/v1/auth/session", {
            method: "POST",
            credentials: "same-origin",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ user, token }),
          });
          if (r.ok) {
            location.reload();
            return;
          }
          let body = null;
          try { body = await r.json(); } catch { /* non-JSON error page */ }
          msg.textContent = reasonOf(body) || `Sign-in failed (${r.status}).`;
        } catch {
          msg.textContent = "Could not reach the AURA server.";
        }
        btn.disabled = false;
      });
    });
  }

  function showBlocked(reason) {
    openScrim("blocked", (card) => {
      card.innerHTML = `<h2 id="aura-auth-title">Network access is not enabled</h2><p></p>`;
      card.querySelector("p").textContent = reason
        || "This AURA server has no access token configured, so it only answers requests "
         + "from the machine it runs on.";
    });
  }

  function showSignedIn(principal) {
    whenBodyReady(() => {
      injectStyle();
      const chip = document.createElement("div");
      chip.className = "aura-auth-chip";
      const who = document.createElement("span");
      who.textContent = `Signed in as ${principal}`;
      const out = document.createElement("button");
      out.type = "button";
      out.textContent = "Sign out";
      out.addEventListener("click", async () => {
        try {
          await nativeFetch("/v1/auth/session", { method: "DELETE", credentials: "same-origin" });
        } catch { /* reload either way; the next status probe decides what to show */ }
        location.reload();
      });
      chip.append(who, out);
      document.body.appendChild(chip);
    });
  }

  window.fetch = async (input, init) => {
    const resp = await nativeFetch(input, init);
    if ((resp.status === 401 || resp.status === 403) && isApiCall(input)) {
      let body = null;
      try { body = await resp.clone().json(); } catch { /* not a JSON error */ }
      const code = codeOf(body);
      if (code === "unauthenticated") showSignIn("Your session has ended. Sign in again.");
      else if (code === "auth_not_configured") showBlocked(reasonOf(body));
    }
    return resp;
  };

  nativeFetch("/v1/auth/status", { credentials: "same-origin" })
    .then((r) => (r.ok ? r.json() : null))
    .then((s) => {
      if (!s) return;
      if (s.blocked) showBlocked(s.reason);
      else if (s.auth_required && !s.authenticated) showSignIn();
      else if (s.authenticated && s.principal) showSignedIn(s.principal);
    })
    .catch(() => { /* server down: the page's own error handling covers it */ });
})();
