"""End-to-end smoke test: push a real radiograph through the whole pipeline.

Preflight proves the process *can* load its models. This proves it actually
serves a diagnosis: health -> modality router -> upload -> case detail ->
model card.

Two modes, one set of assertions:

    python deploy/smoke_test.py                     # in-process, via TestClient
    python deploy/smoke_test.py --url http://localhost:8000   # against a container

The in-process mode boots the ASGI app through its lifespan, which is where
``install_router()`` runs. That matters: ``gateway/app.py`` catches a router
failure and only prints a warning, so an import-only check passes while every
``/v1/studies/analyze`` and brain-MRI route 404s.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
AURA_ROOT = REPO_ROOT / "aura"
SAMPLE = AURA_ROOT / "sample.jpg"

_checks: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    _checks.append((name, passed, detail))
    print(f"  {'PASS' if passed else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))


class _Client:
    """Thin shim so the same assertions run in-process or over HTTP."""

    def __init__(self, base: str | None, db_path: str | None = None):
        self.base = base.rstrip("/") if base else None
        self._tc = None
        if self.base is None:
            # Point the case store at a throwaway DB before importing anything:
            # common/config.py reads AURA_DB_PATH at import time, and without
            # this the smoke run appends cases to the developer's real worklist.
            if db_path:
                os.environ["AURA_DB_PATH"] = db_path
            sys.path.insert(0, str(AURA_ROOT))
            from fastapi.testclient import TestClient
            from aura.gateway.app import app

            self._ctx = TestClient(app)
            self._tc = self._ctx.__enter__()

    def close(self) -> None:
        if self._tc is not None:
            self._ctx.__exit__(None, None, None)

    def get(self, path: str):
        if self._tc is not None:
            r = self._tc.get(path)
            return r.status_code, (r.json() if r.content else None), r.text
        import urllib.error
        import urllib.request

        try:
            with urllib.request.urlopen(f"{self.base}{path}", timeout=120) as resp:  # noqa: S310
                raw = resp.read().decode()
                return resp.status, (json.loads(raw) if raw else None), raw
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode()
            return exc.code, None, raw

    def post_file(self, path: str, file_path: Path):
        if self._tc is not None:
            with file_path.open("rb") as fh:
                r = self._tc.post(path, files={"file": (file_path.name, fh, "image/jpeg")})
            return r.status_code, (r.json() if r.content else None), r.text

        import urllib.error
        import urllib.request
        import uuid

        boundary = uuid.uuid4().hex
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'
            "Content-Type: image/jpeg\r\n\r\n"
        ).encode() + file_path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()

        req = urllib.request.Request(  # noqa: S310
            f"{self.base}{path}",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:  # noqa: S310
                raw = resp.read().decode()
                return resp.status, (json.loads(raw) if raw else None), raw
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode()
            return exc.code, None, raw


def wait_for_health(base: str, timeout: int) -> bool:
    """Poll until the gateway answers. Cold start loads DenseNet-121 on CPU."""
    import urllib.error
    import urllib.request

    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base}/v1/health", timeout=5) as resp:  # noqa: S310
                if resp.status == 200:
                    return True
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(2)
    print(f"  gateway never became healthy within {timeout}s (last error: {last})")
    return False


def run(client: _Client, expect_brain: bool) -> int:
    print("\n[1/5] Health")
    code, body, text = client.get("/v1/health")
    check("GET /v1/health -> 200", code == 200, text[:200] if code != 200 else "")
    if code == 200:
        check("status ok", body.get("status") == "ok", str(body))
        check("fusion model trained", body.get("trained") is True,
              "artifacts/fusion_*.npz missing or unreadable"
              if body.get("trained") is not True else f"backend={body.get('backend')}")

    print("\n[2/5] Modality router")
    # gateway/app.py installs this inside a try/except that only warns, so a
    # failure here is otherwise invisible until a request 404s.
    code, body, text = client.get("/v1/engines")
    check("GET /v1/engines -> 200", code == 200,
          "router failed to install (gateway only logs a WARNING)" if code != 200 else "")
    if code == 200:
        names = json.dumps(body)
        check("thorax engine registered", "thorax" in names, names[:200])
        if expect_brain:
            check("neuromind engine registered", "neuromind" in names, names[:200])

    print("\n[3/5] Live inference on a real radiograph")
    check("sample.jpg present", SAMPLE.is_file(), str(SAMPLE))
    case_id = None
    if SAMPLE.is_file():
        t0 = time.time()
        code, body, text = client.post_file("/v1/studies/upload", SAMPLE)
        dt = time.time() - t0
        check("POST /v1/studies/upload -> 200", code == 200,
              text[:300] if code != 200 else f"{dt:.1f}s")
        if code == 200:
            case_id = body.get("case_id")
            check("case_id returned", bool(case_id), str(body)[:200])

    print("\n[4/5] Case detail")
    if case_id:
        code, body, text = client.get(f"/v1/cases/{case_id}")
        check(f"GET /v1/cases/{case_id} -> 200", code == 200, text[:200] if code != 200 else "")
        if code == 200:
            fusion = body.get("fusion") or {}
            posterior = fusion.get("posterior") or {}
            check("case carries a fusion posterior", bool(posterior),
                  f"keys={list(body)[:10]}")
            if posterior:
                total = sum(posterior.values())
                top = max(posterior, key=posterior.get)
                check("posterior is a normalised distribution",
                      abs(total - 1.0) < 1e-3, f"sum={total:.6f}")
                # A degenerate all-equal posterior is what an untrained or
                # fallback fusion head emits; a real one concentrates somewhere.
                check("posterior is not uniform",
                      max(posterior.values()) > 1.5 / len(posterior),
                      f"top={top} p={posterior[top]:.3f} over {len(posterior)} classes")
            check("vision findings present", bool(body.get("vision")),
                  f"keys={list(body)[:10]}")
    else:
        check("case detail", False, "skipped - no case_id from upload")

    print("\n[5/5] Model card")
    code, body, text = client.get("/v1/model-card")
    check("GET /v1/model-card -> 200", code == 200, text[:200] if code != 200 else "")
    if code == 200:
        # model_version is None when the numpy fallback served the request, so
        # this doubles as a "the real DenseNet answered" assertion.
        check("model_version populated (real backbone served)",
              bool(body.get("model_version")), str(body.get("model_version")))

    failed = [n for n, ok, _ in _checks if not ok]
    print("\n" + "=" * 74)
    if failed:
        print(f"SMOKE TEST FAILED - {len(failed)} check(s): {', '.join(failed)}")
        print("=" * 74)
        return 1
    print(f"SMOKE TEST OK - {len(_checks)} checks passed.")
    print("=" * 74)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="AURA end-to-end smoke test")
    ap.add_argument("--url", default=None,
                    help="base URL of a running gateway; omit to run in-process")
    ap.add_argument("--wait", type=int, default=180,
                    help="seconds to wait for --url to become healthy (default 180)")
    ap.add_argument("--expect-brain", action="store_true",
                    help="also require the neuromind engine to be registered")
    ap.add_argument("--use-real-db", action="store_true",
                    help="in-process mode: write to the real case store instead "
                         "of a temporary one")
    args = ap.parse_args()

    print("=" * 74)
    print(f"AURA smoke test - target: {args.url or 'in-process (TestClient)'}")
    print("=" * 74)

    if args.url and not wait_for_health(args.url, args.wait):
        return 1

    # ignore_cleanup_errors: on Windows the store keeps the SQLite handle open,
    # so rmtree raises WinError 32 *after* a passing run and turns a green smoke
    # test into a non-zero exit.
    with tempfile.TemporaryDirectory(prefix="aura-smoke-",
                                     ignore_cleanup_errors=True) as td:
        db = None if (args.url or args.use_real_db) else str(Path(td) / "smoke.db")
        client = _Client(args.url, db)
        try:
            return run(client, args.expect_brain)
        finally:
            client.close()


if __name__ == "__main__":
    sys.exit(main())
