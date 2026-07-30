"""Session-wide test isolation.

Two problems this solves, both of which have already caused real damage:

**1. The suite used to run against the live case store.** `DB_PATH` defaults to
`aura/artifacts/aura.db` — the same SQLite file the running app serves its worklist
from. Tests seeded fixture cases into it (`CASE-TEST-1`, `R1`, `R2`, ...), left
`outcomes` rows behind, and — worst — wrote `conformal_state`, which is the online
adaptive-conformal threshold q̂ that decides how wide a conformal set is in
production. A test run therefore moved a served safety parameter. `AURA_DB_PATH` is
redirected here, before `aura.common.config` is imported, so every test gets a
throwaway database and the real one is never opened.

**2. Nothing noticed when a test overwrote a calibration artifact.** A prior audit
traced an operating-point degeneracy — pneumothorax never firing, effusion always
firing — to a test-written n=16 calibration fit that had clobbered a validated
n=2099 one in `aura/artifacts/`. The suite passed the whole time. The session hook
at the bottom hashes every git-tracked artifact before and after the run and fails
the session if any changed, so that class of bug can never again be invisible.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

_PKG_ROOT = Path(__file__).resolve().parent          # .../aura-main/aura
_REPO_ROOT = _PKG_ROOT.parent
_ARTIFACTS = _PKG_ROOT / "artifacts"

# ---------------------------------------------------------------------------- #
# 1. Redirect the case store BEFORE anything imports aura.common.config, which
#    resolves DB_PATH once at module import time.
# ---------------------------------------------------------------------------- #
if not os.environ.get("AURA_KEEP_TEST_DB"):
    _tmp_db = Path(tempfile.mkdtemp(prefix="aura-test-db-")) / "aura.db"
    os.environ["AURA_DB_PATH"] = str(_tmp_db)


# ---------------------------------------------------------------------------- #
# 2. Fail the session if a test mutated a served artifact.
# ---------------------------------------------------------------------------- #
def _tracked_artifacts() -> list[Path]:
    """git-tracked files under aura/artifacts — the ones a clone actually serves."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z", "aura/artifacts"],
            cwd=str(_REPO_ROOT), capture_output=True, timeout=60,
        )
    except Exception:                                # git absent / not a repo
        return []
    if out.returncode != 0:
        return []
    names = out.stdout.decode("utf-8", "replace").split("\0")
    return [_REPO_ROOT / n for n in names if n and (_REPO_ROOT / n).is_file()]


def _digest(paths: list[Path]) -> dict[str, str]:
    fingerprints: dict[str, str] = {}
    for p in paths:
        try:
            h = hashlib.md5()                        # nosec B324 — change detection only
            with p.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            fingerprints[str(p)] = h.hexdigest()
        except OSError:
            continue
    return fingerprints


def pytest_configure(config: pytest.Config) -> None:
    config._aura_artifacts_before = _digest(_tracked_artifacts())   # type: ignore[attr-defined]


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    before: dict[str, str] = getattr(session.config, "_aura_artifacts_before", {})
    if not before:
        return
    after = _digest(_tracked_artifacts())

    changed = sorted(k for k, v in after.items() if k in before and before[k] != v)
    removed = sorted(k for k in before if k not in after)
    if not (changed or removed):
        return

    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    lines = ["", "=" * 74,
             "TESTS MUTATED SERVED ARTIFACTS - this is a bug in the test, not the app.",
             "=" * 74]
    for p in changed:
        lines.append(f"  modified: {Path(p).relative_to(_REPO_ROOT)}")
    for p in removed:
        lines.append(f"  removed : {Path(p).relative_to(_REPO_ROOT)}")
    lines += [
        "",
        "These files are the calibration and registry state the app serves. A test",
        "that writes them silently changes production behaviour: an n=16 fit once",
        "overwrote a validated n=2099 one this way and no test failed.",
        "",
        "Fix the test to write under tmp_path; do not delete this check.",
        f"Restore with:  git checkout -- aura/artifacts/",
        "=" * 74, "",
    ]
    msg = "\n".join(lines)
    if reporter is not None:
        reporter.write_line(msg, red=True)
    else:                                            # pragma: no cover
        print(msg)

    session.exitstatus = 1
