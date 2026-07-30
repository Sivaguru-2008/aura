"""One import root. Keep it that way.

For most of this project's life the package only imported cleanly with *both*
`aura-main/` and `aura-main/aura/` on `sys.path`, because modules mixed
`from aura.services.x import ...` with bare `from services.x import ...`. pytest
papered over it (`pythonpath = ["."]` plus rootdir insertion), so the suite passed
while anything importing the package directly — a script, a notebook, an ASGI loader
configured slightly differently — hit `ModuleNotFoundError: No module named
'knowledge'`. That cost enough time to produce a 175 KB IMPORT_REPAIR_REPORT.md.

The repair is complete: every first-party import is now absolute under `aura.`.
These tests keep it complete. They are cheap and they fail on the *first* regression,
which is the only moment the fix is still a one-line change.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

PKG_ROOT = Path(__file__).resolve().parents[1]      # .../aura-main/aura
REPO_ROOT = PKG_ROOT.parent                          # .../aura-main

# Top-level directories inside aura/ that are importable as bare names only when the
# package root itself is on sys.path. Importing any of these without the `aura.`
# prefix reintroduces the two-root requirement.
FIRST_PARTY = {
    "services", "gateway", "schemas", "knowledge", "common",
    "ml", "mimic", "backend", "apps", "tests",
}

SKIP_DIRS = {"__pycache__", ".pytest_cache", "pytest-tmp", "artifacts",
             ".venv", "venv", "change", "demo_data", "data"}


def _source_files() -> list[Path]:
    out = []
    for p in PKG_ROOT.rglob("*.py"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        out.append(p)
    return out


def _bare_root_imports(path: Path) -> list[tuple[int, str]]:
    """(lineno, statement) for every first-party import missing the `aura.` prefix."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:                      # pragma: no cover - would fail elsewhere
        pytest.fail(f"{path} does not parse: {exc}")

    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            # level > 0 is an explicit relative import (`from .x import y`) — fine.
            if node.level == 0 and node.module:
                root = node.module.split(".")[0]
                if root in FIRST_PARTY:
                    found.append((node.lineno, f"from {node.module} import ..."))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in FIRST_PARTY:
                    found.append((node.lineno, f"import {alias.name}"))
    return found


def test_no_bare_root_first_party_imports():
    """Every first-party import is absolute under `aura.` or explicitly relative."""
    offenders: list[str] = []
    for path in _source_files():
        for lineno, stmt in _bare_root_imports(path):
            rel = path.relative_to(REPO_ROOT)
            offenders.append(f"  {rel}:{lineno}  {stmt}")

    assert not offenders, (
        "These imports resolve only when aura/ is *itself* on sys.path, which "
        "reintroduces the two-import-root requirement the codebase was repaired to "
        "remove:\n" + "\n".join(sorted(offenders)) +
        "\n\nPrefix them with `aura.` (or make them explicitly relative)."
    )


def test_package_imports_with_only_the_repo_root_on_syspath():
    """The property the rule above exists to protect.

    Runs in a subprocess with a clean sys.path so the ambient test environment (which
    has both roots) cannot mask a regression. If this fails, someone reintroduced a
    bare-root import that ast alone did not catch — a runtime __import__, an importlib
    call, or a sys.path mutation.
    """
    code = (
        "import sys; sys.path.insert(0, r'%s')\n"
        "import aura.gateway.app, aura.backend.api.routes, aura.services.safety\n"
        "import aura.services.fusion, aura.services.vision, aura.aura_cli\n"
        "print('ok')\n" % REPO_ROOT
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
        env={"PATH": "", "SYSTEMROOT": "C:\\Windows"} if sys.platform == "win32" else {},
    )
    assert proc.returncode == 0 and "ok" in proc.stdout, (
        "aura no longer imports with only the repo root on sys.path.\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr[-2000:]}"
    )
