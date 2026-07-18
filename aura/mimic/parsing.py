"""Shared parsing helpers for the MIMIC-CXR aug CSVs.

The aug CSVs store several columns (``image``, ``view``, ``AP``, ``PA``,
``Lateral``, ``text``, ``text_augment``) as *stringified Python lists*. These
helpers turn them back into real lists safely — malformed or empty cells become
``[]`` rather than raising — so every downstream consumer parses them the same way.
"""
from __future__ import annotations

import ast
from typing import Any

_EMPTY_TOKENS = {"", "[]", "nan", "none", "null"}


def safe_list(value: Any) -> list:
    """Parse a stringified Python list; return ``[]`` for empty/malformed input.

    Accepts already-parsed lists/tuples, ``NaN`` floats, ``None``, and strings.
    Never raises.
    """
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value is None:
        return []
    if isinstance(value, float):                 # pandas NaN
        return []
    s = str(value).strip()
    if s.lower() in _EMPTY_TOKENS:
        return []
    try:
        parsed = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []
    if isinstance(parsed, (list, tuple)):
        return [x for x in parsed]
    return [parsed]


def safe_str_list(value: Any) -> list[str]:
    """Like :func:`safe_list` but coerces every item to a stripped ``str`` and
    drops blanks — used for image paths and report texts."""
    out: list[str] = []
    for item in safe_list(value):
        s = str(item).strip()
        if s:
            out.append(s)
    return out
