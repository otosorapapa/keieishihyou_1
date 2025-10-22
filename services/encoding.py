"""Utilities for encoding detection with an optional chardet dependency."""

from __future__ import annotations

import importlib

_COMMON_ENCODINGS: tuple[str, ...] = ("cp932", "utf-8", "utf-8-sig")


def _import_chardet():
    spec = importlib.util.find_spec("chardet")
    if spec is None:
        return None
    return importlib.import_module("chardet")


def _fallback_detect(data: bytes) -> dict[str, str | None]:
    for encoding in _COMMON_ENCODINGS:
        try:
            data.decode(encoding)
            return {"encoding": encoding}
        except UnicodeDecodeError:
            continue
    return {"encoding": "cp932"}


_chardet = _import_chardet()


def detect_bytes(data: bytes) -> dict[str, str | None]:
    """Return chardet-like detection results for the provided bytes."""
    if _chardet is not None:
        return _chardet.detect(data)
    return _fallback_detect(data)

