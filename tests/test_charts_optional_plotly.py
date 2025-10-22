"""Ensure the dashboard can operate without Plotly installed."""

from __future__ import annotations

import importlib
import importlib.util
import sys

import pandas as pd


def test_charts_import_without_plotly(monkeypatch):
    """The charts module should fall back gracefully when Plotly is absent."""

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, *args, **kwargs):  # type: ignore[unused-argument]
        if name.startswith("plotly"):
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    sys.modules.pop("ui.charts", None)

    charts = importlib.import_module("ui.charts")

    assert charts.HAS_PLOTLY is False
    assert isinstance(charts.PLOTLY_IMPORT_ERROR, ModuleNotFoundError)

    empty = pd.DataFrame()
    assert charts.sales_and_profit_chart(empty, empty, empty) is None

    sys.modules.pop("ui.charts", None)
