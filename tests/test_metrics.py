import numpy as np
import pandas as pd

from services.metrics import calculate_metrics


def test_safe_division_handles_zero_denominator():
    df = pd.DataFrame(
        {
            "集計年": [2020, 2021],
            "売上高（百万円）": [100.0, 0.0],
            "経常利益（経常損失）（百万円）": [10.0, 5.0],
        }
    )
    metrics = calculate_metrics(df)
    assert np.isnan(metrics.loc[metrics["集計年"] == 2021, "経常利益率"].iloc[0])


def test_depreciation_columns_are_combined_for_ebitda():
    df = pd.DataFrame(
        {
            "集計年": [2020, 2021],
            "営業利益（百万円）": [50.0, 40.0],
            "減価償却費（百万円）": [10.0, 12.0],
            "減価償却費（百万円）.1": [5.0, 4.0],
        }
    )
    metrics = calculate_metrics(df)
    ebitda_latest = metrics.loc[metrics["集計年"] == 2021, "EBITDA"].iloc[0]
    assert ebitda_latest == 40.0 + 12.0 + 4.0


def test_fte_uses_available_columns():
    df = pd.DataFrame(
        {
            "集計年": [2020],
            "付加価値額（百万円）": [200.0],
            "常用雇用者": [10.0],
            "合計_正社員・正職員以外（就業時間換算人数）": [2.0],
        }
    )
    metrics = calculate_metrics(df)
    assert metrics.loc[0, "FTE"] == 12.0
    assert metrics.loc[0, "労働生産性"] == 200.0 / 12.0
