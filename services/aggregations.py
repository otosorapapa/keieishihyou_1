from __future__ import annotations

import pandas as pd

from .metrics import calculate_metrics, compute_yearly_aggregates


def get_major_average(df: pd.DataFrame, major_code: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    major_df = df[df["産業大分類コード"] == major_code]
    yearly = compute_yearly_aggregates(major_df)
    return calculate_metrics(yearly)


def get_overall_average(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    yearly = compute_yearly_aggregates(df)
    return calculate_metrics(yearly)


def align_years(*dfs: pd.DataFrame) -> list[pd.DataFrame]:
    years = sorted({int(year) for df in dfs if not df.empty for year in df.get("集計年", [])})
    if not years:
        return list(dfs)
    aligned = []
    for df in dfs:
        if df.empty:
            aligned.append(df)
            continue
        df = df.copy()
        df["集計年"] = df["集計年"].astype(int)
        df = df[df["集計年"].isin(years)]
        aligned.append(df)
    return aligned
