from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

REVENUE_COL = "売上高（百万円）"
OPERATING_PROFIT_COL = "営業利益（百万円）"
ORDINARY_PROFIT_COL = "経常利益（経常損失）（百万円）"
GROSS_PROFIT_COL = "売上総利益（百万円）"
NET_PROFIT_COL = "税引後当期純利益（百万円）"
VALUE_ADDED_COL = "付加価値額（百万円）"
PERSONNEL_COST_COL = "人件費（百万円）"
TOTAL_ASSETS_COL = "資産（百万円）"
CURRENT_ASSETS_COL = "流動資産（百万円）"
NON_CURRENT_ASSETS_COL = "固定資産（百万円）"
TOTAL_LIABILITIES_COL = "負債（百万円）"
CURRENT_LIABILITIES_COL = "流動負債（百万円）"
NON_CURRENT_LIABILITIES_COL = "固定負債（百万円）"
SHORT_LOAN_BANK_COL = "短期借入金（金融機関）（百万円）"
SHORT_LOAN_OTHER_COL = "短期借入金（金融機関以外）（百万円）"
LONG_LOAN_BANK_COL = "長期借入金（金融機関）（百万円）"
LONG_LOAN_OTHER_COL = "長期借入金（金融機関以外）（百万円）"
BONDS_COL = "社債（百万円）"
EQUITY_COL = "純資産（百万円）"
INTEREST_COL = "支払利息・割引料（百万円）"

DEPRECIATION_COLUMNS = [
    "減価償却費（百万円）",
    "減価償却費（百万円）.1",
]
FTE_COLUMNS = [
    "常用雇用者",
    "合計_正社員・正職員以外（就業時間換算人数）",
    "他社からの出向従業者（出向役員を含む）及び派遣従業者の合計",
]
INTEREST_BEARING_DEBT_COLUMNS = [
    SHORT_LOAN_BANK_COL,
    SHORT_LOAN_OTHER_COL,
    LONG_LOAN_BANK_COL,
    LONG_LOAN_OTHER_COL,
    BONDS_COL,
]


@dataclass
class KPIConfig:
    name: str
    column: str
    value_type: str  # currency, percentage, ratio
    yoy_column: Optional[str]
    yoy_type: str  # pct or diff
    required_columns: Iterable[str]
    description: str


@dataclass
class KPIResult:
    name: str
    value: Optional[float]
    value_type: str
    yoy: Optional[float]
    yoy_type: str
    sparkline: pd.Series
    missing_columns: List[str]
    description: str


KPI_CONFIGS: List[KPIConfig] = [
    KPIConfig(
        name="売上高",
        column=REVENUE_COL,
        value_type="currency",
        yoy_column="売上高YoY",
        yoy_type="pct",
        required_columns=[REVENUE_COL],
        description="年間平均の売上高（百万円）",
    ),
    KPIConfig(
        name="営業利益",
        column=OPERATING_PROFIT_COL,
        value_type="currency",
        yoy_column="営業利益YoY",
        yoy_type="pct",
        required_columns=[OPERATING_PROFIT_COL],
        description="営業活動による利益（百万円）",
    ),
    KPIConfig(
        name="EBITDA",
        column="EBITDA",
        value_type="currency",
        yoy_column="EBITDAYoY",
        yoy_type="pct",
        required_columns=[OPERATING_PROFIT_COL, "減価償却費合計"],
        description="営業利益と減価償却費の合計（百万円）",
    ),
    KPIConfig(
        name="自己資本比率",
        column="自己資本比率",
        value_type="percentage",
        yoy_column="自己資本比率Diff",
        yoy_type="diff",
        required_columns=[EQUITY_COL, TOTAL_ASSETS_COL],
        description="純資産 ÷ 総資産",
    ),
    KPIConfig(
        name="総資本回転率",
        column="総資本回転率",
        value_type="ratio",
        yoy_column="総資本回転率Diff",
        yoy_type="diff",
        required_columns=[REVENUE_COL, TOTAL_ASSETS_COL],
        description="売上高 ÷ 総資産",
    ),
    KPIConfig(
        name="労働生産性",
        column="労働生産性",
        value_type="currency",
        yoy_column="労働生産性YoY",
        yoy_type="pct",
        required_columns=[VALUE_ADDED_COL, "FTE"],
        description="FTE 当たり付加価値額（百万円）",
    ),
    KPIConfig(
        name="労働分配率",
        column="労働分配率",
        value_type="percentage",
        yoy_column="労働分配率Diff",
        yoy_type="diff",
        required_columns=[PERSONNEL_COST_COL, VALUE_ADDED_COL],
        description="人件費 ÷ 付加価値額",
    ),
    KPIConfig(
        name="経常利益率",
        column="経常利益率",
        value_type="percentage",
        yoy_column="経常利益率Diff",
        yoy_type="diff",
        required_columns=[ORDINARY_PROFIT_COL, REVENUE_COL],
        description="経常利益 ÷ 売上高",
    ),
]


def combine_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    existing = [col for col in columns if col in df.columns]
    if not existing:
        return pd.Series(np.nan, index=df.index)
    return df[existing].sum(axis=1, skipna=True)




def get_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series(np.nan, index=df.index)

def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan)


def calculate_cagr(series: pd.Series, periods: int = 3) -> pd.Series:
    ratio = series / series.shift(periods)
    result = np.where(ratio <= 0, np.nan, np.power(ratio, 1 / periods) - 1)
    return pd.Series(result, index=series.index)


def compute_yearly_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "集計年" not in df.columns:
        return pd.DataFrame(columns=["集計年"])
    numeric_cols = df.select_dtypes(include=["number", "float", "int"]).columns
    grouped = (
        df.groupby("集計年")[numeric_cols]
        .mean(numeric_only=True)
        .sort_index()
        .reset_index()
    )
    grouped["集計年"] = grouped["集計年"].astype(int)
    return grouped


def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df = df.sort_values("集計年")

    df["減価償却費合計"] = combine_columns(df, DEPRECIATION_COLUMNS)
    df["FTE"] = combine_columns(df, FTE_COLUMNS)
    df["有利子負債"] = combine_columns(df, INTEREST_BEARING_DEBT_COLUMNS)

    df["EBITDA"] = get_series(df, OPERATING_PROFIT_COL) + df["減価償却費合計"]
    df["EBITDA"] = df["EBITDA"].replace({np.inf: np.nan, -np.inf: np.nan})

    df["売上高YoY"] = get_series(df, REVENUE_COL).pct_change()
    df["営業利益YoY"] = get_series(df, OPERATING_PROFIT_COL).pct_change()
    df["EBITDAYoY"] = df["EBITDA"].pct_change()

    df["売上高CAGR3Y"] = calculate_cagr(get_series(df, REVENUE_COL), periods=3)

    df["総利益率"] = safe_divide(get_series(df, GROSS_PROFIT_COL), get_series(df, REVENUE_COL))
    df["営業利益率"] = safe_divide(get_series(df, OPERATING_PROFIT_COL), get_series(df, REVENUE_COL))
    df["経常利益率"] = safe_divide(get_series(df, ORDINARY_PROFIT_COL), get_series(df, REVENUE_COL))

    depreciation = df["減価償却費合計"].fillna(0)
    interest = get_series(df, INTEREST_COL)
    df["インタレスト・カバレッジ"] = safe_divide(df["EBITDA"], interest)

    df["自己資本比率"] = safe_divide(get_series(df, EQUITY_COL), get_series(df, TOTAL_ASSETS_COL))
    df["自己資本比率Diff"] = df["自己資本比率"].diff()

    total_debt = df["有利子負債"]
    df["有利子負債依存度"] = safe_divide(total_debt, get_series(df, TOTAL_ASSETS_COL))

    df["総資本回転率"] = safe_divide(get_series(df, REVENUE_COL), get_series(df, TOTAL_ASSETS_COL))
    df["総資本回転率Diff"] = df["総資本回転率"].diff()

    df["労働生産性"] = safe_divide(get_series(df, VALUE_ADDED_COL), df["FTE"])
    df["労働生産性YoY"] = df["労働生産性"].pct_change()

    df["労働分配率"] = safe_divide(get_series(df, PERSONNEL_COST_COL), get_series(df, VALUE_ADDED_COL))
    df["労働分配率Diff"] = df["労働分配率"].diff()

    df["経常利益率Diff"] = df["経常利益率"].diff()

    df["ROE"] = safe_divide(get_series(df, NET_PROFIT_COL), get_series(df, EQUITY_COL))
    df["純利益率"] = safe_divide(get_series(df, NET_PROFIT_COL), get_series(df, REVENUE_COL))
    df["レバレッジ"] = safe_divide(get_series(df, TOTAL_ASSETS_COL), get_series(df, EQUITY_COL))

    df["常用雇用者"] = df.get("常用雇用者")
    df["正社員以外換算"] = df.get("合計_正社員・正職員以外（就業時間換算人数）")
    df["派遣等合計"] = df.get("他社からの出向従業者（出向役員を含む）及び派遣従業者の合計")

    return df


def build_kpi_results(df: pd.DataFrame) -> List[KPIResult]:
    if df.empty:
        return []
    results: List[KPIResult] = []
    available_columns = set(df.columns)
    latest = df.iloc[-1]
    for config in KPI_CONFIGS:
        missing = [col for col in config.required_columns if col not in available_columns]
        value = latest.get(config.column) if not missing else np.nan
        yoy_series = df.get(config.yoy_column) if config.yoy_column else None
        yoy_value = yoy_series.iloc[-1] if yoy_series is not None and len(yoy_series) else np.nan
        sparkline_series = df.set_index("集計年").get(config.column, pd.Series(dtype=float))
        if isinstance(sparkline_series, pd.Series):
            sparkline_series = sparkline_series.dropna()
        results.append(
            KPIResult(
                name=config.name,
                value=None if pd.isna(value) else float(value),
                value_type=config.value_type,
                yoy=None if pd.isna(yoy_value) else float(yoy_value),
                yoy_type=config.yoy_type,
                sparkline=sparkline_series,
                missing_columns=missing,
                description=config.description,
            )
        )
    return results


def build_metric_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    columns = [
        "集計年",
        REVENUE_COL,
        OPERATING_PROFIT_COL,
        ORDINARY_PROFIT_COL,
        NET_PROFIT_COL,
        "EBITDA",
        "総利益率",
        "営業利益率",
        "経常利益率",
        "自己資本比率",
        "有利子負債依存度",
        "総資本回転率",
        "労働生産性",
        "労働分配率",
        "インタレスト・カバレッジ",
        "ROE",
        "売上高YoY",
        "営業利益YoY",
        "売上高CAGR3Y",
    ]
    existing = [col for col in columns if col in df.columns]
    return df[existing].copy()
