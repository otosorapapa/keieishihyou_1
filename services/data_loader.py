from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from .duckdb_store import DuckDBStore
from .encoding import detect_bytes

KEY_COLUMNS = [
    "集計年",
    "産業大分類コード",
    "産業大分類名",
    "業種中分類コード",
    "業種中分類名",
    "集計形式",
]


@st.cache_data(show_spinner=False)
def detect_encoding(path: str | Path, sample_size: int = 1_000_000) -> str:
    """Detect file encoding prioritising cp932."""
    path = Path(path)
    raw = path.read_bytes()[:sample_size]
    guess = detect_bytes(raw)
    encoding = guess.get("encoding") or ""
    if encoding:
        encoding = encoding.lower()
    if encoding in {"shift_jis", "cp932", "sjis"}:
        return "cp932"
    if encoding:
        return encoding
    return "cp932"


def _read_csv_with_fallback(path: str | Path, encoding: Optional[str]) -> pd.DataFrame:
    errors = "ignore" if encoding and "cp932" in encoding.lower() else "strict"
    try:
        return pd.read_csv(path, encoding=encoding, dtype=str, na_values=["", "-"])
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8-sig", dtype=str, na_values=["", "-"])


@st.cache_data(show_spinner=False)
def load_csv_data(path: str | Path) -> pd.DataFrame:
    """Load CSV data with encoding detection and type conversion."""
    path = Path(path)
    encoding = detect_encoding(path)
    df = _read_csv_with_fallback(path, encoding)
    return preprocess_dataframe(df)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    df = df.copy()
    if "集計年" in df.columns:
        df["集計年"] = pd.to_numeric(df["集計年"], errors="coerce").astype("Int64")

    for col in df.columns:
        if col in KEY_COLUMNS:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_dataset(csv_path: str | Path, db_path: str | Path = "app.duckdb") -> pd.DataFrame:
    """Load dataset from DuckDB, initialising from CSV if necessary."""
    store = DuckDBStore(db_path)
    base_df = load_csv_data(csv_path)
    if not base_df.empty:
        store.upsert_dataframe(base_df, KEY_COLUMNS)
    stored_df = store.fetch_all()
    if stored_df.empty:
        return base_df
    return stored_df


def upsert_uploaded_file(data: bytes, db_path: str | Path = "app.duckdb") -> pd.DataFrame:
    """Upsert data from an uploaded CSV file and return the refreshed dataset."""
    if not data:
        return pd.DataFrame()

    detected = detect_bytes(data)
    encoding = (detected.get("encoding") or "").lower()
    if encoding in {"shift_jis", "cp932", "sjis"}:
        encoding = "cp932"
    elif not encoding:
        encoding = "cp932"

    buffer = io.BytesIO(data)
    try:
        df = pd.read_csv(buffer, encoding=encoding, dtype=str, na_values=["", "-"])
    except UnicodeDecodeError:
        buffer.seek(0)
        df = pd.read_csv(buffer, encoding="utf-8-sig", dtype=str, na_values=["", "-"])

    df = preprocess_dataframe(df)
    if df.empty:
        return df

    store = DuckDBStore(db_path)
    store.upsert_dataframe(df, KEY_COLUMNS)
    return store.fetch_all()


def get_major_options(df: pd.DataFrame) -> list[tuple[str, str]]:
    majors = (
        df[["産業大分類コード", "産業大分類名"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["産業大分類コード", "産業大分類名"])
    )
    return [
        (row["産業大分類コード"], row["産業大分類名"])
        for _, row in majors.iterrows()
    ]


def get_mid_options(df: pd.DataFrame, major_code: str) -> list[tuple[str, str]]:
    filtered = df[df["産業大分類コード"] == major_code]
    mids = (
        filtered[["業種中分類コード", "業種中分類名"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["業種中分類コード", "業種中分類名"])
    )
    return [
        (str(row["業種中分類コード"]), row["業種中分類名"])
        for _, row in mids.iterrows()
    ]


def filter_dataset(
    df: pd.DataFrame,
    major_code: str,
    mid_name: str,
) -> pd.DataFrame:
    if df.empty:
        return df
    mask = (df["産業大分類コード"] == major_code) & (df["業種中分類名"] == mid_name)
    return df[mask].copy()


def get_year_bounds(df: pd.DataFrame) -> tuple[int, int]:
    if df.empty or "集計年" not in df.columns:
        return (0, 0)
    years = df["集計年"].dropna().astype(int)
    return int(years.min()), int(years.max())
