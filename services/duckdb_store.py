from __future__ import annotations

from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd


class DuckDBStore:
    def __init__(self, db_path: str | Path = "app.duckdb", table_name: str = "financials") -> None:
        self.db_path = Path(db_path)
        self.table_name = table_name
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.db_path))

    def fetch_all(self) -> pd.DataFrame:
        with self._connect() as conn:
            table_exists = conn.execute(
                """
                SELECT count(*)
                FROM information_schema.tables
                WHERE table_schema = 'main'
                  AND table_name = ?
                """,
                [self.table_name],
            ).fetchone()[0]
            if not table_exists:
                return pd.DataFrame()
            df = conn.execute(f"SELECT * FROM {self.table_name}").df()
        return df

    def upsert_dataframe(self, df: pd.DataFrame, key_columns: Iterable[str]) -> None:
        if df.empty:
            return
        key_columns = list(key_columns)
        all_columns = df.columns.tolist()
        with self._connect() as conn:
            table_exists = conn.execute(
                """
                SELECT count(*)
                FROM information_schema.tables
                WHERE table_schema = 'main'
                  AND table_name = ?
                """,
                [self.table_name],
            ).fetchone()[0]
            conn.register("incoming_df", df)
            if not table_exists:
                conn.execute(
                    f"CREATE TABLE {self.table_name} AS SELECT * FROM incoming_df LIMIT 0"
                )
            key_condition = " AND ".join(
                f"target.{col} <=> incoming.{col}" for col in key_columns
            )
            delete_sql = (
                f"DELETE FROM {self.table_name} AS target USING incoming_df AS incoming "
                f"WHERE {key_condition}"
            )
            conn.execute(delete_sql)
            insert_cols = ", ".join(all_columns)
            conn.execute(
                f"INSERT INTO {self.table_name} ({insert_cols}) "
                f"SELECT {insert_cols} FROM incoming_df"
            )
            conn.unregister("incoming_df")

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute(f"DROP TABLE IF EXISTS {self.table_name}")
