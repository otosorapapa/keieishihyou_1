from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
import streamlit as st

try:
    from st_aggrid import AgGrid, GridOptionsBuilder  # type: ignore[import]
    HAS_AGGRID = True
except ModuleNotFoundError:
    AgGrid = None  # type: ignore[assignment]
    GridOptionsBuilder = None  # type: ignore[assignment]
    HAS_AGGRID = False

from services import aggregations, data_loader, metrics
from ui import charts, components

DATA_PATH = Path("/mnt/data/産業構造マップ_中小企業経営分析_推移　bs pl　従業員数.csv")
DEFAULT_MAJOR_CODE = "D"
DEFAULT_MID_NAME = "設備工事業"
DB_PATH = Path("app.duckdb")


def filter_by_year(df: pd.DataFrame, year_range: Tuple[int, int]) -> pd.DataFrame:
    if df.empty or "集計年" not in df.columns:
        return df
    start, end = year_range
    return df[(df["集計年"] >= start) & (df["集計年"] <= end)]


def update_query_params(major: str, mid: str, year_range: Tuple[int, int]) -> None:
    try:
        params = st.query_params
        params["maj"] = major
        params["mid"] = mid
        params["y1"] = str(year_range[0])
        params["y2"] = str(year_range[1])
    except Exception:
        st.experimental_set_query_params(maj=major, mid=mid, y1=year_range[0], y2=year_range[1])


def get_initial_params() -> Tuple[str | None, str | None, Tuple[int | None, int | None]]:
    try:
        params = st.query_params
        get_value = params.get
    except Exception:
        params = st.experimental_get_query_params()
        get_value = params.get
    major_param = get_value("maj")
    mid_param = get_value("mid")
    y1_param = get_value("y1")
    y2_param = get_value("y2")

    def parse_param(value):
        if value is None:
            return None
        if isinstance(value, list):
            return value[0] if value else None
        return value

    major = parse_param(major_param)
    mid = parse_param(mid_param)
    y1 = parse_param(y1_param)
    y2 = parse_param(y2_param)

    start = int(y1) if y1 and str(y1).isdigit() else None
    end = int(y2) if y2 and str(y2).isdigit() else None
    return major, mid, (start, end)


def render_table(table_df: pd.DataFrame) -> None:
    if table_df.empty:
        st.info("年次 KPI テーブルを表示できるデータがありません。")
        return
    if not HAS_AGGRID or AgGrid is None or GridOptionsBuilder is None:
        st.warning(
            "インタラクティブな表の表示には streamlit-aggrid のインストールが必要です。"
            "簡易表示モードでテーブルを表示しています。"
        )
        st.dataframe(table_df)
    else:
        grid_builder = GridOptionsBuilder.from_dataframe(table_df)
        grid_builder.configure_default_column(filter=True, sortable=True, resizable=True)
        grid_builder.configure_pagination(paginationAutoPageSize=False, paginationPageSize=10)
        grid_options = grid_builder.build()
        AgGrid(table_df, gridOptions=grid_options, height=320, theme="balham")
    csv = table_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "CSV ダウンロード",
        data=csv,
        file_name="kpi_table.csv",
        mime="text/csv",
        use_container_width=False,
    )


def main() -> None:
    st.set_page_config(page_title="中小企業経営分析ダッシュボード", layout="wide")
    components.load_css("assets/styles.css")

    st.title("中小企業 経営分析ダッシュボード")

    if not DATA_PATH.exists():
        st.error("初期データファイルが見つかりません。管理者にお問い合わせください。")
        return

    data = data_loader.load_dataset(DATA_PATH, DB_PATH)

    if data.empty:
        st.warning("データが読み込めませんでした。CSV を確認してください。")
        return

    initial_major, initial_mid, (initial_start, initial_end) = get_initial_params()

    major_options = data_loader.get_major_options(data)
    major_codes = [code for code, _ in major_options]

    if initial_major and initial_major in major_codes:
        default_major = initial_major
    elif DEFAULT_MAJOR_CODE in major_codes:
        default_major = DEFAULT_MAJOR_CODE
    else:
        default_major = major_codes[0]

    mid_options = data_loader.get_mid_options(data, default_major)
    mid_names = [name for _, name in mid_options]

    if initial_mid and initial_mid in mid_names:
        default_mid = initial_mid
    elif DEFAULT_MID_NAME in mid_names:
        default_mid = DEFAULT_MID_NAME
    else:
        default_mid = mid_names[0] if mid_names else ""

    min_year, max_year = data_loader.get_year_bounds(data)
    if initial_start and initial_end:
        year_range_default = (
            max(min_year, initial_start),
            min(max_year, initial_end),
        )
    else:
        year_range_default = (max(min_year, max_year - 4), max_year)

    top_cols = st.columns([1.2, 1.2, 2.2, 1.2])
    with top_cols[0]:
        major_selection = st.selectbox(
            "産業大分類コード",
            options=major_options,
            index=major_codes.index(default_major) if default_major in major_codes else 0,
            format_func=lambda opt: f"{opt[0]} : {opt[1]}",
            key="major-select",
        )[0]
    mid_options = data_loader.get_mid_options(data, major_selection)
    mid_display = [f"{name} ({code})" for code, name in mid_options]
    mid_lookup = {f"{name} ({code})": name for code, name in mid_options}
    default_mid_display = next((label for label, name in mid_lookup.items() if name == default_mid), mid_display[0] if mid_display else "")
    with top_cols[1]:
        mid_selection_label = st.selectbox(
            "業種中分類",
            options=mid_display,
            index=mid_display.index(default_mid_display) if default_mid_display in mid_display else 0,
            key="mid-select",
        )
        mid_selection = mid_lookup[mid_selection_label]
    with top_cols[2]:
        year_range = st.slider(
            "表示年範囲",
            min_value=int(min_year),
            max_value=int(max_year),
            value=(int(year_range_default[0]), int(year_range_default[1])),
            step=1,
        )
    with top_cols[3]:
        if st.button("データ更新", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        uploaded = st.file_uploader("データ取り込み", type=["csv"], label_visibility="collapsed")
        if uploaded is not None:
            updated = data_loader.upsert_uploaded_file(uploaded.getvalue(), DB_PATH)
            if updated.empty:
                components.show_toast("取り込めるデータが見つかりませんでした", icon="⚠️")
            else:
                components.show_toast("データを取り込みました", icon="✅")
                st.rerun()

    update_query_params(major_selection, mid_selection, year_range)

    filtered = data_loader.filter_dataset(data, major_selection, mid_selection)
    if filtered.empty:
        components.show_toast("選択された組合せのデータは未登録です", icon="⚠️")
        suggestions = [label for label in mid_display if mid_lookup[label] != mid_selection]
        if suggestions:
            st.info("代替候補: " + "、".join(suggestions[:5]))
        return

    current_yearly = metrics.compute_yearly_aggregates(filtered)
    current_metrics = metrics.calculate_metrics(current_yearly)
    current_metrics_range = filter_by_year(current_metrics, year_range)

    major_average = aggregations.get_major_average(data, major_selection)
    overall_average = aggregations.get_overall_average(data)

    major_metrics_range = filter_by_year(major_average, year_range)
    overall_metrics_range = filter_by_year(overall_average, year_range)

    kpis = metrics.build_kpi_results(current_metrics_range)
    components.render_kpi_cards(kpis)

    st.subheader("主要指標のトレンド")

    sales_profit_fig = charts.sales_and_profit_chart(
        current_metrics_range,
        major_metrics_range,
        overall_metrics_range,
    )
    st.plotly_chart(sales_profit_fig, use_container_width=True)
    components.download_chart_button(sales_profit_fig, "PNG ダウンロード", "sales_profit")

    profitability_fig = charts.profitability_chart(
        current_metrics_range,
        major_metrics_range,
        overall_metrics_range,
    )
    st.plotly_chart(profitability_fig, use_container_width=True)
    components.download_chart_button(profitability_fig, "PNG ダウンロード", "profitability")

    bs_fig = charts.balance_sheet_structure_chart(current_metrics_range)
    st.plotly_chart(bs_fig, use_container_width=True)
    components.download_chart_button(bs_fig, "PNG ダウンロード", "balance_sheet")

    ebitda_fig = charts.ebitda_interest_chart(
        current_metrics_range,
        major_metrics_range,
        overall_metrics_range,
    )
    st.plotly_chart(ebitda_fig, use_container_width=True)
    components.download_chart_button(ebitda_fig, "PNG ダウンロード", "ebitda_interest")

    productivity_fig = charts.productivity_distribution_chart(
        current_metrics_range,
        major_metrics_range,
        overall_metrics_range,
    )
    st.plotly_chart(productivity_fig, use_container_width=True)
    components.download_chart_button(productivity_fig, "PNG ダウンロード", "productivity")

    dupont_fig = charts.dupont_chart(
        current_metrics_range,
        major_metrics_range,
        overall_metrics_range,
    )
    st.plotly_chart(dupont_fig, use_container_width=True)
    components.download_chart_button(dupont_fig, "PNG ダウンロード", "dupont")

    st.subheader("年次 KPI 一覧")
    table_df = metrics.build_metric_table(current_metrics_range)
    render_table(table_df)


if __name__ == "__main__":
    main()
