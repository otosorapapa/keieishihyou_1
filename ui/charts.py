from __future__ import annotations

from typing import Optional, TYPE_CHECKING, Any

import importlib.util

import numpy as np
import pandas as pd

PLOTLY_IMPORT_ERROR: ModuleNotFoundError | None

_graph_objects_spec = importlib.util.find_spec("plotly.graph_objects")
_subplots_spec = importlib.util.find_spec("plotly.subplots")

if _graph_objects_spec is not None and _subplots_spec is not None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
    PLOTLY_IMPORT_ERROR = None
else:  # pragma: no cover - executed when Plotly is unavailable
    HAS_PLOTLY = False
    PLOTLY_IMPORT_ERROR = ModuleNotFoundError(
        "Plotly is not installed. Install plotly to enable interactive charts.",
        name="plotly",
    )
    go = Any  # type: ignore[assignment]

    def make_subplots(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("Plotly is required to create subplots.")

if TYPE_CHECKING:  # pragma: no cover - only for static analysis
    import plotly.graph_objects as go  # noqa: F401  # pylint: disable=unused-import

FONT_FAMILY = "Hiragino Kaku Gothic ProN, Hiragino Sans, Noto Sans JP, Meiryo, sans-serif"
PRIMARY_COLOR = "#2c7be5"
SECONDARY_COLOR = "#d6336c"
TERTIARY_COLOR = "#2f9e44"
GREY_MAJOR = "#6c757d"
GREY_OVERALL = "#adb5bd"


def _base_layout(fig: "go.Figure", title: str) -> "go.Figure":
    fig.update_layout(
        title=title,
        template="plotly_white",
        font=dict(family=FONT_FAMILY),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=30, t=60, b=40),
        hovermode="x unified",
    )
    return fig


def _line(
    fig: "go.Figure",
    x,
    y,
    name: str,
    color: str,
    secondary_y: bool = False,
    dash: Optional[str] = None,
) -> None:
    trace = go.Scatter(
        x=x,
        y=y,
        name=name,
        mode="lines+markers",
        line=dict(color=color, width=2, dash=dash),
    )
    if secondary_y:
        fig.add_trace(trace, secondary_y=True)
    else:
        fig.add_trace(trace)


def sales_and_profit_chart(
    current: pd.DataFrame,
    major: pd.DataFrame,
    overall: pd.DataFrame,
) -> Optional["go.Figure"]:
    if not HAS_PLOTLY:
        return None
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if not current.empty:
        x = current["集計年"]
        _line(fig, x, current["売上高（百万円）"], "売上高", PRIMARY_COLOR)
        _line(fig, x, current["営業利益（百万円）"], "営業利益", SECONDARY_COLOR, secondary_y=True)
        _line(fig, x, current["経常利益（百万円）"], "経常利益", TERTIARY_COLOR, secondary_y=True)
    if not major.empty:
        x = major["集計年"]
        _line(fig, x, major.get("売上高（百万円）"), "大分類平均 売上高", GREY_MAJOR, dash="dot")
        _line(fig, x, major.get("営業利益（百万円）"), "大分類平均 営業利益", GREY_MAJOR, secondary_y=True, dash="dot")
        _line(fig, x, major.get("経常利益（百万円）"), "大分類平均 経常利益", GREY_MAJOR, secondary_y=True, dash="dot")
    if not overall.empty:
        x = overall["集計年"]
        _line(fig, x, overall.get("売上高（百万円）"), "全体平均 売上高", GREY_OVERALL, dash="dash")
        _line(fig, x, overall.get("営業利益（百万円）"), "全体平均 営業利益", GREY_OVERALL, secondary_y=True, dash="dash")
        _line(fig, x, overall.get("経常利益（百万円）"), "全体平均 経常利益", GREY_OVERALL, secondary_y=True, dash="dash")
    fig.update_yaxes(title_text="売上高（百万円）", secondary_y=False)
    fig.update_yaxes(title_text="利益（百万円）", secondary_y=True)
    return _base_layout(fig, "売上高と利益の推移")


def profitability_chart(
    current: pd.DataFrame,
    major: pd.DataFrame,
    overall: pd.DataFrame,
) -> Optional["go.Figure"]:
    if not HAS_PLOTLY:
        return None
    fig = go.Figure()
    if not current.empty:
        x = current["集計年"]
        _line(fig, x, current.get("総利益率"), "総利益率", PRIMARY_COLOR)
        _line(fig, x, current.get("営業利益率"), "営業利益率", SECONDARY_COLOR)
        _line(fig, x, current.get("経常利益率"), "経常利益率", TERTIARY_COLOR)
    if not major.empty:
        x = major["集計年"]
        _line(fig, x, major.get("総利益率"), "大分類平均", GREY_MAJOR, dash="dot")
    if not overall.empty:
        x = overall["集計年"]
        _line(fig, x, overall.get("総利益率"), "全体平均", GREY_OVERALL, dash="dash")
    fig.update_yaxes(tickformat=".0%")
    return _base_layout(fig, "利益率の推移")


def _stacked_bar(
    x,
    y,
    name: str,
    color: str,
    row: int,
    col: int,
    fig: "go.Figure",
    orientation: str = "h",
) -> None:
    fig.add_trace(
        go.Bar(
            x=x if orientation == "h" else y,
            y=y if orientation == "h" else x,
            name=name,
            orientation=orientation,
            marker_color=color,
        ),
        row=row,
        col=col,
    )


def balance_sheet_structure_chart(current: pd.DataFrame) -> Optional["go.Figure"]:
    if not HAS_PLOTLY:
        return None
    if current.empty:
        return go.Figure()
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_y=True,
        horizontal_spacing=0.1,
        subplot_titles=("資産構成", "負債・純資産構成"),
    )
    years = current["集計年"]
    total_assets = current.get("資産（百万円）")
    current_assets = current.get("流動資産（百万円）")
    fixed_assets = current.get("固定資産（百万円）")
    current_liabilities = current.get("流動負債（百万円）")
    fixed_liabilities = current.get("固定負債（百万円）")
    equity = current.get("純資産（百万円）")

    assets_ratio = (current_assets / total_assets).replace([np.inf, -np.inf], np.nan)
    fixed_ratio = (fixed_assets / total_assets).replace([np.inf, -np.inf], np.nan)

    current_ratio = (current_liabilities / total_assets).replace([np.inf, -np.inf], np.nan)
    fixed_l_ratio = (fixed_liabilities / total_assets).replace([np.inf, -np.inf], np.nan)
    equity_ratio = (equity / total_assets).replace([np.inf, -np.inf], np.nan)

    _stacked_bar(assets_ratio, years, "流動資産", PRIMARY_COLOR, 1, 1, fig)
    _stacked_bar(fixed_ratio, years, "固定資産", SECONDARY_COLOR, 1, 1, fig)
    _stacked_bar(current_ratio, years, "流動負債", PRIMARY_COLOR, 1, 2, fig)
    _stacked_bar(fixed_l_ratio, years, "固定負債", SECONDARY_COLOR, 1, 2, fig)
    _stacked_bar(equity_ratio, years, "純資産", TERTIARY_COLOR, 1, 2, fig)

    fig.update_layout(barmode="stack")
    fig.update_xaxes(range=[0, 1], tickformat=".0%", row=1, col=1)
    fig.update_xaxes(range=[0, 1], tickformat=".0%", row=1, col=2)
    fig.update_yaxes(type="category", autorange="reversed")
    return _base_layout(fig, "バランスシート構成比")


def ebitda_interest_chart(
    current: pd.DataFrame,
    major: pd.DataFrame,
    overall: pd.DataFrame,
) -> Optional["go.Figure"]:
    if not HAS_PLOTLY:
        return None
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if not current.empty:
        x = current["集計年"]
        fig.add_trace(
            go.Bar(x=x, y=current.get("EBITDA"), name="EBITDA", marker_color=PRIMARY_COLOR),
            secondary_y=False,
        )
        fig.add_trace(
            go.Bar(
                x=x,
                y=current.get("支払利息・割引料（百万円）"),
                name="利払",
                marker_color=SECONDARY_COLOR,
                opacity=0.6,
            ),
            secondary_y=False,
        )
        _line(fig, x, current.get("インタレスト・カバレッジ"), "Interest Coverage", TERTIARY_COLOR, secondary_y=True)
    if not major.empty:
        x = major["集計年"]
        _line(fig, x, major.get("インタレスト・カバレッジ"), "大分類平均 Coverage", GREY_MAJOR, secondary_y=True, dash="dot")
    if not overall.empty:
        x = overall["集計年"]
        _line(fig, x, overall.get("インタレスト・カバレッジ"), "全体平均 Coverage", GREY_OVERALL, secondary_y=True, dash="dash")
    fig.update_yaxes(title_text="金額（百万円）", secondary_y=False)
    fig.update_yaxes(title_text="倍率", secondary_y=True)
    return _base_layout(fig, "EBITDA と利払負担")


def productivity_distribution_chart(
    current: pd.DataFrame,
    major: pd.DataFrame,
    overall: pd.DataFrame,
) -> Optional["go.Figure"]:
    if not HAS_PLOTLY:
        return None
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if not current.empty:
        x = current["集計年"]
        _line(fig, x, current.get("労働生産性"), "労働生産性", PRIMARY_COLOR)
        _line(fig, x, current.get("労働分配率"), "労働分配率", SECONDARY_COLOR, secondary_y=True)
    if not major.empty:
        x = major["集計年"]
        _line(fig, x, major.get("労働生産性"), "大分類平均 生産性", GREY_MAJOR, dash="dot")
        _line(fig, x, major.get("労働分配率"), "大分類平均 分配率", GREY_MAJOR, secondary_y=True, dash="dot")
    if not overall.empty:
        x = overall["集計年"]
        _line(fig, x, overall.get("労働生産性"), "全体平均 生産性", GREY_OVERALL, dash="dash")
        _line(fig, x, overall.get("労働分配率"), "全体平均 分配率", GREY_OVERALL, secondary_y=True, dash="dash")
    fig.update_yaxes(title_text="労働生産性（百万円/人）", secondary_y=False)
    fig.update_yaxes(title_text="労働分配率", secondary_y=True, tickformat=".0%")
    return _base_layout(fig, "労働生産性と労働分配率")


def dupont_chart(
    current: pd.DataFrame,
    major: pd.DataFrame,
    overall: pd.DataFrame,
) -> Optional["go.Figure"]:
    if not HAS_PLOTLY:
        return None
    fig = go.Figure()
    if not current.empty:
        x = current["集計年"]
        _line(fig, x, current.get("純利益率"), "純利益率", PRIMARY_COLOR)
        _line(fig, x, current.get("総資本回転率"), "総資本回転率", SECONDARY_COLOR)
        _line(fig, x, current.get("レバレッジ"), "レバレッジ", TERTIARY_COLOR)
    if not major.empty:
        x = major["集計年"]
        _line(fig, x, major.get("純利益率"), "大分類平均 純利益率", GREY_MAJOR, dash="dot")
        _line(fig, x, major.get("総資本回転率"), "大分類平均 総資本回転率", GREY_MAJOR, dash="dot")
        _line(fig, x, major.get("レバレッジ"), "大分類平均 レバレッジ", GREY_MAJOR, dash="dot")
    if not overall.empty:
        x = overall["集計年"]
        _line(fig, x, overall.get("純利益率"), "全体平均 純利益率", GREY_OVERALL, dash="dash")
        _line(fig, x, overall.get("総資本回転率"), "全体平均 総資本回転率", GREY_OVERALL, dash="dash")
        _line(fig, x, overall.get("レバレッジ"), "全体平均 レバレッジ", GREY_OVERALL, dash="dash")
    fig.update_yaxes(title_text="値")
    return _base_layout(fig, "DuPont 分解 (ROE 要因)")
