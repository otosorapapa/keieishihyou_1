from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import plotly.graph_objects as go
import streamlit as st

from services.metrics import KPIResult

FONT_FAMILY = "Hiragino Kaku Gothic ProN, Hiragino Sans, Noto Sans JP, Meiryo, sans-serif"


def load_css(path: str | Path) -> None:
    css_path = Path(path)
    if not css_path.exists():
        return
    with open(css_path, "r", encoding="utf-8") as file:
        st.markdown(f"<style>{file.read()}</style>", unsafe_allow_html=True)


def format_value(value: Optional[float], value_type: str) -> str:
    if value is None:
        return "—"
    if value_type == "currency":
        return f"{value:,.0f} 百万円"
    if value_type == "percentage":
        return f"{value * 100:.1f}%"
    if value_type == "ratio":
        return f"{value:.2f}x"
    return f"{value:,.2f}"


def format_yoy(yoy: Optional[float], value_type: str, yoy_type: str) -> str:
    if yoy is None:
        return "—"
    if yoy_type == "pct":
        return f"{yoy * 100:+.1f}%"
    if yoy_type == "diff" and value_type == "percentage":
        return f"{yoy * 100:+.1f}pt"
    return f"{yoy:+.2f}"


def sparkline(series: Iterable[float], color: str = "#2c7be5") -> go.Figure:
    fig = go.Figure()
    if series is not None:
        fig.add_trace(
            go.Scatter(
                x=list(series.index) if hasattr(series, "index") else list(range(len(series))),
                y=list(series),
                mode="lines",
                line=dict(color=color, width=2),
                hovertemplate="%{y:,}<extra></extra>",
            )
        )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=60,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_FAMILY),
    )
    return fig


def render_kpi_cards(kpis: Iterable[KPIResult], columns_per_row: int = 4) -> None:
    kpis = list(kpis)
    if not kpis:
        return
    for i in range(0, len(kpis), columns_per_row):
        row = kpis[i : i + columns_per_row]
        cols = st.columns(len(row))
        for col, kpi in zip(cols, row):
            with col:
                value_text = format_value(kpi.value, kpi.value_type)
                yoy_text = format_yoy(kpi.yoy, kpi.value_type, kpi.yoy_type)
                if kpi.yoy is None:
                    badge_class = ""
                    icon = "—"
                else:
                    badge_class = "up" if kpi.yoy >= 0 else "down"
                    icon = "▲" if kpi.yoy >= 0 else "▼"
                if kpi.missing_columns:
                    tooltip_text = "未計算: " + " / ".join(kpi.missing_columns)
                else:
                    tooltip_text = kpi.description
                tooltip_icon = "<span title=\"{}\">ℹ️</span>".format(tooltip_text.replace(""", "'"))
                badge_html = (
                    f"<span class='kpi-badge {badge_class}'>{icon} {yoy_text}</span>"
                    if badge_class
                    else f"<span class='kpi-badge'>{yoy_text}</span>"
                )
                st.markdown(
                    f"<div class='kpi-card'>"
                    f"<div class='kpi-title'>{kpi.name} {tooltip_icon}</div>"
                    f"<div class='kpi-value'>{value_text}</div>"
                    f"<div class='kpi-meta'>{badge_html}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                if kpi.sparkline is not None and len(kpi.sparkline) > 1:
                    st.plotly_chart(
                        sparkline(kpi.sparkline),
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )
                else:
                    st.write(" ")


def show_toast(message: str, icon: str = "ℹ️") -> None:
    st.toast(f"{icon} {message}")


def download_chart_button(fig: go.Figure, label: str, key: str) -> None:
    try:
        image_bytes = fig.to_image(format="png", width=1280, height=720, scale=2)
        st.download_button(
            label=label,
            data=image_bytes,
            file_name=f"{key}.png",
            mime="image/png",
            key=key,
            help="PNG 画像をダウンロード",
        )
    except Exception as exc:  # noqa: BLE001
        st.warning(f"画像の生成に失敗しました: {exc}")
