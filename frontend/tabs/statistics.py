"""
tabs/statistics.py — Вкладка «Описательная статистика».

Секции: расширенные метрики, анализ распределения, ECDF, корреляции.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from helpers import API_BASE_URL, safe_post, data_payload
from explanations import COLUMN_HELP

# URL-адреса эндпоинтов
STATS_EXTENDED_URL = f"{API_BASE_URL}/stats/extended"
STATS_FIT_DIST_URL = f"{API_BASE_URL}/stats/fit-distribution"
STATS_CORR_URL = f"{API_BASE_URL}/stats/correlation"


def render(df: pd.DataFrame) -> None:
    """Отрисовывает содержимое вкладки «Описательная статистика»."""

    st.header("Описательная статистика")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # ==============================================================
    # Секция 1: Общий обзор — расширенные метрики
    # ==============================================================
    st.subheader("Общий обзор")

    payload_stats = {**data_payload()}
    resp_stats = safe_post(STATS_EXTENDED_URL, payload_stats)

    if resp_stats.status_code == 200:
        stats_data = resp_stats.json()["columns"]

        stats_df = pd.DataFrame(stats_data).T

        is_constant_flags = stats_df.get("is_constant", pd.Series(False, index=stats_df.index))

        display_df = stats_df.drop(columns=["is_constant"], errors="ignore")

        display_df.columns = [
            {
                "count": "Кол-во",
                "mean": "Среднее",
                "median": "Медиана",
                "std": "СКО",
                "min": "Мин",
                "max": "Макс",
                "q25": "Q1 (25%)",
                "q75": "Q3 (75%)",
                "skewness": "Асимметрия",
                "kurtosis": "Эксцесс",
                "sem": "SEM",
            }.get(c, c)
            for c in display_df.columns
        ]
        display_df.index.name = "Столбец"

        interpretations = []
        for idx, row in display_df.iterrows():
            if is_constant_flags.get(idx, False):
                interpretations.append("Константа (std = 0)")
                continue

            parts = []
            skew = row.get("Асимметрия", 0)
            kurt = row.get("Эксцесс", 0)

            if skew is not None:
                if skew > 0.5:
                    parts.append("Правосторонняя асимм.")
                elif skew < -0.5:
                    parts.append("Левосторонняя асимм.")
                else:
                    parts.append("Симметрично")

            if kurt is not None:
                if kurt > 0.5:
                    parts.append("Островершинное")
                elif kurt < -0.5:
                    parts.append("Плосковершинное")
                else:
                    parts.append("Мезокуртич.")

            interpretations.append("; ".join(parts) if parts else "—")

        display_df["Интерпретация"] = interpretations

        st.dataframe(
            display_df,
            use_container_width=True,
            column_config={
                "СКО": st.column_config.Column(help=COLUMN_HELP["СКО"]),
                "Асимметрия": st.column_config.Column(help=COLUMN_HELP["Асимметрия"]),
                "Эксцесс": st.column_config.Column(help=COLUMN_HELP["Эксцесс"]),
                "SEM": st.column_config.Column(help=COLUMN_HELP["SEM"]),
                "Интерпретация": st.column_config.Column(help=COLUMN_HELP["Интерпретация"]),
            },
        )
    else:
        st.error("Ошибка при расчёте расширенных статистик.")

    st.divider()

    # ==============================================================
    # Секция 2: Анализ распределения
    # ==============================================================
    st.subheader("Анализ распределения")

    selected_col = st.selectbox(
        "Выберите столбец для анализа распределения",
        numeric_cols,
        key="dist_col",
    )

    if selected_col:
        payload_fit = {**data_payload(), "column": selected_col}
        resp_fit = safe_post(STATS_FIT_DIST_URL, payload_fit)

        if resp_fit.status_code == 200:
            fit_data = resp_fit.json()

            fig_hist = px.histogram(
                df,
                x=selected_col,
                nbins=30,
                histnorm="probability density",
                title=f"Распределение: {selected_col}",
                labels={selected_col: selected_col, "y": "Плотность"},
                opacity=0.7,
            )
            fig_hist.update_layout(bargap=0.05, template="plotly_white")

            if fit_data["pdf_curve_x"] and fit_data["pdf_curve_y"]:
                dist_label = fit_data["best_distribution_ru"] or fit_data["best_distribution"]
                fig_hist.add_trace(go.Scatter(
                    x=fit_data["pdf_curve_x"],
                    y=fit_data["pdf_curve_y"],
                    mode="lines",
                    line=dict(color="#EF553B", width=2.5),
                    name=f"PDF ({dist_label})",
                ))

            st.plotly_chart(fig_hist, use_container_width=True)
            st.caption("Для вставки в ВКР: наведите на график и нажмите кнопку камеры (Download plot as PNG) в панели инструментов.")

            if fit_data["best_distribution"]:
                dist_name_ru = fit_data["best_distribution_ru"]
                p_val = fit_data["best_p_value"]
                ks_stat = fit_data["best_ks_statistic"]

                st.info(
                    f"Данные наиболее соответствуют **{dist_name_ru}** распределению "
                    f"(p = {p_val:.4f} по критерию Колмогорова-Смирнова, "
                    f"KS-статистика = {ks_stat:.4f})."
                )

                with st.expander("Все протестированные распределения"):
                    _dist_ru = {
                        "norm": "Нормальное",
                        "lognorm": "Логнормальное",
                        "expon": "Экспоненциальное",
                        "poisson": "Пуассона",
                    }
                    fit_table = []
                    for r in fit_data["all_results"]:
                        fit_table.append({
                            "Распределение": _dist_ru.get(r["distribution"], r["distribution"]),
                            "KS-статистика": f"{r['ks_statistic']:.4f}",
                            "p-значение": f"{r['p_value']:.4f}",
                            "Лучшее": "***" if r["distribution"] == fit_data["best_distribution"] else "",
                        })
                    if fit_table:
                        st.dataframe(
                            pd.DataFrame(fit_table),
                            use_container_width=True,
                            hide_index=True,
                        )
            else:
                st.warning("Не удалось подобрать распределение для данной колонки.")
        else:
            st.error("Ошибка при подборе распределения.")

    st.divider()

    # ==============================================================
    # Секция 3: ECDF
    # ==============================================================
    st.subheader("Эмпирическая функция распределения (ECDF)")

    ecdf_col = st.selectbox(
        "Выберите столбец для ECDF",
        numeric_cols,
        key="ecdf_col",
    )

    if ecdf_col:
        fig_ecdf = px.ecdf(
            df,
            x=ecdf_col,
            title=f"ECDF: {ecdf_col}",
            labels={ecdf_col: ecdf_col, "y": "Кумулятивная вероятность"},
        )
        fig_ecdf.update_layout(template="plotly_white")
        st.plotly_chart(fig_ecdf, use_container_width=True)
        st.caption("Для вставки в ВКР: наведите на график и нажмите кнопку камеры (Download plot as PNG) в панели инструментов.")

    st.divider()

    # ==============================================================
    # Секция 4: Тепловая карта корреляций
    # ==============================================================
    st.subheader("Матрица корреляций")

    corr_method = st.selectbox(
        "Метод корреляции",
        ["pearson", "spearman"],
        format_func=lambda x: {
            "pearson": "Пирсон (линейная)",
            "spearman": "Спирмен (ранговая)",
        }[x],
        key="corr_method",
    )

    payload_corr = {**data_payload(), "method": corr_method}
    resp_corr = safe_post(STATS_CORR_URL, payload_corr)

    if resp_corr.status_code == 200:
        corr_data = resp_corr.json()
        columns = corr_data["columns"]
        matrix = corr_data["matrix"]

        fig_heatmap = go.Figure(data=go.Heatmap(
            z=matrix,
            x=columns,
            y=columns,
            colorscale="RdBu_r",
            zmin=-1, zmax=1,
            text=np.round(matrix, 2).tolist(),
            texttemplate="%{text}",
            textfont={"size": 12},
            hoverongaps=False,
        ))
        method_ru = "Пирсон" if corr_method == "pearson" else "Спирмен"
        fig_heatmap.update_layout(
            title=f"Матрица корреляций ({method_ru})",
            template="plotly_white",
            height=500,
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        st.caption("Для вставки в ВКР: наведите на график и нажмите кнопку камеры (Download plot as PNG) в панели инструментов.")
    else:
        st.error("Ошибка при расчёте корреляций.")
