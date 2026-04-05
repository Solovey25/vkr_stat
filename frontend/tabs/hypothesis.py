"""
tabs/hypothesis.py — Вкладка «Сравнение выборок».

Автоматический выбор теста (Стьюдент / Уэлч / Манна-Уитни),
проверка предпосылок, размер эффекта.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from helpers import API_BASE_URL, safe_post, data_payload
from explanations import (
    LATEX, format_decision_step,
    help_pvalue, help_cohens_d, help_rank_biserial,
    help_normality, help_levene, help_statistic,
)

INFERENCE_COMPARE_URL = f"{API_BASE_URL}/inference/compare"


def render(df: pd.DataFrame) -> None:
    """Отрисовывает содержимое вкладки «Сравнение выборок»."""

    st.header("Сравнение двух выборок")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("Для сравнения необходимо минимум два числовых столбца.")
        return

    col1, col2 = st.columns(2)

    with col1:
        column_a = st.selectbox(
            "Первая выборка (столбец)",
            numeric_cols,
            key="hyp_col_a",
        )
    with col2:
        column_b = st.selectbox(
            "Вторая выборка (столбец)",
            numeric_cols,
            index=min(1, len(numeric_cols) - 1),
            key="hyp_col_b",
        )

    if st.button("Выполнить тест", key="run_test"):
        payload = {
            **data_payload(),
            "column_a": column_a,
            "column_b": column_b,
        }
        with st.spinner("Выполняется сравнение выборок..."):
            response = safe_post(INFERENCE_COMPARE_URL, payload)

        if response.status_code == 200:
            result = response.json()
            assumptions = result["assumptions"]

            st.subheader(f"Тест: {result['test_name']}")
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric("Статистика", f"{result['statistic']:.4f}",
                          help=help_statistic(result["statistic"], result["test_name"]))
                st.metric("Нормальность A", "Да" if assumptions["is_norm_a"] else "Нет",
                          help=help_normality(assumptions["is_norm_a"],
                                              assumptions.get("norm_test_name"),
                                              assumptions.get("shapiro_a_p")))
            with res_col2:
                st.metric("p-значение", f"{result['p_value']:.4f}",
                          help=help_pvalue(result["p_value"], "различие средних"))
                st.metric("Нормальность B", "Да" if assumptions["is_norm_b"] else "Нет",
                          help=help_normality(assumptions["is_norm_b"],
                                              assumptions.get("norm_test_name"),
                                              assumptions.get("shapiro_b_p")))

            if result.get("effect_size") is not None:
                if result.get("effect_size_metric") == "rank_biserial":
                    _es_label = "Размер эффекта (ранг-бисериальная r)"
                    _es_help = help_rank_biserial(result["effect_size"])
                else:
                    _es_label = "Размер эффекта (d Коэна)"
                    _es_help = help_cohens_d(result["effect_size"])
                st.metric(
                    _es_label,
                    f"{result['effect_size']:.4f}",
                    help=_es_help,
                )

            st.info(f"**Научное заключение:** {result['conclusion']}")

            with st.expander("Ход анализа"):
                for step in result.get("decision_chain", []):
                    st.markdown(format_decision_step(step))
                _test = result["test_name"]
                if "Стьюдент" in _test:
                    st.latex(LATEX["t_test"])
                elif "Уэлч" in _test:
                    st.latex(LATEX["welch"])
                elif "Манна-Уитни" in _test:
                    st.latex(LATEX["mann_whitney"])

            with st.expander("Проверка предпосылок"):
                _norm_test = assumptions.get("norm_test_name")
                p_col1, p_col2 = st.columns(2)
                with p_col1:
                    if _norm_test is None:
                        st.caption("Тест нормальности не проводился (N < 8)")
                    else:
                        st.metric(
                            f"{_norm_test} (A)",
                            f"p = {assumptions['shapiro_a_p']:.4f}",
                            help=help_normality(assumptions["is_norm_a"],
                                                _norm_test, assumptions["shapiro_a_p"]),
                        )
                        st.metric(
                            f"{_norm_test} (B)",
                            f"p = {assumptions['shapiro_b_p']:.4f}",
                            help=help_normality(assumptions["is_norm_b"],
                                                _norm_test, assumptions["shapiro_b_p"]),
                        )
                        st.latex(LATEX["shapiro_wilk"])
                with p_col2:
                    if assumptions.get("levene_p") is not None:
                        st.metric(
                            "Тест Левене",
                            f"p = {assumptions['levene_p']:.4f}",
                            help=help_levene(assumptions["equal_variances"],
                                             assumptions["levene_p"]),
                        )
                        st.metric(
                            "Равенство дисперсий",
                            "Да" if assumptions["equal_variances"] else "Нет",
                            help="Результат теста Левене: равны ли дисперсии двух выборок.",
                        )
                    else:
                        st.caption("Тест Левене не проводился (N < 8)")

            fig_box = go.Figure()
            fig_box.add_trace(go.Box(y=df[column_a].dropna(), name=column_a))
            fig_box.add_trace(go.Box(y=df[column_b].dropna(), name=column_b))
            fig_box.update_layout(title="Сравнение распределений (Box Plot)")
            st.plotly_chart(fig_box, use_container_width=True)
            st.caption("Для вставки в ВКР: наведите на график и нажмите кнопку камеры (Download plot as PNG) в панели инструментов.")
        else:
            st.error(f"Ошибка API: {response.json().get('detail', 'Неизвестная ошибка')}")
