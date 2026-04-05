"""
tabs/preprocessing.py — Вкладка «Предобработка данных».

Секции: пропуски, выбросы, масштабирование, кодирование, экспорт, лог.
"""

from __future__ import annotations

import io

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from helpers import (
    API_BASE_URL, safe_post, data_payload,
    add_log_entry, refresh_metadata,
)
from explanations import (
    help_outlier_count, help_outlier_pct,
    help_quartile, help_iqr_bound,
)

# URL-адреса эндпоинтов предобработки
SANITIZE_MISSING_URL = f"{API_BASE_URL}/sanitize/missing"
SANITIZE_OUTLIERS_URL = f"{API_BASE_URL}/sanitize/outliers"
SANITIZE_REMOVE_OUTLIERS_URL = f"{API_BASE_URL}/sanitize/remove-outliers"


def render(df: pd.DataFrame) -> None:
    """Отрисовывает содержимое вкладки «Предобработка»."""

    st.header("Предобработка данных")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # ==============================================================
    # Секция 1: Обработка пропусков
    # ==============================================================
    st.subheader("Пропуски (NaN)")

    missing_data = []
    for col in df.columns:
        miss = int(df[col].isna().sum())
        if miss > 0:
            missing_data.append({
                "Столбец": col,
                "Пропуски": miss,
                "Процент": f"{miss / len(df) * 100:.1f}%",
                "Тип": str(df[col].dtype),
            })

    if missing_data:
        st.dataframe(
            pd.DataFrame(missing_data),
            use_container_width=True,
            hide_index=True,
        )

        miss_col1, miss_col2 = st.columns(2)
        with miss_col1:
            miss_method = st.selectbox(
                "Метод обработки",
                ["drop", "fill"],
                format_func=lambda x: {
                    "drop": "Удалить строки с пропусками",
                    "fill": "Заполнить пропуски",
                }[x],
                key="miss_method",
            )
        with miss_col2:
            fill_strategy = st.selectbox(
                "Стратегия заполнения",
                ["mean", "median", "most_frequent"],
                format_func=lambda x: {
                    "mean": "Среднее (mean)",
                    "median": "Медиана (median)",
                    "most_frequent": "Мода (most_frequent)",
                }[x],
                key="fill_strategy",
                disabled=(miss_method != "fill"),
            )

        if st.button("Применить очистку пропусков", key="apply_missing"):
            payload = {
                **data_payload(),
                "method": miss_method,
                "strategy": fill_strategy,
                "columns": None,
            }
            with st.spinner("Обработка пропусков..."):
                resp = safe_post(SANITIZE_MISSING_URL, payload)

            if resp.status_code == 200:
                result = resp.json()
                new_df = pd.DataFrame(result["data"])
                st.session_state["main_df"] = new_df
                refresh_metadata()

                affected = result["affected_count"]
                if miss_method == "drop":
                    add_log_entry(f"Удалено {affected} строк с пропусками")
                else:
                    add_log_entry(
                        f"Заполнено {affected} пропусков "
                        f"(стратегия: {fill_strategy})"
                    )
                st.rerun()
            else:
                st.error(f"Ошибка: {resp.json().get('detail', 'Неизвестная ошибка')}")
    else:
        st.success("Пропуски не обнаружены.")

    st.divider()

    # ==============================================================
    # Секция 2: Выбросы (метод IQR)
    # ==============================================================
    st.subheader("Выбросы (метод IQR)")

    if numeric_cols:
        outlier_col = st.selectbox(
            "Выберите числовую колонку для анализа выбросов",
            numeric_cols,
            key="outlier_col",
        )

        if outlier_col:
            _outlier_payload = {**data_payload(), "columns": [outlier_col]}
            _outlier_resp = safe_post(SANITIZE_OUTLIERS_URL, _outlier_payload)
            outlier_data = (
                _outlier_resp.json()["outliers"]
                if _outlier_resp.status_code == 200
                else None
            )

            if outlier_data is not None:

                if outlier_data:
                    info = outlier_data[0]

                    box_col, info_col = st.columns([2, 1])

                    with box_col:
                        fig_box = go.Figure()
                        fig_box.add_trace(go.Box(
                            y=df[outlier_col].dropna(),
                            name=outlier_col,
                            boxpoints="outliers",
                            marker=dict(color="#636EFA"),
                        ))
                        fig_box.update_layout(
                            title=f"Box Plot: {outlier_col}",
                            yaxis_title=outlier_col,
                            template="plotly_white",
                        )
                        st.plotly_chart(fig_box, use_container_width=True)
                        st.caption("Для вставки в ВКР: наведите на график и нажмите кнопку камеры (Download plot as PNG) в панели инструментов.")

                    with info_col:
                        st.metric("Выбросов", info["outliers_count"],
                                  help=help_outlier_count(info["outliers_count"], info["outliers_percent"]))
                        st.metric("Процент", f"{info['outliers_percent']}%",
                                  help=help_outlier_pct(info["outliers_percent"]))
                        st.metric("Q1", f"{info['q1']:.4f}",
                                  help=help_quartile(info["q1"], "Q1"))
                        st.metric("Q3", f"{info['q3']:.4f}",
                                  help=help_quartile(info["q3"], "Q3"))
                        st.metric("Нижняя граница", f"{info['lower_bound']:.4f}",
                                  help=help_iqr_bound(info["lower_bound"], is_lower=True))
                        st.metric("Верхняя граница", f"{info['upper_bound']:.4f}",
                                  help=help_iqr_bound(info["upper_bound"], is_lower=False))

                    if info["outliers_count"] > 0:
                        if st.button(
                            f"Удалить {info['outliers_count']} выбросов "
                            f"в «{outlier_col}»",
                            key="remove_outliers",
                        ):
                            payload_rm = {
                                **data_payload(),
                                "columns": [outlier_col],
                            }
                            with st.spinner("Удаление выбросов..."):
                                resp_rm = safe_post(
                                    SANITIZE_REMOVE_OUTLIERS_URL, payload_rm,
                                )

                            if resp_rm.status_code == 200:
                                result_rm = resp_rm.json()
                                st.session_state["main_df"] = pd.DataFrame(
                                    result_rm["data"]
                                )
                                refresh_metadata()
                                removed = result_rm["removed_count"]
                                add_log_entry(
                                    f"Удалено {removed} выбросов "
                                    f"в колонке «{outlier_col}»"
                                )
                                st.rerun()
                            else:
                                st.error("Ошибка при удалении выбросов.")
                    else:
                        st.success(f"Выбросы в «{outlier_col}» не обнаружены.")
            else:
                st.error("Ошибка при расчёте выбросов. Проверьте, запущен ли сервер.")
    else:
        st.warning("Нет числовых столбцов для анализа выбросов.")

    st.divider()

    # ==============================================================
    # Секция 3: Масштабирование
    # ==============================================================
    st.subheader("Масштабирование признаков")

    if numeric_cols:
        scale_cols = st.multiselect(
            "Выберите столбцы для масштабирования",
            numeric_cols,
            key="scale_cols",
        )

        scale_method = st.selectbox(
            "Метод масштабирования",
            ["standard", "minmax"],
            format_func=lambda x: {
                "standard": "StandardScaler (Z-score: μ=0, σ=1)",
                "minmax": "MinMaxScaler (диапазон [0, 1])",
            }[x],
            key="scale_method",
        )

        if scale_cols and st.button("Применить масштабирование", key="apply_scale"):
            payload_scale = {
                **data_payload(),
                "columns": scale_cols,
                "method": scale_method,
            }
            with st.spinner("Масштабирование..."):
                resp_scale = safe_post(
                    f"{API_BASE_URL}/sanitize/scale",
                    payload_scale,
                )

            if resp_scale.status_code == 200:
                scaled_df = pd.DataFrame(resp_scale.json()["data"])
                st.session_state["main_df"] = scaled_df
                refresh_metadata()
                add_log_entry(
                    f"Масштабирование ({scale_method}) применено "
                    f"к столбцам: {', '.join(scale_cols)}"
                )
                st.rerun()
            else:
                st.error("Ошибка при масштабировании.")
    else:
        st.warning("Нет числовых столбцов для масштабирования.")

    st.divider()

    # ==============================================================
    # Секция 4: Кодирование категориальных столбцов
    # ==============================================================
    st.subheader("Кодирование категорий")

    categorical_cols = df.select_dtypes(
        exclude=["number", "datetime"]
    ).columns.tolist()

    if categorical_cols:
        encode_cols = st.multiselect(
            "Выберите категориальные столбцы для кодирования",
            categorical_cols,
            key="encode_cols",
        )

        if encode_cols:
            for col in encode_cols:
                unique_vals = df[col].dropna().unique()[:10]
                st.caption(
                    f"**{col}**: {', '.join(str(v) for v in unique_vals)}"
                    + (" ..." if df[col].nunique() > 10 else "")
                )

            if st.button("Закодировать выбранные столбцы", key="apply_encode"):
                payload_enc = {
                    **data_payload(),
                    "columns": encode_cols,
                }
                with st.spinner("Кодирование категорий..."):
                    resp_enc = safe_post(
                        f"{API_BASE_URL}/sanitize/encode",
                        payload_enc,
                    )

                if resp_enc.status_code == 200:
                    enc_result = resp_enc.json()
                    encoded_df = pd.DataFrame(enc_result["data"])
                    st.session_state["main_df"] = encoded_df
                    refresh_metadata()

                    mapping = enc_result["mapping"]
                    details = []
                    for col_name, col_map in mapping.items():
                        pairs = [
                            f"{code} = {val}"
                            for code, val in col_map.items()
                        ]
                        details.append(f"{col_name}: {', '.join(pairs)}")
                    add_log_entry(
                        f"Закодированы столбцы: {'; '.join(details)}"
                    )
                    st.rerun()
                else:
                    st.error("Ошибка при кодировании категорий.")
    else:
        st.info("Нет категориальных столбцов для кодирования.")

    st.divider()

    # ==============================================================
    # Секция 5: Экспорт данных
    # ==============================================================
    st.subheader("Экспорт данных")

    export_col1, export_col2 = st.columns(2)

    with export_col1:
        csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="Скачать CSV",
            data=csv_bytes,
            file_name="данные.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with export_col2:
        xlsx_buffer = io.BytesIO()
        with pd.ExcelWriter(xlsx_buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Данные")
        st.download_button(
            label="Скачать XLSX",
            data=xlsx_buffer.getvalue(),
            file_name="данные.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    st.divider()

    # ==============================================================
    # Секция 6: История действий
    # ==============================================================
    st.subheader("История действий")

    log = st.session_state.get("processing_log", [])
    if log:
        for entry in reversed(log):
            st.text(entry)
    else:
        st.caption("Действия ещё не выполнялись.")
