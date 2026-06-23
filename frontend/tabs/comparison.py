"""
tabs/comparison.py — Вкладка «Сравнение датасетов».

Полный отчёт: структура, качество, числовой дрифт, категориальный дрифт.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from helpers import API_BASE_URL, safe_post, df_to_records
from explanations import (
    COLUMN_HELP, LATEX,
    help_pvalue, help_psi, help_ks, help_mean, help_delta,
    help_verdict, help_shape_drifted, help_chi2, help_cramers_v,
)

COMPARE_DATASETS_URL = f"{API_BASE_URL}/compare/datasets"


def render(df: pd.DataFrame) -> None:
    """Отрисовывает содержимое вкладки «Сравнение датасетов»."""

    st.header("Сравнение двух датасетов")

    if not st.session_state.get("compare_mode", False):
        st.info(
            "Включите **«Режим сравнения»** в боковой панели, "
            "чтобы загрузить второй датасет и провести сравнительный анализ."
        )
        return

    if st.session_state.get("comp_df") is None:
        st.info(
            "Загрузите второй файл (Dataset B) через боковую панель "
            "в разделе «Режим сравнения»."
        )
        return

    comp_df = st.session_state["comp_df"]

    # --- Поиск совпадающих колонок ---
    num_a = set(df.select_dtypes(include="number").columns)
    num_b = set(comp_df.select_dtypes(include="number").columns)
    common_num_cols = sorted(num_a & num_b)

    cat_a = set(df.select_dtypes(include="object").columns)
    cat_b = set(comp_df.select_dtypes(include="object").columns)
    common_cat_cols = sorted(cat_a & cat_b)

    all_common_cols = sorted(set(df.columns) & set(comp_df.columns))

    info_parts = []
    if common_num_cols:
        info_parts.append(
            f"**{len(common_num_cols)}** числовых: {', '.join(common_num_cols)}"
        )
    if common_cat_cols:
        info_parts.append(
            f"**{len(common_cat_cols)}** категориальных: {', '.join(common_cat_cols)}"
        )
    if info_parts:
        st.info("Совпадающие колонки — " + " | ".join(info_parts))
    else:
        st.warning("Совпадающих колонок не найдено. Структурный отчёт всё равно доступен.")

    # --- Выбор ID-колонки ---
    valid_id_cols = [
        c for c in all_common_cols
        if (
            pd.api.types.is_integer_dtype(df[c])
            or pd.api.types.is_object_dtype(df[c])
        )
        and not pd.api.types.is_datetime64_any_dtype(df[c])
    ]
    id_col_options = ["(нет)"] + valid_id_cols
    id_column_choice = st.selectbox(
        "ID-колонка для парного сравнения (необязательно)",
        id_col_options,
        index=0,
        key="compare_id_col",
        help=(
            "Если в обоих датасетах есть общий идентификатор (ID клиента, "
            "номер записи и т.д.), выберите его для парных тестов (paired t-test / "
            "Wilcoxon). Если не указано — используются независимые тесты."
        ),
    )
    selected_id_column = None if id_column_choice == "(нет)" else id_column_choice

    # --- Кнопка запуска ---
    if st.button("Запустить сравнение", key="run_compare"):
        payload = {}
        fid_a = st.session_state.get("file_id")
        fid_b = st.session_state.get("file_id_b")
        if fid_a:
            payload["file_id_a"] = fid_a
        else:
            payload["data_a"] = df_to_records(df)
        if fid_b:
            payload["file_id_b"] = fid_b
        else:
            payload["data_b"] = df_to_records(comp_df)
        if selected_id_column is not None:
            payload["id_column"] = selected_id_column

        with st.spinner("Выполняется сравнение датасетов..."):
            resp = safe_post(COMPARE_DATASETS_URL, payload)

        if resp.status_code == 200:
            st.session_state["compare_result"] = resp.json()
        elif resp.status_code == 422:
            st.error(resp.json().get("detail", "Ошибка"))
        else:
            st.error("Ошибка API. Проверьте, запущен ли сервер.")

    # --- Отображение результатов ---
    compare_result = st.session_state.get("compare_result")
    if compare_result is None:
        return

    structure = compare_result["structure_report"]
    quality = compare_result["quality_report"]
    stat_comparison = compare_result["statistical_comparison"]
    cat_drift = compare_result["categorical_drift"]
    results = stat_comparison["results"]

    if selected_id_column is not None and results:
        any_paired = any(r["is_paired"] for r in results)
        if not any_paired:
            st.warning(
                "Парное сравнение невозможно: совпадений по ID не найдено. "
                "Используются независимые тесты."
            )

    # ==================== СТРУКТУРНЫЙ ОТЧЁТ ====================
    with st.expander("Структурный отчёт", expanded=True):
        s1, s2, s3 = st.columns(3)
        with s1:
            st.metric("Строк (база A)", f"{structure['rows_base']:,}",
                      help="Количество строк в базовом датасете A.")
        with s2:
            st.metric("Строк (сравнение B)", f"{structure['rows_compare']:,}",
                      help="Количество строк в сравниваемом датасете B.")
        with s3:
            delta_pct = (
                f"{structure['rows_delta_percent']:+.1f}%"
                if structure["rows_delta_percent"] is not None
                else "—"
            )
            st.metric(
                "Изменение строк",
                f"{structure['rows_delta']:+,}",
                delta=delta_pct,
                help="Разница в количестве строк между датасетами B и A.",
            )

        if structure["added_columns"]:
            st.success(
                f"Добавленные колонки ({len(structure['added_columns'])}): "
                f"{', '.join(structure['added_columns'])}"
            )
        if structure["removed_columns"]:
            st.error(
                f"Удалённые колонки ({len(structure['removed_columns'])}): "
                f"{', '.join(structure['removed_columns'])}"
            )
        if structure["type_changed_columns"]:
            type_msgs = []
            for col, info in structure["type_changed_columns"].items():
                type_msgs.append(
                    f"**{col}**: {info['base']} → {info['compare']}"
                )
            st.warning(
                "Изменённые типы колонок: " + " | ".join(type_msgs)
            )
        if (not structure["added_columns"]
                and not structure["removed_columns"]
                and not structure["type_changed_columns"]):
            st.info("Структура датасетов идентична.")

    # ==================== КАЧЕСТВО ДАННЫХ ====================
    if quality["columns"]:
        with st.expander("Отчёт о качестве данных"):
            q_rows = []
            for q in quality["columns"]:
                q_rows.append({
                    "Колонка": q["column"],
                    "Пропуски A (%)": f"{q['missing_base_pct']:.1f}%",
                    "Пропуски B (%)": f"{q['missing_compare_pct']:.1f}%",
                    "Δ пропусков (п.п.)": f"{q['missing_delta_pct']:+.1f}",
                    "Деградация": "Да" if q["quality_degraded"] else "Нет",
                    "Новые категории": (
                        ", ".join(q["new_categories"][:5])
                        + ("..." if len(q["new_categories"]) > 5 else "")
                        if q["new_categories"] else "—"
                    ),
                })
            q_df = pd.DataFrame(q_rows)

            def _style_quality(row):
                if row["Деградация"] == "Да":
                    return ["background-color: #fff3cd; color: #212529"] * len(row)
                return [""] * len(row)

            st.dataframe(
                q_df.style.apply(_style_quality, axis=1),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Деградация": st.column_config.Column(help=COLUMN_HELP["Деградация"]),
                },
            )

    # ==================== ЧИСЛОВОЙ ДРИФТ ====================
    if results:
        st.subheader("Числовой дрифт (PSI / KS / статистические тесты)")

        # Уведомление о FDR-коррекции
        correction_method = stat_comparison.get("correction_method")
        if correction_method == "benjamini-hochberg":
            st.info(
                "Для снижения риска ложных открытий при множественных сравнениях "
                "p-значения скорректированы методом Бенджамини-Хохберга (FDR)."
            )

        # Уведомление об отмене парного теста из-за дубликатов ID
        if results and results[0].get("paired_cancelled_reason"):
            st.warning(results[0]["paired_cancelled_reason"])

        polarity_options = ["Нейтрально", "Рост = хорошо", "Падение = хорошо"]
        col_polarities = {}
        with st.expander("Настройка бизнес-полярности колонок"):
            st.caption(
                "Укажите, как интерпретировать изменения: рост — "
                "это хорошо или плохо? Это влияет на цвет строк в таблице."
            )
            pol_cols = st.columns(min(len(results), 4))
            for idx, r in enumerate(results):
                with pol_cols[idx % len(pol_cols)]:
                    col_polarities[r["column"]] = st.radio(
                        r["column"],
                        polarity_options,
                        index=0,
                        key=f"pol_{r['column']}",
                        horizontal=True,
                    )

        sorted_results = sorted(results, key=lambda x: x["psi"], reverse=True)

        summary_rows = []
        for r in sorted_results:
            summary_rows.append({
                "Признак": r["column"],
                "PSI": f"{r['psi']:.4f}",
                "PSI (оценка)": r["psi_interpretation"],
                "KS p-value": f"{r['ks_p_value']:.4f}",
                "Среднее (A)": f"{r['mean_a']:.4f}",
                "Среднее (B)": f"{r['mean_b']:.4f}",
                "Изменение %": (
                    f"{r['delta_percent']:+.2f}%"
                    if r["delta_percent"] is not None
                    else "—"
                ),
                "p-value": (
                    f"{r['p_value']:.4f}"
                    if r["p_value"] is not None
                    else "—"
                ),
                "Скорр. p-value": (
                    f"{r['p_value_corrected']:.4f}"
                    if r.get("p_value_corrected") is not None
                    else "—"
                ),
                "Парный": "Да" if r["is_paired"] else "Нет",
                "Вердикт": r["verdict"],
            })

        summary_df = pd.DataFrame(summary_rows)

        _PSI_COLS = {"PSI", "PSI (оценка)"}

        def _style_drift_row(row):
            col_name = row["Признак"]
            psi_label = row["PSI (оценка)"]
            verdict = row["Вердикт"]
            polarity = col_polarities.get(col_name, "Нейтрально")

            row_bg = ""
            is_growth = verdict == "Значимый рост"
            is_decline = verdict == "Значимое падение"
            is_small = verdict == "Различия значимы, но эффект мал"

            if is_small:
                row_bg = "background-color: #d1ecf1; color: #212529"
            elif polarity == "Рост = хорошо":
                if is_growth:
                    row_bg = "background-color: #d4edda; color: #212529"
                elif is_decline:
                    row_bg = "background-color: #f8d7da; color: #212529"
            elif polarity == "Падение = хорошо":
                if is_decline:
                    row_bg = "background-color: #d4edda; color: #212529"
                elif is_growth:
                    row_bg = "background-color: #f8d7da; color: #212529"
            else:
                if is_growth:
                    row_bg = "background-color: #d4edda; color: #212529"
                elif is_decline:
                    row_bg = "background-color: #f8d7da; color: #212529"

            if not row_bg and verdict == "Различия отсутствуют":
                row_bg = "color: #9e9e9e"

            styles = []
            psi_highlight = (
                "background-color: #fff3cd; color: #212529"
                if psi_label == "Значительный дрифт"
                else ""
            )
            for col_label in row.index:
                if col_label in _PSI_COLS and psi_highlight:
                    styles.append(psi_highlight)
                else:
                    styles.append(row_bg)
            return styles

        styled = summary_df.style.apply(_style_drift_row, axis=1)
        st.dataframe(
            styled,
            use_container_width=True,
            hide_index=True,
            column_config={
                "PSI": st.column_config.Column(help=COLUMN_HELP["PSI"]),
                "PSI (оценка)": st.column_config.Column(help=COLUMN_HELP["PSI (оценка)"]),
                "KS p-value": st.column_config.Column(help=COLUMN_HELP["KS p-value"]),
                "Вердикт": st.column_config.Column(help=COLUMN_HELP["Вердикт"]),
                "Скорр. p-value": st.column_config.Column(help="p-значение после FDR-коррекции методом Бенджамини-Хохберга. Используется для контроля доли ложных открытий при множественных сравнениях."),
                "Парный": st.column_config.Column(help=COLUMN_HELP["Парный"]),
            },
        )

        st.divider()

        # --- Детализация по числовой колонке ---
        st.subheader("Детализация по числовой колонке")

        selected_col = st.selectbox(
            "Выберите колонку для визуализации",
            [r["column"] for r in results],
            key="compare_detail_col",
        )

        if selected_col:
            col_result = next(
                r for r in results if r["column"] == selected_col
            )

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Среднее (A)", f"{col_result['mean_a']:.4f}",
                          help=help_mean(col_result["mean_a"], "A"))
            with m2:
                st.metric("Среднее (B)", f"{col_result['mean_b']:.4f}",
                          help=help_mean(col_result["mean_b"], "B"))
            with m3:
                st.metric("Дельта", f"{col_result['delta']:.4f}",
                          help=help_delta(col_result["delta"], col_result.get("delta_percent")))
            with m4:
                st.metric("Вердикт", col_result["verdict"],
                          help=help_verdict(col_result["verdict"]))

            m5, m6, m7, m8 = st.columns(4)
            with m5:
                st.metric("PSI", f"{col_result['psi']:.4f}",
                          help=help_psi(col_result["psi"], col_result.get("psi_interpretation")))
            with m6:
                st.metric("PSI (оценка)", col_result["psi_interpretation"],
                          help="Текстовая интерпретация PSI: Незначительный / Умеренный / Значительный дрифт.")
            with m7:
                st.metric("KS p-value", f"{col_result['ks_p_value']:.4f}",
                          help=help_ks(col_result["ks_p_value"]))
            with m8:
                drift_label = "Да" if col_result["is_shape_drifted"] else "Нет"
                st.metric("Форма дрифтовала", drift_label,
                          help=help_shape_drifted(col_result["is_shape_drifted"]))

            if col_result["test_name"]:
                test_info = (
                    f"**Тест:** {col_result['test_name']} | "
                    f"**Статистика:** {col_result['statistic']:.4f} | "
                    f"**p-value:** {col_result['p_value']:.4f}"
                )
                if col_result.get("cohens_d") is not None:
                    _tn = col_result.get("test_name", "")
                    _es_name = "ранг-бисер. r" if "Манна-Уитни" in (_tn or "") else "d Коэна"
                    test_info += (
                        f" | **{_es_name}:** {col_result['cohens_d']:.4f}"
                    )
                st.info(test_info)

            if col_result["is_paired"]:
                st.success(
                    f"Использован **парный тест**: {col_result['paired_test_name']}"
                )

            # --- Графики ---
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=df[selected_col].dropna(),
                    name="Dataset A (база)",
                    opacity=0.5,
                    marker_color="#636EFA",
                    nbinsx=30,
                ))
                fig_hist.add_trace(go.Histogram(
                    x=comp_df[selected_col].dropna(),
                    name="Dataset B (сравнение)",
                    opacity=0.5,
                    marker_color="#EF553B",
                    nbinsx=30,
                ))
                fig_hist.update_layout(
                    title=f"Гистограмма: {selected_col}",
                    barmode="overlay",
                    template="plotly_white",
                    xaxis_title=selected_col,
                    yaxis_title="Частота",
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            with chart_col2:
                fig_box = go.Figure()
                fig_box.add_trace(go.Box(
                    y=df[selected_col].dropna(),
                    name="Dataset A (база)",
                    marker_color="#636EFA",
                ))
                fig_box.add_trace(go.Box(
                    y=comp_df[selected_col].dropna(),
                    name="Dataset B (сравнение)",
                    marker_color="#EF553B",
                ))
                fig_box.update_layout(
                    title=f"Box Plot: {selected_col}",
                    template="plotly_white",
                    yaxis_title=selected_col,
                )
                st.plotly_chart(fig_box, use_container_width=True)

            st.caption(
                "Для вставки в ВКР: наведите на график и нажмите "
                "кнопку камеры (Download plot as PNG) в панели инструментов."
            )
    else:
        st.info("Числовых колонок для сравнения не найдено.")

    # ==================== КАТЕГОРИАЛЬНЫЙ ДРИФТ ====================
    cat_results = cat_drift.get("columns", [])
    if cat_results:
        st.divider()
        st.subheader("Категориальный дрифт (χ²-тест)")

        cat_rows = []
        for cr in cat_results:
            cat_rows.append({
                "Колонка": cr["column"],
                "χ² стат.": f"{cr['chi2_stat']:.4f}",
                "p-value": f"{cr['chi2_p_value']:.4f}",
                "V Крамера": f"{cr['cramers_v']:.4f}",
                "Дрифт": "Да" if cr["is_drifted"] else "Нет",
            })
        cat_df = pd.DataFrame(cat_rows)

        def _style_cat_drift(row):
            if row["Дрифт"] == "Да":
                return ["background-color: #fff3cd; color: #212529"] * len(row)
            return [""] * len(row)

        st.dataframe(
            cat_df.style.apply(_style_cat_drift, axis=1),
            use_container_width=True,
            hide_index=True,
            column_config={
                "χ² стат.": st.column_config.Column(help=COLUMN_HELP["χ² стат."]),
                "V Крамера": st.column_config.Column(help=COLUMN_HELP["V Крамера"]),
                "Дрифт": st.column_config.Column(help=COLUMN_HELP["Дрифт"]),
            },
        )

        # --- Детализация категориальной колонки ---
        st.subheader("Детализация категориальной колонки")

        selected_cat_col = st.selectbox(
            "Выберите категориальную колонку",
            [cr["column"] for cr in cat_results],
            key="compare_cat_detail_col",
        )

        if selected_cat_col:
            cat_result = next(
                cr for cr in cat_results
                if cr["column"] == selected_cat_col
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("χ² статистика", f"{cat_result['chi2_stat']:.4f}",
                          help=help_chi2(cat_result["chi2_stat"], cat_result["chi2_p_value"]))
            with c2:
                st.metric("p-value", f"{cat_result['chi2_p_value']:.4f}",
                          help=help_pvalue(cat_result["chi2_p_value"], "однородность категорий"))
            with c3:
                st.metric("V Крамера", f"{cat_result['cramers_v']:.4f}",
                          help=help_cramers_v(cat_result["cramers_v"]))
            st.latex(LATEX["chi_squared"])
            st.latex(LATEX["cramers_v"])

            if cat_result.get("cochran_warning"):
                st.warning(cat_result["cochran_warning"])

            if cat_result["is_drifted"]:
                st.warning(
                    f"Обнаружен значимый дрифт (p = {cat_result['chi2_p_value']:.4f} < 0.05). "
                    f"Распределение категорий в колонке **{selected_cat_col}** "
                    f"статистически значимо изменилось."
                )
            else:
                st.success(
                    f"Дрифт не обнаружен (p = {cat_result['chi2_p_value']:.4f} ≥ 0.05). "
                    f"Распределение категорий в колонке **{selected_cat_col}** "
                    f"не изменилось значимо."
                )

            base_props = cat_result["base_proportions"]
            comp_props = cat_result["compare_proportions"]
            all_categories = sorted(
                set(list(base_props.keys()) + list(comp_props.keys()))
            )

            fig_cat = go.Figure()
            fig_cat.add_trace(go.Bar(
                x=all_categories,
                y=[base_props.get(c, 0) for c in all_categories],
                name="Dataset A (база)",
                marker_color="#636EFA",
            ))
            fig_cat.add_trace(go.Bar(
                x=all_categories,
                y=[comp_props.get(c, 0) for c in all_categories],
                name="Dataset B (сравнение)",
                marker_color="#EF553B",
            ))
            fig_cat.update_layout(
                title=f"Распределение категорий: {selected_cat_col}",
                barmode="group",
                template="plotly_white",
                xaxis_title="Категория",
                yaxis_title="Доля",
                yaxis_tickformat=".0%",
            )
            st.plotly_chart(fig_cat, use_container_width=True)

            st.caption(
                "Для вставки в ВКР: наведите на график и нажмите "
                "кнопку камеры (Download plot as PNG) в панели инструментов."
            )
