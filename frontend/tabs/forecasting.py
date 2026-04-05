"""
tabs/forecasting.py — Вкладка «Прогнозирование».

Два подраздела: причинно-следственный анализ (регрессия) и анализ динамики (временные ряды).
"""

from __future__ import annotations

import json as _json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from helpers import API_BASE_URL, safe_post, data_payload
from explanations import (
    LATEX,
    help_pvalue, help_r2, help_r2_adj, help_mae, help_rmse,
    help_f_stat, help_aic, help_durbin_watson, help_shapiro_residuals,
    help_mape,
    COLUMN_HELP,
)

ANALYZE_URL = f"{API_BASE_URL}/analyze/regression"
FORECAST_URL = f"{API_BASE_URL}/forecast/timeseries"


def _interpret_r2(r2: float) -> tuple[str, str]:
    """Возвращает (текст интерпретации, цвет-метод) по значению R²."""
    if r2 >= 0.9:
        return "Очень сильная связь", "success"
    if r2 >= 0.7:
        return "Сильная связь", "success"
    if r2 >= 0.5:
        return "Умеренная связь", "warning"
    if r2 >= 0.3:
        return "Слабая связь", "warning"
    return "Связь практически отсутствует", "error"


def render(df: pd.DataFrame) -> None:
    """Отрисовывает содержимое вкладки «Прогнозирование»."""

    st.header("Прогнозирование")

    meta = st.session_state.get("metadata")

    sub_tab_regression, sub_tab_ts = st.tabs([
        "Причинно-следственный анализ (Регрессия)",
        "Анализ динамики (Временные ряды)",
    ])

    # ==================== СУБРОЗДЕЛ 1: РЕГРЕССИЯ ====================
    with sub_tab_regression:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        if len(numeric_cols) < 2:
            st.warning("Для регрессии необходимо минимум два числовых столбца.")
        else:
            st.subheader("Выбор переменных")

            target_col = st.selectbox(
                "Зависимая переменная (Y)",
                numeric_cols,
                index=min(1, len(numeric_cols) - 1),
                key="reg_target",
            )

            available_features = [c for c in numeric_cols if c != target_col]
            feature_cols = st.multiselect(
                "Независимые переменные (X)",
                available_features,
                default=available_features[:1],
                key="reg_features",
            )

            if not feature_cols:
                st.info("Выберите хотя бы одну независимую переменную.")
            else:
                if st.button("Построить модель", key="run_regression"):
                    payload = {
                        **data_payload(),
                        "target_column": target_col,
                        "feature_columns": feature_cols,
                    }

                    with st.spinner("Обучение модели..."):
                        response = safe_post(ANALYZE_URL, payload)

                    if response.status_code != 200:
                        detail = response.json().get("detail", "Неизвестная ошибка")
                        st.error(f"Ошибка API: {detail}")
                    else:
                        result = response.json()
                        st.session_state["reg_result"] = result
                        st.session_state["reg_saved_features"] = feature_cols
                        st.session_state["reg_saved_target"] = target_col

                if "reg_result" in st.session_state:
                    result = st.session_state["reg_result"]
                    ols = result["ols"]
                    skl = result["sklearn"]

                    st.divider()

                    # ---------- Блок 1: Сводные метрики ----------
                    st.subheader("Результаты модели")

                    r2_val = ols["r_squared"]
                    r2_adj = ols["r_squared_adj"]
                    interp_text, interp_type = _interpret_r2(r2_val)

                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("R\u00b2", f"{r2_val:.4f}", help=help_r2(r2_val))
                    with m2:
                        st.metric("R\u00b2 (скорр.)", f"{r2_adj:.4f}",
                                  help=help_r2_adj(r2_adj, r2_val))
                    with m3:
                        st.metric("MAE", f"{skl['mae']:.4f}", help=help_mae(skl["mae"]))
                    with m4:
                        st.metric("RMSE", f"{skl['rmse']:.4f}", help=help_rmse(skl["rmse"]))

                    getattr(st, interp_type)(
                        f"**Интерпретация R\u00b2 = {r2_val:.4f}:** {interp_text}"
                    )

                    info_parts = [
                        f"Модель: **{skl['model_name']}**",
                        f"Наблюдений: **{result['n_samples']}**",
                    ]
                    if result["small_sample"]:
                        info_parts.append(
                            "Малая выборка (<50) — использована **Ridge-регуляризация**"
                        )
                    st.info(" | ".join(info_parts))

                    cleaning = result["cleaning"]
                    if cleaning["dropped_rows"] > 0:
                        st.warning(
                            f"Очистка данных: удалено **{cleaning['dropped_rows']}** "
                            f"строк (NaN: {cleaning['dropped_nan']}, "
                            f"inf: {cleaning['dropped_inf']}). "
                            f"Осталось: {cleaning['cleaned_rows']} "
                            f"из {cleaning['original_rows']}."
                        )

                    # ---------- Блок 1.5: Адекватность модели ----------
                    st.subheader("Адекватность модели")

                    if ols.get("is_model_reliable"):
                        st.success(ols["reliability_text"])
                    else:
                        st.warning(ols["reliability_text"])

                    if ols.get("vif_warnings"):
                        for _vif_warn in ols["vif_warnings"]:
                            st.warning(_vif_warn)

                    rel_col1, rel_col2 = st.columns(2)
                    with rel_col1:
                        st.metric(
                            "Шапиро-Уилк (остатки)",
                            f"W = {ols['residuals_shapiro_stat']:.4f}",
                            help=help_shapiro_residuals(
                                ols["residuals_shapiro_stat"],
                                ols["residuals_shapiro_p"]),
                        )
                    with rel_col2:
                        st.metric(
                            "p-значение",
                            f"{ols['residuals_shapiro_p']:.4f}",
                            help=help_pvalue(ols["residuals_shapiro_p"],
                                             "нормальность остатков"),
                        )

                    # ---------- Блок 2: Таблица коэффициентов ----------
                    st.subheader("Коэффициенты модели (OLS)")

                    factors = ols["factor_stats"]
                    coef_data = []
                    for f in factors:
                        sig = ""
                        p = f["p_value"]
                        if p < 0.001:
                            sig = "***"
                        elif p < 0.01:
                            sig = "**"
                        elif p < 0.05:
                            sig = "*"

                        row = {
                            "Фактор": f["name"],
                            "Коэффициент": f"{f['coefficient']:.4f}",
                            "Стд. ошибка": f"{f['std_error']:.4f}",
                            "t-статистика": f"{f['t_statistic']:.4f}",
                            "p-значение": f"{p:.4f}",
                            "Значимость": sig,
                        }
                        if f.get("vif") is not None:
                            row["VIF"] = f"{f['vif']:.2f}"
                        else:
                            row["VIF"] = "—"
                        coef_data.append(row)

                    coef_df = pd.DataFrame(coef_data)
                    st.dataframe(
                        coef_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Стд. ошибка": st.column_config.Column(help=COLUMN_HELP["Стд. ошибка"]),
                            "t-статистика": st.column_config.Column(help=COLUMN_HELP["t-статистика"]),
                            "p-значение": st.column_config.Column(help=COLUMN_HELP["p-значение"]),
                            "Значимость": st.column_config.Column(help=COLUMN_HELP["Значимость"]),
                            "VIF": st.column_config.Column(help="Variance Inflation Factor — мера мультиколлинеарности. VIF = 1 = нет корреляции с другими факторами. VIF > 5 = умеренная, VIF > 10 = сильная мультиколлинеарность."),
                        },
                    )
                    st.caption(
                        "Значимость: \\*\\*\\* p<0.001, \\*\\* p<0.01, \\* p<0.05"
                    )

                    if ols.get("has_multicollinearity"):
                        st.warning(
                            "Внимание: обнаружена мультиколлинеарность (VIF > 10). "
                            "Коэффициенты модели могут быть нестабильны."
                        )

                    with st.expander("Дополнительные показатели модели"):
                        d1, d2, d3, d4 = st.columns(4)
                        with d1:
                            st.metric("F-статистика", f"{ols['f_statistic']:.4f}",
                                      help=help_f_stat(ols["f_statistic"], ols["f_p_value"]))
                        with d2:
                            st.metric("F p-значение", f"{ols['f_p_value']:.4f}",
                                      help=help_pvalue(ols["f_p_value"], "значимость модели"))
                        with d3:
                            st.metric("AIC", f"{ols['aic']:.2f}",
                                      help=help_aic(ols["aic"]))
                        with d4:
                            st.metric("Дарбин-Уотсон", f"{ols['durbin_watson']:.4f}",
                                      help=help_durbin_watson(ols["durbin_watson"]))
                        st.latex(LATEX["r_squared"])
                        st.latex(LATEX["f_stat"])

                    # ---------- Блок 3: График ----------
                    st.subheader("График регрессии")

                    plot_data = _json.loads(result["plot_json"])
                    fig_regression = go.Figure(plot_data)
                    st.plotly_chart(fig_regression, use_container_width=True)
                    st.caption("Для вставки в ВКР: наведите на график и нажмите кнопку камеры (Download plot as PNG) в панели инструментов.")

                    # ---------- Блок 4: Ручной прогноз ----------
                    st.divider()
                    st.subheader("Ручной прогноз")
                    st.caption(
                        "Введите значения факторов — модель рассчитает "
                        "прогнозное значение целевой переменной."
                    )

                    saved_features = st.session_state.get("reg_saved_features", [])
                    saved_target = st.session_state.get("reg_saved_target", "")
                    coefficients = skl["coefficients"]
                    intercept = skl["intercept"]

                    input_cols = st.columns(min(len(saved_features), 4))
                    input_values = {}

                    for i, feat in enumerate(saved_features):
                        col_idx = i % min(len(saved_features), 4)
                        feat_values = skl["feature_values"].get(feat, [])
                        default_val = 0.0
                        if feat_values:
                            sorted_vals = sorted(feat_values)
                            mid = len(sorted_vals) // 2
                            default_val = sorted_vals[mid]

                        with input_cols[col_idx]:
                            input_values[feat] = st.number_input(
                                feat,
                                value=float(default_val),
                                format="%.4f",
                                key=f"predict_input_{feat}",
                            )

                    if st.button("Рассчитать прогноз", key="run_predict"):
                        prediction = intercept
                        calc_parts = [f"{intercept:.4f}"]
                        for feat in saved_features:
                            coef = coefficients.get(feat, 0.0)
                            val = input_values[feat]
                            prediction += coef * val
                            sign = "+" if coef * val >= 0 else "-"
                            calc_parts.append(
                                f"{sign} {abs(coef):.4f} \u00d7 {val:.4f}"
                            )

                        formula = " ".join(calc_parts)
                        st.success(
                            f"**Прогноз {saved_target} = {prediction:.4f}**"
                        )
                        st.caption(f"Расчёт: {formula} = {prediction:.4f}")

    # ==================== СУБРОЗДЕЛ 2: ВРЕМЕННЫЕ РЯДЫ ====================
    with sub_tab_ts:
        if meta is None:
            st.info("Загрузите файл через боковую панель.")
            return

        datetime_cols = meta.get("datetime_columns", [])
        numeric_cols_ts = df.select_dtypes(include="number").columns.tolist()

        if not datetime_cols:
            st.warning(
                "В загруженном файле не обнаружено колонок с датами. "
                "Анализ временных рядов невозможен."
            )
        elif not numeric_cols_ts:
            st.warning("Нет числовых колонок для прогнозирования.")
        else:
            st.subheader("Настройка прогнозирования")

            ts_col1, ts_col2 = st.columns(2)
            with ts_col1:
                ts_date_col = st.selectbox(
                    "Колонка-дата",
                    datetime_cols,
                    key="ts_date_col",
                )
            with ts_col2:
                ts_value_col = st.selectbox(
                    "Целевая колонка",
                    numeric_cols_ts,
                    key="ts_value_col",
                )

            ts_steps = st.slider(
                "Горизонт прогноза (шаги)",
                min_value=1, max_value=60, value=14,
                key="ts_steps",
            )

            ts_model = st.radio(
                "Алгоритм",
                ["Auto-ARIMA", "Экспоненциальное сглаживание"],
                horizontal=True,
                key="ts_model_radio",
            )
            model_type = "arima" if ts_model == "Auto-ARIMA" else "hw"

            ts_info_list = meta.get("time_series_info", [])
            _ts_info_for_col = None
            for _tsi in ts_info_list:
                if _tsi["column"] == ts_date_col:
                    _ts_info_for_col = _tsi
                    break

            if _ts_info_for_col:
                if not _ts_info_for_col["is_regular"]:
                    st.warning(
                        f"Временной ряд по колонке «{ts_date_col}» нерегулярен "
                        f"(пропусков: {_ts_info_for_col['gaps_count']}). "
                        "Рекомендуется выполнить ресемплирование во вкладке «Предобработка» "
                        "перед построением прогноза."
                    )

            if st.button("Построить прогноз", key="ts_forecast_btn", type="primary"):
                payload = {
                    **data_payload(),
                    "date_column": ts_date_col,
                    "value_column": ts_value_col,
                    "steps": ts_steps,
                    "model_type": model_type,
                }
                with st.spinner("Обучение модели и построение прогноза..."):
                    resp = safe_post(FORECAST_URL, payload)

                if resp.status_code == 200:
                    fc = resp.json()

                    st.subheader("Прогноз")

                    fig_fc = go.Figure()

                    fig_fc.add_trace(go.Scatter(
                        x=fc["historical_dates"],
                        y=fc["historical_values"],
                        mode="lines",
                        name="Исторические данные",
                        line=dict(color="#636EFA"),
                    ))

                    fig_fc.add_trace(go.Scatter(
                        x=fc["forecast_dates"],
                        y=fc["forecast_values"],
                        mode="lines",
                        name="Прогноз",
                        line=dict(color="#EF553B", dash="dash"),
                    ))

                    if fc.get("ci_lower") and fc.get("ci_upper"):
                        fig_fc.add_trace(go.Scatter(
                            x=fc["forecast_dates"],
                            y=fc["ci_upper"],
                            mode="lines",
                            line=dict(width=0),
                            showlegend=False,
                        ))
                        fig_fc.add_trace(go.Scatter(
                            x=fc["forecast_dates"],
                            y=fc["ci_lower"],
                            mode="lines",
                            line=dict(width=0),
                            fill="tonexty",
                            fillcolor="rgba(239, 85, 59, 0.15)",
                            name="Доверительный интервал (95%)",
                        ))

                    fig_fc.update_layout(
                        title=f"Прогноз: {ts_value_col}",
                        xaxis_title="Дата",
                        yaxis_title=ts_value_col,
                        template="plotly_white",
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig_fc, use_container_width=True)
                    st.caption(
                        "Для вставки в ВКР: наведите на график и нажмите "
                        "кнопку камеры (Download plot as PNG) в панели инструментов."
                    )

                    if not fc.get("ci_lower"):
                        st.info(
                            "Доверительный интервал недоступен для метода "
                            "экспоненциального сглаживания Холта. "
                            "Для визуализации неопределённости используйте ARIMA."
                        )

                    metrics = fc["metrics"]
                    mc1, mc2, mc3 = st.columns(3)
                    with mc1:
                        st.metric("MAE (на истории)", f"{metrics['mae']:.4f}",
                                  help=help_mae(metrics["mae"]))
                    with mc2:
                        st.metric("RMSE (на истории)", f"{metrics['rmse']:.4f}",
                                  help=help_rmse(metrics["rmse"]))
                    with mc3:
                        st.metric("MAPE (на истории)", f"{metrics['mape']:.2f}%",
                                  help=help_mape(metrics["mape"]))

                    if fc.get("aic") is not None:
                        st.metric("AIC", f"{fc['aic']:.2f}",
                                  help=help_aic(fc["aic"]))

                    st.success(fc["explanation_text"])

                    with st.expander("Дополнительные показатели модели"):
                        if model_type == "arima":
                            st.latex(LATEX["arima"])
                        else:
                            st.latex(LATEX["holt"])
                        st.latex(LATEX["mape"])

                else:
                    detail = resp.json().get("detail", "Неизвестная ошибка")
                    st.error(f"Ошибка API: {detail}")
