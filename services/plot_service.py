"""
plot_service.py — Построение интерактивных графиков регрессии (Plotly).

Функция build_regression_plot строит научный график с:
    - Scatter-точками реальных данных
    - Линией регрессии (trend line)
    - 95%-м доверительным интервалом (полупрозрачная заливка)

Доверительный интервал рассчитывается через statsmodels OLS get_prediction,
что даёт корректные границы на основе стандартной ошибки прогноза.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm


def build_regression_plot(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    confidence: float = 0.95,
) -> go.Figure:
    """
    Строит интерактивный график регрессии с доверительным интервалом.

    Параметры:
        df         — очищенный DataFrame (без NaN/inf).
        target     — имя целевой переменной (Y).
        features   — список независимых переменных (X).
        confidence — уровень доверия (по умолчанию 0.95).

    Возвращает:
        plotly.graph_objects.Figure — готовый к отображению график.
    """
    clean = df[features + [target]].dropna()
    y = clean[target].values
    X_with_const = sm.add_constant(clean[features])

    # --- Подгоняем OLS для доверительного интервала ---
    ols_model = sm.OLS(y, X_with_const).fit()
    prediction = ols_model.get_prediction(X_with_const)
    pred_summary = prediction.summary_frame(alpha=1 - confidence)

    y_pred = pred_summary["mean"].values
    ci_lower = pred_summary["mean_ci_lower"].values
    ci_upper = pred_summary["mean_ci_upper"].values

    # --- Ось X для графика ---
    # Для одного фактора — используем его значения.
    # Для множественной регрессии — используем предсказанные Y (ŷ),
    # чтобы визуализировать «факт vs прогноз» в одном измерении.
    if len(features) == 1:
        x_vals = clean[features[0]].values
        x_label = features[0]
    else:
        x_vals = y_pred
        x_label = "Прогнозное значение (Y\u0302)"

    # Сортируем всё по оси X для корректной отрисовки линии и заливки
    sort_idx = np.argsort(x_vals)
    x_sorted = x_vals[sort_idx]
    y_sorted = y[sort_idx]
    y_pred_sorted = y_pred[sort_idx]
    ci_lower_sorted = ci_lower[sort_idx]
    ci_upper_sorted = ci_upper[sort_idx]

    # --- Строим фигуру ---
    fig = go.Figure()

    # 1. Доверительный интервал (заливка)
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_sorted, x_sorted[::-1]]),
        y=np.concatenate([ci_upper_sorted, ci_lower_sorted[::-1]]),
        fill="toself",
        fillcolor="rgba(99, 110, 250, 0.15)",
        line=dict(color="rgba(99, 110, 250, 0)"),
        hoverinfo="skip",
        showlegend=True,
        name=f"{int(confidence * 100)}% доверительный интервал",
    ))

    # 2. Реальные данные (scatter)
    fig.add_trace(go.Scatter(
        x=x_sorted,
        y=y_sorted,
        mode="markers",
        marker=dict(
            color="#636EFA",
            size=7,
            opacity=0.7,
            line=dict(width=1, color="white"),
        ),
        name="Наблюдения",
    ))

    # 3. Линия регрессии
    fig.add_trace(go.Scatter(
        x=x_sorted,
        y=y_pred_sorted,
        mode="lines",
        line=dict(color="#EF553B", width=2.5),
        name="Линия регрессии",
    ))

    # --- Формула в аннотации ---
    r2 = ols_model.rsquared
    n = len(y)
    equation_parts = [f"{ols_model.params.iloc[0]:.3f}"]
    param_names = list(ols_model.params.index)
    for i, feat_name in enumerate(param_names[1:], start=1):
        coef = ols_model.params.iloc[i]
        sign = "+" if coef >= 0 else "-"
        equation_parts.append(f" {sign} {abs(coef):.3f}\u00b7{feat_name}")
    equation = f"{target} = {''.join(equation_parts)}"

    # --- Оформление ---
    fig.update_layout(
        title=dict(
            text=(
                f"Регрессионный анализ: {target}<br>"
                f"<sub>{equation} &nbsp;|&nbsp; "
                f"R\u00b2 = {r2:.4f} &nbsp;|&nbsp; n = {n}</sub>"
            ),
            x=0.5,
            xanchor="center",
        ),
        xaxis_title=x_label,
        yaxis_title=target,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        template="plotly_white",
        hovermode="x unified",
        margin=dict(t=100),
    )

    return fig


def regression_plot_to_json(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    confidence: float = 0.95,
) -> str:
    """
    Строит график и возвращает его как JSON-строку (для передачи по API).
    Фронтенд может отрисовать результат через Plotly.react().
    """
    fig = build_regression_plot(df, target, features, confidence)
    return fig.to_json()
