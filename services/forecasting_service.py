"""
forecasting_service.py — Сервис прогнозирования временных рядов.

Реализует два метода:
    1. ARIMA (auto_arima stepwise) — для стационарных / приводимых к стационарности рядов.
    2. Holt (экспоненциальное сглаживание) — безусловный трендовый метод.
"""

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing


from typing import Optional


def _mape(actual: np.ndarray, predicted: np.ndarray) -> Optional[float]:
    """MAPE, исключая нулевые фактические значения. None если все y_i = 0."""
    mask = actual != 0
    if not mask.any():
        return None
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


class ForecastingService:
    """Прогнозирование временных рядов (ARIMA / Holt)."""

    def fit_predict_arima(
        self,
        series: pd.Series,
        steps: int,
        confidence_level: float = 0.95,
    ) -> dict:
        """Stepwise auto_arima (pmdarima), прогноз на *steps* шагов."""

        import pmdarima as pm

        auto_model = pm.auto_arima(
            series,
            start_p=0, max_p=3,
            start_q=0, max_q=3,
            d=None,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
        )

        if auto_model is None:
            raise ValueError(
                "Не удалось подобрать ни одну модель ARIMA. "
                "Проверьте данные: возможно, ряд слишком короткий или содержит аномалии."
            )

        best_order = auto_model.order
        best_aic = float(auto_model.aic())

        # Прогноз
        forecast_values, ci = auto_model.predict(
            n_periods=steps,
            return_conf_int=True,
            alpha=1 - confidence_level,
        )

        # Генерация индекса прогноза
        if hasattr(series.index, 'freq') and series.index.freq is not None:
            forecast_index = pd.date_range(
                start=series.index[-1] + series.index.freq,
                periods=steps,
                freq=series.index.freq,
            )
        else:
            forecast_index = list(range(len(series), len(series) + steps))

        # Метрики на обучающих данных
        fitted = auto_model.predict_in_sample()
        actual = series.values
        n = min(len(actual), len(fitted))
        actual_trim = actual[-n:]
        fitted_trim = fitted[-n:]

        mae = float(mean_absolute_error(actual_trim, fitted_trim))
        rmse = float(np.sqrt(mean_squared_error(actual_trim, fitted_trim)))
        mape = _mape(actual_trim, fitted_trim)

        p, d, q = best_order
        explanation = (
            f"Лучшая модель ARIMA({p},{d},{q}) выбрана по критерию Акаике "
            f"(AIC = {best_aic:.2f}) с помощью stepwise-алгоритма auto_arima. "
            f"Точность на обучающих данных: MAE = {mae:.4f}, RMSE = {rmse:.4f}, MAPE = {mape:.2f}%."
        )

        return {
            "forecast_dates": [str(d) for d in forecast_index],
            "forecast_values": [float(v) for v in forecast_values],
            "ci_lower": [float(v) for v in ci[:, 0]],
            "ci_upper": [float(v) for v in ci[:, 1]],
            "metrics": {"mae": mae, "rmse": rmse, "mape": mape if mape is not None else 0.0},
            "aic": best_aic,
            "order": list(best_order),
            "explanation_text": explanation,
        }

    def fit_predict_hw(
        self,
        series: pd.Series,
        steps: int,
    ) -> dict:
        """Экспоненциальное сглаживание Холта (аддитивный тренд)."""

        try:
            model = ExponentialSmoothing(
                series,
                trend="add",
                initialization_method="estimated",
            )
            fit = model.fit()
        except (ValueError, LinAlgError, np.linalg.LinAlgError) as exc:
            raise ValueError(
                "Не удалось обучить модель Холта. "
                f"Возможно, ряд содержит константные значения или аномалии: {exc}"
            ) from exc

        forecast_values = fit.forecast(steps)

        # Метрики на обучающих данных
        fitted = fit.fittedvalues
        actual = series.values
        fitted_arr = fitted.values if hasattr(fitted, 'values') else np.array(fitted)

        mae = float(mean_absolute_error(actual, fitted_arr))
        rmse = float(np.sqrt(mean_squared_error(actual, fitted_arr)))
        mape = _mape(actual, fitted_arr)

        explanation = (
            f"Модель экспоненциального сглаживания Холта (аддитивный тренд). "
            f"Точность на обучающих данных: MAE = {mae:.4f}, RMSE = {rmse:.4f}, MAPE = {mape:.2f}%."
        )

        forecast_index = forecast_values.index

        return {
            "forecast_dates": [str(d) for d in forecast_index],
            "forecast_values": [float(v) for v in forecast_values],
            "ci_lower": None,
            "ci_upper": None,
            "metrics": {"mae": mae, "rmse": rmse, "mape": mape if mape is not None else 0.0},
            "aic": None,
            "order": None,
            "explanation_text": explanation,
        }
