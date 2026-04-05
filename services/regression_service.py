"""
regression_service.py — Математическое ядро регрессионного анализа.

Класс RegressionService выполняет два параллельных расчёта:
    1. statsmodels OLS — коэффициенты, p-values, R², стандартные ошибки
       (для научного обоснования в тексте ВКР).
    2. scikit-learn — предиктивная модель. При выборке < 50 записей
       автоматически используется Ridge-регрессия для предотвращения
       переобучения.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sp_stats

from config import VIF_THRESHOLD
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SMALL_SAMPLE_THRESHOLD = 50


class RegressionService:
    """Сервис регрессионного анализа (statsmodels + scikit-learn)."""

    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        features: list[str],
    ) -> None:
        self.df = df
        self.target = target
        self.features = features

        # Убираем пропуски в используемых столбцах
        self.clean_df = df[features + [target]].dropna()

        self.X = self.clean_df[features].values
        self.y = self.clean_df[target].values
        self.n_samples = len(self.clean_df)

    # ------------------------------------------------------------------
    # statsmodels OLS
    # ------------------------------------------------------------------
    def _fit_ols(self) -> dict:
        """Подгоняет OLS-модель и возвращает статистические показатели."""
        X_with_const = sm.add_constant(self.clean_df[self.features])
        ols_model = sm.OLS(self.y, X_with_const).fit()

        # Собираем коэффициенты по каждому фактору
        factor_stats: list[dict] = []
        # Первый элемент — константа, далее факторы
        param_names = list(ols_model.params.index)
        for name in param_names:
            factor_stats.append({
                "name": name,
                "coefficient": float(ols_model.params[name]),
                "std_error": float(ols_model.bse[name]),
                "t_statistic": float(ols_model.tvalues[name]),
                "p_value": float(ols_model.pvalues[name]),
            })

        # --- VIF (мультиколлинеарность) ---
        vif_warnings: list[str] = []
        vif_map: dict[str, float] = {}
        has_multicollinearity = False
        if len(self.features) > 1:
            X_df = self.clean_df[self.features]
            for i, feat in enumerate(self.features):
                vif_val = float(variance_inflation_factor(X_df.values, i))
                vif_map[feat] = round(vif_val, 2)
                if vif_val > VIF_THRESHOLD:
                    has_multicollinearity = True
                    vif_warnings.append(
                        f"Риск мультиколлинеарности: VIF({feat}) = {vif_val:.1f} > {VIF_THRESHOLD}. "
                        f"Рекомендация: удалите фактор «{feat}» из модели и повторите расчёт."
                    )
        elif len(self.features) == 1:
            vif_map[self.features[0]] = 1.0

        # Добавляем VIF в factor_stats (для константы VIF не считаем)
        for fs in factor_stats:
            fs["vif"] = vif_map.get(fs["name"])

        # --- Анализ адекватности: нормальность остатков (Шапиро-Уилк) ---
        residuals = ols_model.resid
        shapiro_stat, shapiro_p = sp_stats.shapiro(residuals)
        is_reliable = shapiro_p >= 0.05

        if is_reliable:
            reliability_text = (
                f"Остатки модели нормально распределены "
                f"(тест Шапиро-Уилка: W = {shapiro_stat:.4f}, "
                f"p = {shapiro_p:.4f} ≥ 0.05). "
                "Модель адекватна — выводы о значимости коэффициентов корректны."
            )
        else:
            reliability_text = (
                f"Остатки модели НЕ подчиняются нормальному распределению "
                f"(тест Шапиро-Уилка: W = {shapiro_stat:.4f}, "
                f"p = {shapiro_p:.4f} < 0.05). "
                "Модель может быть неадекватна — интерпретируйте p-значения "
                "коэффициентов с осторожностью."
            )

        # --- Автокорреляция остатков (Дарбин-Уотсон) ---
        dw = float(sm.stats.durbin_watson(ols_model.resid))

        if dw < 1.5:
            dw_interp = (
                f"\nТест Дарбина-Уотсона: DW = {dw:.4f} < 1.5 — "
                "обнаружена положительная автокорреляция остатков. "
                "Стандартные ошибки коэффициентов могут быть занижены."
            )
        elif dw > 2.5:
            dw_interp = (
                f"\nТест Дарбина-Уотсона: DW = {dw:.4f} > 2.5 — "
                "обнаружена отрицательная автокорреляция остатков. "
                "Стандартные ошибки коэффициентов могут быть завышены."
            )
        else:
            dw_interp = (
                f"\nТест Дарбина-Уотсона: DW = {dw:.4f} ∈ [1.5, 2.5] — "
                "значимой автокорреляции остатков не обнаружено."
            )
        reliability_text += dw_interp

        return {
            "r_squared": float(ols_model.rsquared),
            "r_squared_adj": float(ols_model.rsquared_adj),
            "f_statistic": float(ols_model.fvalue),
            "f_p_value": float(ols_model.f_pvalue),
            "aic": float(ols_model.aic),
            "bic": float(ols_model.bic),
            "durbin_watson": dw,
            "factor_stats": factor_stats,
            "residuals_shapiro_stat": round(float(shapiro_stat), 6),
            "residuals_shapiro_p": round(float(shapiro_p), 6),
            "is_model_reliable": is_reliable,
            "reliability_text": reliability_text,
            "vif_warnings": vif_warnings,
            "has_multicollinearity": has_multicollinearity,
        }

    # ------------------------------------------------------------------
    # scikit-learn (LinearRegression / Ridge)
    # ------------------------------------------------------------------
    def _fit_sklearn(self) -> dict:
        """Обучает предиктивную модель и возвращает метрики + предсказания."""
        if self.n_samples < SMALL_SAMPLE_THRESHOLD:
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=1.0)),
            ])
            model_name = "Ridge Regression (StandardScaler)"
        else:
            model = LinearRegression()
            model_name = "Linear Regression (OLS)"

        model.fit(self.X, self.y)
        predictions = model.predict(self.X)

        # Извлекаем коэффициенты (для Pipeline — из вложенной модели)
        if isinstance(model, Pipeline):
            _estimator = model.named_steps["ridge"]
        else:
            _estimator = model

        return {
            "model_name": model_name,
            "intercept": float(_estimator.intercept_),
            "coefficients": {
                feat: float(coef)
                for feat, coef in zip(self.features, _estimator.coef_)
            },
            "r2": float(r2_score(self.y, predictions)),
            "mae": float(mean_absolute_error(self.y, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(self.y, predictions))),
            "predictions": predictions.tolist(),
            "feature_values": {
                feat: self.clean_df[feat].tolist() for feat in self.features
            },
            "target_values": self.clean_df[self.target].tolist(),
        }

    # ------------------------------------------------------------------
    # Публичный метод
    # ------------------------------------------------------------------
    def run(self) -> dict:
        """Выполняет полный анализ и возвращает сводный словарь."""
        return {
            "n_samples": self.n_samples,
            "small_sample": self.n_samples < SMALL_SAMPLE_THRESHOLD,
            "ols": self._fit_ols(),
            "sklearn": self._fit_sklearn(),
        }
