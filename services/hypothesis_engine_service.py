"""
hypothesis_engine_service.py — Сервис интеллектуальной проверки статистических гипотез.

Реализует «Дерево принятия решений» (Decision Tree) для автоматического
выбора наиболее подходящего статистического критерия на основе свойств данных.

Модуль содержит два основных направления анализа:

1. Сравнение двух числовых выборок (compare_two_groups):
   - Проверка нормальности → Шапиро-Уилк
   - Проверка гомоскедастичности → Критерий Левене
   - Выбор критерия:
       * t-тест Стьюдента (равные дисперсии)
       * t-тест Уэлча (неравные дисперсии)
       * U-критерий Манна-Уитни (ненормальные данные)
   - Расчёт размера эффекта — d Коэна (Cohen's d)

2. Анализ связи категориальных переменных (analyze_categorical_association):
   - Таблица сопряжённости (Crosstab)
   - Критерий хи-квадрат Пирсона
   - Коэффициент V Крамера (Cramér's V)

Используется в рамках ВКР: «Веб-приложение для статистического анализа и прогнозирования».
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from statsmodels.tsa.stattools import adfuller

from config import (
    DEFAULT_ALPHA,
    EFFECT_NEGLIGIBLE,
    EFFECT_SMALL,
    EFFECT_MEDIUM,
    RANK_BISERIAL_LARGE,
    CRAMERS_V_STRONG,
)

# Уровень значимости по умолчанию
ALPHA: float = DEFAULT_ALPHA


class HypothesisEngineService:
    """
    Сервис автоматического выбора и проведения статистических тестов.

    Все методы — статические: принимают данные, возвращают структурированный
    результат (словарь), не хранят состояния между вызовами.
    """

    # ==================================================================
    # 1. Сравнение двух числовых выборок
    # ==================================================================

    @staticmethod
    def compare_two_groups(
        sample_a: pd.Series,
        sample_b: pd.Series,
        alpha: float = ALPHA,
    ) -> dict[str, Any]:
        """
        Сравнивает две независимые числовые выборки, автоматически выбирая
        наиболее подходящий статистический критерий.

        Алгоритм (Дерево принятия решений):
        ─────────────────────────────────────────────────────────────────
        1. Валидация: обе выборки должны содержать ≥ 3 непустых значений.
        2. Проверка нормальности каждой выборки (тест Шапиро-Уилка):
           H₀: выборка взята из нормального распределения.
           Если p < α → H₀ отвергается (распределение не нормальное).
        3. Если обе нормальны — проверяем равенство дисперсий (тест Левене):
           H₀: дисперсии двух выборок равны (гомоскедастичность).
           Если p < α → H₀ отвергается (дисперсии неравны).
        4. Выбор критерия сравнения средних:
           ┌───────────────────────────────────────────────────────────┐
           │ Обе нормальны + дисперсии равны   → t-тест Стьюдента     │
           │ Обе нормальны + дисперсии неравны → t-тест Уэлча         │
           │ Хотя бы одна ненормальна          → U-тест Манна-Уитни   │
           └───────────────────────────────────────────────────────────┘
        5. Расчёт размера эффекта (d Коэна) — только для t-тестов.

        Параметры:
            sample_a — первая числовая выборка (pd.Series).
            sample_b — вторая числовая выборка (pd.Series).
            alpha    — уровень значимости (по умолчанию 0.05).

        Возвращает:
            Словарь с полным описанием результатов тестирования.

        Исключения:
            ValueError — если в какой-либо выборке менее 3 значений.
        """
        # --- Шаг 1: Валидация входных данных ---
        a = sample_a.dropna().astype(float)
        b = sample_b.dropna().astype(float)

        if len(a) < 3:
            raise ValueError(
                f"Выборка A содержит менее 3 значений ({len(a)}). "
                "Статистический тест невозможен."
            )
        if len(b) < 3:
            raise ValueError(
                f"Выборка B содержит менее 3 значений ({len(b)}). "
                "Статистический тест невозможен."
            )

        # --- Шаг 2: Проверка нормальности (адаптивный выбор теста) ---
        #
        # Выбор теста нормальности зависит от объёма выборки:
        #   N < 8    — тест невозможен (мощность ≈ 0), принудительно непараметрика.
        #   8 ≤ N ≤ 300 — тест Шапиро-Уилка (оптимален для малых/средних выборок).
        #   N > 300  — тест Д'Агостино-Пирсон (Шапиро при большом N даёт
        #              ложные отклонения даже при минимальных отклонениях от нормы).
        n_a, n_b = len(a), len(b)
        force_nonparametric = n_a < 8 or n_b < 8

        if force_nonparametric:
            # Слишком мало наблюдений — любой тест нормальности бесполезен
            norm_a, norm_b = False, False
            norm_a_stat = norm_a_p = float("nan")
            norm_b_stat = norm_b_p = float("nan")
            norm_test_name: str | None = None
        elif n_a > 300 or n_b > 300:
            # Д'Агостино-Пирсон — устойчив при больших объёмах
            norm_a_stat, norm_a_p = sp_stats.normaltest(a)
            norm_b_stat, norm_b_p = sp_stats.normaltest(b)
            norm_test_name = "Д'Агостино-Пирсон"
            norm_a = bool(norm_a_p >= alpha)
            norm_b = bool(norm_b_p >= alpha)
        else:
            # Шапиро-Уилк — стандарт для 8 ≤ N ≤ 300
            norm_a_stat, norm_a_p = sp_stats.shapiro(a)
            norm_b_stat, norm_b_p = sp_stats.shapiro(b)
            norm_test_name = "Шапиро-Уилк"
            norm_a = bool(norm_a_p >= alpha)
            norm_b = bool(norm_b_p >= alpha)

        # --- Шаг 3: Проверка равенства дисперсий (тест Левене) ---
        #
        # Тест Левене проверяет гипотезу H₀ о равенстве дисперсий
        # (гомоскедастичность). Пропускается при force_nonparametric.
        if force_nonparametric:
            levene_stat = levene_p = float("nan")
            equal_var = False
        else:
            levene_stat, levene_p = sp_stats.levene(a, b)
            equal_var = bool(levene_p >= alpha)

        # --- Шаг 4: Выбор и проведение статистического критерия ---
        decision_path: list[str] = []
        cohens_d: float | None = None

        if force_nonparametric:
            # Принудительно непараметрический тест при малых выборках
            decision_path.append(
                f"Выборка слишком мала (N_a={n_a}, N_b={n_b}, порог=8) — "
                "принудительно непараметрический тест. "
                "Внимание: мощность теста крайне низкая."
            )

            stat, p_value = sp_stats.mannwhitneyu(a, b, alternative="two-sided")
            test_name = "U-критерий Манна-Уитни"

            rank_biserial = _compute_rank_biserial(float(stat), n_a, n_b)
            cohens_d = round(rank_biserial, 6)

        elif norm_a and norm_b:
            # Обе выборки нормально распределены
            decision_path.append(f"Обе выборки нормальны ({norm_test_name})")

            if equal_var:
                # Дисперсии равны → классический t-тест Стьюдента
                decision_path.append("Дисперсии равны (Левене) → t-тест Стьюдента")
                stat, p_value = sp_stats.ttest_ind(a, b, equal_var=True)
                test_name = "t-тест Стьюдента"
            else:
                # Дисперсии неравны → t-тест Уэлча (робастная модификация)
                decision_path.append("Дисперсии неравны (Левене) → t-тест Уэлча")
                stat, p_value = sp_stats.ttest_ind(a, b, equal_var=False)
                test_name = "t-тест Уэлча"

            # --- Шаг 5: Размер эффекта (d Коэна) ---
            #
            # d Коэна (Cohen's d) — стандартизированная мера различия средних.
            # Формула:
            #              M₁ − M₂
            #   d = ─────────────────────────
            #       √((s₁²·(n₁−1) + s₂²·(n₂−1)) / (n₁ + n₂ − 2))
            #
            # где M — средние, s² — дисперсии, n — размеры выборок.
            #
            # Интерпретация (по Коэну):
            #   |d| < 0.2  — незначительный эффект
            #   |d| ≈ 0.5  — средний эффект
            #   |d| ≈ 0.8  — большой эффект
            #   |d| > 1.2  — очень большой эффект
            cohens_d = _compute_cohens_d(a.values, b.values)

        else:
            # Хотя бы одна выборка ненормальна → непараметрический тест
            if not norm_a and not norm_b:
                decision_path.append(
                    f"Обе выборки ненормальны ({norm_test_name}) → U-тест Манна-Уитни"
                )
            elif not norm_a:
                decision_path.append(
                    f"Выборка A ненормальна ({norm_test_name}) → U-тест Манна-Уитни"
                )
            else:
                decision_path.append(
                    f"Выборка B ненормальна ({norm_test_name}) → U-тест Манна-Уитни"
                )

            stat, p_value = sp_stats.mannwhitneyu(a, b, alternative="two-sided")
            test_name = "U-критерий Манна-Уитни"

            # --- Размер эффекта: ранг-бисериальная корреляция ---
            #
            # Для Манна-Уитни d Коэна неприменим. Используется
            # ранг-бисериальная корреляция: r = 1 − 2U/(n₁·n₂).
            # Диапазон [−1, 1]: 0 — нет эффекта, ±1 — максимальный.
            rank_biserial = _compute_rank_biserial(float(stat), len(a), len(b))
            cohens_d = round(rank_biserial, 6)

        # --- Формируем текстовый вывод ---
        if p_value < alpha:
            conclusion = (
                f"p-значение = {p_value:.4f} < {alpha}. "
                "Различия между выборками статистически значимы."
            )
        else:
            conclusion = (
                f"p-значение = {p_value:.4f} ≥ {alpha}. "
                "Статистически значимых различий не обнаружено."
            )

        # --- Интерпретация размера эффекта ---
        is_rank_biserial = test_name == "U-критерий Манна-Уитни"
        if cohens_d is not None:
            effect_interp = (
                _interpret_rank_biserial(cohens_d)
                if is_rank_biserial
                else _interpret_cohens_d(cohens_d)
            )
        else:
            effect_interp = None

        # --- Безопасное округление (NaN → None для JSON) ---
        def _safe_round(val: float, digits: int = 6) -> float | None:
            if val != val:  # NaN check
                return None
            return round(float(val), digits)

        return {
            "test_name": test_name,
            "statistic": round(float(stat), 6),
            "p_value": round(float(p_value), 6),
            # Нормальность
            "normality_a": norm_a,
            "normality_b": norm_b,
            "shapiro_a_stat": _safe_round(norm_a_stat),
            "shapiro_a_p": _safe_round(norm_a_p),
            "shapiro_b_stat": _safe_round(norm_b_stat),
            "shapiro_b_p": _safe_round(norm_b_p),
            "norm_test_name": norm_test_name,
            # Дисперсии
            "equal_variance": equal_var,
            "levene_stat": _safe_round(levene_stat),
            "levene_p": _safe_round(levene_p),
            # Размер эффекта
            "cohens_d": round(cohens_d, 6) if cohens_d is not None else None,
            "effect_size_interpretation": effect_interp,
            "effect_size_metric": "rank_biserial" if is_rank_biserial else "cohens_d",
            # Решение
            "decision_path": decision_path,
            "conclusion": conclusion,
            # Описательные статистики выборок
            "sample_a_n": len(a),
            "sample_b_n": len(b),
            "sample_a_mean": round(float(a.mean()), 6),
            "sample_b_mean": round(float(b.mean()), 6),
            "sample_a_std": round(float(a.std()), 6),
            "sample_b_std": round(float(b.std()), 6),
        }

    # ==================================================================
    # 2. Тест стационарности (ADF-тест)
    # ==================================================================

    @staticmethod
    def test_stationarity(
        series: pd.Series,
        alpha: float = ALPHA,
    ) -> dict[str, Any]:
        """
        Проверяет стационарность временного ряда расширенным тестом Дики-Фуллера.

        Стационарность — обязательное требование для модели ARIMA.
        Нестационарный ряд необходимо дифференцировать (d ≥ 1) перед моделированием.

        Тест проверяет гипотезу:
            H₀: ряд имеет единичный корень (нестационарен).
            H₁: ряд стационарен.
        Если p < α → H₀ отвергается → ряд стационарен.

        Параметры:
            series — числовой pd.Series (значения временного ряда).
            alpha  — уровень значимости (по умолчанию 0.05).

        Возвращает:
            Словарь:
            {
                "test_name": str,          — "Расширенный тест Дики-Фуллера (ADF)"
                "adf_statistic": float,    — ADF-статистика
                "p_value": float,          — p-значение
                "used_lag": int,           — количество использованных лагов
                "n_observations": int,     — число наблюдений
                "critical_values": dict,   — критические значения (1%, 5%, 10%)
                "is_stationary": bool,     — True если ряд стационарен
                "conclusion": str,         — текстовый вывод на русском
            }

        Исключения:
            ValueError — при недостаточном количестве данных.
        """
        data = series.dropna().astype(float)

        if len(data) < 10:
            raise ValueError(
                f"Недостаточно данных для ADF-теста ({len(data)} значений). "
                "Минимум — 10 наблюдений."
            )

        adf_stat, p_value, used_lag, n_obs, critical_values, _ = adfuller(
            data.values, autolag="AIC",
        )

        is_stationary = p_value < alpha

        # Критические значения — округляем для читаемости
        crit_formatted = {
            k: round(float(v), 4) for k, v in critical_values.items()
        }

        if is_stationary:
            conclusion = (
                f"ADF-статистика = {adf_stat:.4f}, p-значение = {p_value:.4f} < {alpha}. "
                "Ряд стационарен — можно применять ARIMA без дифференцирования (d = 0)."
            )
        else:
            conclusion = (
                f"ADF-статистика = {adf_stat:.4f}, p-значение = {p_value:.4f} ≥ {alpha}. "
                "Ряд нестационарен — необходимо дифференцирование (d ≥ 1) перед ARIMA."
            )

        return {
            "test_name": "Расширенный тест Дики-Фуллера (ADF)",
            "adf_statistic": round(float(adf_stat), 6),
            "p_value": round(float(p_value), 6),
            "used_lag": int(used_lag),
            "n_observations": int(n_obs),
            "critical_values": crit_formatted,
            "is_stationary": is_stationary,
            "conclusion": conclusion,
        }

    # ==================================================================
    # 3. Анализ связи категориальных переменных
    # ==================================================================

    @staticmethod
    def analyze_categorical_association(
        series_a: pd.Series,
        series_b: pd.Series,
    ) -> dict[str, Any]:
        """
        Анализирует статистическую связь между двумя категориальными переменными.

        Алгоритм:
        ─────────────────────────────────────────────────────────────────
        1. Построение таблицы сопряжённости (Contingency Table / Crosstab):
           Двумерная таблица частот, где строки — категории A,
           столбцы — категории B, ячейки — количество наблюдений.

        2. Критерий хи-квадрат Пирсона (χ² test of independence):
           H₀: переменные A и B статистически независимы.
           Критерий сравнивает наблюдаемые частоты (O) с ожидаемыми (E):
               χ² = Σ (Oᵢⱼ − Eᵢⱼ)² / Eᵢⱼ
           При p < α связь считается статистически значимой.

        3. Коэффициент V Крамера (Cramér's V):
           Нормированная мера силы связи на основе χ²:
               V = √(χ² / (n · (min(r, c) − 1)))
           где n — число наблюдений, r — число строк, c — число столбцов.
           V ∈ [0, 1]: 0 — нет связи, 1 — полная связь.

        Параметры:
            series_a — первая категориальная переменная (pd.Series).
            series_b — вторая категориальная переменная (pd.Series).

        Возвращает:
            Словарь с таблицей сопряжённости, χ², p-значением и V Крамера.

        Исключения:
            ValueError — если в данных недостаточно категорий для анализа.
        """
        # Убираем строки, где хотя бы одна переменная — NaN
        mask = series_a.notna() & series_b.notna()
        a_clean = series_a[mask]
        b_clean = series_b[mask]

        if len(a_clean) < 2:
            raise ValueError(
                "Недостаточно данных для анализа (менее 2 наблюдений)."
            )

        # --- Шаг 1: Таблица сопряжённости ---
        crosstab = pd.crosstab(a_clean, b_clean)

        if crosstab.shape[0] < 2 or crosstab.shape[1] < 2:
            raise ValueError(
                "Таблица сопряжённости должна содержать минимум 2 строки "
                "и 2 столбца. Убедитесь, что обе переменные имеют "
                "≥ 2 уникальных значения."
            )

        # --- Шаг 2: Критерий хи-квадрат Пирсона ---
        chi2, p_value, dof, expected = sp_stats.chi2_contingency(crosstab)

        # --- Проверка условия Кохрана ---
        #
        # Аппроксимация χ² корректна только если все ожидаемые частоты ≥ 5.
        # При нарушении — p-value ненадёжно.
        cochran_warning: str | None = None
        if (expected < 5).any():
            low_count = int((expected < 5).sum())
            total_cells = expected.size
            cochran_warning = (
                f"Внимание: {low_count} из {total_cells} ожидаемых частот < 5 "
                f"({low_count / total_cells * 100:.0f}%). "
                "Результаты χ²-теста могут быть неточными "
                "(нарушено условие Кохрана)."
            )

        # --- Шаг 3: Коэффициент V Крамера ---
        #
        # V = √(χ² / (n · (k − 1)))
        # где k = min(число строк, число столбцов) таблицы сопряжённости,
        # n — общее число наблюдений.
        n = int(crosstab.values.sum())
        k = min(crosstab.shape[0], crosstab.shape[1])
        cramers_v = float(np.sqrt(chi2 / (n * (k - 1)))) if k > 1 else 0.0

        # --- Формируем вывод ---
        if p_value < ALPHA:
            conclusion = (
                f"p-значение = {p_value:.4f} < {ALPHA}. "
                "Связь между переменными статистически значима."
            )
        else:
            conclusion = (
                f"p-значение = {p_value:.4f} ≥ {ALPHA}. "
                "Статистически значимой связи не обнаружено."
            )

        return {
            "test_name": "Критерий хи-квадрат Пирсона",
            "chi2_statistic": round(float(chi2), 6),
            "p_value": round(float(p_value), 6),
            "degrees_of_freedom": int(dof),
            "cramers_v": round(cramers_v, 6),
            "cramers_v_interpretation": _interpret_cramers_v(cramers_v),
            "cochran_warning": cochran_warning,
            "conclusion": conclusion,
            "n_observations": n,
            # Таблица сопряжённости — в формате для JSON-сериализации
            "crosstab_index": crosstab.index.astype(str).tolist(),
            "crosstab_columns": crosstab.columns.astype(str).tolist(),
            "crosstab_values": crosstab.values.tolist(),
        }


# ==================================================================
# Вспомогательные функции (модульный уровень)
# ==================================================================

def _compute_cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """
    Рассчитывает d Коэна (Cohen's d) — стандартизированный размер эффекта.

    Формула (pooled standard deviation):
                     M₁ − M₂
        d = ─────────────────────────────────
            √((s₁²·(n₁−1) + s₂²·(n₂−1)) / (n₁ + n₂ − 2))

    Параметры:
        a — массив значений первой выборки.
        b — массив значений второй выборки.

    Возвращает:
        Значение d Коэна (float). Положительное — среднее A > среднего B.
    """
    n1, n2 = len(a), len(b)
    var1, var2 = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))

    # Объединённое стандартное отклонение (pooled SD)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return float((np.mean(a) - np.mean(b)) / pooled_std)


def _interpret_cohens_d(d: float) -> str:
    """
    Возвращает текстовую интерпретацию размера эффекта по шкале Коэна.

    Пороги (Cohen, 1988):
        |d| < 0.2  — незначительный эффект
        |d| < 0.5  — малый эффект
        |d| < 0.8  — средний эффект
        |d| < 1.2  — большой эффект
        |d| ≥ 1.2  — очень большой эффект

    Параметры:
        d — значение d Коэна.

    Возвращает:
        Строка с интерпретацией на русском языке.
    """
    abs_d = abs(d)
    if abs_d < EFFECT_NEGLIGIBLE:
        return "Незначительный эффект"
    if abs_d < EFFECT_SMALL:
        return "Малый эффект"
    if abs_d < EFFECT_MEDIUM:
        return "Средний эффект"
    if abs_d < 1.2:
        return "Большой эффект"
    return "Очень большой эффект"


def _compute_rank_biserial(U: float, n1: int, n2: int) -> float:
    """
    Рассчитывает ранг-бисериальную корреляцию — размер эффекта для U-теста
    Манна-Уитни.

    Формула:
        r = 1 − 2U / (n₁ · n₂)

    Диапазон: [−1, 1].
        r ≈ 0   — нет различий между группами.
        |r| → 1 — полное разделение групп.

    Параметры:
        U  — U-статистика Манна-Уитни.
        n1 — размер первой выборки.
        n2 — размер второй выборки.

    Возвращает:
        Значение ранг-бисериальной корреляции (float).
    """
    if n1 * n2 == 0:
        return 0.0
    return float(1.0 - (2.0 * U) / (n1 * n2))


def _interpret_rank_biserial(r: float) -> str:
    """
    Возвращает текстовую интерпретацию ранг-бисериальной корреляции.

    Пороги (аналогично корреляции Пирсона):
        |r| < 0.1  — незначительный эффект
        |r| < 0.3  — малый эффект
        |r| < 0.5  — средний эффект
        |r| ≥ 0.5  — большой эффект

    Параметры:
        r — значение ранг-бисериальной корреляции.

    Возвращает:
        Строка с интерпретацией на русском языке.
    """
    abs_r = abs(r)
    if abs_r < 0.1:
        return "Незначительный эффект"
    if abs_r < 0.3:
        return "Малый эффект"
    if abs_r < RANK_BISERIAL_LARGE:
        return "Средний эффект"
    return "Большой эффект"


def _interpret_cramers_v(v: float) -> str:
    """
    Возвращает текстовую интерпретацию коэффициента V Крамера.

    Пороги (Cohen, 1988):
        V < 0.1  — связь отсутствует / пренебрежимо мала
        V < 0.3  — слабая связь
        V < 0.5  — умеренная связь
        V ≥ 0.5  — сильная связь

    Параметры:
        v — значение V Крамера.

    Возвращает:
        Строка с интерпретацией на русском языке.
    """
    if v < 0.1:
        return "Связь отсутствует"
    if v < 0.3:
        return "Слабая связь"
    if v < CRAMERS_V_STRONG:
        return "Умеренная связь"
    return "Сильная связь"
