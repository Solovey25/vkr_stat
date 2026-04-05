"""
statistics_analyzer_service.py — Сервис расширенного статистического анализа.

Предоставляет продвинутые инструменты анализа данных:
    1. Расширенные описательные метрики: асимметрия, эксцесс, SEM.
    2. Smart Distribution Fitting (MLE): автоподбор наилучшего распределения
       с оценкой по критерию Колмогорова-Смирнова.
    3. Генерация кривой PDF для визуализации теоретического распределения.
    4. Матрица корреляций (Пирсон / Спирмен).

Используется в рамках ВКР: «Веб-приложение для статистического анализа и прогнозирования».
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# Набор распределений для автоподбора (название → объект scipy.stats)
_CANDIDATE_DISTRIBUTIONS: dict[str, sp_stats.rv_continuous] = {
    "norm": sp_stats.norm,
    "lognorm": sp_stats.lognorm,
    "expon": sp_stats.expon,
}


class StatisticsAnalyzerService:
    """
    Сервис расширенного статистического анализа табличных данных.

    Все методы — статические: принимают данные, возвращают результат,
    не хранят внутреннего состояния.
    """

    # ==================================================================
    # 1. Расширенные описательные метрики
    # ==================================================================

    @staticmethod
    def compute_extended_stats(df: pd.DataFrame) -> dict[str, dict[str, float]]:
        """
        Рассчитывает расширенные описательные статистики для всех числовых столбцов.

        Метрики для каждого столбца:
            - count   — количество непустых значений
            - mean    — среднее арифметическое
            - median  — медиана (50-й перцентиль)
            - std     — стандартное отклонение (выборочное)
            - min/max — минимум и максимум
            - q25/q75 — первый и третий квартили
            - skewness — асимметрия (Skewness): > 0 — правый хвост длиннее,
                         < 0 — левый хвост длиннее, ≈ 0 — симметрично
            - kurtosis — эксцесс (Kurtosis): > 0 — острая вершина,
                         < 0 — плоская вершина, ≈ 0 — нормальное распределение
            - sem      — стандартная ошибка среднего (SEM = std / √n)

        Параметры:
            df — исходный DataFrame.

        Возвращает:
            Словарь {имя_столбца: {метрика: значение}}.
        """
        numeric_df = df.select_dtypes(include="number")
        result: dict[str, dict[str, float]] = {}

        for col in numeric_df.columns:
            series = numeric_df[col].dropna()

            if len(series) == 0:
                continue

            std_val = float(series.std())
            is_constant = series.nunique() <= 1 or std_val == 0.0

            result[col] = {
                "count": float(len(series)),
                "mean": float(series.mean()),
                "median": float(series.median()),
                "std": std_val,
                "min": float(series.min()),
                "max": float(series.max()),
                "q25": float(series.quantile(0.25)),
                "q75": float(series.quantile(0.75)),
                "skewness": 0.0 if is_constant else float(series.skew()),
                "kurtosis": 0.0 if is_constant else float(series.kurtosis()),
                "sem": 0.0 if is_constant else float(series.sem()),
                "is_constant": is_constant,
            }

        return result

    # ==================================================================
    # 2. Smart Distribution Fitting (MLE + KS-тест)
    # ==================================================================

    @staticmethod
    def fit_best_distribution(
        series: pd.Series,
    ) -> dict[str, Any]:
        """
        Подбирает наилучшее теоретическое распределение для выборки методом MLE.

        Алгоритм:
            1. Для каждого кандидатного распределения (norm, lognorm, expon)
               подгоняем параметры методом максимального правдоподобия (MLE).
            2. Оцениваем качество подгонки критерием Колмогорова-Смирнова (KS-тест).
            3. Выбираем распределение с наибольшим p-value (лучшее совпадение).

        Особенность для Пуассона: так как Пуассон — дискретное распределение,
        оно обрабатывается отдельно через подсчёт среднего и KS-тест
        против генерированной выборки.

        Параметры:
            series — числовой pd.Series (колонка DataFrame).

        Возвращает:
            Словарь с результатами:
            {
                "best_distribution": str,       # Название лучшего распределения
                "best_params": dict,            # Параметры распределения
                "best_ks_statistic": float,     # KS-статистика лучшего
                "best_p_value": float,          # p-value лучшего
                "all_results": [...]            # Результаты по всем распределениям
            }
        """
        data = series.dropna().values.astype(float)

        if len(data) < 5:
            return {
                "best_distribution": None,
                "best_params": {},
                "best_ks_statistic": None,
                "best_p_value": None,
                "all_results": [],
            }

        all_results: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None

        # --- Непрерывные распределения ---
        for name, dist in _CANDIDATE_DISTRIBUTIONS.items():
            try:
                # Для lognorm данные должны быть > 0
                if name == "lognorm" and np.any(data <= 0):
                    continue

                # Подгоняем параметры методом MLE
                params = dist.fit(data)

                # KS-тест: сравниваем эмпирическое распределение с теоретическим
                ks_stat, p_value = sp_stats.kstest(data, name, args=params)

                # Сохраняем параметры в читаемом формате
                param_names = _get_param_names(name, params)

                entry = {
                    "distribution": name,
                    "params": param_names,
                    "ks_statistic": round(float(ks_stat), 6),
                    "p_value": round(float(p_value), 6),
                }
                all_results.append(entry)

                # Лучшее — максимальный p-value (наименьшее отклонение)
                if best is None or p_value > best["p_value"]:
                    best = entry

            except (RuntimeError, ValueError, OverflowError):
                # Оптимизация MLE не сошлась — пропускаем распределение
                continue

        # --- Дискретное распределение Пуассона ---
        # Пуассон применим если данные — неотрицательные целые числа.
        # Важно: критерий Колмогорова-Смирнова предназначен для непрерывных
        # распределений и даёт заниженные p-value для дискретных.
        # Поэтому используем критерий хи-квадрат (chi-squared goodness-of-fit),
        # который корректно работает с дискретными данными.
        if np.all(data >= 0) and np.allclose(data, np.round(data)):
            try:
                lam = float(np.mean(data))
                int_data = np.round(data).astype(int)

                # Подсчитываем наблюдаемые частоты для каждого уникального значения
                unique_vals, observed_freq = np.unique(int_data, return_counts=True)

                # Рассчитываем ожидаемые частоты по закону Пуассона
                expected_freq = (
                    sp_stats.poisson.pmf(unique_vals, mu=lam) * len(int_data)
                )

                # Объединяем ячейки с ожидаемой частотой < 5
                # (требование корректности хи-квадрат теста)
                obs_merged: list[float] = []
                exp_merged: list[float] = []
                obs_acc, exp_acc = 0.0, 0.0

                for o, e in zip(observed_freq, expected_freq):
                    obs_acc += o
                    exp_acc += e
                    if exp_acc >= 5:
                        obs_merged.append(obs_acc)
                        exp_merged.append(exp_acc)
                        obs_acc, exp_acc = 0.0, 0.0

                # Дописываем остаток в последнюю ячейку
                if exp_acc > 0:
                    if exp_merged:
                        obs_merged[-1] += obs_acc
                        exp_merged[-1] += exp_acc
                    else:
                        obs_merged.append(obs_acc)
                        exp_merged.append(exp_acc)

                # Хи-квадрат тест (df = кол-во ячеек − 1 − 1 оценённый параметр)
                if len(obs_merged) > 2:
                    chi2_stat, p_value = sp_stats.chisquare(
                        obs_merged, f_exp=exp_merged, ddof=1,
                    )
                    entry = {
                        "distribution": "poisson",
                        "params": {"lambda": round(lam, 6)},
                        "ks_statistic": round(float(chi2_stat), 6),
                        "p_value": round(float(p_value), 6),
                    }
                    all_results.append(entry)

                    if best is None or p_value > best["p_value"]:
                        best = entry

            except (RuntimeError, ValueError):
                pass

        if best is None:
            return {
                "best_distribution": None,
                "best_params": {},
                "best_ks_statistic": None,
                "best_p_value": None,
                "all_results": [],
            }

        return {
            "best_distribution": best["distribution"],
            "best_params": best["params"],
            "best_ks_statistic": best["ks_statistic"],
            "best_p_value": best["p_value"],
            "all_results": all_results,
        }

    # ==================================================================
    # 3. Генерация кривой PDF
    # ==================================================================

    @staticmethod
    def generate_pdf_curve(
        series: pd.Series,
        distribution: str,
        params: dict[str, float],
        n_points: int = 200,
    ) -> dict[str, list[float]]:
        """
        Генерирует координаты (X, Y) кривой плотности вероятности (PDF)
        для указанного распределения.

        Используется для наложения теоретической кривой поверх гистограммы
        на фронтенде. Кривая строится в диапазоне [min − δ, max + δ],
        где δ = 10% от размаха данных.

        Параметры:
            series       — исходные данные (для определения диапазона по X).
            distribution — название распределения ("norm", "lognorm", "expon", "poisson").
            params       — параметры распределения (из fit_best_distribution).
            n_points     — количество точек кривой (по умолчанию 200).

        Возвращает:
            {"x": [float, ...], "y": [float, ...]} — координаты для построения.
        """
        data = series.dropna().values.astype(float)

        if len(data) == 0:
            return {"x": [], "y": []}

        # Определяем диапазон по X с запасом 10%
        data_min, data_max = float(np.min(data)), float(np.max(data))
        margin = (data_max - data_min) * 0.1 if data_max > data_min else 1.0
        x = np.linspace(data_min - margin, data_max + margin, n_points)

        # Рассчитываем PDF в зависимости от распределения
        if distribution == "poisson":
            # Пуассон — дискретный: используем PMF округлённых X
            lam = params.get("lambda", 1.0)
            x_int = np.arange(max(0, int(data_min)), int(data_max) + 2)
            y = sp_stats.poisson.pmf(x_int, mu=lam)
            return {
                "x": x_int.tolist(),
                "y": y.tolist(),
            }

        # Непрерывные распределения — восстанавливаем параметры scipy из dict
        if distribution == "norm":
            y = sp_stats.norm.pdf(x, loc=params.get("loc", 0), scale=params.get("scale", 1))
        elif distribution == "lognorm":
            y = sp_stats.lognorm.pdf(
                x, params.get("s", 1), loc=params.get("loc", 0), scale=params.get("scale", 1),
            )
        elif distribution == "expon":
            y = sp_stats.expon.pdf(x, loc=params.get("loc", 0), scale=params.get("scale", 1))
        else:
            return {"x": [], "y": []}

        return {
            "x": x.tolist(),
            "y": y.tolist(),
        }

    # ==================================================================
    # 4. Матрица корреляций
    # ==================================================================

    @staticmethod
    def compute_correlation_matrix(
        df: pd.DataFrame,
        method: Literal["pearson", "spearman"] = "pearson",
    ) -> dict[str, Any]:
        """
        Рассчитывает матрицу корреляций для всех числовых столбцов.

        Методы:
            - "pearson"  — коэффициент Пирсона (линейная корреляция).
            - "spearman" — коэффициент Спирмена (ранговая корреляция,
                          устойчив к выбросам и нелинейным зависимостям).

        Параметры:
            df     — исходный DataFrame.
            method — метод расчёта корреляции.

        Возвращает:
            Словарь:
            {
                "method": str,
                "columns": list[str],
                "matrix": list[list[float]],  — матрица значений (строка за строкой)
            }
        """
        numeric_df = df.select_dtypes(include="number")

        if numeric_df.empty:
            return {"method": method, "columns": [], "matrix": []}

        corr_matrix = numeric_df.corr(method=method)

        # Заменяем NaN на 0.0 (может появиться при нулевой дисперсии)
        corr_matrix = corr_matrix.fillna(0.0)

        return {
            "method": method,
            "columns": corr_matrix.columns.tolist(),
            "matrix": corr_matrix.round(4).values.tolist(),
        }


# ==================================================================
# Вспомогательные функции (модульный уровень)
# ==================================================================

def _get_param_names(dist_name: str, params: tuple) -> dict[str, float]:
    """
    Преобразует кортеж параметров scipy.stats.fit() в именованный словарь.

    Для каждого распределения scipy возвращает параметры в определённом порядке:
        - norm: (loc, scale)
        - expon: (loc, scale)
        - lognorm: (s, loc, scale)

    Параметры:
        dist_name — название распределения.
        params    — кортеж параметров от .fit().

    Возвращает:
        Словарь {имя_параметра: значение}.
    """
    result: dict[str, float] = {}

    if dist_name == "norm":
        result["loc"] = round(float(params[0]), 6)
        result["scale"] = round(float(params[1]), 6)
    elif dist_name == "expon":
        result["loc"] = round(float(params[0]), 6)
        result["scale"] = round(float(params[1]), 6)
    elif dist_name == "lognorm":
        result["s"] = round(float(params[0]), 6)
        result["loc"] = round(float(params[1]), 6)
        result["scale"] = round(float(params[2]), 6)
    else:
        # Общий случай — нумеруем параметры
        for i, p in enumerate(params):
            result[f"param_{i}"] = round(float(p), 6)

    return result
