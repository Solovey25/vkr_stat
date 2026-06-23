"""
comparative_service.py — Сервис попарного сравнения двух датасетов (Data Drift Detection).

Позволяет загрузить второй файл и сопоставить его с текущим рабочим файлом,
рассчитав математическую разницу и доказав её статистическую значимость.

Основные возможности:
    1. Сравнение числовых колонок: разница средних, PSI, KS-тест, парные тесты.
    2. Сравнение категориальных колонок: хи-квадрат, V Крамера, доли категорий.
    3. Анализ структурных изменений: добавление/удаление колонок, смена типов.
    4. Анализ качества данных: рост пропусков, новые категории.

Используется в рамках ВКР: «Веб-приложение для статистического анализа и прогнозирования».
"""

from __future__ import annotations

from typing import Any

import numpy as np

from config import PSI_MODERATE, PSI_SIGNIFICANT, EFFECT_NEGLIGIBLE
import pandas as pd
from scipy import stats as sp_stats
from statsmodels.stats.multitest import multipletests

from services.hypothesis_engine_service import HypothesisEngineService


class ComparativeService:
    """
    Сервис попарного сравнения двух датасетов.

    Все методы — статические: принимают данные, возвращают результат,
    не хранят внутреннего состояния.
    """

    # ==================================================================
    # 1. Поиск совпадающих колонок
    # ==================================================================

    @staticmethod
    def find_common_numeric_columns(
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
    ) -> list[str]:
        """
        Находит числовые колонки с одинаковыми именами в обоих датасетах.

        Параметры:
            df_a — первый DataFrame (база).
            df_b — второй DataFrame (сравнение).

        Возвращает:
            Отсортированный список имён совпадающих числовых колонок.
        """
        numeric_a = set(df_a.select_dtypes(include="number").columns)
        numeric_b = set(df_b.select_dtypes(include="number").columns)
        return sorted(numeric_a & numeric_b)

    @staticmethod
    def find_common_categorical_columns(
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
    ) -> list[str]:
        """
        Находит категориальные (object) колонки с одинаковыми именами в обоих датасетах.

        Параметры:
            df_a — первый DataFrame (база).
            df_b — второй DataFrame (сравнение).

        Возвращает:
            Отсортированный список имён совпадающих категориальных колонок.
        """
        cat_a = set(
            df_a.select_dtypes(include="object")
            .select_dtypes(exclude=["datetime", "datetime64"])
            .columns
        )
        cat_b = set(
            df_b.select_dtypes(include="object")
            .select_dtypes(exclude=["datetime", "datetime64"])
            .columns
        )
        return sorted(cat_a & cat_b)

    # ==================================================================
    # 2. Расчёт Population Stability Index (PSI)
    # ==================================================================

    @staticmethod
    def calculate_psi(
        base_series: pd.Series,
        curr_series: pd.Series,
        buckets: int = 10,
    ) -> float:
        """
        Рассчитывает Population Stability Index (PSI) для оценки дрифта
        распределения числовой переменной.

        Алгоритм:
            1. Квантильное разбиение по base_series на buckets корзин.
            2. Подсчёт долей наблюдений в каждой корзине для обеих выборок.
            3. Замена нулевых долей на 0.0001 (избежание log(0)).
            4. PSI = Σ (actual% − expected%) × ln(actual% / expected%).

        Интерпретация (стандартные пороги):
            PSI < 0.1  — нет дрифта (распределение стабильно).
            PSI < 0.2  — умеренный дрифт (требует внимания).
            PSI ≥ 0.2  — значительный дрифт (распределение изменилось).

        Параметры:
            base_series — базовая выборка (expected distribution).
            curr_series — сравниваемая выборка (actual distribution).
            buckets     — количество корзин квантильного разбиения (по умолчанию 10).

        Возвращает:
            Значение PSI (float ≥ 0).
        """
        base = base_series.dropna().values.astype(float)
        curr = curr_series.dropna().values.astype(float)

        if len(base) < buckets or len(curr) < buckets:
            return 0.0

        # Квантильные границы по базовому распределению
        quantiles = np.linspace(0, 100, buckets + 1)
        bins = np.percentile(base, quantiles)
        bins = np.unique(bins)  # Убираем дубликаты при константных значениях

        if len(bins) < 2:
            return 0.0

        # Подсчёт долей в корзинах
        base_counts = np.histogram(base, bins=bins)[0]
        curr_counts = np.histogram(curr, bins=bins)[0]

        base_pct = base_counts / len(base)
        curr_pct = curr_counts / len(curr)

        # Замена нулей для избежания деления на 0 и log(0)
        base_pct = np.where(base_pct == 0, 0.0001, base_pct)
        curr_pct = np.where(curr_pct == 0, 0.0001, curr_pct)

        psi = float(np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct)))
        return round(psi, 6)

    # ==================================================================
    # 3. Сравнение числовых колонок (Numerical Drift)
    # ==================================================================

    @staticmethod
    def compare_datasets(
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
        alpha: float = 0.05,
        id_column: str | None = None,
    ) -> dict[str, Any]:
        """
        Сравнивает два датасета по всем совпадающим числовым колонкам.

        Для каждой пары колонок рассчитывает:
            - Дельта-статистику: абсолютное и процентное отклонение средних.
            - PSI (Population Stability Index) для оценки дрифта распределения.
            - KS-тест (Колмогорова-Смирнова) для проверки сдвига формы.
            - Статистический тест сравнения средних: автовыбор критерия
              (независимые или парные выборки).
            - Размер эффекта: d Коэна (для t-тестов).
            - Вердикт: «Значимый рост», «Значимое падение» или «Различия отсутствуют».

        Параметры:
            df_a      — первый DataFrame (база, Dataset A).
            df_b      — второй DataFrame (сравнение, Dataset B).
            alpha     — уровень значимости (по умолчанию 0.05).
            id_column — колонка-идентификатор для парного сравнения (опционально).

        Возвращает:
            Словарь:
            {
                "common_columns": list[str],  — совпавшие числовые колонки
                "results": list[dict],        — результаты по каждой колонке
            }
        """
        common_cols = ComparativeService.find_common_numeric_columns(df_a, df_b)

        if not common_cols:
            return {"common_columns": [], "results": []}

        # --- Подготовка парного режима ---
        is_paired_mode = False
        paired_cancelled_reason: str | None = None
        if (
            id_column
            and id_column in df_a.columns
            and id_column in df_b.columns
        ):
            # Проверка уникальности ID — дубликаты исказят merge и парный тест
            if not df_a[id_column].is_unique or not df_b[id_column].is_unique:
                paired_cancelled_reason = (
                    "Парный тест отменён: обнаружены дубликаты в ID-колонке"
                )
            else:
                common_ids = (
                    df_a[[id_column]].drop_duplicates()
                    .merge(df_b[[id_column]].drop_duplicates(), on=id_column)
                )
                if len(common_ids) >= 3:
                    is_paired_mode = True

        results: list[dict[str, Any]] = []

        for col in common_cols:
            sample_a = df_a[col].dropna()
            sample_b = df_b[col].dropna()

            # --- Дельта-статистика ---
            mean_a = float(sample_a.mean()) if len(sample_a) > 0 else 0.0
            mean_b = float(sample_b.mean()) if len(sample_b) > 0 else 0.0
            std_a = float(sample_a.std()) if len(sample_a) > 1 else 0.0
            std_b = float(sample_b.std()) if len(sample_b) > 1 else 0.0

            delta = round(float(mean_b - mean_a), 6)

            if mean_a != 0:
                delta_percent = round(float(((mean_b - mean_a) / mean_a) * 100), 6)
            else:
                delta_percent = None

            # --- PSI ---
            psi_value = ComparativeService.calculate_psi(df_a[col], df_b[col])

            # --- KS-тест (Колмогорова-Смирнова) ---
            if len(sample_a) >= 2 and len(sample_b) >= 2:
                ks_stat_val, ks_p_val = sp_stats.ks_2samp(sample_a, sample_b)
                ks_stat_val = round(float(ks_stat_val), 6)
                ks_p_val = round(float(ks_p_val), 6)
            else:
                ks_stat_val, ks_p_val = 0.0, 1.0

            is_shape_drifted = ks_p_val < alpha or psi_value >= PSI_SIGNIFICANT

            # --- Тест сравнения средних ---
            is_paired = False
            paired_test_name: str | None = None
            test_result: dict[str, Any] | None = None
            verdict: str = "Недостаточно данных"

            # Попытка парного теста
            if is_paired_mode and col != id_column:
                try:
                    paired_df = (
                        df_a[[id_column, col]]
                        .merge(
                            df_b[[id_column, col]],
                            on=id_column,
                            suffixes=("_a", "_b"),
                        )
                        .dropna()
                    )
                    if len(paired_df) >= 3:
                        diff = paired_df[f"{col}_b"] - paired_df[f"{col}_a"]
                        _, diff_p = sp_stats.shapiro(diff)

                        if diff_p >= alpha:
                            # Разности нормальны → парный t-тест
                            stat, p_val = sp_stats.ttest_rel(
                                paired_df[f"{col}_a"], paired_df[f"{col}_b"],
                            )
                            paired_test_name = "Парный t-тест"
                        else:
                            # Разности ненормальны → критерий Уилкоксона
                            stat, p_val = sp_stats.wilcoxon(
                                paired_df[f"{col}_a"], paired_df[f"{col}_b"],
                            )
                            paired_test_name = "Критерий Уилкоксона"

                        # Размер эффекта d Коэна для парного теста
                        diff_mean = float(diff.mean())
                        diff_std = float(diff.std())
                        paired_cohens_d = (
                            round(abs(diff_mean / diff_std), 6)
                            if diff_std > 0 else 0.0
                        )

                        is_paired = True
                        test_result = {
                            "test_name": paired_test_name,
                            "statistic": round(float(stat), 6),
                            "p_value": round(float(p_val), 6),
                            "cohens_d": paired_cohens_d,
                        }
                except (ValueError, KeyError):
                    pass  # Fallback на независимый тест

            # Fallback: независимый тест
            if test_result is None:
                try:
                    test_result = HypothesisEngineService.compare_two_groups(
                        df_a[col], df_b[col], alpha=alpha,
                    )
                except ValueError:
                    pass  # verdict останется "Недостаточно данных"

            # --- Вердикт ---
            if test_result is not None:
                p_value = test_result["p_value"]
                effect_d = test_result.get("cohens_d")

                if p_value < alpha:
                    # Защита от ложной значимости: p < 0.05, но эффект мал
                    if effect_d is not None and abs(effect_d) < EFFECT_NEGLIGIBLE:
                        verdict = "Различия значимы, но эффект мал"
                    elif delta > 0:
                        verdict = "Значимый рост"
                    elif delta < 0:
                        verdict = "Значимое падение"
                    else:
                        verdict = "Значимое различие"
                else:
                    verdict = "Различия отсутствуют"

            results.append({
                "column": col,
                "n_a": len(sample_a),
                "n_b": len(sample_b),
                "mean_a": round(mean_a, 6),
                "mean_b": round(mean_b, 6),
                "std_a": round(std_a, 6),
                "std_b": round(std_b, 6),
                "delta": delta,
                "delta_percent": delta_percent,
                "test_name": test_result["test_name"] if test_result else None,
                "statistic": test_result["statistic"] if test_result else None,
                "p_value": test_result["p_value"] if test_result else None,
                "cohens_d": test_result["cohens_d"] if test_result else None,
                "psi": psi_value,
                "psi_interpretation": _interpret_psi(psi_value),
                "ks_stat": ks_stat_val,
                "ks_p_value": ks_p_val,
                "is_shape_drifted": is_shape_drifted,
                "is_paired": is_paired,
                "paired_test_name": paired_test_name,
                "paired_cancelled_reason": paired_cancelled_reason,
                "verdict": verdict,
            })

        # --- FDR-коррекция Benjamini-Hochberg ---
        raw_pvals = [r["p_value"] for r in results if r["p_value"] is not None]
        correction_method = None
        if len(raw_pvals) > 1:
            correction_method = "benjamini-hochberg"
            reject, corrected, _, _ = multipletests(raw_pvals, alpha=alpha, method="fdr_bh")
            idx = 0
            for r in results:
                if r["p_value"] is not None:
                    r["p_value_corrected"] = round(float(corrected[idx]), 6)
                    # Пересчитать вердикт по скорректированному p-value
                    if not reject[idx]:
                        r["verdict"] = "Различия отсутствуют"
                    idx += 1

        return {
            "common_columns": common_cols,
            "results": results,
            "correction_method": correction_method,
        }

    # ==================================================================
    # 4. Анализ структурных изменений
    # ==================================================================

    @staticmethod
    def analyze_structural_changes(
        df_base: pd.DataFrame,
        df_compare: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Анализирует структурные различия между двумя датасетами.

        Определяет:
            - Изменение числа строк (абсолютное и процентное).
            - Добавленные колонки (есть в compare, нет в base).
            - Удалённые колонки (есть в base, нет в compare).
            - Колонки с изменённым типом данных.

        Параметры:
            df_base    — базовый DataFrame (точка отсчёта).
            df_compare — сравниваемый DataFrame.

        Возвращает:
            Словарь с результатами структурного анализа.
        """
        rows_base = len(df_base)
        rows_compare = len(df_compare)
        rows_delta = rows_compare - rows_base

        if rows_base != 0:
            rows_delta_percent = round((rows_delta / rows_base) * 100, 2)
        else:
            rows_delta_percent = None

        cols_base = set(df_base.columns)
        cols_compare = set(df_compare.columns)

        added_columns = sorted(cols_compare - cols_base)
        removed_columns = sorted(cols_base - cols_compare)

        # Колонки с изменённым типом данных
        common_cols = sorted(cols_base & cols_compare)
        type_changed: dict[str, dict[str, str]] = {}
        for col in common_cols:
            dtype_base = str(df_base[col].dtype)
            dtype_compare = str(df_compare[col].dtype)
            if dtype_base != dtype_compare:
                type_changed[col] = {
                    "base": dtype_base,
                    "compare": dtype_compare,
                }

        return {
            "rows_base": rows_base,
            "rows_compare": rows_compare,
            "rows_delta": rows_delta,
            "rows_delta_percent": rows_delta_percent,
            "added_columns": added_columns,
            "removed_columns": removed_columns,
            "type_changed_columns": type_changed,
        }

    # ==================================================================
    # 5. Анализ качества данных
    # ==================================================================

    @staticmethod
    def analyze_quality_changes(
        df_base: pd.DataFrame,
        df_compare: pd.DataFrame,
        common_columns: list[str],
    ) -> list[dict[str, Any]]:
        """
        Сравнивает качество данных между двумя датасетами по общим колонкам.

        Для каждой общей колонки:
            - Рассчитывает процент пропусков (NaN) в обоих датасетах.
            - Если в df_compare пропусков стало больше на > 5 п.п., помечает
              quality_degraded = True.
            - Для категориальных (object) колонок находит новые уникальные
              значения, которые появились только в df_compare.

        Параметры:
            df_base        — базовый DataFrame.
            df_compare     — сравниваемый DataFrame.
            common_columns — список общих колонок для анализа.

        Возвращает:
            Список словарей с отчётом по каждой колонке.
        """
        results: list[dict[str, Any]] = []
        rows_base = len(df_base)
        rows_compare = len(df_compare)

        for col in common_columns:
            # --- Процент пропусков ---
            missing_base = int(df_base[col].isna().sum())
            missing_compare = int(df_compare[col].isna().sum())

            missing_base_pct = round(
                (missing_base / rows_base * 100) if rows_base > 0 else 0.0, 2,
            )
            missing_compare_pct = round(
                (missing_compare / rows_compare * 100) if rows_compare > 0 else 0.0, 2,
            )
            missing_delta_pct = round(missing_compare_pct - missing_base_pct, 2)
            quality_degraded = missing_delta_pct > 5.0

            # --- Новые категории (только для object-колонок) ---
            new_categories: list[str] = []
            if (
                df_base[col].dtype == object
                and df_compare[col].dtype == object
            ):
                cats_base = set(df_base[col].dropna().unique())
                cats_compare = set(df_compare[col].dropna().unique())
                new_categories = sorted(
                    str(v) for v in (cats_compare - cats_base)
                )

            results.append({
                "column": col,
                "missing_base_pct": missing_base_pct,
                "missing_compare_pct": missing_compare_pct,
                "missing_delta_pct": missing_delta_pct,
                "quality_degraded": quality_degraded,
                "new_categories": new_categories,
            })

        return results

    # ==================================================================
    # 6. Анализ дрифта категориальных колонок
    # ==================================================================

    @staticmethod
    def compare_categorical_columns(
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
        alpha: float = 0.05,
    ) -> list[dict[str, Any]]:
        """
        Сравнивает распределения категориальных колонок между двумя датасетами
        с помощью критерия хи-квадрат для обнаружения категориального дрифта.

        Алгоритм для каждой общей категориальной колонки:
            1. Вычислить value_counts (доли) в обоих датасетах.
            2. Объединить все уникальные категории.
            3. Построить таблицу сопряжённости 2×K (base counts vs compare counts).
            4. Применить scipy.stats.chi2_contingency.
            5. Рассчитать коэффициент V Крамера.
            6. Пометить is_drifted = True если p < alpha.

        Параметры:
            df_a  — базовый DataFrame.
            df_b  — сравниваемый DataFrame.
            alpha — уровень значимости (по умолчанию 0.05).

        Возвращает:
            Список словарей с результатами по каждой категориальной колонке.
        """
        common_cats = ComparativeService.find_common_categorical_columns(df_a, df_b)
        results: list[dict[str, Any]] = []

        for col in common_cats:
            vals_a = df_a[col].dropna()
            vals_b = df_b[col].dropna()

            if len(vals_a) == 0 or len(vals_b) == 0:
                continue

            # Доли категорий
            counts_a = vals_a.value_counts()
            counts_b = vals_b.value_counts()

            props_a = (counts_a / counts_a.sum()).to_dict()
            props_b = (counts_b / counts_b.sum()).to_dict()

            base_proportions = {str(k): round(float(v), 6) for k, v in props_a.items()}
            compare_proportions = {str(k): round(float(v), 6) for k, v in props_b.items()}

            # Объединённый набор категорий
            all_categories = sorted(set(counts_a.index) | set(counts_b.index))

            if len(all_categories) < 2:
                continue  # Нельзя провести хи-квадрат с одной категорией

            # Таблица сопряжённости: 2 строки (base, compare) × K категорий
            row_base = [int(counts_a.get(cat, 0)) for cat in all_categories]
            row_compare = [int(counts_b.get(cat, 0)) for cat in all_categories]
            contingency = np.array([row_base, row_compare])

            try:
                chi2, p_value, dof, expected = sp_stats.chi2_contingency(contingency)
            except ValueError:
                continue

            # Проверка условия Кохрана
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

            # V Крамера
            n = int(contingency.sum())
            k = min(contingency.shape[0], contingency.shape[1])
            cramers_v = float(np.sqrt(chi2 / (n * (k - 1)))) if k > 1 else 0.0

            results.append({
                "column": col,
                "chi2_stat": round(float(chi2), 6),
                "chi2_p_value": round(float(p_value), 6),
                "cramers_v": round(cramers_v, 6),
                "is_drifted": p_value < alpha,
                "base_proportions": base_proportions,
                "compare_proportions": compare_proportions,
                "cochran_warning": cochran_warning,
            })

        # --- FDR-коррекция Benjamini-Hochberg ---
        raw_pvals = [r["chi2_p_value"] for r in results]
        correction_method = None
        if len(raw_pvals) > 1:
            correction_method = "benjamini-hochberg"
            reject, corrected, _, _ = multipletests(raw_pvals, alpha=alpha, method="fdr_bh")
            for i, r in enumerate(results):
                r["p_value_corrected"] = round(float(corrected[i]), 6)
                r["is_drifted"] = bool(reject[i])

        return {"items": results, "correction_method": correction_method}


# ==================================================================
# Вспомогательные функции (модульный уровень)
# ==================================================================

def _interpret_psi(psi: float) -> str:
    """
    Возвращает текстовую интерпретацию PSI.

    Пороги:
        PSI < 0.1  — нет дрифта
        PSI < 0.2  — умеренный дрифт
        PSI ≥ 0.2  — значительный дрифт
    """
    if psi < PSI_MODERATE:
        return "Нет дрифта"
    if psi < PSI_SIGNIFICANT:
        return "Умеренный дрифт"
    return "Значительный дрифт"
