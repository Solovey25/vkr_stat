"""
data_sanitizer_service.py — Сервис предобработки (очистки) данных.

Предоставляет атомарные операции над DataFrame:
    1. Обработка пропусков (NaN): удаление строк или заполнение (mean/median/mode).
    2. Детекция выбросов по методу IQR (межквартильный размах).
    3. Удаление выбросов за пределами [Q1 − 1.5·IQR, Q3 + 1.5·IQR].
    4. Масштабирование признаков: StandardScaler (Z-score) и MinMaxScaler.

Принцип работы — «сначала рассчитываем, потом применяем»:
    - Методы get_*  возвращают информацию (количество пропусков, границы выбросов).
    - Методы drop_* / fill_* / remove_* / scale_* возвращают новый DataFrame,
      не изменяя исходный (иммутабельность).

Используется в рамках ВКР: «Веб-приложение для статистического анализа и прогнозирования».
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class DataSanitizerService:
    """
    Сервис предобработки табличных данных.

    Все методы принимают DataFrame и возвращают результат (новый DataFrame
    или информационный словарь), не мутируя исходные данные.
    """

    # ==================================================================
    # 1. Обработка пропусков (NaN)
    # ==================================================================

    @staticmethod
    def get_missing_info(df: pd.DataFrame) -> list[dict[str, Any]]:
        """
        Рассчитывает статистику пропусков для каждого столбца.

        Параметры:
            df — исходный DataFrame.

        Возвращает:
            Список словарей с информацией о пропусках:
            [{"column": str, "missing_count": int, "missing_percent": float, "dtype": str}, ...]
        """
        total_rows = len(df)
        result: list[dict[str, Any]] = []

        for col in df.columns:
            missing = int(df[col].isna().sum())
            percent = round(missing / total_rows * 100, 2) if total_rows > 0 else 0.0
            result.append({
                "column": col,
                "missing_count": missing,
                "missing_percent": percent,
                "dtype": str(df[col].dtype),
            })

        return result

    @staticmethod
    def drop_missing(
        df: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> tuple[pd.DataFrame, int]:
        """
        Удаляет строки, содержащие пропуски (NaN).

        Параметры:
            df      — исходный DataFrame.
            columns — список столбцов для проверки (None = все столбцы).

        Возвращает:
            Кортеж (очищенный DataFrame, количество удалённых строк).
        """
        original_count = len(df)

        if columns:
            clean_df = df.dropna(subset=columns).reset_index(drop=True)
        else:
            clean_df = df.dropna().reset_index(drop=True)

        dropped = original_count - len(clean_df)
        return clean_df, dropped

    @staticmethod
    def fill_missing(
        df: pd.DataFrame,
        strategy: Literal["mean", "median", "most_frequent"] = "mean",
        columns: list[str] | None = None,
    ) -> tuple[pd.DataFrame, int]:
        """
        Заполняет пропуски в числовых столбцах с помощью SimpleImputer.

        Параметры:
            df       — исходный DataFrame.
            strategy — стратегия заполнения: "mean" (среднее), "median" (медиана),
                       "most_frequent" (мода).
            columns  — список числовых столбцов для заполнения (None = все числовые).

        Возвращает:
            Кортеж (DataFrame с заполненными пропусками, количество заполненных значений).
        """
        result_df = df.copy()

        # Определяем целевые числовые столбцы
        numeric_cols = result_df.select_dtypes(include="number").columns.tolist()
        if columns:
            target_cols = [c for c in columns if c in numeric_cols]
        else:
            target_cols = numeric_cols

        if not target_cols:
            return result_df, 0

        # Считаем количество пропусков до заполнения
        missing_before = int(result_df[target_cols].isna().sum().sum())

        # Применяем SimpleImputer из sklearn
        imputer = SimpleImputer(strategy=strategy)
        result_df[target_cols] = imputer.fit_transform(result_df[target_cols])

        # Считаем сколько значений было заполнено
        missing_after = int(result_df[target_cols].isna().sum().sum())
        filled_count = missing_before - missing_after

        return result_df, filled_count

    # ==================================================================
    # 2. Детекция выбросов (метод IQR)
    # ==================================================================

    @staticmethod
    def get_outliers_info(
        df: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Рассчитывает информацию о выбросах для числовых столбцов методом IQR.

        Метод межквартильного размаха (Interquartile Range):
            IQR = Q3 − Q1
            Нижняя граница = Q1 − 1.5 · IQR
            Верхняя граница = Q3 + 1.5 · IQR
            Выбросы — значения за пределами этих границ.

        Параметры:
            df      — исходный DataFrame.
            columns — список числовых столбцов (None = все числовые).

        Возвращает:
            Список словарей с информацией о выбросах для каждого столбца:
            [{"column": str, "q1": float, "q3": float, "iqr": float,
              "lower_bound": float, "upper_bound": float,
              "outliers_count": int, "outliers_percent": float, "total_rows": int}, ...]
        """
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if columns:
            target_cols = [c for c in columns if c in numeric_cols]
        else:
            target_cols = numeric_cols

        total_rows = len(df)
        result: list[dict[str, Any]] = []

        for col in target_cols:
            series = df[col].dropna()

            q1 = float(np.percentile(series, 25))
            q3 = float(np.percentile(series, 75))
            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            # Считаем выбросы — значения строго за пределами границ
            outliers_mask = (series < lower) | (series > upper)
            outliers_count = int(outliers_mask.sum())
            outliers_percent = (
                round(outliers_count / total_rows * 100, 2) if total_rows > 0 else 0.0
            )

            result.append({
                "column": col,
                "q1": round(q1, 4),
                "q3": round(q3, 4),
                "iqr": round(iqr, 4),
                "lower_bound": round(lower, 4),
                "upper_bound": round(upper, 4),
                "outliers_count": outliers_count,
                "outliers_percent": outliers_percent,
                "total_rows": total_rows,
            })

        return result

    # ==================================================================
    # 3. Удаление выбросов
    # ==================================================================

    @staticmethod
    def remove_outliers(
        df: pd.DataFrame,
        columns: list[str],
    ) -> tuple[pd.DataFrame, int]:
        """
        Удаляет строки, содержащие выбросы по методу IQR, для указанных столбцов.

        Строка удаляется, если хотя бы в одном из указанных столбцов
        значение выходит за границы [Q1 − 1.5·IQR, Q3 + 1.5·IQR].

        Параметры:
            df      — исходный DataFrame.
            columns — список числовых столбцов для проверки выбросов.

        Возвращает:
            Кортеж (очищенный DataFrame, количество удалённых строк).
        """
        original_count = len(df)
        mask = pd.Series(True, index=df.index)

        for col in columns:
            if col not in df.columns:
                continue

            series = df[col]
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            # Оставляем только значения внутри границ (NaN тоже оставляем)
            mask &= ((series >= lower) & (series <= upper)) | series.isna()

        clean_df = df[mask].reset_index(drop=True)
        dropped = original_count - len(clean_df)

        return clean_df, dropped

    # ==================================================================
    # 4. Кодирование категориальных столбцов
    # ==================================================================

    @staticmethod
    def encode_categorical_columns(
        df: pd.DataFrame,
        columns: list[str],
    ) -> tuple[pd.DataFrame, dict[str, dict[int, str]]]:
        """
        Кодирует категориальные (текстовые) столбцы числовыми кодами
        через pd.factorize.

        Параметры:
            df      — исходный DataFrame.
            columns — список категориальных столбцов для кодирования.

        Возвращает:
            Кортеж:
                - DataFrame с закодированными столбцами (коды 0, 1, 2, …;
                  NaN остаётся как -1).
                - Словарь маппингов {имя_столбца: {код: исходное_значение}}.
        """
        result_df = df.copy()
        mapping: dict[str, dict[int, str]] = {}

        for col in columns:
            if col not in result_df.columns:
                continue

            codes, uniques = pd.factorize(result_df[col], sort=True)
            result_df[col] = codes
            mapping[col] = {int(i): str(v) for i, v in enumerate(uniques)}

        return result_df, mapping

    # ==================================================================
    # 5. Масштабирование (Scaling)
    # ==================================================================

    @staticmethod
    def scale_standard(
        df: pd.DataFrame,
        columns: list[str],
    ) -> pd.DataFrame:
        """
        Применяет Z-преобразование (StandardScaler) к указанным числовым столбцам.

        Формула: z = (x − μ) / σ

        После масштабирования каждый признак имеет среднее ≈ 0 и СКО ≈ 1.

        Параметры:
            df      — исходный DataFrame.
            columns — список числовых столбцов для масштабирования.

        Возвращает:
            Новый DataFrame с масштабированными столбцами.
        """
        result_df = df.copy()
        valid_cols = [c for c in columns if c in result_df.select_dtypes(include="number").columns]

        if valid_cols:
            scaler = StandardScaler()
            result_df[valid_cols] = scaler.fit_transform(result_df[valid_cols])

        return result_df

    @staticmethod
    def scale_minmax(
        df: pd.DataFrame,
        columns: list[str],
    ) -> pd.DataFrame:
        """
        Применяет MinMax-масштабирование к указанным числовым столбцам.

        Формула: x' = (x − min) / (max − min)

        После масштабирования значения лежат в диапазоне [0, 1].

        Параметры:
            df      — исходный DataFrame.
            columns — список числовых столбцов для масштабирования.

        Возвращает:
            Новый DataFrame с масштабированными столбцами.
        """
        result_df = df.copy()
        valid_cols = [c for c in columns if c in result_df.select_dtypes(include="number").columns]

        if valid_cols:
            scaler = MinMaxScaler()
            result_df[valid_cols] = scaler.fit_transform(result_df[valid_cols])

        return result_df
