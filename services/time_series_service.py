"""
time_series_service.py — Сервис подготовки данных для анализа временных рядов.

Обеспечивает валидацию, диагностику и ресемплирование временных рядов
перед применением моделей прогнозирования (ARIMA и др.).

Основные возможности:
    1. Валидация временного ряда: проверка формата дат, сортировка,
       определение частоты (pd.infer_freq), подсчёт пропусков.
    2. Ресемплирование: агрегация данных по заданной частоте
       с заполнением пропущенных дат.

Используется в рамках ВКР: «Веб-приложение для статистического анализа и прогнозирования».
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd


# Человекочитаемые описания частот pandas
_FREQ_DESCRIPTIONS: dict[str, str] = {
    "D": "Дневная",
    "B": "Рабочие дни",
    "W": "Недельная",
    "MS": "Месячная (начало)",
    "ME": "Месячная (конец)",
    "M": "Месячная",
    "QS": "Квартальная (начало)",
    "QE": "Квартальная (конец)",
    "Q": "Квартальная",
    "YS": "Годовая (начало)",
    "YE": "Годовая (конец)",
    "Y": "Годовая",
    "h": "Часовая",
    "min": "Минутная",
    "s": "Секундная",
}

# Словарь агрегирующих функций для ресемплирования
_AGG_FUNCTIONS: dict[str, str] = {
    "mean": "Среднее",
    "sum": "Сумма",
    "median": "Медиана",
    "min": "Минимум",
    "max": "Максимум",
}


class TimeSeriesService:
    """
    Сервис подготовки временных рядов к прогнозированию.

    Все методы — статические: принимают данные, возвращают результат,
    не хранят внутреннего состояния.
    """

    # ==================================================================
    # 1. Валидация временного ряда
    # ==================================================================

    @staticmethod
    def validate_time_series(
        df: pd.DataFrame,
        date_col: str,
        value_col: str,
    ) -> dict[str, Any]:
        """
        Проверяет пригодность данных для анализа временных рядов.

        Алгоритм:
            1. Проверяет, что колонка дат может быть преобразована в datetime.
            2. Сортирует данные по дате.
            3. Определяет частоту ряда через pd.infer_freq.
            4. Подсчитывает пропуски в датах.
            5. Формирует диагностический отчёт.

        Параметры:
            df        — исходный DataFrame.
            date_col  — имя колонки с датами.
            value_col — имя колонки с числовыми значениями.

        Возвращает:
            Словарь с диагностикой:
            {
                "is_valid": bool,           — пригоден ли ряд для анализа
                "date_column": str,         — имя колонки дат
                "value_column": str,        — имя колонки значений
                "total_points": int,        — общее число точек
                "date_range_start": str,    — начало диапазона дат
                "date_range_end": str,      — конец диапазона дат
                "inferred_freq": str|None,  — определённая частота
                "freq_description": str,    — описание частоты на русском
                "is_regular": bool,         — регулярный ли ряд
                "gaps_count": int,          — количество пропусков
                "missing_values": int,      — пропуски в значениях (NaN)
                "suggestion": str|None,     — рекомендация
                "sorted_data": list[dict],  — отсортированные данные
            }

        Исключения:
            ValueError — если колонка дат не может быть преобразована.
        """
        # Проверяем наличие колонок
        if date_col not in df.columns:
            raise ValueError(f"Колонка «{date_col}» не найдена в данных.")
        if value_col not in df.columns:
            raise ValueError(f"Колонка «{value_col}» не найдена в данных.")

        # Пробуем преобразовать колонку дат
        work_df = df[[date_col, value_col]].copy()
        try:
            work_df[date_col] = pd.to_datetime(work_df[date_col], errors="coerce")
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Не удалось преобразовать колонку «{date_col}» в даты: {e}"
            )

        # Убираем строки с NaT в дате
        nat_count = int(work_df[date_col].isna().sum())
        work_df = work_df.dropna(subset=[date_col])

        if len(work_df) < 3:
            return {
                "is_valid": False,
                "date_column": date_col,
                "value_column": value_col,
                "total_points": len(work_df),
                "date_range_start": None,
                "date_range_end": None,
                "inferred_freq": None,
                "freq_description": "Недостаточно данных",
                "is_regular": False,
                "gaps_count": 0,
                "missing_values": 0,
                "suggestion": "Для анализа временных рядов необходимо минимум 3 точки с валидными датами.",
                "sorted_data": [],
            }

        # Сортируем по дате
        work_df = work_df.sort_values(date_col).reset_index(drop=True)

        # Подсчитываем NaN в значениях
        missing_values = int(work_df[value_col].isna().sum())

        # Определяем частоту
        try:
            idx = pd.DatetimeIndex(work_df[date_col].values)
            inferred_freq = pd.infer_freq(idx)
        except (ValueError, TypeError):
            inferred_freq = None

        freq_desc = "Не удалось определить"
        if inferred_freq is not None:
            base_freq = inferred_freq.lstrip("0123456789")
            freq_desc = _FREQ_DESCRIPTIONS.get(base_freq, inferred_freq)

        # Подсчитываем пропуски в датах
        date_series = work_df[date_col]
        diffs = date_series.diff().dropna()
        gaps_count = 0
        if len(diffs) > 0:
            median_diff = diffs.median()
            if median_diff > pd.Timedelta(0):
                gaps_count = int((diffs > median_diff * 1.5).sum())

        is_regular = inferred_freq is not None and gaps_count == 0

        # Формируем рекомендацию
        suggestion = None
        if not is_regular:
            if inferred_freq is None:
                suggestion = (
                    "Частота ряда не определена автоматически. "
                    "Выполните ресемплирование с нужной частотой "
                    "перед применением ARIMA."
                )
            elif gaps_count > 0:
                suggestion = (
                    f"Обнаружено {gaps_count} пропусков в датах. "
                    "Выполните ресемплирование для заполнения пропусков."
                )

        if missing_values > 0:
            mv_text = (
                f"В колонке значений обнаружено {missing_values} пропусков (NaN). "
                "Они будут заполнены при ресемплировании."
            )
            suggestion = f"{suggestion} {mv_text}" if suggestion else mv_text

        # Подготавливаем отсортированные данные для возврата
        sorted_df = work_df.copy()
        sorted_df[date_col] = sorted_df[date_col].dt.strftime("%Y-%m-%d %H:%M:%S")
        sorted_data = sorted_df.to_dict(orient="records")

        return {
            "is_valid": True,
            "date_column": date_col,
            "value_column": value_col,
            "total_points": len(work_df),
            "date_range_start": str(work_df[date_col].iloc[0].date()),
            "date_range_end": str(work_df[date_col].iloc[-1].date()),
            "inferred_freq": inferred_freq,
            "freq_description": freq_desc,
            "is_regular": is_regular,
            "gaps_count": gaps_count,
            "missing_values": missing_values,
            "suggestion": suggestion,
            "sorted_data": sorted_data,
        }

    # ==================================================================
    # 2. Ресемплирование временного ряда
    # ==================================================================

    @staticmethod
    def resample_data(
        df: pd.DataFrame,
        date_col: str,
        value_col: str,
        freq: str = "D",
        agg_func: Literal["mean", "sum", "median", "min", "max"] = "mean",
        fill_method: Literal["interpolate", "zero", "ffill"] = "interpolate",
    ) -> dict[str, Any]:
        """
        Ресемплирует временной ряд по указанной частоте.

        Алгоритм:
            1. Преобразует колонку дат в datetime и устанавливает как индекс.
            2. Агрегирует данные по заданной частоте (mean/sum/median/min/max).
            3. Заполняет пропущенные даты выбранным методом.

        Параметры:
            df          — исходный DataFrame.
            date_col    — имя колонки с датами.
            value_col   — имя колонки со значениями.
            freq        — целевая частота ("D", "W", "MS", "h" и т.д.).
            agg_func    — функция агрегации ("mean", "sum", "median", "min", "max").
            fill_method — метод заполнения пропусков:
                          "interpolate" — линейная интерполяция,
                          "zero"        — заполнение нулями,
                          "ffill"       — прямое заполнение (Forward Fill).

        Возвращает:
            Словарь:
            {
                "data": list[dict],      — ресемплированные данные
                "rows_before": int,      — строк до ресемплирования
                "rows_after": int,       — строк после ресемплирования
                "freq": str,             — применённая частота
                "freq_description": str, — описание частоты
                "agg_func": str,         — применённая функция агрегации
                "fill_method": str,      — метод заполнения
                "filled_count": int,     — количество заполненных пропусков
            }

        Исключения:
            ValueError — при невалидных входных данных.
        """
        if date_col not in df.columns:
            raise ValueError(f"Колонка «{date_col}» не найдена в данных.")
        if value_col not in df.columns:
            raise ValueError(f"Колонка «{value_col}» не найдена в данных.")

        work_df = df[[date_col, value_col]].copy()
        work_df[date_col] = pd.to_datetime(work_df[date_col], errors="coerce")
        work_df = work_df.dropna(subset=[date_col])

        if len(work_df) < 2:
            raise ValueError(
                "Недостаточно данных для ресемплирования (менее 2 точек с валидными датами)."
            )

        rows_before = len(work_df)

        # Устанавливаем дату как индекс
        work_df = work_df.set_index(date_col).sort_index()

        # Агрегируем по заданной частоте
        resampled = work_df[value_col].resample(freq).agg(agg_func)

        # Считаем пропуски после ресемплирования (до заполнения)
        nan_count = int(resampled.isna().sum())

        # Заполняем пропуски
        if fill_method == "interpolate":
            resampled = resampled.interpolate(method="linear")
        elif fill_method == "zero":
            resampled = resampled.fillna(0.0)
        elif fill_method == "ffill":
            resampled = resampled.ffill()

        # Оставшиеся NaN (в начале ряда при ffill/interpolate) — заполняем bfill
        resampled = resampled.bfill()

        rows_after = len(resampled)

        # Описание частоты
        base_freq = freq.lstrip("0123456789")
        freq_desc = _FREQ_DESCRIPTIONS.get(base_freq, freq)

        # Формируем результат
        result_df = resampled.reset_index()
        result_df.columns = [date_col, value_col]
        result_df[date_col] = result_df[date_col].dt.strftime("%Y-%m-%d %H:%M:%S")

        return {
            "data": result_df.to_dict(orient="records"),
            "rows_before": rows_before,
            "rows_after": rows_after,
            "freq": freq,
            "freq_description": freq_desc,
            "agg_func": agg_func,
            "fill_method": fill_method,
            "filled_count": nan_count,
        }
