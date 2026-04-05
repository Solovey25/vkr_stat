"""
report_service.py — Генерация PDF-отчёта (протокола анализа).

Собирает результаты работы подсистем в единый PDF-документ,
оформленный по академическим стандартам.

Особенности:
    - In-Memory Streaming: вся работа в оперативной памяти (io.BytesIO).
    - Font Subsetting: fpdf2 автоматически зашивает только использованные глифы.
    - Zero-Image Payload: графики рендерятся на бэкенде через plotly + kaleido.
"""

from __future__ import annotations

import io
import os
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from fpdf import FPDF

# Системные шрифты Windows (пропорциональные, с кириллицей)
_FONT_DIR = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
_FONT_REGULAR = os.path.join(_FONT_DIR, "times.ttf")
_FONT_BOLD = os.path.join(_FONT_DIR, "timesbd.ttf")
_FONT_ITALIC = os.path.join(_FONT_DIR, "timesi.ttf")

# Цвета (R, G, B)
_CLR_HEADER_BG = (41, 65, 122)     # тёмно-синий фон заголовков таблиц
_CLR_HEADER_FG = (255, 255, 255)   # белый текст заголовков
_CLR_ROW_EVEN = (234, 239, 250)    # чередование строк — голубоватый
_CLR_ROW_ODD = (255, 255, 255)     # белый
_CLR_TITLE = (30, 50, 100)         # цвет заголовков разделов
_CLR_ACCENT = (180, 30, 30)        # акцент (предупреждения)
_CLR_BLACK = (0, 0, 0)
_CLR_GRAY = (100, 100, 100)
_CLR_LINE = (180, 190, 210)        # линия-разделитель


class PDFReportService:
    """Генератор PDF-отчёта с поддержкой кириллицы и встроенными графиками."""

    def __init__(self, filename: str, df: pd.DataFrame) -> None:
        self._filename = filename
        self._df = df
        self._pdf = FPDF(orientation="P", unit="mm", format="A4")
        self._pdf.set_auto_page_break(auto=True, margin=20)

        # Подключение TTF-шрифтов с поддержкой UTF-8
        self._pdf.add_font("Report", "", _FONT_REGULAR, uni=True)
        self._pdf.add_font("Report", "B", _FONT_BOLD, uni=True)
        self._pdf.add_font("Report", "I", _FONT_ITALIC, uni=True)

        self._pdf.set_font("Report", "", 11)

    # ==================================================================
    # Утилиты
    # ==================================================================

    def _hr(self) -> None:
        """Горизонтальная линия-разделитель."""
        self._pdf.set_draw_color(*_CLR_LINE)
        self._pdf.set_line_width(0.4)
        y = self._pdf.get_y()
        self._pdf.line(self._pdf.l_margin, y, self._pdf.w - self._pdf.r_margin, y)
        self._pdf.ln(4)

    def _add_title(self, text: str) -> None:
        """Заголовок раздела."""
        self._pdf.ln(3)
        self._hr()
        self._pdf.set_font("Report", "B", 15)
        self._pdf.set_text_color(*_CLR_TITLE)
        self._pdf.cell(0, 10, text, new_x="LMARGIN", new_y="NEXT")
        self._pdf.set_text_color(*_CLR_BLACK)
        self._pdf.ln(2)
        self._pdf.set_font("Report", "", 11)

    def _add_subtitle(self, text: str) -> None:
        """Подзаголовок."""
        self._pdf.ln(2)
        self._pdf.set_font("Report", "B", 12)
        self._pdf.set_text_color(*_CLR_TITLE)
        self._pdf.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
        self._pdf.set_text_color(*_CLR_BLACK)
        self._pdf.ln(1)
        self._pdf.set_font("Report", "", 11)

    def _add_text(self, text: str) -> None:
        """Обычный текст с переносом."""
        self._pdf.set_font("Report", "", 11)
        self._pdf.multi_cell(0, 6, text)
        self._pdf.ln(2)

    def _add_kv(self, key: str, value: str) -> None:
        """Строка «ключ: значение»."""
        self._pdf.set_font("Report", "B", 11)
        self._pdf.cell(65, 7, f"{key}:", new_x="END", new_y="TOP")
        self._pdf.set_font("Report", "", 11)
        self._pdf.cell(0, 7, str(value), new_x="LMARGIN", new_y="NEXT")

    def _add_table(self, headers: list[str], rows: list[list[str]],
                   col_widths: list[float] | None = None) -> None:
        """Таблица с цветным заголовком и чередованием строк."""
        usable = self._pdf.w - self._pdf.l_margin - self._pdf.r_margin
        if col_widths is None:
            col_widths = [usable / len(headers)] * len(headers)
        else:
            # Масштабируем до ширины страницы
            total = sum(col_widths)
            col_widths = [w / total * usable for w in col_widths]

        row_h = 7

        # --- Заголовок ---
        self._pdf.set_fill_color(*_CLR_HEADER_BG)
        self._pdf.set_text_color(*_CLR_HEADER_FG)
        self._pdf.set_font("Report", "B", 10)
        for i, h in enumerate(headers):
            self._pdf.cell(col_widths[i], row_h, h, border=0, fill=True,
                           new_x="END", new_y="TOP", align="C")
        self._pdf.ln()

        # --- Строки ---
        self._pdf.set_text_color(*_CLR_BLACK)
        self._pdf.set_font("Report", "", 10)
        for row_idx, row in enumerate(rows):
            if row_idx % 2 == 0:
                self._pdf.set_fill_color(*_CLR_ROW_EVEN)
            else:
                self._pdf.set_fill_color(*_CLR_ROW_ODD)
            for i, cell in enumerate(row):
                txt = str(cell) if cell is not None else "—"
                # Первый столбец — выравнивание влево, остальные — по центру
                align = "L" if i == 0 else "C"
                self._pdf.cell(col_widths[i], row_h, txt[:50], border=0, fill=True,
                               new_x="END", new_y="TOP", align=align)
            self._pdf.ln()

        # Нижняя линия таблицы
        self._pdf.set_draw_color(*_CLR_LINE)
        self._pdf.set_line_width(0.3)
        y = self._pdf.get_y()
        self._pdf.line(self._pdf.l_margin, y, self._pdf.l_margin + usable, y)
        self._pdf.ln(4)

    def _add_plotly_image(self, fig, width_mm: int = 170) -> None:
        """Рендерит Plotly-фигуру в PNG in-memory и вставляет в PDF."""
        try:
            img_bytes = fig.to_image(format="png", width=900, height=500, scale=2)
            buf = io.BytesIO(img_bytes)
            self._pdf.image(buf, w=width_mm)
            self._pdf.ln(5)
        except Exception:
            self._add_text("[График не удалось отрендерить]")

    def _safe_fmt(self, val: Any, decimals: int = 4) -> str:
        """Безопасное форматирование числа."""
        if val is None:
            return "—"
        try:
            return f"{float(val):.{decimals}f}"
        except (ValueError, TypeError):
            return str(val)

    # ==================================================================
    # Блок 1: Паспорт данных (всегда)
    # ==================================================================

    def add_passport(self) -> None:
        """Паспорт данных — базовые метаданные файла."""
        self._pdf.add_page()

        # Заголовок документа
        self._pdf.set_font("Report", "B", 18)
        self._pdf.set_text_color(*_CLR_TITLE)
        self._pdf.cell(0, 14, "Протокол статистического анализа",
                       new_x="LMARGIN", new_y="NEXT", align="C")
        self._pdf.set_text_color(*_CLR_BLACK)
        self._pdf.ln(2)

        # Дата по центру
        now = datetime.now().strftime("%d.%m.%Y %H:%M")
        self._pdf.set_font("Report", "I", 10)
        self._pdf.set_text_color(*_CLR_GRAY)
        self._pdf.cell(0, 6, f"Дата генерации: {now}", new_x="LMARGIN", new_y="NEXT", align="C")
        self._pdf.set_text_color(*_CLR_BLACK)
        self._pdf.ln(4)

        self._add_title("1. Паспорт данных")

        self._add_kv("Имя файла", self._filename)
        self._add_kv("Строк", str(len(self._df)))
        self._add_kv("Столбцов", str(len(self._df.columns)))

        num_cols = self._df.select_dtypes(include="number").columns.tolist()
        cat_cols = self._df.select_dtypes(exclude=["number", "datetime"]).columns.tolist()
        dt_cols = self._df.select_dtypes(include="datetime").columns.tolist()

        self._add_kv("Числовые столбцы", ", ".join(num_cols) if num_cols else "—")
        self._add_kv("Категориальные", ", ".join(cat_cols) if cat_cols else "—")
        self._add_kv("Дата/время", ", ".join(dt_cols) if dt_cols else "—")
        self._pdf.ln(3)

    # ==================================================================
    # Блок 2: Описательная статистика
    # ==================================================================

    def add_statistics_section(self, stats_data: dict[str, dict]) -> None:
        """Таблица расширенных статистик + распределения."""
        self._add_title("2. Описательная статистика")

        headers = ["Столбец", "Среднее", "Медиана", "СКО", "Асимметрия", "Эксцесс"]
        rows = []
        for col, s in stats_data.items():
            rows.append([
                col,
                self._safe_fmt(s.get("mean"), 2),
                self._safe_fmt(s.get("median"), 2),
                self._safe_fmt(s.get("std"), 2),
                self._safe_fmt(s.get("skewness"), 3),
                self._safe_fmt(s.get("kurtosis"), 3),
            ])

        self._add_table(headers, rows, col_widths=[30, 20, 20, 20, 20, 20])

    def add_distribution_info(self, column: str, fit_result: dict) -> None:
        """Результат подбора распределения для одной колонки."""
        best = fit_result.get("best_distribution")
        if best:
            best_ru = fit_result.get("best_distribution_ru", best)
            p_val = fit_result.get("best_p_value", 0)
            ks_stat = fit_result.get("best_ks_statistic", 0)
            self._pdf.set_font("Report", "", 10)
            self._pdf.multi_cell(
                0, 5,
                f"\u2022 {column}: {best_ru} распределение "
                f"(KS = {self._safe_fmt(ks_stat)}, p = {self._safe_fmt(p_val)})",
            )
            self._pdf.ln(1)

    # ==================================================================
    # Блок 3: Проверка гипотез
    # ==================================================================

    def add_hypothesis_section(self, result: dict) -> None:
        """Результат сравнения двух выборок."""
        self._add_title("3. Проверка гипотез")

        assumptions = result.get("assumptions", {})

        self._add_kv("Выбранный тест", result.get("test_name", "—"))
        self._add_kv("Статистика", self._safe_fmt(result.get("statistic")))
        self._add_kv("p-значение", self._safe_fmt(result.get("p_value")))

        if result.get("effect_size") is not None:
            metric = result.get("effect_size_metric", "cohens_d")
            label = "d Коэна" if metric == "cohens_d" else "ранг-бисериальная r"
            self._add_kv(f"Размер эффекта ({label})", self._safe_fmt(result["effect_size"]))

        self._pdf.ln(2)

        # Предпосылки
        self._add_subtitle("Проверка предпосылок")

        norm_test = assumptions.get("norm_test_name")
        if norm_test:
            self._add_kv(f"{norm_test} (A), p", self._safe_fmt(assumptions.get("shapiro_a_p")))
            self._add_kv(f"{norm_test} (B), p", self._safe_fmt(assumptions.get("shapiro_b_p")))
        else:
            self._add_text("Тест нормальности не проводился (N < 8).")

        if assumptions.get("levene_p") is not None:
            self._add_kv("Тест Левене, p", self._safe_fmt(assumptions.get("levene_p")))
            eq_var = "Да" if assumptions.get("equal_variances") else "Нет"
            self._add_kv("Равенство дисперсий", eq_var)

        self._pdf.ln(2)

        # Дерево решений
        chain = result.get("decision_chain", [])
        if chain:
            self._add_subtitle("Ход анализа (дерево решений)")
            for step in chain:
                self._pdf.set_font("Report", "", 11)
                self._pdf.multi_cell(0, 6, f"\u2192 {step}")
                self._pdf.ln(1)

        # Заключение
        conclusion = result.get("conclusion", "")
        if conclusion:
            self._add_subtitle("Заключение")
            self._pdf.set_font("Report", "B", 11)
            self._add_text(conclusion)
            self._pdf.set_font("Report", "", 11)

    # ==================================================================
    # Блок 4: Регрессионный анализ
    # ==================================================================

    def add_regression_section(self, result: dict, plot_fig=None) -> None:
        """Результаты регрессионного анализа с графиком."""
        self._add_title("4. Регрессионный анализ")

        ols = result.get("ols", {})
        skl = result.get("sklearn", {})
        cleaning = result.get("cleaning", {})

        # Метрики в две колонки через таблицу
        metrics_data = [
            ("R\u00b2", self._safe_fmt(ols.get("r_squared"))),
            ("R\u00b2 (скорр.)", self._safe_fmt(ols.get("r_squared_adj"))),
            ("MAE", self._safe_fmt(skl.get("mae"), 2)),
            ("RMSE", self._safe_fmt(skl.get("rmse"), 2)),
            ("F-статистика", self._safe_fmt(ols.get("f_statistic"), 2)),
            ("AIC", self._safe_fmt(ols.get("aic"), 2)),
            ("Дарбин-Уотсон", self._safe_fmt(ols.get("durbin_watson"))),
            ("Наблюдений", str(result.get("n_samples", "—"))),
        ]

        for key, val in metrics_data:
            self._add_kv(key, val)

        if cleaning.get("dropped_rows", 0) > 0:
            self._pdf.ln(1)
            self._pdf.set_font("Report", "I", 10)
            self._pdf.set_text_color(*_CLR_GRAY)
            self._add_text(
                f"Очистка данных: удалено {cleaning['dropped_rows']} строк "
                f"(NaN: {cleaning.get('dropped_nan', 0)}, inf: {cleaning.get('dropped_inf', 0)})."
            )
            self._pdf.set_text_color(*_CLR_BLACK)

        self._pdf.ln(2)

        # Адекватность
        self._add_subtitle("Адекватность модели")
        self._add_text(ols.get("reliability_text", ""))

        # Таблица коэффициентов
        self._add_subtitle("Коэффициенты модели (OLS)")

        factors = ols.get("factor_stats", [])
        headers = ["Фактор", "Коэфф.", "Ст. ошибка", "t-стат.", "p-значение", "VIF"]
        rows = []
        for f in factors:
            sig = ""
            p = f.get("p_value", 1)
            if p < 0.001:
                sig = " ***"
            elif p < 0.01:
                sig = " **"
            elif p < 0.05:
                sig = " *"

            vif = self._safe_fmt(f.get("vif"), 2) if f.get("vif") is not None else "—"
            rows.append([
                f.get("name", ""),
                self._safe_fmt(f.get("coefficient")),
                self._safe_fmt(f.get("std_error")),
                self._safe_fmt(f.get("t_statistic"), 2),
                f"{self._safe_fmt(p)}{sig}",
                vif,
            ])

        self._add_table(headers, rows, col_widths=[25, 18, 18, 15, 22, 12])

        if ols.get("vif_warnings"):
            self._pdf.set_text_color(*_CLR_ACCENT)
            for w in ols["vif_warnings"]:
                self._add_text(f"\u26a0 {w}")
            self._pdf.set_text_color(*_CLR_BLACK)

        # График
        if plot_fig is not None:
            self._add_subtitle("График регрессии с 95% доверительным интервалом")
            self._add_plotly_image(plot_fig)

    # ==================================================================
    # Блок 5: Сравнение датасетов
    # ==================================================================

    def add_comparison_section(self, compare_result: dict) -> None:
        """Результаты сравнения датасетов (PSI, KS, хи-квадрат)."""
        self._add_title("5. Сравнение датасетов (Data Drift)")

        stat_cmp = compare_result.get("statistical_comparison", {})
        results = stat_cmp.get("results", [])
        correction = stat_cmp.get("correction_method")

        if correction:
            self._pdf.set_font("Report", "I", 10)
            self._pdf.set_text_color(*_CLR_GRAY)
            self._add_text(f"Коррекция множественных сравнений: {correction}")
            self._pdf.set_text_color(*_CLR_BLACK)

        # Числовой дрифт
        if results:
            self._add_subtitle("Числовой дрифт")
            headers = ["Признак", "PSI", "Оценка PSI", "p-value", "Вердикт"]
            rows = []
            for r in sorted(results, key=lambda x: x.get("psi", 0), reverse=True):
                rows.append([
                    r.get("column", ""),
                    self._safe_fmt(r.get("psi"), 3),
                    r.get("psi_interpretation", "—"),
                    self._safe_fmt(r.get("p_value")),
                    r.get("verdict", "—"),
                ])
            self._add_table(headers, rows, col_widths=[25, 15, 25, 18, 35])

        # Категориальный дрифт
        cat_drift = compare_result.get("categorical_drift", {})
        cat_items = cat_drift.get("columns", [])
        if cat_items:
            self._add_subtitle("Категориальный дрифт (\u03c7\u00b2-тест)")
            headers = ["Колонка", "\u03c7\u00b2 стат.", "p-value", "V Крамера", "Дрифт"]
            rows = []
            for cr in cat_items:
                rows.append([
                    cr.get("column", ""),
                    self._safe_fmt(cr.get("chi2_stat"), 2),
                    self._safe_fmt(cr.get("chi2_p_value")),
                    self._safe_fmt(cr.get("cramers_v"), 3),
                    "Да" if cr.get("is_drifted") else "Нет",
                ])
            self._add_table(headers, rows, col_widths=[25, 18, 18, 18, 15])

        # Структурный отчёт
        structure = compare_result.get("structure_report", {})
        if structure:
            self._add_subtitle("Структурный отчёт")
            self._add_kv("Строк (A)", str(structure.get("rows_base", "—")))
            self._add_kv("Строк (B)", str(structure.get("rows_compare", "—")))
            delta = structure.get("rows_delta", 0)
            self._add_kv("Изменение", f"{delta:+d}" if isinstance(delta, int) else str(delta))

            if structure.get("added_columns"):
                self._add_kv("Добавленные колонки", ", ".join(structure["added_columns"]))
            if structure.get("removed_columns"):
                self._add_kv("Удалённые колонки", ", ".join(structure["removed_columns"]))

    # ==================================================================
    # Сборка
    # ==================================================================

    def build(self) -> bytes:
        """Возвращает готовый PDF как байты."""
        return self._pdf.output()
