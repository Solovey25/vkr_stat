"""
data_loader_service.py — Сервис профессиональной загрузки данных.

Обеспечивает устойчивый импорт файлов различных форматов (CSV, TXT, XLSX, XLS)
с автоматическим определением кодировки и разделителя.

Основные возможности:
    1. Автоопределение кодировки (chardet) — корректная работа с CP1251, UTF-8 и др.
    2. Автоопределение разделителя (pandas sniffer) — запятая, точка с запятой, табуляция.
    3. Валидация: пустой файл, превышение размера, отсутствие числовых колонок.
    4. Формирование метаданных: типы колонок, количество непустых значений,
       разделение на числовые и категориальные признаки.

Используется в рамках ВКР: «Веб-приложение для статистического анализа и прогнозирования».
"""

from __future__ import annotations

import io
import os
from typing import Any

import chardet
import pandas as pd


class DataLoaderService:
    """
    Сервис загрузки и первичной валидации табличных данных.

    Принимает имя файла и его байтовое содержимое, определяет формат,
    кодировку и разделитель, выполняет валидацию и возвращает DataFrame
    вместе с метаданными для отображения на фронтенде.
    """

    # Максимально допустимый размер файла — 100 МБ
    MAX_FILE_SIZE: int = 100 * 1024 * 1024

    # Поддерживаемые расширения файлов
    SUPPORTED_EXTENSIONS: set[str] = {".csv", ".txt", ".xlsx", ".xls"}

    def __init__(
        self,
        filename: str,
        file_bytes: bytes,
        sheet_name: str | None = None,
    ) -> None:
        """
        Инициализирует сервис загрузки данных.

        Параметры:
            filename   — оригинальное имя загруженного файла (для определения формата).
            file_bytes — байтовое содержимое файла.
            sheet_name — имя листа Excel для чтения (None = первый лист).
        """
        self.filename: str = filename or "unknown"
        self.file_bytes: bytes = file_bytes

        # Определяем расширение файла (приводим к нижнему регистру)
        _, self.extension = os.path.splitext(self.filename.lower())

        # Кодировка — заполняется при чтении CSV/TXT
        self.detected_encoding: str | None = None

        # Имя выбранного листа Excel (если применимо)
        self.sheet_name: str | None = None

        # Список всех листов Excel-файла (если применимо)
        self.sheet_names: list[str] = []

        # Список колонок, автоматически распознанных как даты
        self.datetime_columns: list[str] = []

        # Информация о регулярности временных рядов (по каждой колонке-дате)
        self.time_series_info: list[dict[str, Any]] = []

        # DataFrame — заполняется после успешного чтения
        self.df: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Публичный метод
    # ------------------------------------------------------------------

    def load(self) -> dict[str, Any]:
        """
        Главный метод: читает файл, валидирует и возвращает данные с метаданными.

        Порядок действий:
            1. Проверка размера файла.
            2. Проверка поддерживаемого расширения.
            3. Чтение файла в DataFrame (с автоопределением кодировки/разделителя).
            4. Автоматический парсинг колонок-дат (подготовка к ARIMA).
            5. Валидация содержимого (не пуст, есть числовые колонки).
            6. Формирование метаданных.

        Возвращает:
            Словарь {"df": pd.DataFrame, "metadata": dict} с данными и метаинформацией.

        Исключения:
            ValueError — при любой ошибке валидации или чтения (с описанием на русском).
        """
        # Шаг 1: Проверяем размер файла
        self._validate_size()

        # Шаг 2: Проверяем расширение
        self._validate_extension()

        # Шаг 3: Читаем файл в зависимости от формата
        if self.extension in {".csv", ".txt"}:
            self.df = self._read_csv()
        elif self.extension in {".xlsx", ".xls"}:
            self.df = self._read_excel()

        # Шаг 4: Пытаемся автоматически распознать колонки с датами
        # (подготовка к будущему ARIMA-анализу временных рядов)
        self._detect_dates()

        # Шаг 4.1: Для каждой распознанной колонки-даты проверяем регулярность ряда
        for dt_col in self.datetime_columns:
            info = self.check_time_regularity(self.df, dt_col)
            self.time_series_info.append(info)

        # Шаг 5: Проверяем содержимое DataFrame
        self._validate_dataframe()

        # Шаг 6: Собираем метаданные
        metadata = self._build_metadata()

        return {"df": self.df, "metadata": metadata}

    # ------------------------------------------------------------------
    # Валидация
    # ------------------------------------------------------------------

    def _validate_size(self) -> None:
        """Проверяет, что файл не пустой и не превышает допустимый размер."""
        if len(self.file_bytes) == 0:
            raise ValueError(
                "Загруженный файл пуст. Пожалуйста, выберите файл с данными."
            )

        if len(self.file_bytes) > self.MAX_FILE_SIZE:
            size_mb = len(self.file_bytes) / (1024 * 1024)
            raise ValueError(
                f"Размер файла ({size_mb:.1f} МБ) превышает допустимый лимит "
                f"({self.MAX_FILE_SIZE // (1024 * 1024)} МБ)."
            )

    def _validate_extension(self) -> None:
        """Проверяет, что расширение файла поддерживается системой."""
        if self.extension not in self.SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(self.SUPPORTED_EXTENSIONS))
            raise ValueError(
                f"Формат файла «{self.extension}» не поддерживается. "
                f"Допустимые форматы: {supported}."
            )

    def _validate_dataframe(self) -> None:
        """Проверяет, что DataFrame не пуст и содержит хотя бы одну числовую колонку."""
        if self.df is None or self.df.empty:
            raise ValueError(
                "Не удалось извлечь данные из файла. "
                "Убедитесь, что файл содержит табличные данные."
            )

        # Для статистического анализа необходима хотя бы одна числовая колонка
        numeric_cols = self.df.select_dtypes(include="number").columns
        if len(numeric_cols) == 0:
            raise ValueError(
                "В загруженном файле не найдено числовых столбцов. "
                "Статистический анализ невозможен без числовых данных."
            )

    # ------------------------------------------------------------------
    # Автопарсинг дат
    # ------------------------------------------------------------------

    def _detect_dates(self) -> None:
        """
        Автоматически распознаёт колонки, содержащие даты.

        Перебирает все строковые (object) колонки и пытается преобразовать
        их в datetime64 с помощью pd.to_datetime. Если более 50% значений
        успешно парсятся как даты — колонка конвертируется.

        Это критически важно для будущего ARIMA-анализа временных рядов,
        которому необходим столбец типа datetime в качестве индекса.
        """
        df = self.df
        detected: list[str] = []

        # Проверяем только строковые колонки — числовые и уже datetime пропускаем
        object_cols = df.select_dtypes(include="object").columns

        for col in object_cols:
            # Берём непустые значения для анализа
            non_null = df[col].dropna()
            if len(non_null) == 0:
                continue

            try:
                # infer_datetime_format ускоряет парсинг, errors="coerce"
                # превращает нераспознанные значения в NaT (не выбрасывает исключение)
                parsed = pd.to_datetime(non_null, errors="coerce", dayfirst=False)

                # Считаем долю успешно распознанных значений
                success_rate = parsed.notna().sum() / len(non_null)

                # Если более 50% значений — валидные даты, конвертируем колонку
                if success_rate > 0.5:
                    df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=False)
                    detected.append(col)
            except (ValueError, TypeError, OverflowError):
                # Если pd.to_datetime не справился — пропускаем колонку
                continue

        self.datetime_columns = detected

    # ------------------------------------------------------------------
    # Чтение файлов
    # ------------------------------------------------------------------

    def _read_csv(self) -> pd.DataFrame:
        """
        Читает CSV/TXT файл с автоопределением кодировки и разделителя.

        Алгоритм:
            1. chardet анализирует байтовый поток и определяет кодировку.
            2. Байты декодируются в текст с использованием определённой кодировки.
            3. pandas читает текст с автоопределением разделителя (sep=None).

        Возвращает:
            pd.DataFrame с данными из файла.
        """
        # --- Определяем кодировку ---
        # Стратегия: UTF-8 первым (покрывает ~95% файлов), chardet — fallback.
        # Это устраняет ложные срабатывания chardet (напр. UTF-8 → cp1252).
        text: str | None = None

        # 1. Пробуем UTF-8 (самая распространённая кодировка)
        try:
            text = self.file_bytes.decode("utf-8")
            self.detected_encoding = "utf-8"
        except UnicodeDecodeError:
            pass

        # 2. chardet fallback
        if text is None:
            detection = chardet.detect(self.file_bytes)
            encoding = detection.get("encoding") or "cp1251"
            self.detected_encoding = encoding
            try:
                text = self.file_bytes.decode(encoding)
            except (UnicodeDecodeError, LookupError):
                pass

        # 3. Последний fallback — cp1251 (русская Windows)
        if text is None:
            try:
                text = self.file_bytes.decode("cp1251", errors="replace")
                self.detected_encoding = "cp1251"
            except UnicodeDecodeError:
                raise ValueError(
                    "Не удалось декодировать файл. "
                    "Попробуйте сохранить файл в кодировке UTF-8."
                )

        # --- Читаем CSV с автоопределением разделителя ---
        try:
            df = pd.read_csv(
                io.StringIO(text),
                sep=None,           # автоопределение разделителя
                engine="python",    # Python-движок поддерживает sep=None
            )
        except pd.errors.EmptyDataError:
            raise ValueError(
                "Файл не содержит данных или имеет некорректную структуру."
            )
        except pd.errors.ParserError as e:
            raise ValueError(
                f"Ошибка при разборе файла: {e}. "
                "Проверьте, что файл имеет корректный табличный формат."
            )

        return df

    def _read_excel(self) -> pd.DataFrame:
        """
        Читает Excel-файл (.xlsx или .xls).

        Для формата .xlsx используется движок openpyxl,
        для .xls — стандартный движок pandas (xlrd).

        Если в файле несколько листов, читает указанный лист (sheet_name)
        или первый лист по умолчанию. Список всех листов сохраняется
        в self.sheet_names для передачи фронтенду.

        Возвращает:
            pd.DataFrame с данными из выбранного листа.
        """
        try:
            # openpyxl обрабатывает современный формат .xlsx
            # Для .xls pandas автоматически выберет подходящий движок
            engine = "openpyxl" if self.extension == ".xlsx" else None
            excel_file = pd.ExcelFile(io.BytesIO(self.file_bytes), engine=engine)

            # Сохраняем список всех листов для отображения на фронтенде
            self.sheet_names = excel_file.sheet_names

            # Определяем, какой лист читать
            target_sheet = self.sheet_name
            if target_sheet is not None and target_sheet not in self.sheet_names:
                raise ValueError(
                    f"Лист «{target_sheet}» не найден в файле. "
                    f"Доступные листы: {', '.join(self.sheet_names)}."
                )

            # Если лист не указан — читаем первый
            if target_sheet is None:
                target_sheet = self.sheet_names[0]

            self.sheet_name = target_sheet
            df = excel_file.parse(target_sheet)

        except ValueError:
            # Пробрасываем наши собственные ValueError без изменений
            raise
        except Exception as e:
            raise ValueError(
                f"Ошибка при чтении Excel-файла: {e}. "
                "Убедитесь, что файл не повреждён и имеет корректный формат."
            )

        # Для Excel кодировка не определяется
        self.detected_encoding = None

        return df

    # ------------------------------------------------------------------
    # Проверка регулярности временного ряда
    # ------------------------------------------------------------------

    @staticmethod
    def check_time_regularity(
        df: pd.DataFrame,
        date_column: str,
    ) -> dict[str, Any]:
        """
        Проверяет регулярность временного ряда по указанной колонке-дате.

        Алгоритм:
            1. Сортирует данные по дате.
            2. Пытается определить частоту ряда через pd.infer_freq.
            3. Подсчитывает пропуски (разрывы) в датах.

        Параметры:
            df          — DataFrame с данными.
            date_column — имя колонки с датами (dtype datetime64).

        Возвращает:
            Словарь с результатами:
            {
                "column": str,           — имя колонки-даты
                "freq": str | None,      — определённая частота ("D", "MS", ...) или None
                "freq_description": str,  — человекочитаемое описание частоты
                "is_regular": bool,      — True если ряд регулярный (без пропусков)
                "total_points": int,     — общее количество временных точек
                "gaps_count": int,       — количество пропущенных интервалов
                "suggestion": str | None — рекомендация при нерегулярности
            }
        """
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

        series = df[date_column].dropna().sort_values()
        total_points = len(series)

        if total_points < 3:
            return {
                "column": date_column,
                "freq": None,
                "freq_description": "Недостаточно данных",
                "is_regular": False,
                "total_points": total_points,
                "gaps_count": 0,
                "suggestion": "Для анализа временных рядов необходимо минимум 3 точки.",
            }

        # Пробуем определить частоту через pd.infer_freq
        try:
            idx = pd.DatetimeIndex(series.values)
            inferred_freq = pd.infer_freq(idx)
        except (ValueError, TypeError):
            inferred_freq = None

        freq_desc = "Не удалось определить"
        if inferred_freq is not None:
            # infer_freq может вернуть "2D", "3MS" и т.д., берём базовую часть
            base_freq = inferred_freq.lstrip("0123456789")
            freq_desc = _FREQ_DESCRIPTIONS.get(base_freq, inferred_freq)

        # Подсчитываем пропуски: разницы между соседними датами
        diffs = series.diff().dropna()
        if len(diffs) > 0:
            # Наиболее частый интервал — считаем «нормальным шагом»
            median_diff = diffs.median()
            # Пропуск — если интервал больше 1.5× медианного
            gaps_count = int((diffs > median_diff * 1.5).sum())
        else:
            gaps_count = 0

        is_regular = inferred_freq is not None and gaps_count == 0

        suggestion = None
        if not is_regular:
            if inferred_freq is None:
                suggestion = (
                    "Частота ряда не определена автоматически. "
                    "Рекомендуется привести данные к регулярной сетке "
                    "(ресемплирование) перед ARIMA-анализом."
                )
            else:
                suggestion = (
                    f"Обнаружено {gaps_count} пропусков в датах. "
                    "Рекомендуется заполнить пропуски (интерполяция) "
                    "или выполнить ресемплирование перед ARIMA-анализом."
                )

        return {
            "column": date_column,
            "freq": inferred_freq,
            "freq_description": freq_desc,
            "is_regular": is_regular,
            "total_points": total_points,
            "gaps_count": gaps_count,
            "suggestion": suggestion,
        }

    # ------------------------------------------------------------------
    # Формирование метаданных
    # ------------------------------------------------------------------

    def _build_metadata(self) -> dict[str, Any]:
        """
        Формирует словарь метаданных о загруженном файле.

        Метаданные включают:
            - Общую информацию: имя файла, размер, кодировку.
            - Количество строк и столбцов.
            - Списки числовых и категориальных признаков.
            - Детальную информацию по каждому столбцу (тип, кол-во непустых значений).
            - Превью данных (первые 10 строк).

        Возвращает:
            Словарь с метаданными, готовый к сериализации в JSON.
        """
        df = self.df

        # Разделяем столбцы на числовые, категориальные и datetime
        numeric_columns: list[str] = df.select_dtypes(include="number").columns.tolist()
        datetime_columns: list[str] = df.select_dtypes(
            include="datetime"
        ).columns.tolist()
        categorical_columns: list[str] = df.select_dtypes(
            exclude=["number", "datetime", "datetimetz"]
        ).columns.tolist()

        # Формируем информацию о каждом столбце
        column_info: list[dict[str, Any]] = []
        for col in df.columns:
            column_info.append({
                "name": col,
                "dtype": str(df[col].dtype),
                "non_null_count": int(df[col].notna().sum()),
            })

        # Превью: первые 10 строк в формате списка словарей.
        # Для datetime-колонок конвертируем в ISO-строки для JSON-сериализации.
        preview_df = df.head(10).copy()
        for col in datetime_columns:
            if col in preview_df.columns:
                preview_df[col] = preview_df[col].dt.strftime("%Y-%m-%d %H:%M:%S")

        # Заменяем NaN/NaT на None для корректной JSON-сериализации
        preview: list[dict] = (
            preview_df
            .where(preview_df.notna(), None)
            .to_dict(orient="records")
        )

        return {
            "filename": self.filename,
            "file_size": len(self.file_bytes),
            "encoding": self.detected_encoding,
            "rows": len(df),
            "columns_count": len(df.columns),
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "datetime_columns": datetime_columns,
            "time_series_info": self.time_series_info,
            "sheet_names": self.sheet_names,
            "active_sheet": self.sheet_name,
            "column_info": column_info,
            "preview": preview,
        }
