"""
schemas.py — Pydantic-модели для валидации и обмена данными между клиентом и сервером.

Используется в рамках ВКР: «Веб-приложение для статистического анализа и прогнозирования».
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


# ---------- Множественная регрессия (RegressionService) ----------

class FactorStat(BaseModel):
    """Статистика одного фактора из OLS-модели."""
    name: str
    coefficient: float
    std_error: float
    t_statistic: float
    p_value: float
    vif: float | None = None


class OLSResult(BaseModel):
    """Результаты statsmodels OLS."""
    r_squared: float
    r_squared_adj: float
    f_statistic: float
    f_p_value: float
    aic: float
    bic: float
    durbin_watson: float
    factor_stats: list[FactorStat]
    residuals_shapiro_stat: float       # Статистика Шапиро-Уилка для остатков
    residuals_shapiro_p: float          # p-значение Шапиро-Уилка для остатков
    is_model_reliable: bool             # True если остатки нормальны (p ≥ 0.05)
    reliability_text: str               # Текстовое пояснение адекватности
    vif_warnings: list[str] = []        # Предупреждения о мультиколлинеарности (VIF > 10)
    has_multicollinearity: bool = False  # True если хотя бы у одного фактора VIF > 10


class SklearnResult(BaseModel):
    """Результаты scikit-learn модели."""
    model_name: str
    intercept: float
    coefficients: dict[str, float]
    r2: float
    mae: float
    rmse: float
    predictions: list[float]
    feature_values: dict[str, list[float]]
    target_values: list[float]


# ---------- /analyze/regression (JSON-вход, фронтенд-ответ) ----------

class AnalyzeRegressionRequest(BaseModel):
    """JSON-запрос на регрессионный анализ (без загрузки файла)."""
    file_id: str | None = None                  # ID файла в серверном кэше
    data: list[dict[str, Any]] | None = None    # Строки таблицы (fallback)
    target_column: str                          # Зависимая переменная (Y)
    feature_columns: list[str]                  # Независимые переменные (X)


class CleaningReport(BaseModel):
    """Отчёт об очистке данных перед анализом."""
    original_rows: int       # Строк до очистки
    cleaned_rows: int        # Строк после очистки
    dropped_rows: int        # Удалено строк
    dropped_inf: int         # Из них — содержавших inf
    dropped_nan: int         # Из них — содержавших NaN


class RegressionLinePoint(BaseModel):
    """Одна точка прогнозной линии (для графика)."""
    x: float
    y_actual: float
    y_predicted: float


class AnalyzeRegressionResponse(BaseModel):
    """Структурированный ответ /analyze/regression для фронтенда."""
    cleaning: CleaningReport
    n_samples: int
    small_sample: bool
    ols: OLSResult
    sklearn: SklearnResult
    regression_line: list[RegressionLinePoint]
    plot_json: str  # Plotly-фигура (JSON) с доверительным интервалом


# ---------- Загрузка файлов (/api/upload-file) ----------

class ColumnInfo(BaseModel):
    """Информация об одном столбце загруженного файла."""
    name: str               # Имя столбца
    dtype: str              # Тип данных (например, 'int64', 'float64', 'object')
    non_null_count: int     # Количество непустых значений


class TimeSeriesColumnInfo(BaseModel):
    """Информация о регулярности одного временного ряда (колонки-даты)."""
    column: str                         # Имя колонки-даты
    freq: str | None                    # Определённая частота ("D", "MS", ...) или None
    freq_description: str               # Человекочитаемое описание частоты
    is_regular: bool                    # True если ряд регулярный (без пропусков)
    total_points: int                   # Количество временных точек
    gaps_count: int                     # Количество пропущенных интервалов
    suggestion: str | None              # Рекомендация при нерегулярности


class FileMetadata(BaseModel):
    """Метаданные загруженного файла."""
    filename: str                       # Оригинальное имя файла
    file_size: int                      # Размер файла в байтах
    encoding: str | None                # Определённая кодировка (только для CSV/TXT)
    rows: int                           # Количество строк в таблице
    columns_count: int                  # Количество столбцов
    numeric_columns: list[str]          # Список числовых столбцов
    categorical_columns: list[str]      # Список категориальных столбцов
    datetime_columns: list[str] = []    # Список автоматически распознанных колонок-дат
    time_series_info: list[TimeSeriesColumnInfo] = []  # Регулярность временных рядов
    sheet_names: list[str] = []         # Список листов Excel (пуст для CSV/TXT)
    active_sheet: str | None = None     # Имя прочитанного листа Excel


class UploadResponse(BaseModel):
    """Полный ответ эндпоинта /api/upload-file."""
    file_id: str | None = None          # ID файла в серверном кэше
    metadata: FileMetadata              # Метаданные файла
    column_info: list[ColumnInfo]       # Детальная информация по каждому столбцу
    preview: list[dict]                 # Превью данных (первые 10 строк)


# ---------- Предобработка данных (/api/sanitize/*) ----------

class MissingInfo(BaseModel):
    """Информация о пропусках в одном столбце."""
    column: str                         # Имя столбца
    missing_count: int                  # Количество пропущенных значений
    missing_percent: float              # Процент пропусков
    dtype: str                          # Тип данных столбца


class SanitizeMissingRequest(BaseModel):
    """Запрос на обработку пропусков."""
    file_id: str | None = None                              # ID файла в серверном кэше
    data: list[dict] | None = None                          # Строки таблицы (fallback)
    method: Literal["drop", "fill"]                         # Метод: удалить или заполнить
    strategy: Literal["mean", "median", "most_frequent"] = "mean"  # Стратегия заполнения
    columns: list[str] | None = None                        # Столбцы (None = все)


class SanitizeMissingResponse(BaseModel):
    """Ответ после обработки пропусков."""
    data: list[dict]                    # Обновлённые данные
    rows_before: int                    # Строк до обработки
    rows_after: int                     # Строк после обработки
    affected_count: int                 # Удалено строк / заполнено значений
    method: str                         # Применённый метод


class OutlierInfo(BaseModel):
    """Информация о выбросах в одном числовом столбце (метод IQR)."""
    column: str                         # Имя столбца
    q1: float                           # Первый квартиль (25-й перцентиль)
    q3: float                           # Третий квартиль (75-й перцентиль)
    iqr: float                          # Межквартильный размах
    lower_bound: float                  # Нижняя граница: Q1 − 1.5·IQR
    upper_bound: float                  # Верхняя граница: Q3 + 1.5·IQR
    outliers_count: int                 # Количество выбросов
    outliers_percent: float             # Процент выбросов
    total_rows: int                     # Общее количество строк


class OutliersRequest(BaseModel):
    """Запрос на расчёт информации о выбросах."""
    file_id: str | None = None          # ID файла в серверном кэше
    data: list[dict] | None = None      # Строки таблицы (fallback)
    columns: list[str] | None = None    # Столбцы для анализа (None = все числовые)


class OutliersInfoResponse(BaseModel):
    """Ответ с информацией о выбросах."""
    outliers: list[OutlierInfo]         # Информация по каждому столбцу


class RemoveOutliersRequest(BaseModel):
    """Запрос на удаление выбросов."""
    file_id: str | None = None          # ID файла в серверном кэше
    data: list[dict] | None = None      # Строки таблицы (fallback)
    columns: list[str]                  # Столбцы для удаления выбросов


class RemoveOutliersResponse(BaseModel):
    """Ответ после удаления выбросов."""
    data: list[dict]                    # Очищенные данные
    rows_before: int                    # Строк до удаления
    rows_after: int                     # Строк после удаления
    removed_count: int                  # Удалено строк


class ProcessingReport(BaseModel):
    """Общий отчёт о выполненной операции предобработки."""
    operation: str                      # Название операции
    rows_before: int                    # Строк до
    rows_after: int                     # Строк после
    affected_count: int                 # Затронуто строк/значений
    details: str                        # Текстовое описание на русском


class ScaleRequest(BaseModel):
    """Запрос на масштабирование числовых столбцов."""
    file_id: str | None = None          # ID файла в серверном кэше
    data: list[dict] | None = None      # Строки таблицы (fallback)
    columns: list[str]                  # Столбцы для масштабирования
    method: str = "standard"            # "standard" (Z-score) или "minmax" ([0, 1])


class ScaleResponse(BaseModel):
    """Ответ после масштабирования."""
    data: list[dict]                    # Масштабированные данные


class EncodeRequest(BaseModel):
    """Запрос на кодирование категориальных столбцов."""
    file_id: str | None = None          # ID файла в серверном кэше
    data: list[dict] | None = None      # Строки таблицы (fallback)
    columns: list[str]                  # Категориальные столбцы для кодирования


class EncodeResponse(BaseModel):
    """Ответ после кодирования категорий."""
    data: list[dict]                    # Данные с числовыми кодами
    mapping: dict[str, dict[int, str]]  # {столбец: {код: исходное_значение}}


# ---------- Расширенная статистика (/api/stats/*) ----------

class ExtendedColumnStats(BaseModel):
    """Расширенные статистики одного числового столбца."""
    count: float                        # Количество непустых значений
    mean: float                         # Среднее
    median: float                       # Медиана
    std: float                          # Стандартное отклонение
    min: float                          # Минимум
    max: float                          # Максимум
    q25: float                          # Первый квартиль
    q75: float                          # Третий квартиль
    skewness: float                     # Асимметрия (Skewness)
    kurtosis: float                     # Эксцесс (Kurtosis)
    sem: float                          # Стандартная ошибка среднего (SEM)
    is_constant: bool = False           # True если столбец — константа (std=0)


class ExtendedStatsRequest(BaseModel):
    """Запрос на расчёт расширенных описательных статистик."""
    file_id: str | None = None          # ID файла в серверном кэше
    data: list[dict] | None = None      # Строки таблицы (fallback)


class ExtendedStatsResponse(BaseModel):
    """Ответ /api/stats/extended — расширенные статистики по всем числовым столбцам."""
    columns: dict[str, ExtendedColumnStats]


class DistributionFitResult(BaseModel):
    """Результат подгонки одного распределения."""
    distribution: str                   # Название распределения (англ.)
    params: dict[str, float]            # Параметры распределения
    ks_statistic: float                 # KS-статистика
    p_value: float                      # p-значение KS-теста


class DistributionFitRequest(BaseModel):
    """Запрос на подбор распределения."""
    file_id: str | None = None          # ID файла в серверном кэше
    data: list[dict] | None = None      # Строки таблицы (fallback)
    column: str                         # Имя числовой колонки для анализа


class DistributionFitResponse(BaseModel):
    """Ответ /api/stats/fit-distribution — лучшее распределение + кривая PDF."""
    best_distribution: str | None       # Название лучшего распределения
    best_distribution_ru: str | None    # Название на русском языке
    best_params: dict[str, float]       # Параметры лучшего распределения
    best_ks_statistic: float | None     # KS-статистика лучшего
    best_p_value: float | None          # p-значение лучшего
    all_results: list[DistributionFitResult]  # Результаты по всем распределениям
    pdf_curve_x: list[float]            # Координаты X кривой PDF
    pdf_curve_y: list[float]            # Координаты Y кривой PDF


class CorrelationRequest(BaseModel):
    """Запрос на расчёт матрицы корреляций."""
    file_id: str | None = None          # ID файла в серверном кэше
    data: list[dict] | None = None      # Строки таблицы (fallback)
    method: Literal["pearson", "spearman"] = "pearson"  # Метод корреляции


class CorrelationResponse(BaseModel):
    """Ответ /api/stats/correlation — матрица корреляций."""
    method: str                         # Использованный метод
    columns: list[str]                  # Названия столбцов (оси матрицы)
    matrix: list[list[float]]           # Матрица значений корреляций


# ---------- Проверка гипотез — экспертная система (/api/hypothesis/*) ----------

class AssumptionsCheck(BaseModel):
    """
    Результат проверки предпосылок (assumptions) для выбора статистического критерия.

    Содержит флаги нормальности обеих выборок и равенства дисперсий,
    а также точные значения статистик и p-value тестов-предпосылок,
    чтобы пользователь мог обосновать выбор метода в тексте ВКР.

    Тест нормальности выбирается адаптивно:
        N < 8    — тест не проводится (поля = None).
        8 ≤ N ≤ 300 — Шапиро-Уилк.
        N > 300  — Д'Агостино-Пирсон.
    """
    is_norm_a: bool                     # Нормальность выборки A (p ≥ α)
    is_norm_b: bool                     # Нормальность выборки B (p ≥ α)
    shapiro_a_stat: float | None        # Статистика теста нормальности для A (None если N < 8)
    shapiro_a_p: float | None           # p-значение теста нормальности для A (None если N < 8)
    shapiro_b_stat: float | None        # Статистика теста нормальности для B (None если N < 8)
    shapiro_b_p: float | None           # p-значение теста нормальности для B (None если N < 8)
    norm_test_name: str | None = None   # "Шапиро-Уилк", "Д'Агостино-Пирсон" или None (N < 8)
    equal_variances: bool               # Равенство дисперсий (тест Левене, p ≥ α)
    levene_stat: float | None           # Статистика теста Левене (None если N < 8)
    levene_p: float | None              # p-значение теста Левене (None если N < 8)


class SampleDescriptive(BaseModel):
    """Описательные статистики одной выборки (для контекста сравнения)."""
    n: int                              # Размер выборки
    mean: float                         # Среднее арифметическое
    std: float                          # Стандартное отклонение


class ComparisonResult(BaseModel):
    """
    Полный результат сравнения двух числовых выборок.

    Передаёт на фронтенд не только итоговые цифры, но и полный путь
    принятия решения (decision_chain), по которому экспертная система
    выбрала конкретный статистический критерий.
    """
    test_name: str                      # Название теста на русском
    statistic: float                    # Значение тестовой статистики
    p_value: float                      # p-значение
    effect_size: float | None           # d Коэна (t-тесты) или ранг-бисериальная r (Манна-Уитни)
    effect_size_interpretation: str | None  # Текстовая интерпретация размера эффекта
    effect_size_metric: str = "cohens_d"   # "cohens_d" или "rank_biserial"
    assumptions: AssumptionsCheck        # Результат проверки предпосылок
    decision_chain: list[str]           # Путь принятия решения (пошаговое описание)
    conclusion: str                     # Готовый вывод на русском языке
    sample_a: SampleDescriptive         # Описательные статистики выборки A
    sample_b: SampleDescriptive         # Описательные статистики выборки B


class ComparisonRequest(BaseModel):
    """Запрос на сравнение двух числовых выборок."""
    file_id: str | None = None          # ID файла в серверном кэше
    data: list[dict] | None = None      # Строки таблицы (fallback)
    column_a: str                       # Имя столбца первой выборки
    column_b: str                       # Имя столбца второй выборки


class CategoricalResult(BaseModel):
    """
    Результат анализа связи двух категориальных переменных.

    Включает таблицу сопряжённости, критерий χ² Пирсона и коэффициент
    V Крамера с текстовой интерпретацией силы связи.
    """
    test_name: str                      # "Критерий хи-квадрат Пирсона"
    chi2_stat: float                    # Значение статистики χ²
    p_value: float                      # p-значение
    degrees_of_freedom: int             # Число степеней свободы
    cramers_v: float                    # Коэффициент V Крамера
    strength_interpretation: str        # Интерпретация силы связи (по шкале Коэна)
    cochran_warning: str | None = None  # Предупреждение о нарушении условия Кохрана
    conclusion: str                     # Готовый вывод на русском языке
    n_observations: int                 # Общее число наблюдений
    # Таблица сопряжённости для отображения на фронтенде
    contingency_index: list[str]        # Строки (категории переменной A)
    contingency_columns: list[str]      # Столбцы (категории переменной B)
    contingency_values: list[list[int]] # Матрица частот


class CategoricalRequest(BaseModel):
    """Запрос на анализ связи категориальных переменных."""
    file_id: str | None = None          # ID файла в серверном кэше
    data: list[dict] | None = None      # Строки таблицы (fallback)
    column_a: str                       # Имя первой категориальной переменной
    column_b: str                       # Имя второй категориальной переменной


# ---------- Временные ряды (/api/timeseries/*) ----------

class TimeSeriesValidateRequest(BaseModel):
    """Запрос на валидацию временного ряда."""
    file_id: str | None = None          # ID файла в серверном кэше
    data: list[dict] | None = None      # Строки таблицы (fallback)
    date_column: str                    # Имя колонки с датами
    value_column: str                   # Имя колонки с числовыми значениями


class TimeSeriesValidateResponse(BaseModel):
    """Ответ валидации временного ряда — диагностика «здоровья» ряда."""
    is_valid: bool                      # Пригоден ли ряд для анализа
    date_column: str                    # Имя колонки дат
    value_column: str                   # Имя колонки значений
    total_points: int                   # Общее число точек
    date_range_start: str | None        # Начало диапазона дат
    date_range_end: str | None          # Конец диапазона дат
    inferred_freq: str | None           # Определённая частота
    freq_description: str               # Описание частоты на русском
    is_regular: bool                    # Регулярный ли ряд
    gaps_count: int                     # Количество пропусков в датах
    missing_values: int                 # Пропуски NaN в значениях
    suggestion: str | None              # Рекомендация


class TimeSeriesResampleRequest(BaseModel):
    """Запрос на ресемплирование временного ряда."""
    file_id: str | None = None          # ID файла в серверном кэше
    data: list[dict] | None = None      # Строки таблицы (fallback)
    date_column: str                    # Имя колонки с датами
    value_column: str                   # Имя колонки со значениями
    freq: str = "D"                     # Целевая частота (D, W, MS, h ...)
    agg_func: Literal["mean", "sum", "median", "min", "max"] = "mean"
    fill_method: Literal["interpolate", "zero", "ffill"] = "interpolate"


class TimeSeriesResampleResponse(BaseModel):
    """Ответ после ресемплирования временного ряда."""
    data: list[dict]                    # Ресемплированные данные
    rows_before: int                    # Строк до
    rows_after: int                     # Строк после
    freq: str                           # Применённая частота
    freq_description: str               # Описание частоты
    agg_func: str                       # Функция агрегации
    fill_method: str                    # Метод заполнения
    filled_count: int                   # Количество заполненных пропусков


class StationarityRequest(BaseModel):
    """Запрос на тест стационарности (ADF)."""
    file_id: str | None = None          # ID файла в серверном кэше
    data: list[dict] | None = None      # Строки таблицы (fallback)
    column: str                         # Имя числовой колонки


class StationarityResponse(BaseModel):
    """Ответ теста стационарности (расширенный тест Дики-Фуллера)."""
    test_name: str                      # Название теста
    adf_statistic: float                # ADF-статистика
    p_value: float                      # p-значение
    used_lag: int                       # Использованных лагов
    n_observations: int                 # Число наблюдений
    critical_values: dict[str, float]   # Критические значения (1%, 5%, 10%)
    is_stationary: bool                 # True если ряд стационарен
    conclusion: str                     # Текстовый вывод


# ---------- Прогнозирование временных рядов (/api/forecast/*) ----------

class ForecastMetrics(BaseModel):
    """Метрики качества прогнозной модели."""
    mae: float                          # Средняя абсолютная ошибка
    rmse: float                         # Корень среднеквадратичной ошибки
    mape: float                         # Средняя абсолютная процентная ошибка (%)


class ForecastRequest(BaseModel):
    """Запрос на прогнозирование временного ряда."""
    model_config = {"protected_namespaces": ()}
    file_id: str | None = None          # ID файла в серверном кэше
    data: list[dict] | None = None      # Строки таблицы (fallback)
    date_column: str                    # Имя колонки с датами
    value_column: str                   # Имя колонки с числовыми значениями
    steps: int                          # Горизонт прогноза (количество шагов)
    model_type: Literal["arima", "hw"]  # Тип модели
    confidence_level: float = 0.95      # Уровень доверия для доверительного интервала


class ForecastResponse(BaseModel):
    """Ответ с результатами прогнозирования."""
    historical_dates: list[str]         # Даты исторических наблюдений
    historical_values: list[float]      # Значения исторических наблюдений
    forecast_dates: list[str]           # Даты прогнозных точек
    forecast_values: list[float]        # Прогнозные значения
    ci_lower: list[float] | None       # Нижняя граница доверительного интервала
    ci_upper: list[float] | None       # Верхняя граница доверительного интервала
    metrics: ForecastMetrics            # Метрики качества
    aic: float | None                   # AIC (только для ARIMA)
    explanation_text: str               # Текстовое обоснование модели


# ---------- Сравнение двух датасетов (/api/compare/*) ----------


class DatasetCompareRequest(BaseModel):
    """Запрос на сравнение двух датасетов."""
    file_id_a: str | None = None        # ID первого файла в серверном кэше
    file_id_b: str | None = None        # ID второго файла в серверном кэше
    data_a: list[dict] | None = None    # Строки первого датасета (fallback)
    data_b: list[dict] | None = None    # Строки второго датасета (fallback)
    id_column: str | None = None        # Колонка-идентификатор для парного сравнения


class ColumnComparisonResult(BaseModel):
    """Результат сравнения одной числовой колонки между двумя датасетами."""
    column: str                         # Имя колонки
    n_a: int                            # Количество наблюдений в датасете A
    n_b: int                            # Количество наблюдений в датасете B
    mean_a: float                       # Среднее в датасете A
    mean_b: float                       # Среднее в датасете B
    std_a: float                        # СКО в датасете A
    std_b: float                        # СКО в датасете B
    delta: float                        # Разность средних (B − A)
    delta_percent: float | None         # Изменение в процентах (None если mean_a == 0)
    test_name: str | None               # Название использованного теста
    statistic: float | None             # Значение тестовой статистики
    p_value: float | None               # p-значение
    cohens_d: float | None              # Размер эффекта (d Коэна)
    # --- Дрифт распределения (PSI + KS-тест) ---
    psi: float                          # Population Stability Index
    psi_interpretation: str             # "Нет дрифта" / "Умеренный дрифт" / "Значительный дрифт"
    ks_stat: float                      # Статистика KS-теста (Колмогорова-Смирнова)
    ks_p_value: float                   # p-значение KS-теста
    is_shape_drifted: bool              # True если ks_p_value < 0.05 или PSI ≥ 0.2
    # --- Парное сравнение ---
    is_paired: bool                     # True если использован парный тест
    paired_test_name: str | None        # Название парного теста или None
    verdict: str                        # Вердикт на русском языке
    p_value_corrected: float | None = None  # p-значение после FDR-коррекции (Benjamini-Hochberg)


# --- Категориальный дрифт ---

class CategoricalDriftResult(BaseModel):
    """Результат анализа дрифта одной категориальной колонки."""
    column: str                         # Имя колонки
    chi2_stat: float                    # Статистика хи-квадрат
    chi2_p_value: float                 # p-значение хи-квадрат теста
    cramers_v: float                    # Коэффициент V Крамера
    is_drifted: bool                    # True если p < alpha (после FDR-коррекции)
    base_proportions: dict[str, float]  # Доли категорий в базовом датасете
    compare_proportions: dict[str, float]  # Доли категорий в сравниваемом датасете
    cochran_warning: str | None = None  # Предупреждение о нарушении условия Кохрана
    p_value_corrected: float | None = None  # p-значение после FDR-коррекции (Benjamini-Hochberg)


class CategoricalDrift(BaseModel):
    """Сводный отчёт о дрифте категориальных колонок."""
    columns: list[CategoricalDriftResult]
    correction_method: str | None = None        # Метод коррекции множественных сравнений


# --- Структурный отчёт (Data Drift) ---

class StructureReport(BaseModel):
    """Отчёт о структурных изменениях между двумя датасетами."""
    rows_base: int                      # Строк в базовом датасете
    rows_compare: int                   # Строк в сравниваемом датасете
    rows_delta: int                     # Разница (compare − base)
    rows_delta_percent: float | None    # Изменение в % (None если base == 0)
    added_columns: list[str]            # Колонки, появившиеся в compare
    removed_columns: list[str]          # Колонки, пропавшие в compare
    type_changed_columns: dict[str, dict[str, str]]  # {col: {base: dtype, compare: dtype}}


# --- Отчёт о качестве данных (Data Quality) ---

class ColumnQualityReport(BaseModel):
    """Отчёт о качестве данных одной общей колонки."""
    column: str                         # Имя колонки
    missing_base_pct: float             # % пропусков в базовом датасете
    missing_compare_pct: float          # % пропусков в сравниваемом датасете
    missing_delta_pct: float            # Изменение % пропусков (compare − base)
    quality_degraded: bool              # True если пропусков стало больше на >5 п.п.
    new_categories: list[str]           # Новые значения категориальной колонки


class QualityReport(BaseModel):
    """Сводный отчёт о качестве данных по всем общим колонкам."""
    columns: list[ColumnQualityReport]


# --- Статистическое сравнение (обёртка существующего) ---

class StatisticalComparison(BaseModel):
    """Результаты статистического сравнения числовых колонок."""
    common_columns: list[str]                   # Совпавшие числовые колонки
    results: list[ColumnComparisonResult]        # Результаты по каждой колонке
    correction_method: str | None = None        # Метод коррекции множественных сравнений


# --- Полный ответ ---

class DatasetCompareResponse(BaseModel):
    """Полный отчёт сравнения двух датасетов: структура + качество + статистика + категориальный дрифт."""
    structure_report: StructureReport           # Структурные изменения
    quality_report: QualityReport               # Качество данных
    statistical_comparison: StatisticalComparison  # Статистическое сравнение числовых колонок
    categorical_drift: CategoricalDrift         # Дрифт категориальных колонок


# ---------- Экспорт PDF-отчёта (/api/export/pdf) ----------


class ReportHypothesisParams(BaseModel):
    """Параметры блока проверки гипотез для отчёта."""
    column_a: str
    column_b: str


class ReportRegressionParams(BaseModel):
    """Параметры блока регрессии для отчёта."""
    target_column: str
    feature_columns: list[str]


class ReportComparisonParams(BaseModel):
    """Параметры блока сравнения датасетов для отчёта."""
    file_id_b: str | None = None
    data_b: list[dict] | None = None
    id_column: str | None = None


class ReportRequest(BaseModel):
    """Запрос на генерацию PDF-отчёта. Включаются только переданные блоки."""
    file_id: str                                           # ID основного датасета в кэше
    filename: str = "dataset"                              # Имя файла для заголовка отчёта
    include_stats: bool = False                            # Блок описательной статистики
    hypothesis_params: ReportHypothesisParams | None = None  # Блок проверки гипотез
    regression_params: ReportRegressionParams | None = None  # Блок регрессии
    comparison_params: ReportComparisonParams | None = None  # Блок сравнения датасетов
