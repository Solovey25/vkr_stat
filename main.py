"""
main.py — Точка входа FastAPI-сервера.

Веб-приложение для статистического анализа и прогнозирования.
Предоставляет REST API для:
    1. Загрузки CSV и получения описательных статистик.
    2. Сравнения двух выборок (автовыбор теста).
    3. Построения линейной регрессии.

Запуск:
    uvicorn main:app --reload --port 8001
"""

import io
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse, StreamingResponse

from schemas import (
    AnalyzeRegressionRequest,
    AnalyzeRegressionResponse,
    CorrelationRequest,
    CorrelationResponse,
    DistributionFitRequest,
    DistributionFitResponse,
    ExtendedStatsRequest,
    ExtendedStatsResponse,
    OutliersInfoResponse,
    OutliersRequest,
    RemoveOutliersRequest,
    RemoveOutliersResponse,
    ScaleRequest,
    ScaleResponse,
    EncodeRequest,
    EncodeResponse,
    CategoricalDrift,
    CategoricalDriftResult,
    CategoricalRequest,
    CategoricalResult,
    ColumnComparisonResult,
    ColumnQualityReport,
    ComparisonRequest,
    ComparisonResult,
    DatasetCompareRequest,
    DatasetCompareResponse,
    QualityReport,
    StatisticalComparison,
    StructureReport,
    SanitizeMissingRequest,
    SanitizeMissingResponse,
    StationarityRequest,
    StationarityResponse,
    TimeSeriesResampleRequest,
    TimeSeriesResampleResponse,
    TimeSeriesValidateRequest,
    TimeSeriesValidateResponse,
    UploadResponse,
    ForecastRequest,
    ForecastResponse,
    ReportRequest,
)
from services.dataframe_cache import cache as df_cache
from services.data_loader_service import DataLoaderService
from services.data_sanitizer_service import DataSanitizerService
from services.statistics_analyzer_service import StatisticsAnalyzerService
from services.hypothesis_engine_service import HypothesisEngineService
from services.regression_service import RegressionService
from services.plot_service import regression_plot_to_json, build_regression_plot
from services.report_service import PDFReportService
from services.time_series_service import TimeSeriesService
from services.comparative_service import ComparativeService
from services.forecasting_service import ForecastingService

# ---------- Инициализация приложения ----------

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Статистический анализ и прогнозирование",
    description="API для ВКР: загрузка данных, описательная статистика, "
                "проверка гипотез и линейная регрессия.",
    version="0.1.0",
)

app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Слишком много запросов. Попробуйте позже."},
    )


# ---------- Вспомогательные функции ----------


def _resolve_dataframe(body) -> pd.DataFrame:
    """Извлекает DataFrame из file_id (кэш) или body.data (fallback).

    Приоритет: file_id > data. Если ни то ни другое — HTTP 422.
    """
    if getattr(body, "file_id", None):
        df = df_cache.get(body.file_id)
        if df is None:
            raise HTTPException(
                status_code=422,
                detail="Файл не найден в кэше. Загрузите файл повторно.",
            )
        return df

    if getattr(body, "data", None) is not None:
        try:
            return pd.DataFrame(body.data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Ошибка создания таблицы: {e}")

    raise HTTPException(
        status_code=422,
        detail="Необходимо передать file_id или data.",
    )


# ---------- Эндпоинты ----------


@app.post(
    "/api/upload-file",
    response_model=UploadResponse,
    summary="Загрузка и анализ файла",
)
@limiter.limit("30/minute")
async def upload_file(
    request: Request,  # для rate limiter
    file: UploadFile = File(...),
    sheet_name: Optional[str] = Form(default=None),
):
    """
    Загружает файл (CSV, TXT, XLSX, XLS), автоматически определяет
    кодировку и разделитель, выполняет валидацию и возвращает
    метаданные, информацию о столбцах и превью первых 10 строк.

    Поддерживаемые форматы: .csv, .txt, .xlsx, .xls.
    Максимальный размер файла: 100 МБ.

    Параметры:
        file       — загружаемый файл.
        sheet_name — имя листа Excel (необязательно; по умолчанию — первый лист).

    В случае ошибки (пустой файл, неподдерживаемый формат, отсутствие
    числовых столбцов) возвращает HTTP 422 с описанием на русском языке.
    """
    # Читаем байты загруженного файла и гарантируем закрытие дескриптора
    try:
        file_bytes = await file.read()
    finally:
        await file.close()

    # Загружаем и валидируем данные через сервис
    try:
        service = DataLoaderService(file.filename, file_bytes, sheet_name=sheet_name)
        result = service.load()
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    metadata = result["metadata"]

    # Кэшируем DataFrame для последующих запросов по file_id
    file_id = df_cache.put(result["df"])

    return UploadResponse(
        file_id=file_id,
        metadata=metadata,
        column_info=metadata["column_info"],
        preview=metadata["preview"],
    )


# ---------- Эндпоинты предобработки данных ----------


@app.post(
    "/api/sanitize/missing",
    response_model=SanitizeMissingResponse,
    summary="Обработка пропусков",
)
async def sanitize_missing(body: SanitizeMissingRequest):
    """
    Обрабатывает пропуски (NaN) в данных.

    Поддерживает два метода:
        - "drop"  — удаление строк с пропусками.
        - "fill"  — заполнение пропусков (mean/median/most_frequent).

    Принимает JSON с данными и параметрами, возвращает очищенный датасет.
    """
    try:
        df = _resolve_dataframe(body)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка создания таблицы: {e}")

    rows_before = len(df)

    if body.method == "drop":
        result_df, affected = DataSanitizerService.drop_missing(df, columns=body.columns)
        method_desc = "Удаление строк с пропусками"
    else:
        result_df, affected = DataSanitizerService.fill_missing(
            df, strategy=body.strategy, columns=body.columns,
        )
        method_desc = f"Заполнение пропусков (стратегия: {body.strategy})"

    # Обновляем кэш если данные были из кэша
    if body.file_id:
        df_cache.update(body.file_id, result_df)

    return SanitizeMissingResponse(
        data=result_df.where(result_df.notna(), None).to_dict(orient="records"),
        rows_before=rows_before,
        rows_after=len(result_df),
        affected_count=affected,
        method=method_desc,
    )


@app.post(
    "/api/sanitize/outliers",
    response_model=OutliersInfoResponse,
    summary="Информация о выбросах",
)
async def sanitize_outliers_info(body: OutliersRequest):
    """
    Рассчитывает информацию о выбросах по методу IQR для указанных столбцов.

    Возвращает границы [Q1 − 1.5·IQR, Q3 + 1.5·IQR], количество и процент
    выбросов для каждого числового столбца. Данные не изменяются.
    """
    try:
        df = _resolve_dataframe(body)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка создания таблицы: {e}")

    outliers = DataSanitizerService.get_outliers_info(df, columns=body.columns)

    return OutliersInfoResponse(outliers=outliers)


@app.post(
    "/api/sanitize/remove-outliers",
    response_model=RemoveOutliersResponse,
    summary="Удаление выбросов",
)
async def sanitize_remove_outliers(body: RemoveOutliersRequest):
    """
    Удаляет строки, содержащие выбросы (по методу IQR),
    в указанных числовых столбцах.

    Возвращает очищенный датасет и количество удалённых строк.
    """
    try:
        df = _resolve_dataframe(body)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка создания таблицы: {e}")

    rows_before = len(df)
    result_df, removed = DataSanitizerService.remove_outliers(df, columns=body.columns)

    if body.file_id:
        df_cache.update(body.file_id, result_df)

    return RemoveOutliersResponse(
        data=result_df.where(result_df.notna(), None).to_dict(orient="records"),
        rows_before=rows_before,
        rows_after=len(result_df),
        removed_count=removed,
    )


@app.post(
    "/api/sanitize/scale",
    response_model=ScaleResponse,
    summary="Масштабирование признаков",
)
async def sanitize_scale(body: ScaleRequest):
    """
    Масштабирует указанные числовые столбцы.

    Параметры (в теле запроса):
        method — "standard" (Z-score, по умолчанию) или "minmax" ([0, 1]).
    """
    try:
        df = _resolve_dataframe(body)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка создания таблицы: {e}")

    if body.method == "minmax":
        result_df = DataSanitizerService.scale_minmax(df, body.columns)
    else:
        result_df = DataSanitizerService.scale_standard(df, body.columns)

    if body.file_id:
        df_cache.update(body.file_id, result_df)

    return ScaleResponse(
        data=result_df.where(result_df.notna(), None).to_dict(orient="records"),
    )


@app.post(
    "/api/sanitize/encode",
    response_model=EncodeResponse,
    summary="Кодирование категориальных столбцов",
)
async def sanitize_encode(body: EncodeRequest):
    """
    Кодирует текстовые (категориальные) столбцы числовыми кодами
    через pd.factorize для включения в регрессионную модель.

    Возвращает закодированные данные и маппинг {код → исходное значение}.
    """
    try:
        df = _resolve_dataframe(body)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка создания таблицы: {e}")

    for col in body.columns:
        if col not in df.columns:
            raise HTTPException(
                status_code=422,
                detail=f"Столбец «{col}» не найден в данных.",
            )

    result_df, mapping = DataSanitizerService.encode_categorical_columns(
        df, body.columns,
    )

    if body.file_id:
        df_cache.update(body.file_id, result_df)

    return EncodeResponse(
        data=result_df.where(result_df.notna(), None).to_dict(orient="records"),
        mapping=mapping,
    )


# ---------- Эндпоинты расширенной статистики ----------

# Словарь перевода названий распределений на русский язык
_DIST_NAMES_RU: dict[str, str] = {
    "norm": "Нормальное",
    "lognorm": "Логнормальное",
    "expon": "Экспоненциальное",
    "poisson": "Пуассона",
}


@app.post(
    "/api/stats/extended",
    response_model=ExtendedStatsResponse,
    summary="Расширенные описательные статистики",
)
async def stats_extended(body: ExtendedStatsRequest):
    """
    Рассчитывает расширенные описательные статистики для всех числовых столбцов:
    среднее, медиана, СКО, асимметрия (Skewness), эксцесс (Kurtosis), SEM и др.

    Принимает JSON с данными, возвращает метрики по каждому столбцу.
    """
    try:
        df = _resolve_dataframe(body)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка создания таблицы: {e}")

    stats = StatisticsAnalyzerService.compute_extended_stats(df)

    return ExtendedStatsResponse(columns=stats)


@app.post(
    "/api/stats/fit-distribution",
    response_model=DistributionFitResponse,
    summary="Подбор распределения (MLE + KS-тест)",
)
async def stats_fit_distribution(body: DistributionFitRequest):
    """
    Подбирает наилучшее теоретическое распределение для указанной числовой колонки
    методом максимального правдоподобия (MLE) с оценкой по критерию
    Колмогорова-Смирнова.

    Возвращает лучшее распределение, его параметры, p-value и координаты
    кривой PDF для наложения на гистограмму.
    """
    try:
        df = _resolve_dataframe(body)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка создания таблицы: {e}")

    if body.column not in df.columns:
        raise HTTPException(
            status_code=422,
            detail=f"Столбец «{body.column}» не найден в данных.",
        )

    if not pd.api.types.is_numeric_dtype(df[body.column]):
        raise HTTPException(
            status_code=422,
            detail=f"Столбец «{body.column}» не является числовым.",
        )

    # Подбираем лучшее распределение
    fit_result = StatisticsAnalyzerService.fit_best_distribution(df[body.column])

    # Генерируем кривую PDF для лучшего распределения
    pdf_x: list[float] = []
    pdf_y: list[float] = []
    if fit_result["best_distribution"] is not None:
        curve = StatisticsAnalyzerService.generate_pdf_curve(
            df[body.column],
            fit_result["best_distribution"],
            fit_result["best_params"],
        )
        pdf_x = curve["x"]
        pdf_y = curve["y"]

    # Переводим название распределения на русский
    best_name_ru = _DIST_NAMES_RU.get(
        fit_result["best_distribution"] or "", fit_result["best_distribution"]
    )

    return DistributionFitResponse(
        best_distribution=fit_result["best_distribution"],
        best_distribution_ru=best_name_ru,
        best_params=fit_result["best_params"],
        best_ks_statistic=fit_result["best_ks_statistic"],
        best_p_value=fit_result["best_p_value"],
        all_results=fit_result["all_results"],
        pdf_curve_x=pdf_x,
        pdf_curve_y=pdf_y,
    )


@app.post(
    "/api/stats/correlation",
    response_model=CorrelationResponse,
    summary="Матрица корреляций",
)
async def stats_correlation(body: CorrelationRequest):
    """
    Рассчитывает матрицу корреляций для всех числовых столбцов
    методом Пирсона или Спирмена.

    Возвращает названия столбцов и двумерную матрицу значений
    для построения тепловой карты на фронтенде.
    """
    try:
        df = _resolve_dataframe(body)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка создания таблицы: {e}")

    result = StatisticsAnalyzerService.compute_correlation_matrix(df, method=body.method)

    return CorrelationResponse(**result)


# ---------- Эндпоинты экспертной системы проверки гипотез ----------


@app.post(
    "/api/inference/compare",
    response_model=ComparisonResult,
    summary="Сравнение двух числовых выборок (экспертная система)",
)
async def inference_compare(body: ComparisonRequest):
    """
    Сравнивает две числовые выборки с автоматическим выбором критерия.

    Экспертная система проверяет нормальность (Шапиро-Уилк) и равенство
    дисперсий (Левене), затем выбирает оптимальный тест:
        - t-тест Стьюдента (нормальные, равные дисперсии)
        - t-тест Уэлча (нормальные, неравные дисперсии)
        - U-критерий Манна-Уитни (ненормальные данные)

    Дополнительно рассчитывает размер эффекта (d Коэна) для t-тестов.
    """
    try:
        df = _resolve_dataframe(body)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка создания таблицы: {e}")

    # Проверяем наличие столбцов
    for col_name in (body.column_a, body.column_b):
        if col_name not in df.columns:
            raise HTTPException(
                status_code=422,
                detail=f"Столбец «{col_name}» не найден в данных.",
            )

    # Проверяем, что оба столбца числовые
    for col_name in (body.column_a, body.column_b):
        if not pd.api.types.is_numeric_dtype(df[col_name]):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Столбец «{col_name}» не является числовым. "
                    "Для сравнения двух выборок оба столбца должны быть числовыми. "
                    "Для категориальных данных используйте /api/inference/categorical."
                ),
            )

    try:
        result = HypothesisEngineService.compare_two_groups(df[body.column_a], df[body.column_b])
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return ComparisonResult(
        test_name=result["test_name"],
        statistic=result["statistic"],
        p_value=result["p_value"],
        effect_size=result["cohens_d"],
        effect_size_interpretation=result["effect_size_interpretation"],
        effect_size_metric=result.get("effect_size_metric", "cohens_d"),
        assumptions={
            "is_norm_a": result["normality_a"],
            "is_norm_b": result["normality_b"],
            "shapiro_a_stat": result["shapiro_a_stat"],
            "shapiro_a_p": result["shapiro_a_p"],
            "shapiro_b_stat": result["shapiro_b_stat"],
            "shapiro_b_p": result["shapiro_b_p"],
            "norm_test_name": result.get("norm_test_name"),
            "equal_variances": result["equal_variance"],
            "levene_stat": result["levene_stat"],
            "levene_p": result["levene_p"],
        },
        decision_chain=result["decision_path"],
        conclusion=result["conclusion"],
        sample_a={"n": result["sample_a_n"], "mean": result["sample_a_mean"], "std": result["sample_a_std"]},
        sample_b={"n": result["sample_b_n"], "mean": result["sample_b_mean"], "std": result["sample_b_std"]},
    )


@app.post(
    "/api/inference/categorical",
    response_model=CategoricalResult,
    summary="Анализ связи категориальных переменных",
)
async def inference_categorical(body: CategoricalRequest):
    """
    Анализирует связь между двумя категориальными переменными.

    Строит таблицу сопряжённости, применяет критерий хи-квадрат Пирсона
    и рассчитывает коэффициент V Крамера для оценки силы связи.
    """
    try:
        df = _resolve_dataframe(body)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка создания таблицы: {e}")

    # Проверяем наличие столбцов
    for col_name in (body.column_a, body.column_b):
        if col_name not in df.columns:
            raise HTTPException(
                status_code=422,
                detail=f"Столбец «{col_name}» не найден в данных.",
            )

    # Предупреждение: числовой столбец для категориального анализа
    for col_name in (body.column_a, body.column_b):
        if pd.api.types.is_numeric_dtype(df[col_name]):
            nunique = df[col_name].nunique()
            if nunique > 20:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Столбец «{col_name}» является числовым и содержит "
                        f"{nunique} уникальных значений. Для категориального анализа "
                        "столбец должен содержать дискретные категории. "
                        "Для сравнения числовых выборок используйте /api/inference/compare."
                    ),
                )

    try:
        result = HypothesisEngineService.analyze_categorical_association(df[body.column_a], df[body.column_b])
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return CategoricalResult(
        test_name=result["test_name"],
        chi2_stat=result["chi2_statistic"],
        p_value=result["p_value"],
        degrees_of_freedom=result["degrees_of_freedom"],
        cramers_v=result["cramers_v"],
        strength_interpretation=result["cramers_v_interpretation"],
        cochran_warning=result.get("cochran_warning"),
        conclusion=result["conclusion"],
        n_observations=result["n_observations"],
        contingency_index=result["crosstab_index"],
        contingency_columns=result["crosstab_columns"],
        contingency_values=result["crosstab_values"],
    )


# ---------- Эндпоинты временных рядов ----------


@app.post(
    "/api/timeseries/validate",
    response_model=TimeSeriesValidateResponse,
    summary="Валидация временного ряда",
)
async def timeseries_validate(body: TimeSeriesValidateRequest):
    """
    Проверяет пригодность данных для анализа временных рядов.

    Определяет частоту ряда (pd.infer_freq), подсчитывает пропуски в датах,
    формирует диагностический отчёт с рекомендациями.
    """
    try:
        df = _resolve_dataframe(body)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка создания таблицы: {e}")

    try:
        result = TimeSeriesService.validate_time_series(
            df, body.date_column, body.value_column,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return TimeSeriesValidateResponse(**{
        k: v for k, v in result.items() if k != "sorted_data"
    })


@app.post(
    "/api/timeseries/resample",
    response_model=TimeSeriesResampleResponse,
    summary="Ресемплирование временного ряда",
)
async def timeseries_resample(body: TimeSeriesResampleRequest):
    """
    Ресемплирует временной ряд по указанной частоте с выбранной
    функцией агрегации и методом заполнения пропусков.
    """
    try:
        df = _resolve_dataframe(body)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка создания таблицы: {e}")

    try:
        result = TimeSeriesService.resample_data(
            df,
            date_col=body.date_column,
            value_col=body.value_column,
            freq=body.freq,
            agg_func=body.agg_func,
            fill_method=body.fill_method,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return TimeSeriesResampleResponse(**result)


@app.post(
    "/api/timeseries/stationarity",
    response_model=StationarityResponse,
    summary="Тест стационарности (ADF)",
)
async def timeseries_stationarity(body: StationarityRequest):
    """
    Выполняет расширенный тест Дики-Фуллера (ADF) для проверки
    стационарности временного ряда — обязательное условие для ARIMA.
    """
    try:
        df = _resolve_dataframe(body)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка создания таблицы: {e}")

    if body.column not in df.columns:
        raise HTTPException(
            status_code=422,
            detail=f"Столбец «{body.column}» не найден в данных.",
        )

    if not pd.api.types.is_numeric_dtype(df[body.column]):
        raise HTTPException(
            status_code=422,
            detail=f"Столбец «{body.column}» не является числовым.",
        )

    try:
        result = HypothesisEngineService.test_stationarity(df[body.column])
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return StationarityResponse(**result)


# ---------- Эндпоинт сравнения двух датасетов ----------


@app.post(
    "/api/compare/datasets",
    response_model=DatasetCompareResponse,
    summary="Сравнение двух датасетов: структура, качество, статистика и категориальный дрифт",
)
async def compare_datasets(body: DatasetCompareRequest):
    """
    Комплексное сравнение двух датасетов. Ответ состоит из четырёх блоков:

    1. **structure_report** — структурные изменения:
       изменение числа строк, добавленные/удалённые колонки, смена типов данных.

    2. **quality_report** — качество данных:
       изменение процента пропусков (NaN) по каждой общей колонке,
       появление новых категорий в категориальных колонках.

    3. **statistical_comparison** — статистическое сравнение числовых колонок
       (включая PSI, KS-тест и парное сравнение при указании id_column).

    4. **categorical_drift** — дрифт категориальных колонок (хи-квадрат, V Крамера).

    Dataset A — базовый (точка отсчёта), Dataset B — сравниваемый.
    """
    # Резолвим оба датасета: file_id из кэша или data из тела
    try:
        if body.file_id_a:
            df_a = df_cache.get(body.file_id_a)
            if df_a is None:
                raise HTTPException(status_code=422, detail="Файл A не найден в кэше. Загрузите повторно.")
        elif body.data_a is not None:
            df_a = pd.DataFrame(body.data_a)
        else:
            raise HTTPException(status_code=422, detail="Необходимо передать file_id_a или data_a.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка создания таблицы A: {e}")

    try:
        if body.file_id_b:
            df_b = df_cache.get(body.file_id_b)
            if df_b is None:
                raise HTTPException(status_code=422, detail="Файл B не найден в кэше. Загрузите повторно.")
        elif body.data_b is not None:
            df_b = pd.DataFrame(body.data_b)
        else:
            raise HTTPException(status_code=422, detail="Необходимо передать file_id_b или data_b.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка создания таблицы B: {e}")

    # 1. Структурный анализ
    structure = ComparativeService.analyze_structural_changes(df_a, df_b)

    # 2. Анализ качества данных (по всем общим колонкам)
    all_common_columns = sorted(set(df_a.columns) & set(df_b.columns))
    quality_items = ComparativeService.analyze_quality_changes(
        df_a, df_b, all_common_columns,
    )

    # 3. Статистическое сравнение числовых колонок (PSI, KS, парный тест)
    stat_result = ComparativeService.compare_datasets(
        df_a, df_b, id_column=body.id_column,
    )

    # 4. Дрифт категориальных колонок (хи-квадрат)
    cat_drift_result = ComparativeService.compare_categorical_columns(df_a, df_b)

    return DatasetCompareResponse(
        structure_report=StructureReport(**structure),
        quality_report=QualityReport(
            columns=[ColumnQualityReport(**q) for q in quality_items],
        ),
        statistical_comparison=StatisticalComparison(
            common_columns=stat_result["common_columns"],
            results=[ColumnComparisonResult(**r) for r in stat_result["results"]],
            correction_method=stat_result.get("correction_method"),
        ),
        categorical_drift=CategoricalDrift(
            columns=[CategoricalDriftResult(**c) for c in cat_drift_result["items"]],
            correction_method=cat_drift_result.get("correction_method"),
        ),
    )


# ---------- /analyze/regression (JSON-вход) ----------

def _clean_dataframe(
    df: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, dict]:
    """
    Очищает DataFrame от NaN и inf в указанных столбцах.
    Возвращает (очищенный DataFrame, отчёт об очистке).
    """
    subset = df[columns].copy()
    original_rows = len(subset)

    # Считаем строки с inf (до замены)
    inf_mask = subset.isin([np.inf, -np.inf]).any(axis=1)
    dropped_inf = int(inf_mask.sum())

    # Заменяем inf на NaN, чтобы dropna убрал и те и другие
    subset.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Считаем строки с NaN (включая бывшие inf, которые уже посчитаны)
    nan_mask = subset.isna().any(axis=1)
    dropped_nan = int(nan_mask.sum()) - dropped_inf

    # Убираем все «грязные» строки
    clean = subset.dropna()
    cleaned_rows = len(clean)

    report = {
        "original_rows": original_rows,
        "cleaned_rows": cleaned_rows,
        "dropped_rows": original_rows - cleaned_rows,
        "dropped_inf": dropped_inf,
        "dropped_nan": max(dropped_nan, 0),
    }
    return clean, report


@app.post(
    "/api/analyze/regression",
    response_model=AnalyzeRegressionResponse,
    summary="Регрессионный анализ (JSON-вход)",
)
async def analyze_regression(body: AnalyzeRegressionRequest):
    """
    Принимает JSON с данными, очищает от NaN/inf, выполняет
    регрессионный анализ (OLS + sklearn) и возвращает
    структурированный ответ для фронтенда с прогнозной линией.
    """
    # 1. Собираем DataFrame из входных данных
    try:
        df = _resolve_dataframe(body)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Не удалось создать таблицу из переданных данных: {e}",
        )

    # 2. Проверяем наличие всех запрошенных столбцов
    all_columns = body.feature_columns + [body.target_column]
    missing = [c for c in all_columns if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Столбцы не найдены в данных: {missing}",
        )

    # 3. Проверяем, что столбцы числовые
    non_numeric = [
        c for c in all_columns if not pd.api.types.is_numeric_dtype(df[c])
    ]
    if non_numeric:
        raise HTTPException(
            status_code=422,
            detail=f"Столбцы должны быть числовыми: {non_numeric}",
        )

    # 4. Очищаем данные от NaN и inf
    clean_df, cleaning_report = _clean_dataframe(df, all_columns)

    if len(clean_df) < 3:
        raise HTTPException(
            status_code=422,
            detail="После очистки осталось менее 3 строк — анализ невозможен.",
        )

    # 5. Запускаем RegressionService
    service = RegressionService(
        clean_df,
        target=body.target_column,
        features=body.feature_columns,
    )
    result = service.run()

    # 6. Формируем прогнозную линию (координаты для графика).
    #    Для множественной регрессии — сортируем по предсказанным
    #    значениям, чтобы линия шла по возрастанию.
    predictions = result["sklearn"]["predictions"]
    target_values = result["sklearn"]["target_values"]

    # Собираем пары (actual, predicted) и сортируем по predicted
    pairs = sorted(
        zip(range(len(predictions)), target_values, predictions),
        key=lambda t: t[2],
    )

    regression_line = [
        {"x": float(idx), "y_actual": float(actual), "y_predicted": float(pred)}
        for idx, actual, pred in pairs
    ]

    # 7. Строим Plotly-график с 95% доверительным интервалом
    plot_json = regression_plot_to_json(
        clean_df,
        target=body.target_column,
        features=body.feature_columns,
    )

    return AnalyzeRegressionResponse(
        cleaning=cleaning_report,
        n_samples=result["n_samples"],
        small_sample=result["small_sample"],
        ols=result["ols"],
        sklearn=result["sklearn"],
        regression_line=regression_line,
        plot_json=plot_json,
    )


# ---------- Прогнозирование временных рядов ----------


@app.post(
    "/api/forecast/timeseries",
    response_model=ForecastResponse,
    summary="Прогнозирование временного ряда (ARIMA / Holt)",
)
@limiter.limit("10/minute")
async def forecast_timeseries(request: Request, body: ForecastRequest):
    """Строит прогноз временного ряда на заданный горизонт."""
    df = _resolve_dataframe(body)

    if body.date_column not in df.columns:
        raise HTTPException(status_code=422, detail=f"Колонка дат «{body.date_column}» не найдена.")
    if body.value_column not in df.columns:
        raise HTTPException(status_code=422, detail=f"Колонка значений «{body.value_column}» не найдена.")

    df[body.date_column] = pd.to_datetime(df[body.date_column], errors="coerce")
    df = df.dropna(subset=[body.date_column, body.value_column])
    df = df.sort_values(body.date_column)
    df = df.set_index(body.date_column)

    series = df[body.value_column].astype(float)

    if len(series) < 8:
        raise HTTPException(status_code=422, detail="Недостаточно данных для прогнозирования (нужно ≥ 8 точек).")

    # Пытаемся определить частоту и заполнить пропуски
    if series.index.freq is None:
        inferred = pd.infer_freq(series.index)
        if inferred:
            refreqed = series.asfreq(inferred)
            nan_ratio = refreqed.isna().sum() / len(refreqed) if len(refreqed) > 0 else 0
            if nan_ratio > 0.2:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Автоматическое определение частоты невозможно "
                        f"(после приведения к частоте «{inferred}» доля пропусков составила "
                        f"{nan_ratio:.0%}). Пожалуйста, выполните явное ресемплирование "
                        "во вкладке «Предобработка»."
                    ),
                )
            series = refreqed
            if series.isna().any():
                series = series.interpolate(method="linear").bfill().ffill()

    svc = ForecastingService()
    try:
        if body.model_type == "arima":
            result = svc.fit_predict_arima(series, steps=body.steps, confidence_level=body.confidence_level)
        else:
            result = svc.fit_predict_hw(series, steps=body.steps)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return ForecastResponse(
        historical_dates=[str(d) for d in series.index],
        historical_values=[float(v) for v in series.values],
        forecast_dates=result["forecast_dates"],
        forecast_values=result["forecast_values"],
        ci_lower=result["ci_lower"],
        ci_upper=result["ci_upper"],
        metrics=result["metrics"],
        aic=result["aic"],
        explanation_text=result["explanation_text"],
    )


# ---------- Экспорт PDF-отчёта ----------


@app.post(
    "/api/export/pdf",
    summary="Генерация PDF-отчёта (протокол анализа)",
    response_class=StreamingResponse,
)
async def export_pdf(body: ReportRequest):
    """
    Генерирует PDF-отчёт (протокол статистического анализа).

    Динамически включает блоки в зависимости от переданных параметров:
        1. Паспорт данных (всегда)
        2. Описательная статистика (include_stats=True)
        3. Проверка гипотез (hypothesis_params != None)
        4. Регрессионный анализ (regression_params != None)
        5. Сравнение датасетов (comparison_params != None)

    Возвращает PDF как StreamingResponse (application/pdf).
    """
    # 1. Извлекаем основной датасет из кэша
    main_df = df_cache.get(body.file_id)
    if main_df is None:
        raise HTTPException(
            status_code=422,
            detail="Файл не найден в кэше. Загрузите файл повторно.",
        )

    report = PDFReportService(filename=body.filename, df=main_df)

    # Блок 1: Паспорт данных (всегда)
    report.add_passport()

    # Блок 2: Описательная статистика
    if body.include_stats:
        stats = StatisticsAnalyzerService.compute_extended_stats(main_df)
        # Преобразуем ExtendedColumnStats в dict для report_service
        stats_dict = {}
        for col_name, col_stats in stats.items():
            if isinstance(col_stats, dict):
                stats_dict[col_name] = col_stats
            else:
                stats_dict[col_name] = col_stats.__dict__ if hasattr(col_stats, '__dict__') else col_stats
        report.add_statistics_section(stats_dict)

        # Подбор распределений для каждого числового столбца
        num_cols = main_df.select_dtypes(include="number").columns.tolist()
        for col in num_cols:
            try:
                fit_result = StatisticsAnalyzerService.fit_best_distribution(main_df[col])
                # Добавляем русское название
                fit_result["best_distribution_ru"] = _DIST_NAMES_RU.get(
                    fit_result.get("best_distribution") or "", fit_result.get("best_distribution")
                )
                report.add_distribution_info(col, fit_result)
            except Exception:
                pass

    # Блок 3: Проверка гипотез
    if body.hypothesis_params is not None:
        hp = body.hypothesis_params

        for col_name in (hp.column_a, hp.column_b):
            if col_name not in main_df.columns:
                raise HTTPException(
                    status_code=422,
                    detail=f"Столбец «{col_name}» не найден в данных.",
                )
            if not pd.api.types.is_numeric_dtype(main_df[col_name]):
                raise HTTPException(
                    status_code=422,
                    detail=f"Столбец «{col_name}» не является числовым.",
                )

        try:
            hyp_result = HypothesisEngineService.compare_two_groups(
                main_df[hp.column_a], main_df[hp.column_b],
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

        # Формируем dict для report_service
        report.add_hypothesis_section({
            "test_name": hyp_result["test_name"],
            "statistic": hyp_result["statistic"],
            "p_value": hyp_result["p_value"],
            "effect_size": hyp_result["cohens_d"],
            "effect_size_metric": hyp_result.get("effect_size_metric", "cohens_d"),
            "assumptions": {
                "norm_test_name": hyp_result.get("norm_test_name"),
                "shapiro_a_p": hyp_result.get("shapiro_a_p"),
                "shapiro_b_p": hyp_result.get("shapiro_b_p"),
                "levene_p": hyp_result.get("levene_p"),
                "equal_variances": hyp_result.get("equal_variance"),
            },
            "decision_chain": hyp_result.get("decision_path", []),
            "conclusion": hyp_result.get("conclusion", ""),
        })

    # Блок 4: Регрессионный анализ
    if body.regression_params is not None:
        rp = body.regression_params
        all_columns = rp.feature_columns + [rp.target_column]

        missing = [c for c in all_columns if c not in main_df.columns]
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"Столбцы не найдены в данных: {missing}",
            )

        non_numeric = [c for c in all_columns if not pd.api.types.is_numeric_dtype(main_df[c])]
        if non_numeric:
            raise HTTPException(
                status_code=422,
                detail=f"Столбцы должны быть числовыми: {non_numeric}",
            )

        clean_df, cleaning_report = _clean_dataframe(main_df, all_columns)

        if len(clean_df) < 3:
            raise HTTPException(
                status_code=422,
                detail="После очистки осталось менее 3 строк — регрессия невозможна.",
            )

        reg_service = RegressionService(
            clean_df, target=rp.target_column, features=rp.feature_columns,
        )
        reg_result = reg_service.run()
        reg_result["cleaning"] = cleaning_report

        # Строим Plotly-фигуру для вставки в PDF (PNG через kaleido)
        try:
            plot_fig = build_regression_plot(
                clean_df, target=rp.target_column, features=rp.feature_columns,
            )
        except Exception:
            plot_fig = None

        report.add_regression_section(reg_result, plot_fig=plot_fig)

    # Блок 5: Сравнение датасетов
    if body.comparison_params is not None:
        cp = body.comparison_params

        # Резолвим второй датасет
        if cp.file_id_b:
            df_b = df_cache.get(cp.file_id_b)
            if df_b is None:
                raise HTTPException(
                    status_code=422,
                    detail="Файл B не найден в кэше. Загрузите повторно.",
                )
        elif cp.data_b is not None:
            try:
                df_b = pd.DataFrame(cp.data_b)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Ошибка создания таблицы B: {e}")
        else:
            raise HTTPException(
                status_code=422,
                detail="Для блока сравнения необходимо передать file_id_b или data_b.",
            )

        # Структурный отчёт
        structure = ComparativeService.analyze_structural_changes(main_df, df_b)

        # Статистическое сравнение
        stat_result = ComparativeService.compare_datasets(
            main_df, df_b, id_column=cp.id_column,
        )

        # Категориальный дрифт
        cat_drift = ComparativeService.compare_categorical_columns(main_df, df_b)

        report.add_comparison_section({
            "structure_report": structure,
            "statistical_comparison": stat_result,
            "categorical_drift": {
                "columns": cat_drift.get("items", []),
                "correction_method": cat_drift.get("correction_method"),
            },
        })

    # Сборка и отправка PDF
    pdf_bytes = report.build()
    buffer = io.BytesIO(pdf_bytes)

    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="report_{body.file_id[:8]}.pdf"',
        },
    )


# ---------- Запуск сервера ----------

if __name__ == "__main__":
    import uvicorn
    from config import API_HOST, API_PORT, API_WORKERS

    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        workers=API_WORKERS,
    )
