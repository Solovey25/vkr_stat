"""
frontend/app.py — Клиентская часть на Streamlit.

Интерфейс веб-приложения для статистического анализа и прогнозирования.
Взаимодействует с FastAPI-бэкендом через HTTP-запросы.

Запуск:
    streamlit run frontend/app.py
"""

import io
import os
import uuid

import requests
import streamlit as st
import pandas as pd

from helpers import API_BASE_URL
from tabs import preprocessing, statistics, hypothesis, forecasting, comparison

# ---------- Настройки ----------

UPLOAD_URL = f"{API_BASE_URL}/upload-file"

st.set_page_config(
    page_title="Статистический анализ и прогнозирование",
    layout="wide",
)


# ---------- Аутентификация (опциональная) ----------


def _check_password() -> bool:
    """Проверяет пароль через st.secrets (если задан).

    Для активации создайте файл .streamlit/secrets.toml:
        [auth]
        password = "ваш_пароль"
    """
    try:
        expected = st.secrets["auth"]["password"]
    except (KeyError, FileNotFoundError):
        return True  # Аутентификация не настроена — пропускаем

    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if st.session_state["authenticated"]:
        return True

    pwd = st.text_input("Введите пароль для доступа:", type="password")
    if pwd == expected:
        st.session_state["authenticated"] = True
        st.rerun()
    elif pwd:
        st.error("Неверный пароль.")
    return False


if not _check_password():
    st.stop()


# ---------- Идентификатор сессии ----------

if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())


st.title("Веб-приложение для статистического анализа и прогнозирования")


# ---------- Локальный парсинг файла ----------


def _parse_file_locally(filename: str, file_bytes: bytes) -> pd.DataFrame:
    """
    Парсит файл локально на стороне фронтенда для получения полного DataFrame.

    Backend возвращает только метаданные и превью (10 строк), но остальным
    вкладкам (статистика, гипотезы, регрессия) нужен полный набор данных.
    Эта функция воспроизводит логику чтения DataLoaderService локально.
    """
    import chardet

    _, ext = os.path.splitext(filename.lower())

    if ext in {".csv", ".txt"}:
        detection = chardet.detect(file_bytes)
        encoding = detection.get("encoding") or "utf-8"

        try:
            text = file_bytes.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            text = file_bytes.decode("utf-8")

        return pd.read_csv(io.StringIO(text), sep=None, engine="python")

    elif ext in {".xlsx", ".xls"}:
        engine = "openpyxl" if ext == ".xlsx" else None
        return pd.read_excel(io.BytesIO(file_bytes), engine=engine)

    # Запасной вариант — попытка прочитать как CSV
    return pd.read_csv(io.BytesIO(file_bytes), sep=None, engine="python")


# ---------- Сайдбар: загрузка файла ----------

st.sidebar.header("Загрузка данных")

uploaded_file = st.sidebar.file_uploader(
    "Выберите файл с данными",
    type=["csv", "xlsx", "xls", "txt"],
    help="Поддерживаемые форматы: CSV, Excel (XLSX/XLS), TXT. "
         "Кодировка и разделитель определяются автоматически.",
)

if uploaded_file is not None:
    file_key = f"{uploaded_file.name}_{uploaded_file.size}"

    if st.session_state.get("_file_key") != file_key:
        file_bytes = uploaded_file.getvalue()
        file_size_mb = len(file_bytes) / (1024 * 1024)

        if file_size_mb > 20:
            st.sidebar.warning(
                f"Файл большого размера ({file_size_mb:.1f} МБ). "
                "Обработка может занять некоторое время."
            )

        with st.sidebar.status("Загрузка и анализ файла..."):
            response = requests.post(
                UPLOAD_URL,
                files={"file": (uploaded_file.name, file_bytes)},
                timeout=120,
            )

        if response.status_code == 200:
            result = response.json()

            st.session_state["metadata"] = result["metadata"]
            st.session_state["file_id"] = result.get("file_id")
            st.session_state["_file_key"] = file_key

            parsed_df = _parse_file_locally(uploaded_file.name, file_bytes)
            st.session_state["main_df"] = parsed_df
            st.session_state["original_df"] = parsed_df.copy()

            for key in ["reg_result", "reg_features", "reg_saved_features", "reg_target", "reg_saved_target",
                        "processing_log", "compare_result"]:
                st.session_state.pop(key, None)

            st.rerun()
        elif response.status_code == 422:
            detail = response.json().get("detail", "Неизвестная ошибка")
            st.sidebar.error(f"Ошибка: {detail}")
        else:
            st.sidebar.error("Не удалось загрузить файл. Проверьте, запущен ли сервер.")

# Кнопка отката правок
if st.session_state.get("original_df") is not None:
    if st.sidebar.button("Сбросить все правки", use_container_width=True):
        original = st.session_state["original_df"].copy()
        st.session_state["main_df"] = original
        st.session_state["processing_log"] = []
        for key in ["reg_result", "reg_features", "reg_saved_features", "reg_target", "reg_saved_target"]:
            st.session_state.pop(key, None)
        m = st.session_state.get("metadata")
        if m is not None:
            m["rows"] = len(original)
            m["columns_count"] = len(original.columns)
            m["numeric_columns"] = original.select_dtypes(include="number").columns.tolist()
            m["categorical_columns"] = original.select_dtypes(exclude=["number", "datetime"]).columns.tolist()
            m["datetime_columns"] = original.select_dtypes(include="datetime").columns.tolist()
        st.rerun()

# Кнопка очистки данных
if st.sidebar.button("Очистить данные", use_container_width=True):
    for key in ["main_df", "metadata", "_file_key", "original_df",
                "reg_result", "reg_features", "reg_saved_features", "reg_target", "reg_saved_target",
                "processing_log",
                "comp_df", "_comp_file_key", "compare_result"]:
        st.session_state.pop(key, None)
    st.rerun()

# ---------- Информационная панель ----------

df = st.session_state.get("main_df")
meta = st.session_state.get("metadata")

if df is not None and meta is not None:
    st.sidebar.success(f"Файл загружен: {meta['filename']}")

    col_s1, col_s2 = st.sidebar.columns(2)
    with col_s1:
        st.metric("Строки", meta["rows"], help="Количество строк (наблюдений) в загруженном датасете.")
    with col_s2:
        st.metric("Столбцы", meta["columns_count"], help="Общее количество столбцов (признаков) в датасете.")

    col_s3, col_s4 = st.sidebar.columns(2)
    with col_s3:
        st.metric("Числовые", len(meta["numeric_columns"]), help="Количество числовых столбцов (int, float), пригодных для статистического анализа.")
    with col_s4:
        encoding_display = meta.get("encoding") or "—"
        st.metric("Кодировка", encoding_display, help="Кодировка файла (UTF-8, cp1251 и т.д.), определённая автоматически.")

    datetime_cols = meta.get("datetime_columns", [])
    if datetime_cols:
        st.sidebar.info(
            f"Распознаны колонки-даты: **{', '.join(datetime_cols)}**"
        )

    ts_info_list = meta.get("time_series_info", [])
    for ts_info in ts_info_list:
        col_name = ts_info["column"]
        freq_desc = ts_info["freq_description"]
        is_regular = ts_info["is_regular"]
        gaps = ts_info["gaps_count"]
        freq = ts_info.get("freq")

        if is_regular:
            st.sidebar.success(
                f"**{col_name}**: регулярный ряд, "
                f"частота — {freq_desc} ({freq})"
            )
        else:
            warning_text = f"**{col_name}**: нерегулярный ряд"
            if freq is not None:
                warning_text += f", частота — {freq_desc} ({freq}), пропусков: {gaps}"
            else:
                warning_text += ", частота не определена"
            st.sidebar.warning(warning_text)
            suggestion = ts_info.get("suggestion")
            if suggestion:
                st.sidebar.caption(suggestion)

    sheet_names = meta.get("sheet_names", [])
    if len(sheet_names) > 1:
        active = meta.get("active_sheet", sheet_names[0])
        st.sidebar.info(
            f"Excel-листы ({len(sheet_names)}): {', '.join(sheet_names)}. "
            f"Активный: **{active}**"
        )

    with st.sidebar.expander("Превью данных (Dataset A)"):
        st.dataframe(df.head(10), use_container_width=True)

    # ---------- Сайдбар: режим сравнения ----------

    st.sidebar.divider()
    st.sidebar.header("Режим сравнения")

    compare_mode = st.sidebar.toggle(
        "Режим сравнения",
        value=st.session_state.get("compare_mode", False),
        key="compare_mode",
    )

    if compare_mode:
        comp_file = st.sidebar.file_uploader(
            "Загрузите второй датасет (B)",
            type=["csv", "xlsx", "xls", "txt"],
            key="comp_file_uploader",
            help="Второй файл для сравнения с основным (Dataset A).",
        )

        if comp_file is not None:
            comp_file_key = f"comp_{comp_file.name}_{comp_file.size}"
            if st.session_state.get("_comp_file_key") != comp_file_key:
                comp_bytes = comp_file.getvalue()
                comp_parsed = _parse_file_locally(comp_file.name, comp_bytes)
                st.session_state["comp_df"] = comp_parsed
                st.session_state["_comp_file_key"] = comp_file_key
                st.session_state["_comp_filename"] = comp_file.name
                st.session_state.pop("compare_result", None)
                try:
                    _comp_resp = requests.post(
                        UPLOAD_URL,
                        files={"file": (comp_file.name, comp_bytes)},
                        timeout=120,
                    )
                    if _comp_resp.status_code == 200:
                        st.session_state["file_id_b"] = _comp_resp.json().get("file_id")
                except Exception:
                    pass
                st.rerun()

        if st.session_state.get("comp_df") is not None:
            if st.sidebar.button("Поменять местами (A ↔ B)", use_container_width=True):
                tmp_df = st.session_state["main_df"].copy()
                st.session_state["main_df"] = st.session_state["comp_df"].copy()
                st.session_state["comp_df"] = tmp_df
                meta_name = st.session_state.get("metadata", {}).get("filename", "?")
                comp_name = st.session_state.get("_comp_filename", "?")
                if st.session_state.get("metadata"):
                    st.session_state["metadata"]["filename"] = comp_name
                st.session_state["_comp_filename"] = meta_name
                st.session_state.pop("compare_result", None)
                st.rerun()

            comp_df_sidebar = st.session_state["comp_df"]
            comp_name = st.session_state.get("_comp_filename", "?")

            st.sidebar.success(f"Файл B загружен: {comp_name}")

            cb1, cb2 = st.sidebar.columns(2)
            with cb1:
                st.metric("Строки (B)", len(comp_df_sidebar), help="Количество строк в сравниваемом датасете B.")
            with cb2:
                st.metric("Столбцы (B)", len(comp_df_sidebar.columns), help="Количество столбцов в датасете B.")

            cb3, cb4 = st.sidebar.columns(2)
            with cb3:
                st.metric(
                    "Числовые (B)",
                    len(comp_df_sidebar.select_dtypes(include="number").columns),
                    help="Количество числовых столбцов в датасете B.",
                )
            with cb4:
                st.metric(
                    "Категориальные (B)",
                    len(comp_df_sidebar.select_dtypes(include="object").columns),
                    help="Количество категориальных (текстовых) столбцов в датасете B.",
                )

            with st.sidebar.expander("Превью данных (Dataset B)"):
                st.dataframe(comp_df_sidebar.head(10), use_container_width=True)


# ---------- Сайдбар: экспорт PDF-отчёта ----------

if df is not None and meta is not None:
    st.sidebar.divider()
    st.sidebar.header("Экспорт отчёта")

    with st.sidebar.expander("Настройки PDF-отчёта", expanded=False):
        pdf_include_stats = st.checkbox(
            "Описательная статистика",
            value=True,
            key="pdf_stats",
            help="Включить таблицу расширенных статистик и подбор распределений.",
        )

        # Проверка гипотез
        num_cols = meta.get("numeric_columns", [])
        pdf_include_hyp = st.checkbox(
            "Проверка гипотез",
            value=False,
            key="pdf_hyp",
            help="Включить сравнение двух числовых столбцов.",
        )
        hyp_col_a, hyp_col_b = None, None
        if pdf_include_hyp and len(num_cols) >= 2:
            hyp_col_a = st.selectbox(
                "Столбец A (гипотезы)", num_cols, index=0, key="pdf_hyp_a",
            )
            hyp_col_b = st.selectbox(
                "Столбец B (гипотезы)", num_cols, index=min(1, len(num_cols) - 1), key="pdf_hyp_b",
            )
        elif pdf_include_hyp:
            st.caption("Нужно минимум 2 числовых столбца.")
            pdf_include_hyp = False

        # Регрессия
        pdf_include_reg = st.checkbox(
            "Регрессионный анализ",
            value=False,
            key="pdf_reg",
            help="Включить OLS-регрессию с графиком.",
        )
        reg_target, reg_features = None, []
        if pdf_include_reg and len(num_cols) >= 2:
            reg_target = st.selectbox(
                "Зависимая переменная (Y)", num_cols, index=0, key="pdf_reg_target",
            )
            available_features = [c for c in num_cols if c != reg_target]
            reg_features = st.multiselect(
                "Независимые переменные (X)", available_features,
                default=available_features[:1] if available_features else [],
                key="pdf_reg_features",
            )
        elif pdf_include_reg:
            st.caption("Нужно минимум 2 числовых столбца.")
            pdf_include_reg = False

        # Сравнение датасетов
        pdf_include_cmp = False
        comp_df_for_pdf = st.session_state.get("comp_df")
        file_id_b_for_pdf = st.session_state.get("file_id_b")
        if comp_df_for_pdf is not None:
            pdf_include_cmp = st.checkbox(
                "Сравнение датасетов",
                value=False,
                key="pdf_cmp",
                help="Включить анализ дрифта (PSI, KS, хи-квадрат).",
            )

    # Кнопка генерации
    if st.sidebar.button("Скачать PDF-отчёт", use_container_width=True, type="primary"):
        from helpers import safe_post, sanitize_obj, df_to_records

        payload: dict = {"file_id": st.session_state.get("file_id", "")}
        payload["filename"] = meta.get("filename", "dataset")
        payload["include_stats"] = pdf_include_stats

        if pdf_include_hyp and hyp_col_a and hyp_col_b:
            payload["hypothesis_params"] = {
                "column_a": hyp_col_a,
                "column_b": hyp_col_b,
            }

        if pdf_include_reg and reg_target and reg_features:
            payload["regression_params"] = {
                "target_column": reg_target,
                "feature_columns": reg_features,
            }

        if pdf_include_cmp and file_id_b_for_pdf:
            payload["comparison_params"] = {
                "file_id_b": file_id_b_for_pdf,
            }
        elif pdf_include_cmp and comp_df_for_pdf is not None:
            payload["comparison_params"] = {
                "data_b": df_to_records(comp_df_for_pdf),
            }

        with st.sidebar.status("Генерация PDF-отчёта..."):
            try:
                resp = requests.post(
                    f"{API_BASE_URL}/export/pdf",
                    json=sanitize_obj(payload),
                    timeout=120,
                )
                if resp.status_code == 200:
                    st.sidebar.download_button(
                        label="Сохранить PDF",
                        data=resp.content,
                        file_name="report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                elif resp.status_code == 422:
                    detail = resp.json().get("detail", "Ошибка генерации отчёта.")
                    st.sidebar.error(detail)
                else:
                    st.sidebar.error("Не удалось сгенерировать отчёт. Проверьте сервер.")
            except requests.exceptions.ConnectionError:
                st.sidebar.error("Нет подключения к серверу.")
            except Exception as exc:
                st.sidebar.error(f"Ошибка: {exc}")


# ---------- Вкладки ----------

# Инициализация лога действий
if "processing_log" not in st.session_state:
    st.session_state["processing_log"] = []

tab_preprocess, tab_stats, tab_hypothesis, tab_regression, tab_compare = st.tabs([
    "Предобработка",
    "Описательная статистика",
    "Сравнение выборок",
    "Прогнозирование",
    "Сравнение датасетов",
])

with tab_preprocess:
    if df is not None:
        preprocessing.render(df)
    else:
        st.info("Загрузите файл через боковую панель для начала предобработки.")

with tab_stats:
    if df is not None:
        statistics.render(df)
    else:
        st.info("Загрузите файл через боковую панель для начала анализа.")

with tab_hypothesis:
    if df is not None:
        hypothesis.render(df)
    else:
        st.info("Загрузите файл через боковую панель.")

with tab_regression:
    if df is not None:
        forecasting.render(df)
    else:
        st.info("Загрузите файл через боковую панель.")

with tab_compare:
    if df is not None:
        comparison.render(df)
    else:
        if not st.session_state.get("compare_mode", False):
            st.info(
                "Включите **«Режим сравнения»** в боковой панели, "
                "чтобы загрузить второй датасет и провести сравнительный анализ."
            )
        else:
            st.info("Загрузите основной файл (Dataset A) через боковую панель.")
