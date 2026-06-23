"""
helpers.py — Общие утилиты фронтенда.

Содержит функции сериализации данных, безопасного HTTP-запроса,
обновления метаданных и ведения лога предобработки.
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ---------- Константы ----------

_host = os.getenv("API_HOST", "localhost")
_port = os.getenv("API_PORT", "8001")
API_BASE_URL = f"http://{_host}:{_port}/api"


# ---------- Сериализация ----------


def sanitize_value(v):
    """Приводит одно значение к JSON-безопасному типу Python."""
    if v is None:
        return None
    if isinstance(v, float):
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    if isinstance(v, (np.floating,)):
        if np.isnan(v) or np.isinf(v):
            return None
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, (pd.Timestamp,)):
        return v.isoformat() if not pd.isna(v) else None
    if isinstance(v, pd.Timedelta):
        return str(v)
    if isinstance(v, np.ndarray):
        return [sanitize_value(x) for x in v.tolist()]
    try:
        if pd.isna(v):
            return None
    except (ValueError, TypeError):
        pass
    return v


def sanitize_obj(obj):
    """Рекурсивно очищает dict/list от нестандартных типов для JSON."""
    if isinstance(obj, dict):
        return {k: sanitize_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_obj(v) for v in obj]
    return sanitize_value(obj)


# ---------- HTTP ----------


def safe_post(url: str, payload: dict, timeout: int = 60, **kwargs) -> requests.Response:
    """requests.post с безопасной JSON-сериализацией (NaN/Inf → null, numpy → python)."""
    return requests.post(url, json=sanitize_obj(payload), timeout=timeout, **kwargs)


# ---------- DataFrame ↔ records ----------


def df_to_records(frame: pd.DataFrame) -> list[dict]:
    """Конвертирует DataFrame в list[dict], безопасный для JSON-сериализации."""
    tmp = frame.copy()
    for col in tmp.select_dtypes(include=["datetime", "datetimetz"]).columns:
        tmp[col] = tmp[col].dt.strftime("%Y-%m-%dT%H:%M:%S").where(tmp[col].notna())
    tmp = tmp.where(tmp.notna(), other=None)
    raw = tmp.to_dict(orient="records")
    return sanitize_obj(raw)


def data_payload() -> dict:
    """Возвращает {'file_id': ...} или {'data': ...} из session_state."""
    fid = st.session_state.get("file_id")
    if fid:
        return {"file_id": fid}
    df = st.session_state.get("main_df")
    if df is not None:
        return {"data": df_to_records(df)}
    return {}


# ---------- Метаданные и лог ----------


def add_log_entry(message: str) -> None:
    """Добавляет запись в лог действий предобработки с временной меткой."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state["processing_log"].append(f"{timestamp} — {message}")


def refresh_metadata() -> None:
    """Пересчитывает метаданные в session_state из текущего main_df."""
    current_df = st.session_state.get("main_df")
    current_meta = st.session_state.get("metadata")
    if current_df is None or current_meta is None:
        return

    current_meta["rows"] = len(current_df)
    current_meta["columns_count"] = len(current_df.columns)
    current_meta["numeric_columns"] = (
        current_df.select_dtypes(include="number").columns.tolist()
    )
    current_meta["categorical_columns"] = (
        current_df.select_dtypes(exclude=["number", "datetime"]).columns.tolist()
    )
    current_meta["datetime_columns"] = (
        current_df.select_dtypes(include="datetime").columns.tolist()
    )

    st.session_state["metadata"] = current_meta
