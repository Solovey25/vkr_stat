"""
dataframe_cache.py — In-memory TTL-кэш для DataFrame.

Хранит загруженные пользователем DataFrame в оперативной памяти с автоматическим
удалением по истечении TTL. Данные НЕ записываются на диск — приватность по умолчанию.

Потокобезопасный: использует threading.Lock для всех операций с хранилищем.
"""

from __future__ import annotations

import threading
import time
import uuid

import pandas as pd


class DataFrameCache:
    """In-memory TTL-кэш для DataFrame."""

    def __init__(self, ttl_seconds: int = 1800, max_items: int = 20) -> None:
        self._ttl = ttl_seconds
        self._max_items = max_items
        self._store: dict[str, dict] = {}  # {file_id: {"df": DataFrame, "ts": float}}
        self._lock = threading.Lock()

        # Фоновый поток очистки
        self._cleaner = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleaner.start()

    def put(self, df: pd.DataFrame) -> str:
        """Сохраняет DataFrame, возвращает file_id."""
        file_id = uuid.uuid4().hex
        with self._lock:
            # Если превышен лимит — удаляем самый старый
            if len(self._store) >= self._max_items:
                oldest = min(self._store, key=lambda k: self._store[k]["ts"])
                del self._store[oldest]
            self._store[file_id] = {"df": df, "ts": time.time()}
        return file_id

    def get(self, file_id: str) -> pd.DataFrame | None:
        """Возвращает DataFrame или None. Обновляет TTL при доступе."""
        with self._lock:
            entry = self._store.get(file_id)
            if entry is None:
                return None
            entry["ts"] = time.time()
            return entry["df"]

    def update(self, file_id: str, df: pd.DataFrame) -> None:
        """Заменяет DataFrame в кэше (после предобработки)."""
        with self._lock:
            if file_id in self._store:
                self._store[file_id] = {"df": df, "ts": time.time()}

    def delete(self, file_id: str) -> None:
        """Удаляет запись из кэша."""
        with self._lock:
            self._store.pop(file_id, None)

    def _cleanup_loop(self) -> None:
        """Фоновый поток: удаляет истёкшие записи каждые 60 секунд."""
        while True:
            time.sleep(60)
            now = time.time()
            with self._lock:
                expired = [
                    fid for fid, entry in self._store.items()
                    if now - entry["ts"] > self._ttl
                ]
                for fid in expired:
                    del self._store[fid]


# Глобальный singleton — используется всеми эндпоинтами
cache = DataFrameCache()
