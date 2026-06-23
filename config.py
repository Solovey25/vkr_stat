"""
config.py — Централизованная конфигурация приложения.

Все параметры можно переопределить через переменные окружения.
"""

from __future__ import annotations

import os

# ── API ──────────────────────────────────────────────────────────
API_HOST: str = os.getenv("API_HOST", "127.0.0.1")
API_PORT: int = int(os.getenv("API_PORT", "8001"))
API_WORKERS: int = int(os.getenv("API_WORKERS", "1"))

# ── Уровень значимости ──────────────────────────────────────────
DEFAULT_ALPHA: float = float(os.getenv("DEFAULT_ALPHA", "0.05"))

# ── VIF (мультиколлинеарность) ──────────────────────────────────
VIF_THRESHOLD: float = float(os.getenv("VIF_THRESHOLD", "10.0"))

# ── PSI (Population Stability Index) ────────────────────────────
PSI_MODERATE: float = float(os.getenv("PSI_MODERATE", "0.1"))
PSI_SIGNIFICANT: float = float(os.getenv("PSI_SIGNIFICANT", "0.2"))

# ── Размер эффекта (Cohen's d) ──────────────────────────────────
EFFECT_NEGLIGIBLE: float = float(os.getenv("EFFECT_NEGLIGIBLE", "0.2"))
EFFECT_SMALL: float = float(os.getenv("EFFECT_SMALL", "0.5"))
EFFECT_MEDIUM: float = float(os.getenv("EFFECT_MEDIUM", "0.8"))

# ── Размер эффекта (rank-biserial r) ────────────────────────────
RANK_BISERIAL_LARGE: float = float(os.getenv("RANK_BISERIAL_LARGE", "0.5"))

# ── Cramér's V ──────────────────────────────────────────────────
CRAMERS_V_STRONG: float = float(os.getenv("CRAMERS_V_STRONG", "0.5"))
