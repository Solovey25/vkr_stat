"""
frontend/explanations.py — База знаний для контекстной интерпретации аналитики.

Модуль содержит:
A. COLUMN_HELP   — статические подсказки для заголовков таблиц (column_config)
B. help_*()      — генераторы help-текста для st.metric (динамические)
C. LATEX          — формулы для st.latex в экспандерах
D. format_decision_step() — emoji-маркеры для цепочки решений
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════
# A.  COLUMN_HELP — статические подсказки для column_config
# ═══════════════════════════════════════════════════════════════════

COLUMN_HELP: dict[str, str] = {
    # Описательная статистика
    "СКО": (
        "Стандартное квадратическое отклонение (σ) — мера разброса значений "
        "вокруг среднего. Чем больше σ, тем сильнее данные «рассеяны»."
    ),
    "Асимметрия": (
        "Skewness — мера скошенности распределения. "
        "0 = симметрично, >0.5 = правый хвост длиннее, <−0.5 = левый хвост длиннее."
    ),
    "Эксцесс": (
        "Kurtosis (избыточный) — мера «остроты» пика распределения. "
        "0 = нормальное, >0 = островершинное (тяжёлые хвосты), <0 = плосковершинное."
    ),
    "SEM": (
        "Standard Error of the Mean — стандартная ошибка среднего. "
        "Показывает, насколько точно выборочное среднее оценивает "
        "истинное среднее генеральной совокупности. SEM = σ / √n."
    ),
    "Интерпретация": (
        "Краткая словесная характеристика формы распределения "
        "на основе значений асимметрии и эксцесса."
    ),
    # Коэффициенты регрессии
    "Стд. ошибка": (
        "Стандартная ошибка коэффициента — мера неопределённости оценки. "
        "Чем меньше ошибка относительно коэффициента, тем точнее оценка."
    ),
    "t-статистика": (
        "t = коэффициент / стд. ошибка. Проверяет гипотезу H₀: β = 0. "
        "|t| > 2 обычно указывает на значимость."
    ),
    "p-значение": (
        "Вероятность получить такой же или более экстремальный результат "
        "при условии, что H₀ истинна. p < 0.05 → фактор значим."
    ),
    "Значимость": (
        "Уровень звёздочек: *** p<0.001, ** p<0.01, * p<0.05. "
        "Без звёздочки — фактор статистически незначим."
    ),
    # Числовой дрифт
    "PSI": (
        "Population Stability Index — мера сдвига распределения. "
        "PSI < 0.1 = стабильно, 0.1–0.25 = умеренный сдвиг, > 0.25 = значительный."
    ),
    "PSI (оценка)": (
        "Текстовая интерпретация PSI: "
        "«Незначительный», «Умеренный» или «Значительный дрифт»."
    ),
    "KS p-value": (
        "p-value теста Колмогорова-Смирнова. "
        "Если p < 0.05, формы распределений статистически различаются."
    ),
    "Вердикт": (
        "Итоговая оценка: есть ли статистически значимые "
        "различия между базовым и сравниваемым набором данных."
    ),
    "Парный": (
        "Был ли применён парный статистический тест "
        "(по совпавшим ID-записям между датасетами)."
    ),
    # Категориальный дрифт
    "χ² стат.": (
        "Статистика хи-квадрат — мера расхождения наблюдаемых "
        "и ожидаемых частот. Чем больше χ², тем сильнее различие."
    ),
    "V Крамера": (
        "Cramér's V ∈ [0, 1] — нормированная мера связи для категорий. "
        "0 = нет связи, 1 = полная зависимость. V > 0.3 — заметная разница."
    ),
    "Дрифт": (
        "Обнаружен ли статистически значимый сдвиг "
        "в распределении категорий (p < 0.05)."
    ),
    # Качество данных
    "Деградация": (
        "Деградация = «Да», если доля пропусков в сравниваемом датасете "
        "выросла на ≥5 п.п. по сравнению с базовым."
    ),
}


# ═══════════════════════════════════════════════════════════════════
# B.  Генераторы help-текста для st.metric
# ═══════════════════════════════════════════════════════════════════

def _fmt(v, decimals=4) -> str:
    """Форматирует число или возвращает '—'."""
    if v is None:
        return "—"
    try:
        return f"{float(v):.{decimals}f}"
    except (TypeError, ValueError):
        return str(v)


# ---------- Общие ----------

def help_pvalue(p, context: str = "гипотезу", alpha: float = 0.05) -> str:
    if p is None:
        return "p-значение недоступно."
    p_f = float(p)
    verdict = "отвергаем H₀" if p_f < alpha else "не отвергаем H₀"
    return (
        f"p-value = {_fmt(p)} — вероятность получить такой же или более "
        f"экстремальный результат, если H₀ истинна.\n\n"
        f"При α = {alpha}: {verdict} ({context}).\n\n"
        f"Формула зависит от выбранного теста."
    )


# ---------- Регрессия ----------

def help_r2(r2) -> str:
    r2_f = float(r2)
    if r2_f >= 0.9:
        quality = "отличное"
    elif r2_f >= 0.7:
        quality = "хорошее"
    elif r2_f >= 0.5:
        quality = "умеренное"
    else:
        quality = "слабое"
    return (
        f"R² = {_fmt(r2)} — коэффициент детерминации.\n\n"
        f"Показывает долю дисперсии зависимой переменной, "
        f"объяснённую моделью.\n\n"
        f"R² = 1 − SS_res / SS_tot\n\n"
        f"Качество: {quality} ({r2_f:.0%} дисперсии объяснено)."
    )


def help_r2_adj(r2_adj, r2=None, n_feat=None) -> str:
    extra = ""
    if r2 is not None and r2_adj is not None:
        diff = float(r2) - float(r2_adj)
        if diff > 0.05:
            extra = (
                f"\n\nРазница R²−R²(скорр.) = {diff:.4f} — "
                "возможно, в модели есть лишние предикторы."
            )
    return (
        f"R²(скорр.) = {_fmt(r2_adj)} — скорректированный R².\n\n"
        f"Штрафует за число предикторов: "
        f"R²_adj = 1 − (1−R²)·(n−1)/(n−k−1)."
        f"{extra}"
    )


_IN_SAMPLE_NOTE = (
    "\n\nВнимание: метрика рассчитана на исторических (обучающих) данных. "
    "Она показывает, насколько хорошо модель описала прошлое, "
    "но реальная ошибка прогноза в будущем может быть выше."
)


def help_mae(mae) -> str:
    return (
        f"MAE = {_fmt(mae)} — средняя абсолютная ошибка.\n\n"
        f"MAE = (1/n) · Σ|yᵢ − ŷᵢ|\n\n"
        f"Показывает, на сколько в среднем прогноз отклоняется "
        f"от реального значения (в единицах зависимой переменной)."
        f"{_IN_SAMPLE_NOTE}"
    )


def help_rmse(rmse) -> str:
    return (
        f"RMSE = {_fmt(rmse)} — корень среднеквадратичной ошибки.\n\n"
        f"RMSE = √((1/n) · Σ(yᵢ − ŷᵢ)²)\n\n"
        f"Сильнее штрафует за большие ошибки, чем MAE."
        f"{_IN_SAMPLE_NOTE}"
    )


def help_mape(mape) -> str:
    if mape is None:
        return "MAPE недоступен."
    m = float(mape)
    if m < 10:
        quality = "высокая точность"
    elif m < 20:
        quality = "хорошая точность"
    elif m < 50:
        quality = "умеренная точность"
    else:
        quality = "низкая точность"
    return (
        f"MAPE = {_fmt(mape, 2)}% — средняя абсолютная процентная ошибка.\n\n"
        f"MAPE = (1/n) · Σ|yᵢ − ŷᵢ|/|yᵢ| · 100%\n\n"
        f"< 10% = высокая, 10–20% = хорошая, 20–50% = умеренная, > 50% = низкая.\n\n"
        f"Результат: {quality}."
        f"{_IN_SAMPLE_NOTE}"
    )


def help_f_stat(f, p) -> str:
    sig = "модель значима" if p is not None and float(p) < 0.05 else "модель НЕ значима"
    return (
        f"F = {_fmt(f)} — F-статистика (тест Фишера).\n\n"
        f"Проверяет H₀: все коэффициенты = 0.\n"
        f"p = {_fmt(p)} → {sig} (α=0.05)."
    )


def help_aic(aic) -> str:
    return (
        f"AIC = {_fmt(aic, 2)} — информационный критерий Акаике.\n\n"
        f"AIC = 2k − 2·ln(L)\n\n"
        f"Меньше = лучше. Используется для сравнения моделей "
        f"(не имеет абсолютной шкалы «хорошо/плохо»)."
    )


def help_durbin_watson(dw) -> str:
    if dw is None:
        return "Статистика Дарбина-Уотсона недоступна."
    dw_f = float(dw)
    if 1.5 <= dw_f <= 2.5:
        interp = "автокорреляция остатков не обнаружена (норма)"
    elif dw_f < 1.5:
        interp = "положительная автокорреляция остатков (проблема)"
    else:
        interp = "отрицательная автокорреляция остатков (редко)"
    return (
        f"DW = {_fmt(dw)} — статистика Дарбина-Уотсона.\n\n"
        f"Проверяет автокорреляцию остатков. Норма ≈ 2.\n"
        f"DW ∈ [0, 4]: <1.5 = полож. автокорр., >2.5 = отриц.\n\n"
        f"Результат: {interp}."
    )


def help_shapiro_residuals(w, p) -> str:
    if p is None:
        return "Тест Шапиро-Уилка не проводился."
    normal = "остатки нормальны" if float(p) >= 0.05 else "остатки НЕ нормальны"
    return (
        f"W = {_fmt(w)} — статистика Шапиро-Уилка для остатков.\n\n"
        f"Проверяет H₀: остатки распределены нормально.\n"
        f"p = {_fmt(p)} → {normal} (α=0.05).\n\n"
        f"Нормальность остатков — условие корректности OLS."
    )


# ---------- Сравнение выборок ----------

def help_cohens_d(d) -> str:
    if d is None:
        return "Размер эффекта недоступен."
    d_abs = abs(float(d))
    if d_abs < 0.2:
        size = "незначительный"
    elif d_abs < 0.5:
        size = "малый"
    elif d_abs < 0.8:
        size = "средний"
    else:
        size = "большой"
    return (
        f"d Коэна = {_fmt(d)} — стандартизированная разница средних.\n\n"
        f"d = (M₁ − M₂) / SD_pooled\n\n"
        f"|d| < 0.2 = незначит., 0.2–0.5 = малый, "
        f"0.5–0.8 = средний, > 0.8 = большой.\n\n"
        f"Результат: {size} эффект."
    )


def help_rank_biserial(r) -> str:
    if r is None:
        return "Размер эффекта недоступен."
    r_abs = abs(float(r))
    if r_abs < 0.1:
        size = "незначительный"
    elif r_abs < 0.3:
        size = "малый"
    elif r_abs < 0.5:
        size = "средний"
    else:
        size = "большой"
    return (
        f"Ранг-бисериальная r = {_fmt(r)} — размер эффекта "
        f"для непараметрического теста (Манна-Уитни).\n\n"
        f"r = 1 − 2U/(n₁·n₂)\n\n"
        f"|r| < 0.1 = незначит., 0.1–0.3 = малый, "
        f"0.3–0.5 = средний, > 0.5 = большой.\n\n"
        f"Результат: {size} эффект."
    )


def help_normality(is_norm: bool, test: str | None, p) -> str:
    if test is None:
        return "Тест нормальности не проводился (N < 8)."
    verdict = "данные нормальны" if is_norm else "данные НЕ нормальны"
    return (
        f"Тест: {test}, p = {_fmt(p)}.\n\n"
        f"H₀: выборка из нормального распределения.\n"
        f"p < 0.05 → отвергаем H₀.\n\n"
        f"Результат: {verdict}."
    )


def help_levene(equal_var: bool, p) -> str:
    if p is None:
        return "Тест Левене не проводился."
    verdict = "дисперсии равны" if equal_var else "дисперсии НЕ равны"
    return (
        f"Тест Левене, p = {_fmt(p)}.\n\n"
        f"H₀: дисперсии двух выборок равны.\n"
        f"p < 0.05 → отвергаем H₀ (неравные дисперсии → тест Уэлча).\n\n"
        f"Результат: {verdict}."
    )


def help_statistic(stat, test_name: str) -> str:
    return (
        f"Статистика теста «{test_name}» = {_fmt(stat)}.\n\n"
        f"Чем больше |значение|, тем сильнее различие между выборками."
    )


# ---------- Выбросы ----------

def help_outlier_count(n, pct) -> str:
    return (
        f"Найдено {n} выбросов ({_fmt(pct, 2)}% от общего числа наблюдений).\n\n"
        f"Выбросы определяются методом IQR: значения за пределами "
        f"[Q1 − 1.5·IQR, Q3 + 1.5·IQR]."
    )


def help_outlier_pct(pct) -> str:
    return (
        f"Процент выбросов = {_fmt(pct, 2)}%.\n\n"
        f"Если > 5% — стоит проверить данные на ошибки ввода."
    )


def help_quartile(value, which: str) -> str:
    desc = "25-й перцентиль (нижний квартиль)" if which == "Q1" else "75-й перцентиль (верхний квартиль)"
    return (
        f"{which} = {_fmt(value)} — {desc}.\n\n"
        f"25% данных {'ниже' if which == 'Q1' else 'выше'} этого значения."
    )


def help_iqr_bound(val, is_lower: bool) -> str:
    kind = "Нижняя" if is_lower else "Верхняя"
    formula = "Q1 − 1.5·IQR" if is_lower else "Q3 + 1.5·IQR"
    return (
        f"{kind} граница = {_fmt(val)}.\n\n"
        f"Формула: {formula}, где IQR = Q3 − Q1.\n"
        f"Значения за этой границей считаются выбросами."
    )


# ---------- Дрифт (числовой) ----------

def help_psi(psi, interp: str | None = None) -> str:
    if psi is None:
        return "PSI недоступен."
    psi_f = float(psi)
    if interp is None:
        if psi_f < 0.1:
            interp = "незначительный"
        elif psi_f < 0.25:
            interp = "умеренный"
        else:
            interp = "значительный"
    return (
        f"PSI = {_fmt(psi)} — Population Stability Index.\n\n"
        f"PSI = Σ (pᵢ − qᵢ) · ln(pᵢ/qᵢ)\n\n"
        f"< 0.1 = стабильно, 0.1–0.25 = умеренный сдвиг, "
        f"> 0.25 = значительный.\n\n"
        f"Оценка: {interp}."
    )


def help_ks(p) -> str:
    if p is None:
        return "KS p-value недоступен."
    verdict = "формы различаются" if float(p) < 0.05 else "формы схожи"
    return (
        f"KS p-value = {_fmt(p)} — тест Колмогорова-Смирнова.\n\n"
        f"Проверяет H₀: обе выборки из одного распределения.\n"
        f"p < 0.05 → формы распределений различаются.\n\n"
        f"Результат: {verdict}."
    )


def help_mean(val, label: str) -> str:
    return (
        f"Среднее ({label}) = {_fmt(val)}.\n\n"
        f"Арифметическое среднее: M = (1/n) · Σxᵢ."
    )


def help_delta(delta, pct=None) -> str:
    extra = f" ({_fmt(pct, 2)}%)" if pct is not None else ""
    return (
        f"Дельта = {_fmt(delta)}{extra} — разница средних (B − A).\n\n"
        f"Положительная дельта = рост, отрицательная = снижение."
    )


def help_verdict(verdict: str) -> str:
    return (
        f"Вердикт: «{verdict}».\n\n"
        f"Основан на комбинации PSI, KS-теста и статистического теста."
    )


def help_shape_drifted(is_drifted: bool) -> str:
    return (
        f"Форма дрифтовала: {'Да' if is_drifted else 'Нет'}.\n\n"
        f"Определяется по KS-тесту (p < 0.05)."
    )


# ---------- Категориальный дрифт ----------

def help_chi2(chi2, p, df=None) -> str:
    verdict = "различие значимо" if p is not None and float(p) < 0.05 else "различие НЕ значимо"
    df_part = f", df = {df}" if df is not None else ""
    return (
        f"χ² = {_fmt(chi2)} — статистика хи-квадрат{df_part}.\n\n"
        f"Сравнивает наблюдаемые частоты категорий "
        f"с ожидаемыми.\n"
        f"p = {_fmt(p)} → {verdict} (α=0.05)."
    )


def help_cramers_v(v) -> str:
    if v is None:
        return "V Крамера недоступен."
    v_f = float(v)
    if v_f < 0.1:
        strength = "связь отсутствует / пренебрежимо мала"
    elif v_f < 0.3:
        strength = "слабая связь"
    elif v_f < 0.5:
        strength = "умеренная связь"
    else:
        strength = "сильная связь"
    return (
        f"V Крамера = {_fmt(v)} ∈ [0, 1].\n\n"
        f"V = √(χ²/(n·(min(r,c)−1)))\n\n"
        f"Нормированная мера силы связи для категориальных "
        f"переменных.\n\n"
        f"Результат: {strength}."
    )


# ═══════════════════════════════════════════════════════════════════
# C.  LATEX — формулы для st.latex() в экспандерах
# ═══════════════════════════════════════════════════════════════════

LATEX: dict[str, str] = {
    "t_test": r"t = \frac{\bar{X}_1 - \bar{X}_2}{S_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}}",
    "welch": r"t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{S_1^2}{n_1} + \frac{S_2^2}{n_2}}}",
    "mann_whitney": r"U = n_1 n_2 + \frac{n_1(n_1+1)}{2} - R_1",
    "shapiro_wilk": r"W = \frac{\left(\sum_{i=1}^{n} a_i x_{(i)}\right)^2}{\sum_{i=1}^{n}(x_i - \bar{x})^2}",
    "levene": r"W = \frac{(N-k)}{(k-1)} \cdot \frac{\sum_{i=1}^{k} N_i (\bar{Z}_{i\cdot} - \bar{Z}_{\cdot\cdot})^2}{\sum_{i=1}^{k} \sum_{j=1}^{N_i} (Z_{ij} - \bar{Z}_{i\cdot})^2}",
    "cohens_d": r"d = \frac{\bar{X}_1 - \bar{X}_2}{S_p}, \quad S_p = \sqrt{\frac{(n_1-1)S_1^2 + (n_2-1)S_2^2}{n_1+n_2-2}}",
    "rank_biserial": r"r = 1 - \frac{2U}{n_1 \cdot n_2}",
    "chi_squared": r"\chi^2 = \sum_{i=1}^{r}\sum_{j=1}^{c} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}",
    "cramers_v": r"V = \sqrt{\frac{\chi^2}{n \cdot (\min(r, c) - 1)}}",
    "psi": r"PSI = \sum_{i=1}^{k} (p_i - q_i) \cdot \ln\!\left(\frac{p_i}{q_i}\right)",
    "r_squared": r"R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}",
    "f_stat": r"F = \frac{R^2 / k}{(1 - R^2) / (n - k - 1)}",
    "aic": r"AIC = 2k - 2\ln(\hat{L})",
    "durbin_watson": r"DW = \frac{\sum_{t=2}^{T}(e_t - e_{t-1})^2}{\sum_{t=1}^{T} e_t^2}",
    "iqr": r"IQR = Q_3 - Q_1, \quad \text{Выброс:}\; x < Q_1 - 1.5 \cdot IQR \;\text{или}\; x > Q_3 + 1.5 \cdot IQR",
    # Прогнозирование временных рядов
    "arima": r"ARIMA(p,d,q): \quad \Phi(B)(1-B)^d y_t = \Theta(B)\varepsilon_t",
    "mape": r"MAPE = \frac{1}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right| \cdot 100\%",
    "holt": r"\hat{y}_{t+h} = \ell_t + h \cdot b_t, \quad \ell_t = \alpha y_t + (1-\alpha)(\ell_{t-1}+b_{t-1}), \quad b_t = \beta(\ell_t - \ell_{t-1}) + (1-\beta)b_{t-1}",
}


# ═══════════════════════════════════════════════════════════════════
# D.  format_decision_step() — emoji-маркеры для цепочки решений
# ═══════════════════════════════════════════════════════════════════

_POSITIVE_MARKERS = ("нормальны", "нормально", "равны", "подтвержд", "принимаем")
_REDIRECT_MARKERS = ("→", "ненормальн", "неравны", "не нормальн", "не равны",
                     "переходим", "выбран", "используем")
_WARNING_MARKERS = ("мал", "Внимание", "Кохран", "принудительно",
                    "недостаточно", "предупрежд", "Warning")


def format_decision_step(step: str) -> str:
    """Добавляет emoji-маркер к шагу цепочки решений."""
    lower = step.lower()

    for marker in _WARNING_MARKERS:
        if marker.lower() in lower:
            return f"- ⚠️ {step}"

    for marker in _REDIRECT_MARKERS:
        if marker.lower() in lower:
            return f"- ⏭️ {step}"

    for marker in _POSITIVE_MARKERS:
        if marker.lower() in lower:
            return f"- ✅ {step}"

    return f"- {step}"
