"""
generate_demo_datasets.py — Генерация показательных датасетов для демонстрации
всех функций приложения статистического анализа.

Создаёт 5 датасетов:

1. demo_regression.csv — Сильные линейные зависимости, мультиколлинеарность,
   выбросы, пропуски. Для: регрессия, корреляция, описательная статистика,
   очистка данных, подбор распределений.

2. demo_comparison_a.csv / demo_comparison_b.csv — Парные датасеты со сдвигом
   средних, категориальным дрифтом и структурными изменениями.
   Для: сравнение датасетов (PSI, KS, t-test, χ², FDR).

3. demo_hypothesis.csv — Два столбца-выборки с чётким различием + два
   категориальных столбца со связью.
   Для: t-test / Mann-Whitney, χ², Cramér's V.

4. demo_timeseries.csv — Временной ряд с трендом, сезонностью и шумом.
   Для: валидация ВР, ресемплинг, стационарность (ADF), прогноз (ARIMA/Holt).
"""

import numpy as np
import pandas as pd

np.random.seed(42)


# =====================================================================
# 1. РЕГРЕССИЯ + КОРРЕЛЯЦИЯ + ОПИСАТЕЛЬНАЯ СТАТИСТИКА
# =====================================================================

def generate_regression_dataset(n=200):
    """
    Датасет «Продажи компании»: 200 записей.

    Целевая переменная: Revenue (выручка).
    Сильные зависимости:
      - Revenue ≈ 50 + 3.5·Advertising + 1.2·Employees - 0.8·Distance + шум
      - Advertising и Marketing_Budget коррелируют (r ≈ 0.92) — мультиколлинеарность
      - Experience сильно коррелирует с Employees (r ≈ 0.85)

    Специально добавлены:
      - 10 пропусков (NaN) в разных столбцах
      - 5 выбросов в Revenue (× 3)
      - Категориальный столбец Region (4 уровня)
      - Столбец Rating — нормальное распределение (для подбора)
      - Столбец Response_Time — логнормальное распределение (для подбора)
    """
    advertising = np.random.uniform(10, 100, n)
    marketing_budget = advertising * np.random.uniform(0.8, 1.3, n) + np.random.normal(0, 5, n)
    employees = np.random.randint(5, 80, n).astype(float)
    experience = employees * np.random.uniform(0.3, 0.6, n) + np.random.normal(0, 3, n)
    distance = np.random.uniform(1, 50, n)
    region = np.random.choice(["Москва", "Санкт-Петербург", "Новосибирск", "Екатеринбург"], n)

    # Целевая: чёткая линейная зависимость + шум
    revenue = (
        50
        + 3.5 * advertising
        + 1.2 * employees
        - 0.8 * distance
        + np.random.normal(0, 15, n)
    )

    # Нормальное распределение (для fit-distribution)
    rating = np.random.normal(4.0, 0.7, n).clip(1, 5)

    # Логнормальное распределение (для fit-distribution)
    response_time = np.random.lognormal(mean=2.0, sigma=0.5, size=n)

    df = pd.DataFrame({
        "Advertising": np.round(advertising, 1),
        "Marketing_Budget": np.round(marketing_budget, 1),
        "Employees": employees,
        "Experience_Years": np.round(experience, 1),
        "Distance_KM": np.round(distance, 1),
        "Region": region,
        "Rating": np.round(rating, 2),
        "Response_Time_Sec": np.round(response_time, 2),
        "Revenue": np.round(revenue, 2),
    })

    # Добавляем пропуски
    for col in ["Employees", "Distance_KM", "Rating"]:
        idx = np.random.choice(n, 3, replace=False)
        df.loc[idx, col] = np.nan

    # Добавляем выбросы в Revenue
    outlier_idx = np.random.choice(n, 5, replace=False)
    df.loc[outlier_idx, "Revenue"] = df.loc[outlier_idx, "Revenue"] * 3

    return df


# =====================================================================
# 2. СРАВНЕНИЕ ДАТАСЕТОВ (ДРИФТ)
# =====================================================================

def generate_comparison_datasets(n=150):
    """
    Два датасета «до/после» со сдвигом.

    Dataset A (базовый):
      - Price: N(100, 15)
      - Quantity: N(50, 10)
      - Category: {A: 40%, B: 35%, C: 25%}
      - Quality_Score: N(7.0, 1.5)

    Dataset B (сравниваемый — через 6 месяцев):
      - Price: N(110, 18)  — значимый рост средней (+10%)
      - Quantity: N(50.5, 10) — почти без изменений (p > 0.05)
      - Category: {A: 25%, B: 35%, C: 30%, D: 10%} — дрифт + новая категория
      - Quality_Score: N(6.5, 2.0) — лёгкое падение
      - Delivery_Days: новый столбец (структурное изменение)

    id_column для парного теста.
    """
    ids = [f"ORD-{i:04d}" for i in range(1, n + 1)]

    # Dataset A
    df_a = pd.DataFrame({
        "order_id": ids,
        "Price": np.round(np.random.normal(100, 15, n), 2),
        "Quantity": np.round(np.random.normal(50, 10, n), 1),
        "Category": np.random.choice(
            ["A", "B", "C"], n, p=[0.40, 0.35, 0.25]
        ),
        "Quality_Score": np.round(np.random.normal(7.0, 1.5, n).clip(1, 10), 1),
    })

    # Dataset B — со сдвигом
    df_b = pd.DataFrame({
        "order_id": ids,
        "Price": np.round(np.random.normal(110, 18, n), 2),  # сдвиг!
        "Quantity": np.round(np.random.normal(50.5, 10, n), 1),  # почти без изменений
        "Category": np.random.choice(
            ["A", "B", "C", "D"], n, p=[0.25, 0.35, 0.30, 0.10]  # дрифт!
        ),
        "Quality_Score": np.round(np.random.normal(6.5, 2.0, n).clip(1, 10), 1),
        "Delivery_Days": np.random.randint(1, 14, n),  # новый столбец
    })

    # Добавляем пропуски в B (ухудшение качества данных)
    for col in ["Price", "Quality_Score"]:
        idx = np.random.choice(n, 8, replace=False)
        df_b.loc[idx, col] = np.nan

    return df_a, df_b


# =====================================================================
# 3. ПРОВЕРКА ГИПОТЕЗ
# =====================================================================

def generate_hypothesis_dataset(n=100):
    """
    Датасет для демонстрации проверки гипотез.

    Числовые (для t-test / Mann-Whitney):
      - Score_Group_A: N(75, 10) — контрольная группа
      - Score_Group_B: N(82, 12) — экспериментальная (значимо выше, d ≈ 0.6)

    Категориальные (для χ² / Cramér's V):
      - Treatment: {Placebo, Drug_A, Drug_B}
      - Outcome: {Улучшение, Без_изменений, Ухудшение}
      - Связь: Drug_A даёт больше улучшений, Placebo — больше «без изменений»
    """
    score_a = np.round(np.random.normal(75, 10, n), 1)
    score_b = np.round(np.random.normal(82, 12, n), 1)

    # Категориальные с зависимостью
    treatments = []
    outcomes = []
    for _ in range(n):
        t = np.random.choice(["Placebo", "Drug_A", "Drug_B"], p=[0.33, 0.34, 0.33])
        treatments.append(t)
        if t == "Drug_A":
            outcomes.append(np.random.choice(
                ["Улучшение", "Без_изменений", "Ухудшение"], p=[0.60, 0.25, 0.15]
            ))
        elif t == "Drug_B":
            outcomes.append(np.random.choice(
                ["Улучшение", "Без_изменений", "Ухудшение"], p=[0.45, 0.35, 0.20]
            ))
        else:  # Placebo
            outcomes.append(np.random.choice(
                ["Улучшение", "Без_изменений", "Ухудшение"], p=[0.20, 0.50, 0.30]
            ))

    df = pd.DataFrame({
        "Score_Group_A": score_a,
        "Score_Group_B": score_b,
        "Treatment": treatments,
        "Outcome": outcomes,
    })

    return df


# =====================================================================
# 4. ВРЕМЕННОЙ РЯД
# =====================================================================

def generate_timeseries_dataset(n_days=365 * 2):
    """
    Временной ряд «Ежедневные продажи» за 2 года.

    Компоненты:
      - Линейный тренд: +0.15 в день
      - Сезонность: sin с периодом 365 дней (амплитуда 30)
      - Недельная сезонность: выходные ниже (множитель 0.7)
      - Белый шум: N(0, 8)

    Специально:
      - 5 пропусков (NaN) — для демонстрации валидации ВР
      - Нерегулярные интервалы (удалены 3 дня) — для gap detection
    """
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    t = np.arange(n_days)

    trend = 100 + 0.15 * t
    seasonal = 30 * np.sin(2 * np.pi * t / 365)
    weekly = np.where(pd.Series(dates).dt.dayofweek >= 5, 0.7, 1.0)
    noise = np.random.normal(0, 8, n_days)

    values = (trend + seasonal) * weekly + noise

    df = pd.DataFrame({
        "Date": dates,
        "Sales": np.round(values, 2),
    })

    # Добавляем пропуски
    nan_idx = np.random.choice(n_days, 5, replace=False)
    df.loc[nan_idx, "Sales"] = np.nan

    # Удаляем 3 дня (создаём gaps)
    drop_idx = np.random.choice(range(30, n_days - 30), 3, replace=False)
    df = df.drop(drop_idx).reset_index(drop=True)

    return df


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    print("Генерация датасетов...")

    reg = generate_regression_dataset()
    reg.to_csv("demo_regression.csv", index=False, encoding="utf-8-sig")
    print(f"  demo_regression.csv — {len(reg)} строк, {len(reg.columns)} столбцов")

    comp_a, comp_b = generate_comparison_datasets()
    comp_a.to_csv("demo_comparison_a.csv", index=False, encoding="utf-8-sig")
    comp_b.to_csv("demo_comparison_b.csv", index=False, encoding="utf-8-sig")
    print(f"  demo_comparison_a.csv — {len(comp_a)} строк")
    print(f"  demo_comparison_b.csv — {len(comp_b)} строк")

    hyp = generate_hypothesis_dataset()
    hyp.to_csv("demo_hypothesis.csv", index=False, encoding="utf-8-sig")
    print(f"  demo_hypothesis.csv — {len(hyp)} строк")

    ts = generate_timeseries_dataset()
    ts.to_csv("demo_timeseries.csv", index=False, encoding="utf-8-sig")
    print(f"  demo_timeseries.csv — {len(ts)} строк")

    print("\nГотово! Все файлы сохранены в текущей директории.")
    print("\nЧто демонстрировать с каждым файлом:")
    print("-" * 60)
    print("demo_regression.csv:")
    print("  -Описательная статистика (все числовые столбцы)")
    print("  -Подбор распределений: Rating (нормальное), Response_Time_Sec (логнормальное)")
    print("  -Корреляция: Advertising<->Revenue (r≈0.85), Advertising<->Marketing_Budget (r≈0.92)")
    print("  -Очистка: пропуски (Employees, Distance_KM, Rating), выбросы в Revenue")
    print("  -Регрессия: Revenue ~ Advertising + Employees + Distance_KM")
    print("  -VIF: Advertising + Marketing_Budget → VIF > 10 (мультиколлинеарность!)")
    print()
    print("demo_comparison_a.csv + demo_comparison_b.csv:")
    print("  -Сравнение датасетов (id_column = order_id)")
    print("  -Price: значимый рост (+10%), PSI > 0.1")
    print("  -Quantity: различия отсутствуют (p > 0.05)")
    print("  -Category: χ²-дрифт (новая категория D)")
    print("  -Структура: новый столбец Delivery_Days в B")
    print("  -Качество: пропуски в Price и Quality_Score в B")
    print("  -FDR-коррекция: видна разница raw vs corrected p-values")
    print()
    print("demo_hypothesis.csv:")
    print("  -Сравнение выборок: Score_Group_A vs Score_Group_B (d ≈ 0.6)")
    print("  -Категориальный анализ: Treatment × Outcome (V Крамера > 0.2)")
    print()
    print("demo_timeseries.csv:")
    print("  -Валидация: пропуски + gaps в датах")
    print("  -Ресемплинг: D → W или MS")
    print("  -Стационарность: ADF-тест (нестационарный из-за тренда)")
    print("  -Прогноз: ARIMA или Holt на 30 дней")
