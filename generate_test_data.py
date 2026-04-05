import pandas as pd
import numpy as np

def generate_vkr_test_data(n_samples=50):
    np.random.seed(42) # Чтобы данные всегда были одинаковыми
    
    # 1. Площадь квартиры (от 30 до 120 кв.м)
    area = np.random.uniform(30, 120, n_samples)
    
    # 2. Удаленность от центра (от 1 до 20 км)
    distance = np.random.uniform(1, 20, n_samples)
    
    # 3. Количество комнат (зависит от площади)
    rooms = np.round(area / 30 + np.random.normal(0, 0.5, n_samples))
    rooms = np.clip(rooms, 1, 5) # Ограничим от 1 до 5 комнат
    
    # 4. ЦЕЛЕВАЯ ПЕРЕМЕННАЯ: Цена (в миллионах)
    # Формула: Цена = 2.0 (база) + 0.15 * площадь - 0.3 * расстояние + шум
    noise = np.random.normal(0, 1.5, n_samples)
    price = 2.0 + (0.15 * area) - (0.3 * distance) + noise
    
    # Собираем в таблицу
    df = pd.DataFrame({
        'Square_Feet': np.round(area, 1),
        'Distance_KM': np.round(distance, 1),
        'Rooms': rooms.astype(int),
        'Price_MLN': np.round(price, 2)
    })
    
    # Сохраняем
    df.to_csv('real_estate_data.csv', index=False)
    print(f"Файл 'real_estate_data.csv' на {n_samples} записей успешно создан!")

if __name__ == "__main__":
    generate_vkr_test_data(50)